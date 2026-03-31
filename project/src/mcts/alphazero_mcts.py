"""PUCT-based MCTS implementation for AlphaZero-style Splendor agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Protocol, TYPE_CHECKING

import numpy as np

from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.action_space_generator import generate_all_legal_actions
from gym_splendor_code.envs.mechanics.game_settings import MAX_NUMBER_OF_MOVES, POINTS_TO_WIN
from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict

from mcts.action_indexer import StableActionIndexer
from nn.tensor_encoder import SplendorTensorEncoder

if TYPE_CHECKING:
    from nn.policy_value_net import SplendorPolicyValueNet


def clone_state(state: State) -> State:
    """Clone state via the canonical StateAsDict round-trip."""
    return StateAsDict(state).to_state()


class PolicyValueFunction(Protocol):
    """Callable policy/value evaluator for legal actions in a state."""

    def __call__(self, state: State, player_id: int, legal_actions: list[Action]) -> tuple[np.ndarray, float]:
        ...


class TorchPolicyValueFunction:
    """Adapter from torch network output to legal-action priors."""

    def __init__(
        self,
        encoder: SplendorTensorEncoder,
        model: "SplendorPolicyValueNet",
        device: str,
        action_indexer: StableActionIndexer,
    ) -> None:
        self.encoder = encoder
        self.model = model
        self.device = device
        self.action_indexer = action_indexer

    def __call__(self, state: State, player_id: int, legal_actions: list[Action]) -> tuple[np.ndarray, float]:
        import torch

        encoded = self.encoder.encode(state, player_id=player_id)
        state_tensor = torch.from_numpy(encoded).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits, value = self.model(state_tensor)

        logits_np = logits.squeeze(0).detach().cpu().numpy()
        value_scalar = float(value.squeeze(0).detach().cpu().item())

        legal_count = len(legal_actions)
        if legal_count == 0:
            return np.zeros(0, dtype=np.float32), value_scalar

        legal_indices = self.action_indexer.legal_indices(legal_actions)
        legal_logits = logits_np[legal_indices]
        legal_logits = legal_logits - np.max(legal_logits)
        priors = np.exp(legal_logits)
        priors = priors / max(1e-8, np.sum(priors))
        return priors.astype(np.float32), value_scalar


@dataclass
class MCTSNode:
    """Search node storing visit statistics and cached legal actions."""

    state: State
    to_play: int
    parent: MCTSNode | None = None
    parent_action_idx: int | None = None
    legal_actions: list[Action] = field(default_factory=list)
    priors: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    children: dict[int, MCTSNode] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0
    expanded: bool = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


@dataclass
class SearchResult:
    """Search output at root for action selection and training targets."""

    legal_actions: list[Action]
    policy: np.ndarray
    selected_action: Action
    root: MCTSNode


class AlphaZeroMCTS:
    """PUCT MCTS with root-noise and root-perspective backup values."""

    def __init__(
        self,
        policy_value_fn: PolicyValueFunction,
        num_simulations: int = 100,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        max_depth: int = MAX_NUMBER_OF_MOVES,
    ) -> None:
        self.policy_value_fn = policy_value_fn
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.max_depth = max_depth

    def search(self, root_state: State, temperature: float = 1.0) -> SearchResult:
        root_player = root_state.active_player_id
        root = MCTSNode(state=clone_state(root_state), to_play=root_player)

        _ = self._expand(root, is_root=True)
        if len(root.legal_actions) == 0:
            raise RuntimeError("No legal actions available at root.")

        for _ in range(self.num_simulations):
            path = [root]
            node = root
            depth = 0

            while node.expanded and len(node.legal_actions) > 0 and depth < self.max_depth:
                action_idx = self._select_child(node, root_player)
                child = node.children.get(action_idx)
                if child is None:
                    next_state = clone_state(node.state)
                    node.legal_actions[action_idx].execute(next_state)
                    child = MCTSNode(
                        state=next_state,
                        to_play=next_state.active_player_id,
                        parent=node,
                        parent_action_idx=action_idx,
                    )
                    node.children[action_idx] = child

                node = child
                path.append(node)
                depth += 1

                if self._is_terminal(node.state):
                    break

            value_root = self._evaluate_leaf(node, root_player, depth)
            self._backup(path, value_root)

        visit_counts = np.array(
            [root.children.get(i).visit_count if i in root.children else 0 for i in range(len(root.legal_actions))],
            dtype=np.float32,
        )
        policy = self._counts_to_policy(visit_counts, temperature)

        action_idx = int(np.random.choice(len(root.legal_actions), p=policy))
        return SearchResult(
            legal_actions=root.legal_actions,
            policy=policy,
            selected_action=root.legal_actions[action_idx],
            root=root,
        )

    def _expand(self, node: MCTSNode, is_root: bool) -> float:
        node.legal_actions = generate_all_legal_actions(node.state)
        priors, value = self.policy_value_fn(node.state, node.to_play, node.legal_actions)

        if len(node.legal_actions) == 0:
            node.priors = np.zeros(0, dtype=np.float32)
            node.expanded = True
            return value

        if is_root and len(priors) > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(priors)).astype(np.float32)
            priors = (1.0 - self.dirichlet_epsilon) * priors + self.dirichlet_epsilon * noise

        node.priors = priors.astype(np.float32)
        node.expanded = True
        return value

    def _select_child(self, node: MCTSNode, root_player: int) -> int:
        best_score = -np.inf
        best_idx = 0
        parent_visits = max(1, node.visit_count)

        for idx in range(len(node.legal_actions)):
            child = node.children.get(idx)
            q_value = child.q_value if child is not None else 0.0
            if node.to_play != root_player:
                q_value = -q_value

            child_visits = child.visit_count if child is not None else 0
            u_value = self.c_puct * node.priors[idx] * sqrt(parent_visits) / (1 + child_visits)
            score = q_value + u_value

            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx

    def _evaluate_leaf(self, node: MCTSNode, root_player: int, depth: int) -> float:
        if self._is_terminal(node.state) or depth >= self.max_depth:
            return self._terminal_value_from_root(node.state, root_player)

        value_current = self._expand(node, is_root=False)
        if node.to_play == root_player:
            return value_current
        return -value_current

    def _backup(self, path: list[MCTSNode], value_root: float) -> None:
        for node in path:
            node.visit_count += 1
            node.value_sum += value_root

    def _counts_to_policy(self, visit_counts: np.ndarray, temperature: float) -> np.ndarray:
        if np.sum(visit_counts) <= 0:
            return np.ones_like(visit_counts, dtype=np.float32) / len(visit_counts)

        if temperature <= 1e-6:
            policy = np.zeros_like(visit_counts, dtype=np.float32)
            policy[int(np.argmax(visit_counts))] = 1.0
            return policy

        scaled = np.power(visit_counts, 1.0 / temperature)
        denom = np.sum(scaled)
        if denom <= 0:
            return np.ones_like(visit_counts, dtype=np.float32) / len(visit_counts)
        return (scaled / denom).astype(np.float32)

    def _is_terminal(self, state: State) -> bool:
        points = [hand.number_of_my_points() for hand in state.list_of_players_hands]
        return max(points) >= POINTS_TO_WIN

    def _terminal_value_from_root(self, state: State, root_player: int) -> float:
        points = [hand.number_of_my_points() for hand in state.list_of_players_hands]
        root_points = points[root_player]
        opp_points = points[1 - root_player]

        if root_points > opp_points:
            return 1.0
        if root_points < opp_points:
            return -1.0
        return 0.0
