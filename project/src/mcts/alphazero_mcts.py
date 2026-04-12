"""PUCT-based MCTS implementation for AlphaZero-style Splendor agents.

Supports two terminal-handling modes:
- Legacy: uses POINTS_TO_WIN instant-end (backward compat).
- Adapter: uses SplendorPlanningAdapter with let_all_move round completion.

Reference: cestpasphoto/alpha-zero-general MCTS.py for FPU pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Optional, Protocol, TYPE_CHECKING

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
    from planning.adapter import SplendorPlanningAdapter


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
    # Optional planning adapter for let_all_move terminal semantics.
    adapter: Optional["SplendorPlanningAdapter"] = field(default=None, repr=False)

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
    """PUCT MCTS with root-noise and root-perspective backup values.

    Supports two modes:
    - Legacy (default): uses POINTS_TO_WIN instant-end terminal logic.
    - Adapter: uses SplendorPlanningAdapter with let_all_move semantics.
      Pass ``use_adapter=True`` to enable.

    FPU (First Play Urgency): when ``fpu_value`` is set, unvisited children
    use this pessimistic Q estimate instead of 0.0. Borrowed from
    cestpasphoto/alpha-zero-general MCTS.py.
    """

    def __init__(
        self,
        policy_value_fn: PolicyValueFunction,
        num_simulations: int = 100,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        max_depth: int = MAX_NUMBER_OF_MOVES,
        use_adapter: bool = False,
        fpu_value: Optional[float] = None,
    ) -> None:
        self.policy_value_fn = policy_value_fn
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.max_depth = max_depth
        self.use_adapter = use_adapter
        self.fpu_value = fpu_value  # None means use 0.0 (legacy behavior)

    # ------------------------------------------------------------------
    # Adapter-mode helpers
    # ------------------------------------------------------------------

    def _make_adapter(self, state: State) -> "SplendorPlanningAdapter":
        from planning.adapter import SplendorPlanningAdapter
        return SplendorPlanningAdapter(state=clone_state(state))

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, root_state: State, temperature: float = 1.0) -> SearchResult:
        root_player = root_state.active_player_id
        root_adapter = self._make_adapter(root_state) if self.use_adapter else None
        root = MCTSNode(
            state=root_adapter.state if root_adapter else clone_state(root_state),
            to_play=root_player,
            adapter=root_adapter,
        )

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
                    if self.use_adapter and node.adapter is not None:
                        child_adapter = node.adapter.clone()
                        child_adapter.step(node.legal_actions[action_idx])
                        child = MCTSNode(
                            state=child_adapter.state,
                            to_play=child_adapter.current_player,
                            parent=node,
                            parent_action_idx=action_idx,
                            adapter=child_adapter,
                        )
                    else:
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

                if self._is_terminal(node):
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
        if self.use_adapter and node.adapter is not None:
            node.legal_actions = node.adapter.legal_actions
        else:
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
        fpu = self.fpu_value if self.fpu_value is not None else 0.0

        for idx in range(len(node.legal_actions)):
            child = node.children.get(idx)
            q_value = child.q_value if child is not None else fpu
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
        if self._is_terminal(node) or depth >= self.max_depth:
            return self._terminal_value_from_root(node, root_player)

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

    def _is_terminal(self, node: MCTSNode) -> bool:
        if self.use_adapter and node.adapter is not None:
            return node.adapter.is_terminal()
        points = [hand.number_of_my_points() for hand in node.state.list_of_players_hands]
        return max(points) >= POINTS_TO_WIN

    def _terminal_value_from_root(self, node: MCTSNode, root_player: int) -> float:
        if self.use_adapter and node.adapter is not None:
            return node.adapter.terminal_value(root_player)
        points = [hand.number_of_my_points() for hand in node.state.list_of_players_hands]
        root_points = points[root_player]
        opp_points = points[1 - root_player]

        if root_points > opp_points:
            return 1.0
        if root_points < opp_points:
            return -1.0
        return 0.0
