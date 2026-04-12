"""Latent-space MCTS for MuZero with hybrid real-state tracking.

Each node stores both a latent state (for network inference) and a real game
state (for legal action queries). This hybrid approach guarantees legal actions
without requiring a learned legality mask.

Reference: werner-duvaud/muzero-general self_play.py MCTS class + MinMaxStats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import List, Optional

import numpy as np
import torch

from gym_splendor_code.envs.mechanics.action import Action

from mcts.action_indexer import StableActionIndexer
from muzero.network import MuZeroNetwork, MuZeroNetworkOutput
from planning.adapter import SplendorPlanningAdapter


class MinMaxStats:
    """Tracks min/max Q-values for normalization during search.

    Reference: werner-duvaud/muzero-general self_play.py MinMaxStats.
    """

    def __init__(self) -> None:
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value: float) -> None:
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


@dataclass
class MuZeroNode:
    """Search tree node for MuZero MCTS."""

    latent_state: Optional[torch.Tensor] = None
    adapter: Optional[SplendorPlanningAdapter] = None
    to_play: int = 0
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    reward: float = 0.0
    children: dict[int, MuZeroNode] = field(default_factory=dict)
    legal_action_indices: Optional[List[int]] = None
    legal_actions: Optional[List[Action]] = None

    @property
    def expanded(self) -> bool:
        return len(self.children) > 0

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


@dataclass
class MuZeroSearchResult:
    """Output of MuZero MCTS search."""

    legal_actions: List[Action]
    policy: np.ndarray
    selected_action: Action
    selected_action_index: int
    root_value: float


class MuZeroMCTS:
    """Latent-space MCTS with hybrid real-state tracking for legal actions."""

    def __init__(
        self,
        model: MuZeroNetwork,
        action_indexer: StableActionIndexer,
        device: str = "cpu",
        num_simulations: int = 50,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.1,
        dirichlet_epsilon: float = 0.25,
        discount: float = 1.0,
    ) -> None:
        self.model = model
        self.action_indexer = action_indexer
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.discount = discount

    def search(
        self,
        adapter: SplendorPlanningAdapter,
        observation: np.ndarray,
        temperature: float = 1.0,
    ) -> MuZeroSearchResult:
        """Run MCTS from the given position and return action selection.

        Args:
            adapter: Planning adapter at the current game position.
            observation: Encoded observation tensor for the current state.
            temperature: Action selection temperature.
        """
        min_max_stats = MinMaxStats()

        # Create root node.
        root = MuZeroNode(adapter=adapter.clone(), to_play=adapter.current_player)

        # Initial inference at root.
        obs_tensor = torch.from_numpy(observation).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            output = self.model.initial_inference(obs_tensor)

        root.latent_state = output.latent_state
        root_value = float(output.value.item())

        # Expand root with legal actions.
        legal_actions = root.adapter.legal_actions
        if not legal_actions:
            raise RuntimeError("No legal actions available at root.")

        legal_indices = self.action_indexer.legal_indices(legal_actions)
        policy_logits = output.policy_logits.squeeze(0).cpu().numpy()
        legal_logits = policy_logits[legal_indices]
        legal_logits = legal_logits - np.max(legal_logits)
        priors = np.exp(legal_logits)
        priors = priors / max(1e-8, np.sum(priors))

        # Add Dirichlet noise at root.
        noise = np.random.dirichlet(
            [self.dirichlet_alpha] * len(priors)
        ).astype(np.float32)
        priors = (1 - self.dirichlet_epsilon) * priors + self.dirichlet_epsilon * noise

        root.legal_actions = legal_actions
        root.legal_action_indices = list(legal_indices)

        for i, (action, action_idx) in enumerate(zip(legal_actions, legal_indices)):
            child = MuZeroNode(prior=float(priors[i]))
            root.children[i] = child

        # Run simulations.
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            current_adapter = root.adapter.clone()

            # Selection: traverse tree until we reach an unexpanded node.
            while node.expanded:
                child_idx = self._select_child(node, min_max_stats)

                # Advance real state.
                if child_idx < len(root.legal_actions) and node is root:
                    action = root.legal_actions[child_idx]
                    action_index = root.legal_action_indices[child_idx]
                elif node.legal_actions is not None and child_idx < len(node.legal_actions):
                    action = node.legal_actions[child_idx]
                    action_index = node.legal_action_indices[child_idx]
                else:
                    break

                current_adapter.step(action)
                node = node.children[child_idx]
                search_path.append(node)

                if current_adapter.is_terminal():
                    break

            # Expansion and evaluation.
            if current_adapter.is_terminal():
                value = current_adapter.terminal_value(root.to_play)
            else:
                # Recurrent inference to get next latent state.
                parent = search_path[-2] if len(search_path) >= 2 else root
                parent_latent = parent.latent_state
                if parent_latent is None:
                    parent_latent = root.latent_state

                action_tensor = torch.tensor(
                    [action_index], dtype=torch.long, device=self.device
                )
                with torch.no_grad():
                    output = self.model.recurrent_inference(parent_latent, action_tensor)

                node.latent_state = output.latent_state
                node.reward = float(output.reward.item())
                node.to_play = current_adapter.current_player
                node.adapter = current_adapter.clone()
                value = float(output.value.item())

                # Expand this node.
                child_legal_actions = current_adapter.legal_actions
                if child_legal_actions:
                    child_legal_indices = self.action_indexer.legal_indices(child_legal_actions)
                    child_logits = output.policy_logits.squeeze(0).cpu().numpy()
                    child_legal_logits = child_logits[child_legal_indices]
                    child_legal_logits = child_legal_logits - np.max(child_legal_logits)
                    child_priors = np.exp(child_legal_logits)
                    child_priors = child_priors / max(1e-8, np.sum(child_priors))

                    node.legal_actions = child_legal_actions
                    node.legal_action_indices = list(child_legal_indices)
                    for i in range(len(child_legal_actions)):
                        node.children[i] = MuZeroNode(prior=float(child_priors[i]))

            # Backpropagate.
            self._backpropagate(search_path, value, root.to_play, min_max_stats)

        # Build policy from visit counts.
        visit_counts = np.array(
            [root.children[i].visit_count for i in range(len(root.legal_actions))],
            dtype=np.float32,
        )
        policy = self._counts_to_policy(visit_counts, temperature)

        action_idx = int(np.random.choice(len(root.legal_actions), p=policy))
        return MuZeroSearchResult(
            legal_actions=root.legal_actions,
            policy=policy,
            selected_action=root.legal_actions[action_idx],
            selected_action_index=root.legal_action_indices[action_idx],
            root_value=root_value,
        )

    def _select_child(self, node: MuZeroNode, min_max_stats: MinMaxStats) -> int:
        """Select child using PUCT with normalized Q-values."""
        best_score = -float("inf")
        best_idx = 0
        parent_visits = max(1, node.visit_count)

        for idx, child in node.children.items():
            q = child.value
            if node.to_play != child.to_play and child.visit_count > 0:
                q = -q  # Flip perspective for opponent.

            # Normalize Q to [0, 1] range.
            normalized_q = min_max_stats.normalize(q) if child.visit_count > 0 else 0.0

            exploration = (
                self.c_puct
                * child.prior
                * sqrt(parent_visits)
                / (1 + child.visit_count)
            )
            score = normalized_q + exploration

            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx

    def _backpropagate(
        self,
        search_path: List[MuZeroNode],
        value: float,
        root_player: int,
        min_max_stats: MinMaxStats,
    ) -> None:
        """Propagate value up the search path.

        Reference: werner-duvaud/muzero-general backpropagate with discount.
        """
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value if node.to_play == root_player else -value
            min_max_stats.update(node.value)
            value = node.reward + self.discount * value

    def _counts_to_policy(
        self, visit_counts: np.ndarray, temperature: float
    ) -> np.ndarray:
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
