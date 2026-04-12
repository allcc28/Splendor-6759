"""Game history container for MuZero training.

Stores complete episode data and builds unroll targets for training.
Reference: werner-duvaud/muzero-general self_play.py GameHistory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class GameHistory:
    """Complete record of a self-play game for MuZero training."""

    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    root_policies: List[np.ndarray] = field(default_factory=list)
    root_values: List[float] = field(default_factory=list)
    to_play: List[int] = field(default_factory=list)

    def store_search_statistics(
        self,
        root_policy: np.ndarray,
        root_value: float,
    ) -> None:
        """Store MCTS search statistics at the current position."""
        self.root_policies.append(root_policy)
        self.root_values.append(root_value)

    def __len__(self) -> int:
        return len(self.actions)

    def make_target(
        self,
        position: int,
        num_unroll_steps: int,
        discount: float = 1.0,
        td_steps: Optional[int] = None,
    ) -> tuple[
        np.ndarray,
        List[int],
        List[np.ndarray],
        List[float],
        List[float],
    ]:
        """Build training targets for a given position with K unroll steps.

        Returns:
            observation: The observation at *position*.
            actions: K action indices following *position* (0-padded if past end).
            target_policies: K policy distributions (uniform if past end).
            target_values: K value targets (0 if past end).
            target_rewards: K reward targets (0 if past end).
        """
        game_len = len(self.actions)
        observation = self.observations[position]

        actions: List[int] = []
        target_policies: List[np.ndarray] = []
        target_values: List[float] = []
        target_rewards: List[float] = []

        for step in range(num_unroll_steps):
            idx = position + step

            if idx < game_len:
                actions.append(self.actions[idx])
                target_rewards.append(self.rewards[idx])
            else:
                actions.append(0)  # padding
                target_rewards.append(0.0)

            if idx < len(self.root_policies):
                target_policies.append(self.root_policies[idx])
            else:
                # Uniform policy for positions past end.
                if self.root_policies:
                    policy_size = len(self.root_policies[0])
                else:
                    policy_size = 1
                target_policies.append(
                    np.ones(policy_size, dtype=np.float32) / policy_size
                )

            # Value target: use stored root value or bootstrap with discount.
            if td_steps is None or td_steps >= game_len - idx:
                # Monte Carlo: use terminal value.
                if idx < len(self.root_values):
                    target_values.append(self.root_values[idx])
                else:
                    target_values.append(0.0)
            else:
                # TD(n) bootstrap.
                bootstrap_idx = idx + td_steps
                value = sum(
                    self.rewards[i] * (discount ** (i - idx))
                    for i in range(idx, min(bootstrap_idx, game_len))
                )
                if bootstrap_idx < len(self.root_values):
                    value += (discount ** td_steps) * self.root_values[bootstrap_idx]
                target_values.append(value)

        return observation, actions, target_policies, target_values, target_rewards
