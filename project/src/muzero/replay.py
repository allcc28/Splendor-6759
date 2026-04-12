"""Uniform replay buffer for MuZero training.

Stores GameHistory objects and samples positions with unroll windows.
Reference: werner-duvaud/muzero-general replay_buffer.py (simplified: no PER).
"""

from __future__ import annotations

import random
from collections import deque
from typing import List, Tuple

import numpy as np

from muzero.history import GameHistory


class MuZeroReplayBuffer:
    """Uniform replay buffer storing complete game histories."""

    def __init__(self, max_games: int = 1000) -> None:
        self.buffer: deque[GameHistory] = deque(maxlen=max_games)

    def add_game(self, game: GameHistory) -> None:
        if len(game) > 0:
            self.buffer.append(game)

    @property
    def num_games(self) -> int:
        return len(self.buffer)

    @property
    def total_positions(self) -> int:
        return sum(len(g) for g in self.buffer)

    def sample_batch(
        self,
        batch_size: int,
        num_unroll_steps: int = 5,
        discount: float = 1.0,
        td_steps: int | None = None,
    ) -> Tuple[
        np.ndarray,
        List[List[int]],
        List[List[np.ndarray]],
        List[List[float]],
        List[List[float]],
    ]:
        """Sample a batch of positions with unroll targets.

        Returns:
            observations: (batch, C, H, W) observation tensors.
            actions_batch: list of K action index lists.
            policies_batch: list of K policy target lists.
            values_batch: list of K value target lists.
            rewards_batch: list of K reward target lists.
        """
        observations = []
        actions_batch = []
        policies_batch = []
        values_batch = []
        rewards_batch = []

        games = list(self.buffer)
        for _ in range(batch_size):
            game = random.choice(games)
            position = random.randint(0, max(0, len(game) - 1))

            obs, actions, policies, values, rewards = game.make_target(
                position=position,
                num_unroll_steps=num_unroll_steps,
                discount=discount,
                td_steps=td_steps,
            )
            observations.append(obs)
            actions_batch.append(actions)
            policies_batch.append(policies)
            values_batch.append(values)
            rewards_batch.append(rewards)

        return (
            np.stack(observations),
            actions_batch,
            policies_batch,
            values_batch,
            rewards_batch,
        )

    def state_dict(self) -> list:
        """Serialize buffer for checkpointing."""
        return list(self.buffer)

    def load_state_dict(self, data: list) -> None:
        """Restore buffer from checkpoint."""
        self.buffer.clear()
        for game in data:
            self.buffer.append(game)
