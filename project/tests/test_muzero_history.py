"""Tests for MuZero GameHistory and target building."""

from __future__ import annotations

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from muzero.history import GameHistory


def make_dummy_history(length: int = 10, policy_size: int = 16) -> GameHistory:
    """Create a simple dummy game history for testing."""
    h = GameHistory()
    for i in range(length):
        h.observations.append(np.random.randn(80, 3, 4).astype(np.float32))
        h.actions.append(i % policy_size)
        h.rewards.append(0.0 if i < length - 1 else 1.0)
        h.to_play.append(i % 2)
        h.store_search_statistics(
            root_policy=np.ones(policy_size, dtype=np.float32) / policy_size,
            root_value=0.5 if i < length - 1 else 1.0,
        )
    return h


class TestGameHistory:
    def test_length(self):
        h = make_dummy_history(10)
        assert len(h) == 10

    def test_store_statistics(self):
        h = GameHistory()
        policy = np.ones(8, dtype=np.float32) / 8
        h.store_search_statistics(policy, 0.5)
        assert len(h.root_policies) == 1
        assert len(h.root_values) == 1
        assert h.root_values[0] == 0.5


class TestMakeTarget:
    def test_target_shapes(self):
        h = make_dummy_history(10, policy_size=16)
        obs, actions, policies, values, rewards = h.make_target(
            position=2, num_unroll_steps=5
        )
        assert obs.shape == (80, 3, 4)
        assert len(actions) == 5
        assert len(policies) == 5
        assert len(values) == 5
        assert len(rewards) == 5

    def test_target_padding_at_end(self):
        h = make_dummy_history(5, policy_size=16)
        obs, actions, policies, values, rewards = h.make_target(
            position=3, num_unroll_steps=5
        )
        # Position 3: steps 0,1 are in bounds (idx 3,4), steps 2,3,4 are padded.
        assert actions[2] == 0  # padding
        assert rewards[2] == 0.0  # padding

    def test_target_at_position_zero(self):
        h = make_dummy_history(10, policy_size=16)
        obs, actions, policies, values, rewards = h.make_target(
            position=0, num_unroll_steps=3
        )
        assert actions[0] == h.actions[0]
        assert rewards[0] == h.rewards[0]

    def test_policy_target_is_normalized(self):
        h = make_dummy_history(10, policy_size=16)
        _, _, policies, _, _ = h.make_target(position=0, num_unroll_steps=3)
        for p in policies:
            assert abs(np.sum(p) - 1.0) < 1e-5
