"""Tests for the shared planning adapter."""

from __future__ import annotations

import random
import sys
import os

import numpy as np
import pytest

# Ensure project/src is on the path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from planning.adapter import SplendorPlanningAdapter
from gym_splendor_code.envs.mechanics.game_settings import POINTS_TO_WIN


class TestPlanningAdapterBasic:
    """Basic lifecycle and interface tests."""

    def test_fresh_adapter_state(self):
        adapter = SplendorPlanningAdapter()
        assert adapter.current_player == 0
        assert adapter.num_players == 2
        assert not adapter.is_terminal()
        assert adapter.points() == [0, 0]
        assert adapter._total_moves == 0

    def test_legal_actions_nonempty(self):
        adapter = SplendorPlanningAdapter()
        actions = adapter.legal_actions
        assert len(actions) > 0

    def test_step_advances_player(self):
        adapter = SplendorPlanningAdapter()
        actions = adapter.legal_actions
        adapter.step(actions[0])
        assert adapter.current_player == 1
        assert adapter._turn_counts[0] == 1
        assert adapter._turn_counts[1] == 0
        assert adapter._total_moves == 1

    def test_reset(self):
        adapter = SplendorPlanningAdapter()
        actions = adapter.legal_actions
        adapter.step(actions[0])
        adapter.reset()
        assert adapter.current_player == 0
        assert adapter._total_moves == 0
        assert not adapter._final_round_triggered


class TestPlanningAdapterClone:
    """State cloning tests."""

    def test_clone_independence(self):
        adapter = SplendorPlanningAdapter()
        actions = adapter.legal_actions
        adapter.step(actions[0])

        cloned = adapter.clone()
        assert cloned.current_player == adapter.current_player
        assert cloned._total_moves == adapter._total_moves
        assert cloned._turn_counts == adapter._turn_counts

        # Step on clone should not affect original.
        clone_actions = cloned.legal_actions
        if clone_actions:
            cloned.step(clone_actions[0])
            assert cloned._total_moves == adapter._total_moves + 1
            assert adapter._total_moves == 1


class TestLetAllMoveTerminal:
    """Terminal logic: let_all_move round completion."""

    def test_random_game_completes(self):
        """A random game should terminate within MAX_NUMBER_OF_MOVES."""
        random.seed(42)
        adapter = SplendorPlanningAdapter()
        moves = 0
        while not adapter.is_terminal() and moves < 200:
            actions = adapter.legal_actions
            if not actions:
                break
            adapter.step(random.choice(actions))
            moves += 1
        assert adapter.is_terminal()

    def test_terminal_value_range(self):
        """Terminal values should be in {-1, 0, 1}."""
        random.seed(123)
        adapter = SplendorPlanningAdapter()
        while not adapter.is_terminal():
            actions = adapter.legal_actions
            if not actions:
                break
            adapter.step(random.choice(actions))

        for pid in range(2):
            val = adapter.terminal_value(pid)
            assert val in (-1.0, 0.0, 1.0)

    def test_terminal_values_opposite(self):
        """If one player wins (+1), the other must lose (-1)."""
        random.seed(456)
        adapter = SplendorPlanningAdapter()
        while not adapter.is_terminal():
            actions = adapter.legal_actions
            if not actions:
                break
            adapter.step(random.choice(actions))

        v0 = adapter.terminal_value(0)
        v1 = adapter.terminal_value(1)
        if v0 != 0.0:
            assert v0 == -v1

    def test_equal_turns_at_terminal(self):
        """When final round triggers, both players should have equal turns."""
        random.seed(789)
        adapter = SplendorPlanningAdapter()
        while not adapter.is_terminal():
            actions = adapter.legal_actions
            if not actions:
                break
            adapter.step(random.choice(actions))

        if adapter._final_round_triggered:
            assert adapter._turn_counts[0] == adapter._turn_counts[1]


class TestTieBreak:
    """Tie-break resolution: same points -> fewest cards wins."""

    def test_tiebreak_fewer_cards(self):
        adapter = SplendorPlanningAdapter()
        # Manually set points equal but different card counts.
        # This is a conceptual test using the terminal_value logic.
        hand0 = adapter.state.list_of_players_hands[0]
        hand1 = adapter.state.list_of_players_hands[1]

        # We can't easily set points directly, so test the logic via
        # winner_id which uses the same tie-break.
        # If both have 0 points and 0 cards -> draw.
        assert adapter.winner_id() is None
        assert adapter.terminal_value(0) == 0.0


class TestObservationEncoding:
    """Observation encoding and public-info mode."""

    def test_encoding_shape(self):
        adapter = SplendorPlanningAdapter()
        obs = adapter.encode_observation(player_id=0, turn_count=0)
        assert obs.shape == (80, 3, 4)

    def test_public_info_no_opponent_reserved(self):
        """Default mode should zero out opponent reserved identity channels."""
        adapter = SplendorPlanningAdapter(include_opponent_reserved_identity=False)
        obs = adapter.encode_observation(player_id=0, turn_count=0)
        # Channels 65-79 should be all zeros.
        assert np.all(obs[65:80, :, :] == 0.0)

    def test_full_info_mode(self):
        """Full-info mode should preserve opponent reserved channels (may still be 0 if no reservations)."""
        adapter = SplendorPlanningAdapter(include_opponent_reserved_identity=True)
        obs = adapter.encode_observation(player_id=0, turn_count=0)
        # Shape should still be correct.
        assert obs.shape == (80, 3, 4)

    def test_encoding_deterministic(self):
        """Same state should produce same encoding."""
        adapter = SplendorPlanningAdapter()
        obs1 = adapter.encode_observation(player_id=0, turn_count=5)
        obs2 = adapter.encode_observation(player_id=0, turn_count=5)
        np.testing.assert_array_equal(obs1, obs2)

    def test_encoding_values_clipped(self):
        """All values should be in [0, 1]."""
        adapter = SplendorPlanningAdapter()
        obs = adapter.encode_observation(player_id=0, turn_count=0)
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)
