"""Smoke tests for AlphaZeroMCTS."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath("modules"))
sys.path.insert(0, os.path.abspath("project/src"))

from gym_splendor_code.envs.mechanics.state import State

from mcts.alphazero_mcts import AlphaZeroMCTS


def _uniform_policy_value(_state, _player_id, legal_actions):
    if len(legal_actions) == 0:
        return np.zeros(0, dtype=np.float32), 0.0
    prior = np.ones(len(legal_actions), dtype=np.float32) / len(legal_actions)
    return prior, 0.0


def test_mcts_search_returns_valid_policy_and_action() -> None:
    mcts = AlphaZeroMCTS(
        policy_value_fn=_uniform_policy_value,
        num_simulations=8,
        c_puct=1.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.0,
    )

    state = State()
    result = mcts.search(state, temperature=1.0)

    assert len(result.legal_actions) > 0
    assert result.policy.shape == (len(result.legal_actions),)
    assert np.isclose(float(result.policy.sum()), 1.0)
    assert result.selected_action in result.legal_actions


def test_mcts_search_does_not_mutate_root_state() -> None:
    mcts = AlphaZeroMCTS(
        policy_value_fn=_uniform_policy_value,
        num_simulations=4,
        c_puct=1.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.0,
    )

    state = State()
    active_before = state.active_player_id
    points_before = [hand.number_of_my_points() for hand in state.list_of_players_hands]

    _ = mcts.search(state, temperature=1.0)

    assert state.active_player_id == active_before
    assert [hand.number_of_my_points() for hand in state.list_of_players_hands] == points_before
