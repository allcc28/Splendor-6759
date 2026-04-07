"""Smoke tests for AlphaZeroMCTS and AlphaZeroTrainer replay buffer."""

import os
import sys
import tempfile

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


def test_replay_buffer_accumulates_and_persists() -> None:
    """Replay buffer grows across iterations and survives checkpoint round-trip."""
    from training.alphazero_trainer import AlphaZeroTrainer

    config = {
        "experiment": {"name": "test_buffer"},
        "network": {"policy_size": 512, "trunk_channels": 32, "num_res_blocks": 1},
        "mcts": {
            "num_simulations": 4,
            "c_puct": 1.0,
            "dirichlet_alpha": 0.3,
            "dirichlet_epsilon": 0.25,
            "max_turns": 10,
            "temperature_cutoff": 5,
        },
        "training": {
            "seed": 0,
            "device": "cpu",
            "episodes_per_iteration": 1,
            "replay_buffer_size": 500,
            "batch_size": 8,
            "train_epochs": 1,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
        },
    }

    trainer = AlphaZeroTrainer(config)
    assert trainer.replay_buffer is not None
    assert len(trainer.replay_buffer) == 0

    m1 = trainer.run_iteration()
    size_after_1 = m1["buffer_size"]
    assert size_after_1 > 0, "buffer should be non-empty after first iteration"

    m2 = trainer.run_iteration()
    size_after_2 = m2["buffer_size"]
    assert size_after_2 >= size_after_1, "buffer should grow or stay after second iteration"

    # Checkpoint round-trip
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        ckpt_path = f.name
    trainer.save_checkpoint(ckpt_path)

    trainer2 = AlphaZeroTrainer(config)
    trainer2.load_checkpoint(ckpt_path)
    assert len(trainer2.replay_buffer) == len(trainer.replay_buffer), (
        "buffer size must be preserved across checkpoint save/load"
    )


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
