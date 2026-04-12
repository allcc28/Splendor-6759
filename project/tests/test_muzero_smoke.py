"""Smoke tests for MuZero MCTS search and trainer."""

from __future__ import annotations

import sys
import os
import tempfile

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mcts.action_indexer import StableActionIndexer
from muzero.mcts import MuZeroMCTS, MinMaxStats
from muzero.network import MuZeroNetwork
from muzero.trainer import MuZeroTrainer
from nn.tensor_encoder import SplendorTensorEncoder
from planning.adapter import SplendorPlanningAdapter


class TestMinMaxStats:
    def test_normalize(self):
        stats = MinMaxStats()
        stats.update(0.0)
        stats.update(1.0)
        assert stats.normalize(0.5) == pytest.approx(0.5)
        assert stats.normalize(0.0) == pytest.approx(0.0)
        assert stats.normalize(1.0) == pytest.approx(1.0)

    def test_no_range(self):
        stats = MinMaxStats()
        # No updates — should return value unchanged.
        assert stats.normalize(0.5) == 0.5


class TestMuZeroMCTS:
    @pytest.fixture
    def setup(self):
        policy_size = 2048
        model = MuZeroNetwork(
            input_channels=80,
            latent_channels=32,
            policy_size=policy_size,
            action_space_size=policy_size,
            action_embed_dim=8,
            repr_res_blocks=1,
            dyn_res_blocks=1,
        )
        action_indexer = StableActionIndexer(policy_size=policy_size)
        mcts = MuZeroMCTS(
            model=model,
            action_indexer=action_indexer,
            device="cpu",
            num_simulations=5,
            c_puct=1.5,
        )
        return mcts

    def test_search_returns_legal_action(self, setup):
        mcts = setup
        adapter = SplendorPlanningAdapter()
        obs = adapter.encode_observation(player_id=0, turn_count=0)

        result = mcts.search(adapter=adapter, observation=obs, temperature=1.0)

        assert result.selected_action is not None
        assert len(result.legal_actions) > 0
        assert len(result.policy) == len(result.legal_actions)

    def test_policy_is_normalized(self, setup):
        mcts = setup
        adapter = SplendorPlanningAdapter()
        obs = adapter.encode_observation(player_id=0, turn_count=0)

        result = mcts.search(adapter=adapter, observation=obs, temperature=1.0)
        assert abs(np.sum(result.policy) - 1.0) < 1e-5

    def test_search_does_not_mutate_adapter(self, setup):
        mcts = setup
        adapter = SplendorPlanningAdapter()
        original_player = adapter.current_player
        original_points = adapter.points()
        obs = adapter.encode_observation(player_id=0, turn_count=0)

        mcts.search(adapter=adapter, observation=obs, temperature=1.0)

        assert adapter.current_player == original_player
        assert adapter.points() == original_points


class TestMuZeroTrainer:
    @pytest.fixture
    def config(self):
        return {
            "experiment": {"name": "muzero_test"},
            "network": {
                "policy_size": 2048,
                "latent_channels": 32,
                "action_embed_dim": 8,
                "repr_res_blocks": 1,
                "dyn_res_blocks": 1,
            },
            "mcts": {
                "num_simulations": 3,
                "c_puct": 1.5,
                "dirichlet_alpha": 0.1,
                "dirichlet_epsilon": 0.25,
                "max_turns": 120,
                "temperature_cutoff": 10,
                "num_unroll_steps": 3,
                "discount": 1.0,
            },
            "training": {
                "seed": 42,
                "device": "cpu",
                "iterations": 1,
                "episodes_per_iteration": 1,
                "replay_buffer_size": 100,
                "batch_size": 4,
                "train_epochs": 1,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
            },
        }

    def test_run_iteration(self, config):
        trainer = MuZeroTrainer(config)
        metrics = trainer.run_iteration()
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "reward_loss" in metrics
        assert metrics["games"] >= 1

    def test_checkpoint_save_load(self, config):
        trainer = MuZeroTrainer(config)
        trainer.run_iteration()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            trainer.save_checkpoint(path)

            trainer2 = MuZeroTrainer(config)
            trainer2.load_checkpoint(path)

            assert trainer2.replay_buffer.num_games > 0
        finally:
            os.unlink(path)
