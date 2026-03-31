"""Checkpoint compatibility tests for AlphaZero trainer."""

import os
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import torch

sys.path.insert(0, os.path.abspath("modules"))
sys.path.insert(0, os.path.abspath("project/src"))

from training.alphazero_trainer import AlphaZeroTrainer

sys.path.insert(0, os.path.abspath("project/scripts"))

from evaluate_alphazero_mcts import create_runtime


def _minimal_config() -> dict:
    return {
        "experiment": {"name": "test"},
        "network": {
            "policy_size": 128,
            "trunk_channels": 32,
            "num_res_blocks": 1,
        },
        "mcts": {
            "num_simulations": 2,
            "c_puct": 1.5,
            "dirichlet_alpha": 0.3,
            "dirichlet_epsilon": 0.25,
            "max_turns": 4,
            "temperature_cutoff": 2,
        },
        "training": {
            "seed": 42,
            "device": "cpu",
            "iterations": 1,
            "episodes_per_iteration": 1,
            "batch_size": 4,
            "train_epochs": 1,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "checkpoint_every": 1,
        },
    }


def test_load_checkpoint_allows_missing_optimizer_state() -> None:
    trainer = AlphaZeroTrainer(_minimal_config())

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "smoke.pt")
        torch.save(
            {
                "model_state_dict": trainer.model.state_dict(),
                "optimizer_state_dict": None,
                "action_indexer_state": None,
                "smoke_checkpoint": True,
                "config": _minimal_config(),
            },
            ckpt_path,
        )

        payload = trainer.load_checkpoint(ckpt_path)
        assert payload.get("smoke_checkpoint", False) is True


def test_evaluator_runtime_restores_action_indexer_state() -> None:
    trainer = AlphaZeroTrainer(_minimal_config())

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = os.path.join(tmp_dir, "model.pt")
        action_indexer_state = {
            "policy_size": 128,
            "key_to_index": {"dummy_action_key": 7},
            "next_index": 8,
        }
        torch.save(
            {
                "model_state_dict": trainer.model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "action_indexer_state": action_indexer_state,
                "config": _minimal_config(),
            },
            ckpt_path,
        )

        args = Namespace(
            simulations=0,
            c_puct=0.0,
            dirichlet_alpha=0.0,
            dirichlet_epsilon=-1.0,
            max_turns=120,
        )
        runtime = create_runtime(Path(ckpt_path), _minimal_config(), args)
        restored = runtime.mcts.policy_value_fn.action_indexer.state_dict()

        assert restored["policy_size"] == action_indexer_state["policy_size"]
        assert restored["next_index"] == action_indexer_state["next_index"]
        assert restored["key_to_index"] == action_indexer_state["key_to_index"]
