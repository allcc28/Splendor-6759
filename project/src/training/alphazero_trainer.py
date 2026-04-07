"""Minimal AlphaZero trainer: self-play collection + supervised updates."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from gym_splendor_code.envs.mechanics.game_settings import MAX_NUMBER_OF_MOVES, POINTS_TO_WIN
from gym_splendor_code.envs.mechanics.state import State

from mcts.action_indexer import StableActionIndexer
from mcts.alphazero_mcts import AlphaZeroMCTS, SearchResult, TorchPolicyValueFunction
from nn.policy_value_net import SplendorPolicyValueNet
from nn.tensor_encoder import SplendorTensorEncoder


@dataclass
class TrainingSample:
    """Single supervised tuple for policy/value learning."""

    state_tensor: np.ndarray
    policy_target: np.ndarray
    value_target: float


class AlphaZeroTrainer:
    """Compact trainer used for smoke-level AlphaZero bootstrapping."""

    def __init__(self, config: dict) -> None:
        self.config = config
        train_cfg = config["training"]
        mcts_cfg = config["mcts"]
        net_cfg = config["network"]

        requested_device = str(train_cfg.get("device", "cpu"))
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = requested_device
        self.policy_size = int(net_cfg.get("policy_size", 200))
        self.action_indexer = StableActionIndexer(policy_size=self.policy_size)

        self.encoder = SplendorTensorEncoder()
        self.model = SplendorPolicyValueNet(
            input_channels=self.encoder.spec.channels,
            policy_size=self.policy_size,
            trunk_channels=int(net_cfg.get("trunk_channels", 128)),
            num_res_blocks=int(net_cfg.get("num_res_blocks", 3)),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(train_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )

        self.max_turns = int(mcts_cfg.get("max_turns", MAX_NUMBER_OF_MOVES))
        self.temperature_cutoff = int(mcts_cfg.get("temperature_cutoff", 20))
        self.batch_size = int(train_cfg.get("batch_size", 64))
        self.train_epochs = int(train_cfg.get("train_epochs", 3))

        buffer_size = int(train_cfg.get("replay_buffer_size", 0))
        self.replay_buffer: deque[TrainingSample] | None = (
            deque(maxlen=buffer_size) if buffer_size > 0 else None
        )

        self._set_seed(int(train_cfg.get("seed", 42)))

    def run_iteration(self) -> dict:
        """Run one iteration: collect self-play games then optimize on them."""
        episodes = int(self.config["training"].get("episodes_per_iteration", 8))
        samples: list[TrainingSample] = []

        policy_fn = TorchPolicyValueFunction(
            encoder=self.encoder,
            model=self.model,
            device=self.device,
            action_indexer=self.action_indexer,
        )
        mcts = AlphaZeroMCTS(
            policy_value_fn=policy_fn,
            num_simulations=int(self.config["mcts"].get("num_simulations", 100)),
            c_puct=float(self.config["mcts"].get("c_puct", 1.5)),
            dirichlet_alpha=float(self.config["mcts"].get("dirichlet_alpha", 0.3)),
            dirichlet_epsilon=float(self.config["mcts"].get("dirichlet_epsilon", 0.25)),
            max_depth=self.max_turns,
        )

        for _ in range(episodes):
            samples.extend(self._play_one_game(mcts))

        if not samples:
            return {"samples": 0, "buffer_size": 0, "policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}

        if self.replay_buffer is not None:
            self.replay_buffer.extend(samples)
            train_samples = list(self.replay_buffer)
        else:
            train_samples = samples

        metrics = self._train_on_samples(train_samples)
        metrics["samples"] = len(samples)
        metrics["buffer_size"] = len(train_samples)
        return metrics

    def save_checkpoint(self, path: str) -> None:
        """Save model and optimizer state for restartability."""
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "action_indexer_state": self.action_indexer.state_dict(),
            "config": self.config,
        }
        if self.replay_buffer is not None:
            payload["replay_buffer"] = list(self.replay_buffer)
        torch.save(payload, path)

    def load_checkpoint(self, path: str) -> dict:
        """Load model/optimizer state from checkpoint and return payload metadata."""
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state_dict"])

        optimizer_state = payload.get("optimizer_state_dict")
        if optimizer_state:
            self.optimizer.load_state_dict(optimizer_state)

        indexer_state = payload.get("action_indexer_state")
        if indexer_state:
            self.action_indexer.load_state_dict(indexer_state)

        if self.replay_buffer is not None:
            saved_buffer = payload.get("replay_buffer", [])
            self.replay_buffer.extend(saved_buffer)

        return payload

    def _play_one_game(self, mcts: AlphaZeroMCTS) -> list[TrainingSample]:
        state = State()
        trajectory: list[tuple[np.ndarray, np.ndarray, int]] = []

        for turn in range(self.max_turns):
            to_play = state.active_player_id
            temperature = 1.0 if turn < self.temperature_cutoff else 1e-8
            try:
                search: SearchResult = mcts.search(state, temperature=temperature)
            except RuntimeError as exc:
                # Splendor can occasionally reach states where no legal action exists.
                # Treat this as an immediate terminal/truncated stop for the current self-play game.
                if "No legal actions available at root" in str(exc):
                    break
                raise

            policy_target = np.zeros(self.policy_size, dtype=np.float32)
            legal_indices = self.action_indexer.legal_indices(search.legal_actions)
            for idx, prob in zip(legal_indices, search.policy, strict=False):
                policy_target[int(idx)] += float(prob)

            total_prob = float(np.sum(policy_target))
            if total_prob > 0:
                policy_target /= total_prob

            encoded = self.encoder.encode(state, player_id=to_play, turn_count=turn)
            trajectory.append((encoded, policy_target, to_play))

            search.selected_action.execute(state)
            if self._is_terminal(state):
                break

        winner = self._winner_id(state)
        samples: list[TrainingSample] = []
        for state_tensor, policy_target, player_id in trajectory:
            if winner is None:
                value_target = 0.0
            elif winner == player_id:
                value_target = 1.0
            else:
                value_target = -1.0
            samples.append(TrainingSample(state_tensor, policy_target, value_target))
        return samples

    def _train_on_samples(self, samples: list[TrainingSample]) -> dict:
        policy_losses = []
        value_losses = []
        total_losses = []

        self.model.train()
        for _ in range(self.train_epochs):
            random.shuffle(samples)
            for start in range(0, len(samples), self.batch_size):
                batch = samples[start : start + self.batch_size]
                states = torch.from_numpy(np.stack([s.state_tensor for s in batch])).to(self.device)
                policy_targets = torch.from_numpy(np.stack([s.policy_target for s in batch])).to(self.device)
                value_targets = torch.tensor([s.value_target for s in batch], dtype=torch.float32, device=self.device)

                logits, values = self.model(states)
                log_probs = F.log_softmax(logits, dim=-1)
                policy_loss = -(policy_targets * log_probs).sum(dim=-1).mean()
                value_loss = F.mse_loss(values, value_targets)
                total_loss = policy_loss + value_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                total_losses.append(float(total_loss.item()))

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "total_loss": float(np.mean(total_losses)) if total_losses else 0.0,
        }

    def _is_terminal(self, state: State) -> bool:
        points = [hand.number_of_my_points() for hand in state.list_of_players_hands]
        return max(points) >= POINTS_TO_WIN

    def _winner_id(self, state: State) -> int | None:
        points = [hand.number_of_my_points() for hand in state.list_of_players_hands]
        if points[0] > points[1]:
            return 0
        if points[1] > points[0]:
            return 1
        return None

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
