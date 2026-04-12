"""Minimal AlphaZero trainer: self-play collection + supervised updates.

Supports two modes:
- Legacy: uses POINTS_TO_WIN instant-end terminal logic.
- Adapter (V2): uses SplendorPlanningAdapter with let_all_move semantics.
  Enable via ``use_adapter: true`` in the config training section.
"""

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
from planning.adapter import SplendorPlanningAdapter
# Lazy imports for reward shaping to avoid circular/relative import issues.
_reward_shaping_imports_loaded = False
__compute_event_reward = None
_DEFAULT_EVENT_WEIGHTS = None
__capture_state_snapshot = None
__detect_events = None


def _load_reward_shaping_imports():
    global _reward_shaping_imports_loaded, _compute_event_reward, _DEFAULT_EVENT_WEIGHTS
    global _capture_state_snapshot, _detect_events
    if _reward_shaping_imports_loaded:
        return
    from reward.event_based_reward import compute_event_reward as cer, DEFAULT_EVENT_WEIGHTS as dew
    from utils.event_detector import capture_state_snapshot as css, detect_events as de
    _compute_event_reward = cer
    _DEFAULT_EVENT_WEIGHTS = dew
    _capture_state_snapshot = css
    _detect_events = de
    _reward_shaping_imports_loaded = True


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

        # V2 features: adapter mode and FPU.
        self.use_adapter = bool(train_cfg.get("use_adapter", False))
        self.fpu_value = mcts_cfg.get("fpu_value", None)
        if self.fpu_value is not None:
            self.fpu_value = float(self.fpu_value)

        # Reward shaping: event-based intermediate rewards blended with terminal outcome.
        reward_cfg = config.get("reward_shaping", {})
        self.use_reward_shaping = bool(reward_cfg.get("enabled", False))
        self.shaping_discount = float(reward_cfg.get("discount", 0.99))
        self.shaping_weight = float(reward_cfg.get("weight", 0.3))
        # weight controls blend: value = (1-w)*terminal + w*shaped_return
        event_weights = reward_cfg.get("event_weights", None)
        if self.use_reward_shaping:
            _load_reward_shaping_imports()
        if event_weights is not None:
            self.event_weights = np.array(event_weights, dtype=np.float32)
        else:
            self.event_weights = _DEFAULT_EVENT_WEIGHTS.copy() if _DEFAULT_EVENT_WEIGHTS is not None else np.zeros(9, dtype=np.float32)
        # Normalize event weights so shaped returns are roughly in [-1, 1].
        self.reward_scale = float(reward_cfg.get("reward_scale", 0.1))

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
            use_adapter=self.use_adapter,
            fpu_value=self.fpu_value,
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
        if self.use_adapter:
            return self._play_one_game_adapter(mcts)
        return self._play_one_game_legacy(mcts)

    def _play_one_game_adapter(self, mcts: AlphaZeroMCTS) -> list[TrainingSample]:
        """Self-play using the shared planning adapter (V2 mode).

        When reward shaping is enabled, each step's event-based reward is
        collected and blended with the terminal win/loss outcome to produce
        denser value targets.
        """
        adapter = SplendorPlanningAdapter()
        trajectory: list[tuple[np.ndarray, np.ndarray, int]] = []
        step_rewards: list[tuple[int, float]] = []  # (player_id, shaped_reward)

        # Capture initial state snapshot for event detection.
        prev_snapshots = {
            pid: _capture_state_snapshot(adapter.state, pid)
            for pid in range(2)
        } if self.use_reward_shaping else {}

        for turn in range(self.max_turns):
            if adapter.is_terminal():
                break

            to_play = adapter.current_player
            temperature = 1.0 if turn < self.temperature_cutoff else 1e-8
            try:
                search: SearchResult = mcts.search(adapter.state, temperature=temperature)
            except RuntimeError as exc:
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

            encoded = adapter.encode_observation(player_id=to_play, turn_count=turn)
            trajectory.append((encoded, policy_target, to_play))

            action = search.selected_action
            adapter.step(action)

            # Detect events and compute shaped reward for this step.
            if self.use_reward_shaping:
                next_snap = _capture_state_snapshot(adapter.state, to_play)
                events = _detect_events(prev_snapshots[to_play], action, next_snap)
                shaped_r = _compute_event_reward(events, self.event_weights) * self.reward_scale
                step_rewards.append((to_play, shaped_r))
                # Update snapshots for both players.
                for pid in range(2):
                    prev_snapshots[pid] = _capture_state_snapshot(adapter.state, pid)
            else:
                step_rewards.append((to_play, 0.0))

        # Compute value targets.
        winner = adapter.winner_id()
        samples: list[TrainingSample] = []

        if self.use_reward_shaping and len(trajectory) > 0:
            # Compute discounted shaped returns per player.
            # shaped_returns[i] = sum of future discounted rewards for the
            # player who moved at step i.
            n = len(trajectory)
            shaped_returns = np.zeros(n, dtype=np.float32)

            # Backward pass: accumulate discounted rewards per player.
            player_accum = {0: 0.0, 1: 0.0}
            for i in range(n - 1, -1, -1):
                pid = trajectory[i][2]  # player who moved at step i
                r_i = step_rewards[i][1]
                player_accum[pid] = r_i + self.shaping_discount * player_accum[pid]
                shaped_returns[i] = player_accum[pid]

            # Normalize shaped returns to roughly [-1, 1].
            max_abs = max(1e-8, np.max(np.abs(shaped_returns)))
            if max_abs > 1.0:
                shaped_returns = shaped_returns / max_abs

            for i, (state_tensor, policy_target, player_id) in enumerate(trajectory):
                if winner is None:
                    terminal_v = 0.0
                elif winner == player_id:
                    terminal_v = 1.0
                else:
                    terminal_v = -1.0

                # Blend: (1 - weight) * terminal + weight * shaped_return
                value_target = (
                    (1.0 - self.shaping_weight) * terminal_v
                    + self.shaping_weight * shaped_returns[i]
                )
                value_target = float(np.clip(value_target, -1.0, 1.0))
                samples.append(TrainingSample(state_tensor, policy_target, value_target))
        else:
            for state_tensor, policy_target, player_id in trajectory:
                if winner is None:
                    value_target = 0.0
                elif winner == player_id:
                    value_target = 1.0
                else:
                    value_target = -1.0
                samples.append(TrainingSample(state_tensor, policy_target, value_target))

        return samples

    def _play_one_game_legacy(self, mcts: AlphaZeroMCTS) -> list[TrainingSample]:
        """Self-play using legacy instant-end terminal logic."""
        state = State()
        trajectory: list[tuple[np.ndarray, np.ndarray, int]] = []

        for turn in range(self.max_turns):
            to_play = state.active_player_id
            temperature = 1.0 if turn < self.temperature_cutoff else 1e-8
            try:
                search: SearchResult = mcts.search(state, temperature=temperature)
            except RuntimeError as exc:
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
