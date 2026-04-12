"""MuZero trainer: self-play + unroll training loop.

Reference: werner-duvaud/muzero-general trainer.py for unroll training,
self_play.py for play_game structure.
"""

from __future__ import annotations

import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from gym_splendor_code.envs.mechanics.game_settings import MAX_NUMBER_OF_MOVES

from mcts.action_indexer import StableActionIndexer
from muzero.history import GameHistory
from muzero.mcts import MuZeroMCTS, MuZeroSearchResult
from muzero.network import MuZeroNetwork
from muzero.replay import MuZeroReplayBuffer
from nn.tensor_encoder import SplendorTensorEncoder
from planning.adapter import SplendorPlanningAdapter


class MuZeroTrainer:
    """Synchronous MuZero trainer for Splendor."""

    def __init__(self, config: dict) -> None:
        self.config = config
        train_cfg = config["training"]
        mcts_cfg = config["mcts"]
        net_cfg = config["network"]

        requested_device = str(train_cfg.get("device", "cpu"))
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = requested_device

        self.policy_size = int(net_cfg.get("policy_size", 2048))
        self.action_indexer = StableActionIndexer(policy_size=self.policy_size)
        self.encoder = SplendorTensorEncoder()

        self.model = MuZeroNetwork(
            input_channels=self.encoder.spec.channels,
            latent_channels=int(net_cfg.get("latent_channels", 64)),
            policy_size=self.policy_size,
            action_space_size=self.policy_size,
            action_embed_dim=int(net_cfg.get("action_embed_dim", 16)),
            repr_res_blocks=int(net_cfg.get("repr_res_blocks", 3)),
            dyn_res_blocks=int(net_cfg.get("dyn_res_blocks", 2)),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(train_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )

        buffer_size = int(train_cfg.get("replay_buffer_size", 1000))
        self.replay_buffer = MuZeroReplayBuffer(max_games=buffer_size)

        self.num_unroll_steps = int(mcts_cfg.get("num_unroll_steps", 5))
        self.discount = float(mcts_cfg.get("discount", 1.0))
        self.td_steps = mcts_cfg.get("td_steps", None)
        if self.td_steps is not None:
            self.td_steps = int(self.td_steps)
        self.max_turns = int(mcts_cfg.get("max_turns", MAX_NUMBER_OF_MOVES))
        self.temperature_cutoff = int(mcts_cfg.get("temperature_cutoff", 20))
        self.batch_size = int(train_cfg.get("batch_size", 64))
        self.train_epochs = int(train_cfg.get("train_epochs", 2))

        # Loss weights (reference: muzero-general trainer.py).
        self.value_loss_weight = float(train_cfg.get("value_loss_weight", 0.25))
        self.reward_loss_weight = float(train_cfg.get("reward_loss_weight", 1.0))
        self.policy_loss_weight = float(train_cfg.get("policy_loss_weight", 1.0))

        self._set_seed(int(train_cfg.get("seed", 42)))

    def run_iteration(self) -> dict:
        """One iteration: collect self-play games, then train on replay buffer."""
        episodes = int(self.config["training"].get("episodes_per_iteration", 4))
        mcts_cfg = self.config["mcts"]

        mcts = MuZeroMCTS(
            model=self.model,
            action_indexer=self.action_indexer,
            device=self.device,
            num_simulations=int(mcts_cfg.get("num_simulations", 50)),
            c_puct=float(mcts_cfg.get("c_puct", 1.5)),
            dirichlet_alpha=float(mcts_cfg.get("dirichlet_alpha", 0.1)),
            dirichlet_epsilon=float(mcts_cfg.get("dirichlet_epsilon", 0.25)),
            discount=self.discount,
        )

        games_collected = 0
        total_moves = 0
        for _ in range(episodes):
            game = self._play_one_game(mcts)
            self.replay_buffer.add_game(game)
            games_collected += 1
            total_moves += len(game)

        if self.replay_buffer.num_games == 0:
            return {
                "games": 0,
                "moves": 0,
                "buffer_games": 0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "reward_loss": 0.0,
                "total_loss": 0.0,
            }

        metrics = self._train_on_buffer()
        metrics["games"] = games_collected
        metrics["moves"] = total_moves
        metrics["buffer_games"] = self.replay_buffer.num_games
        return metrics

    def save_checkpoint(self, path: str) -> None:
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "action_indexer_state": self.action_indexer.state_dict(),
            "replay_buffer": self.replay_buffer.state_dict(),
            "config": self.config,
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str) -> dict:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state_dict"])

        optimizer_state = payload.get("optimizer_state_dict")
        if optimizer_state:
            self.optimizer.load_state_dict(optimizer_state)

        indexer_state = payload.get("action_indexer_state")
        if indexer_state:
            self.action_indexer.load_state_dict(indexer_state)

        buffer_data = payload.get("replay_buffer", [])
        if buffer_data:
            self.replay_buffer.load_state_dict(buffer_data)

        return payload

    def _play_one_game(self, mcts: MuZeroMCTS) -> GameHistory:
        """Self-play one game using MuZero MCTS."""
        adapter = SplendorPlanningAdapter()
        history = GameHistory()

        for turn in range(self.max_turns):
            if adapter.is_terminal():
                break

            to_play = adapter.current_player
            obs = adapter.encode_observation(player_id=to_play, turn_count=turn)
            temperature = 1.0 if turn < self.temperature_cutoff else 1e-8

            try:
                result: MuZeroSearchResult = mcts.search(
                    adapter=adapter,
                    observation=obs,
                    temperature=temperature,
                )
            except RuntimeError:
                break

            # Build full policy target over policy_size.
            policy_target = np.zeros(self.policy_size, dtype=np.float32)
            legal_indices = self.action_indexer.legal_indices(result.legal_actions)
            for idx, prob in zip(legal_indices, result.policy):
                policy_target[int(idx)] += float(prob)
            total = float(np.sum(policy_target))
            if total > 0:
                policy_target /= total

            # Store in history.
            history.observations.append(obs)
            history.actions.append(result.selected_action_index)
            history.to_play.append(to_play)
            history.store_search_statistics(
                root_policy=policy_target,
                root_value=result.root_value,
            )

            # Step the game.
            adapter.step(result.selected_action)

            # Reward: 0 for intermediate steps.
            history.rewards.append(0.0)

        # Assign terminal rewards.
        winner = adapter.winner_id()
        if winner is not None and len(history.rewards) > 0:
            for i in range(len(history.rewards)):
                if history.to_play[i] == winner:
                    history.rewards[i] = 0.0  # intermediate stays 0
                else:
                    history.rewards[i] = 0.0

            # Final step reward.
            if len(history.to_play) > 0:
                last_player = history.to_play[-1]
                history.rewards[-1] = 1.0 if last_player == winner else -1.0

        # Update root_values with actual game outcome for training.
        for i in range(len(history.root_values)):
            if winner is None:
                history.root_values[i] = 0.0
            elif history.to_play[i] == winner:
                history.root_values[i] = 1.0
            else:
                history.root_values[i] = -1.0

        return history

    def _train_on_buffer(self) -> dict:
        """Train on replay buffer samples using unroll training.

        Reference: werner-duvaud/muzero-general trainer.py update_weights.
        """
        policy_losses = []
        value_losses = []
        reward_losses = []
        total_losses = []

        self.model.train()
        for _ in range(self.train_epochs):
            (
                observations,
                actions_batch,
                policies_batch,
                values_batch,
                rewards_batch,
            ) = self.replay_buffer.sample_batch(
                batch_size=self.batch_size,
                num_unroll_steps=self.num_unroll_steps,
                discount=self.discount,
                td_steps=self.td_steps,
            )

            obs_tensor = torch.from_numpy(observations).to(self.device)

            # Initial inference.
            output = self.model.initial_inference(obs_tensor)
            latent_state = output.latent_state

            # Loss accumulation across unroll steps.
            total_policy_loss = torch.tensor(0.0, device=self.device)
            total_value_loss = torch.tensor(0.0, device=self.device)
            total_reward_loss = torch.tensor(0.0, device=self.device)

            for step in range(self.num_unroll_steps):
                # Build targets for this step.
                step_actions = torch.tensor(
                    [actions_batch[b][step] for b in range(self.batch_size)],
                    dtype=torch.long,
                    device=self.device,
                )
                step_policies = torch.from_numpy(
                    np.stack([policies_batch[b][step] for b in range(self.batch_size)])
                ).to(self.device)
                step_values = torch.tensor(
                    [values_batch[b][step] for b in range(self.batch_size)],
                    dtype=torch.float32,
                    device=self.device,
                )
                step_rewards = torch.tensor(
                    [rewards_batch[b][step] for b in range(self.batch_size)],
                    dtype=torch.float32,
                    device=self.device,
                )

                if step == 0:
                    # Use initial inference output for step 0.
                    pred_policy = output.policy_logits
                    pred_value = output.value
                    pred_reward = output.reward
                else:
                    # Recurrent inference.
                    output = self.model.recurrent_inference(latent_state, step_actions)
                    latent_state = output.latent_state
                    pred_policy = output.policy_logits
                    pred_value = output.value
                    pred_reward = output.reward

                # Gradient scaling at dynamics boundary (0.5x).
                # Reference: muzero-general trainer.py.
                if step > 0:
                    latent_state.register_hook(lambda grad: grad * 0.5)

                # Policy loss: cross-entropy.
                log_probs = F.log_softmax(pred_policy, dim=-1)
                policy_loss = -(step_policies * log_probs).sum(dim=-1).mean()

                # Value loss: MSE.
                value_loss = F.mse_loss(pred_value, step_values)

                # Reward loss: MSE.
                reward_loss = F.mse_loss(pred_reward, step_rewards)

                total_policy_loss += policy_loss
                total_value_loss += value_loss
                total_reward_loss += reward_loss

            # Average over unroll steps.
            num_steps = float(self.num_unroll_steps)
            avg_policy = total_policy_loss / num_steps
            avg_value = total_value_loss / num_steps
            avg_reward = total_reward_loss / num_steps

            total_loss = (
                self.policy_loss_weight * avg_policy
                + self.value_loss_weight * avg_value
                + self.reward_loss_weight * avg_reward
            )

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            policy_losses.append(float(avg_policy.item()))
            value_losses.append(float(avg_value.item()))
            reward_losses.append(float(avg_reward.item()))
            total_losses.append(float(total_loss.item()))

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "reward_loss": float(np.mean(reward_losses)) if reward_losses else 0.0,
            "total_loss": float(np.mean(total_losses)) if total_losses else 0.0,
        }

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
