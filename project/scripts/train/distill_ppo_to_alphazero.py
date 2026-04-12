"""Distill PPO policy into AlphaZero network via supervised pre-training.

Plays games with the PPO agent, encodes states with AlphaZero's tensor encoder,
and maps PPO action choices to AlphaZero policy targets. Then pre-trains
AlphaZero's network on this data so it starts self-play with a decent policy.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from tqdm import tqdm

sys.path.insert(0, ".")
sys.path.insert(0, "modules")
sys.path.append("project/src")

from sb3_contrib import MaskablePPO

from gym_splendor_code.envs.mechanics.game_settings import MAX_NUMBER_OF_MOVES
from gym_splendor_code.envs.splendor import SplendorEnv
from mcts.action_indexer import StableActionIndexer
from nn.policy_value_net import SplendorPolicyValueNet
from nn.tensor_encoder import SplendorTensorEncoder
from planning.adapter import SplendorPlanningAdapter
from utils.state_vectorizer import SplendorStateVectorizer


def parse_args():
    parser = argparse.ArgumentParser(description="Distill PPO policy into AlphaZero network")
    parser.add_argument("--ppo-model", type=str, required=True, help="Path to PPO .zip model")
    parser.add_argument("--config", type=str, required=True, help="AlphaZero YAML config")
    parser.add_argument("--output", type=str, required=True, help="Output checkpoint path")
    parser.add_argument("--games", type=int, default=200, help="Games to play for data collection")
    parser.add_argument("--epochs", type=int, default=10, help="Pre-training epochs")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def collect_ppo_data(ppo_model, num_games: int, encoder: SplendorTensorEncoder,
                     action_indexer: StableActionIndexer, vectorizer: SplendorStateVectorizer):
    """Play games with PPO and collect (tensor_state, action_index, value) tuples."""
    samples = []

    for game_idx in tqdm(range(num_games), desc="Collecting PPO games"):
        adapter = SplendorPlanningAdapter()
        env = SplendorEnv()
        env.reset()
        game_trajectory = []

        for turn in range(MAX_NUMBER_OF_MOVES):
            if adapter.is_terminal():
                break

            to_play = adapter.current_player
            state = adapter.state
            legal_actions = adapter.legal_actions
            if not legal_actions:
                break

            # Encode state for AlphaZero.
            tensor = encoder.encode(state, player_id=to_play, turn_count=turn)

            # Get PPO's action: vectorize state for PPO input.
            state_vec = vectorizer.vectorize(state, player_id=to_play)
            # Pad to 204 dims if needed (PPO may expect event features).
            if len(state_vec) < 204:
                state_vec = np.concatenate([state_vec, np.zeros(204 - len(state_vec), dtype=np.float32)])
            state_vec = state_vec[:204]

            # Get PPO action.
            obs_tensor = torch.from_numpy(state_vec).unsqueeze(0)
            with torch.no_grad():
                action_idx_ppo, _ = ppo_model.predict(state_vec, deterministic=True)

            # Map PPO's chosen action to a legal game action.
            # PPO action index maps to the gym wrapper's action space.
            # We need to find which legal action PPO chose.
            chosen_action = None
            action_idx_ppo = int(action_idx_ppo)
            if action_idx_ppo < len(legal_actions):
                chosen_action = legal_actions[action_idx_ppo]
            else:
                # Fallback: pick the first legal action.
                chosen_action = legal_actions[0]

            # Map to AlphaZero action index.
            az_action_idx = action_indexer.action_index(chosen_action)

            game_trajectory.append((tensor, az_action_idx, to_play))
            adapter.step(chosen_action)

        # Assign values based on game outcome.
        winner = adapter.winner_id()
        for tensor, az_idx, pid in game_trajectory:
            if winner is None:
                value = 0.0
            elif winner == pid:
                value = 1.0
            else:
                value = -1.0
            samples.append((tensor, az_idx, value))

    return samples


def pretrain(model, samples, policy_size, epochs, batch_size, lr, device):
    """Pre-train AlphaZero network on distilled PPO data."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()

    for epoch in range(epochs):
        random.shuffle(samples)
        total_ploss, total_vloss, n_batches = 0.0, 0.0, 0

        for start in range(0, len(samples), batch_size):
            batch = samples[start:start + batch_size]
            states = torch.from_numpy(np.stack([s[0] for s in batch])).to(device)
            action_indices = torch.tensor([s[1] for s in batch], dtype=torch.long, device=device)
            values = torch.tensor([s[2] for s in batch], dtype=torch.float32, device=device)

            logits, pred_values = model(states)

            # Policy loss: cross-entropy with PPO's chosen action.
            policy_loss = F.cross_entropy(logits, action_indices)

            # Value loss: MSE.
            value_loss = F.mse_loss(pred_values, values)

            total_loss = policy_loss + value_loss
            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_ploss += policy_loss.item()
            total_vloss += value_loss.item()
            n_batches += 1

        avg_p = total_ploss / max(1, n_batches)
        avg_v = total_vloss / max(1, n_batches)
        print(f"  Epoch {epoch+1}/{epochs}: policy_loss={avg_p:.3f} value_loss={avg_v:.3f}")


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    net_cfg = config["network"]
    policy_size = int(net_cfg.get("policy_size", 2048))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading PPO model from {args.ppo_model}")
    ppo_model = MaskablePPO.load(args.ppo_model)

    encoder = SplendorTensorEncoder()
    action_indexer = StableActionIndexer(policy_size=policy_size)
    vectorizer = SplendorStateVectorizer()

    print(f"Collecting data from {args.games} PPO games...")
    samples = collect_ppo_data(ppo_model, args.games, encoder, action_indexer, vectorizer)
    print(f"Collected {len(samples)} state-action samples")

    model = SplendorPolicyValueNet(
        input_channels=encoder.spec.channels,
        policy_size=policy_size,
        trunk_channels=int(net_cfg.get("trunk_channels", 128)),
        num_res_blocks=int(net_cfg.get("num_res_blocks", 3)),
    ).to(device)

    print(f"Pre-training for {args.epochs} epochs...")
    pretrain(model, samples, policy_size, args.epochs, args.batch_size, args.lr, device)

    # Save as AlphaZero checkpoint.
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": None,
        "action_indexer_state": action_indexer.state_dict(),
        "config": config,
        "distillation_info": {
            "ppo_model": args.ppo_model,
            "games": args.games,
            "samples": len(samples),
            "epochs": args.epochs,
        },
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(payload, args.output)
    print(f"Saved distilled checkpoint: {args.output}")


if __name__ == "__main__":
    main()
