"""Robust evaluation for MuZero-style Splendor checkpoints.

Evaluates a MuZero agent against:
1) RandomAgent via local Splendor duel API
2) GreedyAgentBoost via local Splendor duel API

Outputs:
  project/experiments/evaluation/robust/muzero/<bucket>/
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from math import sqrt
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, ".")
sys.path.insert(0, "modules")
sys.path.append("project/src")

from agents.greedy_agent_boost import GreedyAgentBoost
from agents.random_agent import RandomAgent
from gym_splendor_code.envs.mechanics.game_settings import MAX_NUMBER_OF_MOVES
from gym_splendor_code.envs.splendor import SplendorEnv
from mcts.action_indexer import StableActionIndexer
from muzero.mcts import MuZeroMCTS
from muzero.network import MuZeroNetwork
from nn.tensor_encoder import SplendorTensorEncoder
from planning.adapter import SplendorPlanningAdapter


@dataclass
class MuZeroRuntime:
    """Pack encoder/model/searcher for reuse across moves."""

    encoder: SplendorTensorEncoder
    model: MuZeroNetwork
    mcts: MuZeroMCTS
    action_indexer: StableActionIndexer
    device: str


class MuZeroAgentAdapter:
    """Adapter exposing choose_action() API compatible with legacy arenas."""

    def __init__(self, runtime: MuZeroRuntime, name: str = "MuZeroMCTS") -> None:
        self.runtime = runtime
        self.name = name

    def choose_action(self, observation, previous_actions):
        state = observation.recreate_state()
        adapter = SplendorPlanningAdapter(state=state)
        obs = adapter.encode_observation(
            player_id=adapter.current_player, turn_count=0
        )
        try:
            result = self.runtime.mcts.search(
                adapter=adapter, observation=obs, temperature=1e-8
            )
            return result.selected_action
        except RuntimeError as exc:
            if "No legal actions available at root" in str(exc):
                return None
            raise

    def finish_game(self) -> None:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MuZero-style Splendor checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--config", type=str, default="", help="Optional YAML config override")
    parser.add_argument("--games", type=int, default=100, help="Games per matchup")
    parser.add_argument("--simulations", type=int, default=None, help="Override MCTS simulations")
    parser.add_argument("--bucket", type=str, default="archive", choices=["archive", "canonical"])
    parser.add_argument("--tag", type=str, default="", help="Optional tag for output filename")
    return parser.parse_args()


def wilson_ci(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0, center - margin), min(1, center + margin)


def load_runtime(args) -> MuZeroRuntime:
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("config", {})

    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    net_cfg = config.get("network", {})
    mcts_cfg = config.get("mcts", {})
    device = "cpu"

    policy_size = int(net_cfg.get("policy_size", 2048))
    encoder = SplendorTensorEncoder()
    action_indexer = StableActionIndexer(policy_size=policy_size)

    model = MuZeroNetwork(
        input_channels=encoder.spec.channels,
        latent_channels=int(net_cfg.get("latent_channels", 64)),
        policy_size=policy_size,
        action_space_size=policy_size,
        action_embed_dim=int(net_cfg.get("action_embed_dim", 16)),
        repr_res_blocks=int(net_cfg.get("repr_res_blocks", 3)),
        dyn_res_blocks=int(net_cfg.get("dyn_res_blocks", 2)),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()

    indexer_state = payload.get("action_indexer_state")
    if indexer_state:
        action_indexer.load_state_dict(indexer_state)

    num_sims = args.simulations or int(mcts_cfg.get("num_simulations", 50))
    mcts = MuZeroMCTS(
        model=model,
        action_indexer=action_indexer,
        device=device,
        num_simulations=num_sims,
        c_puct=float(mcts_cfg.get("c_puct", 1.5)),
        dirichlet_alpha=float(mcts_cfg.get("dirichlet_alpha", 0.1)),
        dirichlet_epsilon=float(mcts_cfg.get("dirichlet_epsilon", 0.25)),
    )

    return MuZeroRuntime(
        encoder=encoder,
        model=model,
        mcts=mcts,
        action_indexer=action_indexer,
        device=device,
    )


def run_duel(muzero_adapter, opponent, games: int, env: SplendorEnv) -> dict:
    """Run a series of games between MuZero and an opponent."""
    wins = 0
    losses = 0
    draws = 0
    muzero_points_total = 0
    opp_points_total = 0

    agents = [muzero_adapter, opponent]

    for game_idx in range(games):
        env.reset()
        first_player = game_idx % 2
        if first_player == 1:
            agent_order = [opponent, muzero_adapter]
            muzero_seat = 1
        else:
            agent_order = [muzero_adapter, opponent]
            muzero_seat = 0

        for move in range(MAX_NUMBER_OF_MOVES):
            if env.is_done:
                break
            current = env.active_player_id()
            agent = agent_order[current]
            obs = env.show_observation("deterministic")
            action = agent.choose_action(obs, [])
            if action is None:
                break
            env.step("deterministic", action)

        pts = [env.points_of_player_by_id(i) for i in range(2)]
        muzero_pts = pts[muzero_seat]
        opp_pts = pts[1 - muzero_seat]
        muzero_points_total += muzero_pts
        opp_points_total += opp_pts

        if muzero_pts > opp_pts:
            wins += 1
        elif muzero_pts < opp_pts:
            losses += 1
        else:
            draws += 1

    win_rate = wins / max(1, games)
    ci_lo, ci_hi = wilson_ci(wins, games)

    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": round(win_rate, 4),
        "ci_95_low": round(ci_lo, 4),
        "ci_95_high": round(ci_hi, 4),
        "avg_muzero_points": round(muzero_points_total / max(1, games), 2),
        "avg_opp_points": round(opp_points_total / max(1, games), 2),
    }


def main() -> None:
    args = parse_args()
    runtime = load_runtime(args)
    muzero_adapter = MuZeroAgentAdapter(runtime)

    env = SplendorEnv()
    results = {"algorithm": "muzero_v1", "bucket": args.bucket, "checkpoint": args.checkpoint}

    print(f"Evaluating MuZero checkpoint: {args.checkpoint}")
    print(f"Games per matchup: {args.games}")

    # vs Random
    print("\n--- vs RandomAgent ---")
    random_agent = RandomAgent(distribution="uniform")
    random_results = run_duel(muzero_adapter, random_agent, args.games, env)
    results["vs_random"] = random_results
    print(f"Win rate: {random_results['win_rate']:.1%} "
          f"[{random_results['ci_95_low']:.1%}, {random_results['ci_95_high']:.1%}]")

    # vs Greedy
    print("\n--- vs GreedyAgentBoost ---")
    greedy_agent = GreedyAgentBoost()
    greedy_results = run_duel(muzero_adapter, greedy_agent, args.games, env)
    results["vs_greedy"] = greedy_results
    print(f"Win rate: {greedy_results['win_rate']:.1%} "
          f"[{greedy_results['ci_95_low']:.1%}, {greedy_results['ci_95_high']:.1%}]")

    # Save results
    tag = args.tag or "muzero_v1"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("project/experiments/evaluation/robust/muzero") / args.bucket
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"muzero_eval_{tag}_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON: {json_path}")

    # Markdown summary
    md_path = out_dir / f"muzero_eval_{tag}_{timestamp}.md"
    with open(md_path, "w") as f:
        f.write(f"# MuZero V1 Evaluation — {tag}\n\n")
        f.write(f"- Checkpoint: `{args.checkpoint}`\n")
        f.write(f"- Games per matchup: {args.games}\n")
        f.write(f"- Bucket: {args.bucket}\n\n")
        f.write("| Opponent | Win Rate | 95% CI | Avg MuZero Pts | Avg Opp Pts |\n")
        f.write("|----------|----------|--------|---------------|-------------|\n")
        for name, res in [("Random", random_results), ("Greedy", greedy_results)]:
            f.write(
                f"| {name} | {res['win_rate']:.1%} | "
                f"[{res['ci_95_low']:.1%}, {res['ci_95_high']:.1%}] | "
                f"{res['avg_muzero_points']} | {res['avg_opp_points']} |\n"
            )
    print(f"Saved Markdown: {md_path}")


if __name__ == "__main__":
    main()
