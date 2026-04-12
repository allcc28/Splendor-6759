"""Head-to-head evaluation: PPO+Lookahead vs PPO (pure) direct matchup."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from math import sqrt
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "project/src")
sys.path.insert(0, "modules")

from gym_splendor_code.envs.mechanics.game_settings import MAX_NUMBER_OF_MOVES
from gym_splendor_code.envs.splendor import SplendorEnv

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "ppo_lookahead_agent", "project/src/agents/ppo_lookahead_agent.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
PPOLookaheadAgent = _mod.PPOLookaheadAgent


def wilson_ci(wins, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0, center - margin), min(1, center + margin)


def run_duel(agent, opponent, games, label=""):
    wins, draws, losses = 0, 0, 0
    agent_pts_total, opp_pts_total = 0, 0
    t0 = time.time()

    for g in range(games):
        env = SplendorEnv()
        env.reset()
        seat = g % 2
        agents = [agent, opponent] if seat == 0 else [opponent, agent]

        for move in range(MAX_NUMBER_OF_MOVES):
            if env.is_done:
                break
            pid = env.active_player_id()
            obs = env.show_observation("deterministic")
            action = agents[pid].choose_action(obs, [])
            if action is None:
                break
            env.step("deterministic", action)

        pts = [env.points_of_player_by_id(i) for i in range(2)]
        my_pts = pts[seat]
        their_pts = pts[1 - seat]
        agent_pts_total += my_pts
        opp_pts_total += their_pts
        if my_pts > their_pts:
            wins += 1
        elif my_pts == their_pts:
            draws += 1
        else:
            losses += 1

        if (g + 1) % 100 == 0:
            elapsed = time.time() - t0
            wr = wins / (g + 1)
            print(f"  [{label}] {g+1}/{games}: win={wr:.1%} avg_pts={agent_pts_total/(g+1):.1f} vs {opp_pts_total/(g+1):.1f} ({elapsed:.0f}s)")

    wr = wins / max(1, games)
    ci_lo, ci_hi = wilson_ci(wins, games)
    return {
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": round(wr, 4),
        "ci_95_low": round(ci_lo, 4),
        "ci_95_high": round(ci_hi, 4),
        "avg_agent_pts": round(agent_pts_total / max(1, games), 2),
        "avg_opp_pts": round(opp_pts_total / max(1, games), 2),
        "sec_per_game": round((time.time() - t0) / max(1, games), 2),
    }


def main():
    p = argparse.ArgumentParser(description="Head-to-head: PPO+Lookahead vs PPO")
    p.add_argument("--ppo-model", type=str, required=True)
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--tag", type=str, default="h2h")
    args = p.parse_args()

    print("=== Head-to-Head Evaluation ===")
    print(f"Model: {args.ppo_model}")
    print(f"Games per matchup: {args.games}")
    print()

    # Agent 1: PPO+Lookahead (d=1, K=15) — our best agent
    lookahead = PPOLookaheadAgent(
        ppo_model_path=args.ppo_model,
        top_k=15, search_depth=1,
        alpha=0.3, beta=0.5, gamma=0.2,
    )

    # Agent 2: PPO pure (d=0) — baseline
    ppo_pure = PPOLookaheadAgent(
        ppo_model_path=args.ppo_model,
        top_k=15, search_depth=0,
    )

    results = {
        "matchups": {},
        "model": args.ppo_model,
    }

    # Match 1: PPO+Lookahead vs PPO pure
    print("--- PPO+Lookahead (d=1) vs PPO pure (d=0) ---")
    r = run_duel(lookahead, ppo_pure, args.games, "Lookahead vs PPO")
    results["matchups"]["lookahead_vs_ppo"] = r
    print(f"  Final: {r['win_rate']:.1%} [{r['ci_95_low']:.1%}, {r['ci_95_high']:.1%}]")
    print(f"  Avg pts: {r['avg_agent_pts']} vs {r['avg_opp_pts']}")
    print()

    # Match 2: PPO+Lookahead vs PPO+Lookahead (self-play — sanity check ~50%)
    print("--- PPO+Lookahead vs PPO+Lookahead (self-play) ---")
    lookahead2 = PPOLookaheadAgent(
        ppo_model_path=args.ppo_model,
        top_k=15, search_depth=1,
        alpha=0.3, beta=0.5, gamma=0.2,
    )
    r2 = run_duel(lookahead, lookahead2, min(args.games, 200), "Self-play")
    results["matchups"]["self_play"] = r2
    print(f"  Final: {r2['win_rate']:.1%} [{r2['ci_95_low']:.1%}, {r2['ci_95_high']:.1%}]")
    print()

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("project/experiments/evaluation/robust/ppo_lookahead/head_to_head")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"h2h_{args.tag}_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    md_path = out_dir / f"h2h_{args.tag}_{ts}.md"
    with open(md_path, "w") as f:
        f.write(f"# Head-to-Head Evaluation — {args.tag}\n\n")
        f.write(f"- Games: {args.games}\n")
        f.write(f"- Model: {args.ppo_model}\n\n")
        f.write("| Matchup | Win Rate | 95% CI | Agent Pts | Opp Pts | sec/game |\n")
        f.write("|---------|----------|--------|-----------|---------|----------|\n")
        for name, r in results["matchups"].items():
            f.write(f"| {name} | {r['win_rate']:.1%} | [{r['ci_95_low']:.1%},{r['ci_95_high']:.1%}] | {r['avg_agent_pts']} | {r['avg_opp_pts']} | {r['sec_per_game']} |\n")

    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
