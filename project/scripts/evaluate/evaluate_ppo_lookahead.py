"""Evaluate PPO+Lookahead agent against Random and Greedy opponents."""

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

from agents.greedy_agent_boost import GreedyAgentBoost
from agents.random_agent import RandomAgent
from gym_splendor_code.envs.mechanics.game_settings import MAX_NUMBER_OF_MOVES
from gym_splendor_code.envs.splendor import SplendorEnv

# Import from project/src explicitly to avoid shadowing by modules/agents.
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "ppo_lookahead_agent", "project/src/agents/ppo_lookahead_agent.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
PPOLookaheadAgent = _mod.PPOLookaheadAgent


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate PPO+Lookahead agent")
    p.add_argument("--ppo-model", type=str, required=True)
    p.add_argument("--games", type=int, default=1000)
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.2)
    p.add_argument("--event-weights", type=str, default=None,
                   help="Comma-separated 9 event weights, e.g. '0.01,1.0,0.05,0.8,25.0,0.25,0.5,0.4,2.0'")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--bucket", type=str, default="archive")
    return p.parse_args()


def wilson_ci(wins, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0, center - margin), min(1, center + margin)


def run_duel(agent, opponent, games, label=""):
    wins, draws, agent_pts, opp_pts = 0, 0, 0, 0
    t0 = time.time()

    for g in range(games):
        env = SplendorEnv()
        env.reset()
        seat = g % 2  # alternate first player
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
        agent_pts += my_pts
        opp_pts += their_pts
        if my_pts > their_pts:
            wins += 1
        elif my_pts == their_pts:
            draws += 1

        if (g + 1) % 100 == 0:
            elapsed = time.time() - t0
            wr = wins / (g + 1)
            print(f"  [{label}] {g+1}/{games}: win={wr:.1%} avg_pts={agent_pts/(g+1):.1f} ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    wr = wins / max(1, games)
    ci_lo, ci_hi = wilson_ci(wins, games)
    return {
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": games - wins - draws,
        "win_rate": round(wr, 4),
        "ci_95_low": round(ci_lo, 4),
        "ci_95_high": round(ci_hi, 4),
        "avg_agent_pts": round(agent_pts / max(1, games), 2),
        "avg_opp_pts": round(opp_pts / max(1, games), 2),
        "sec_per_game": round(elapsed / max(1, games), 2),
    }


def main():
    args = parse_args()

    print(f"=== PPO+Lookahead Evaluation ===")
    print(f"Model: {args.ppo_model}")
    print(f"Config: depth={args.depth}, top_k={args.top_k}, α={args.alpha}, β={args.beta}, γ={args.gamma}")
    print(f"Games per matchup: {args.games}")

    event_weights = None
    if args.event_weights:
        event_weights = np.array([float(x) for x in args.event_weights.split(",")])
        print(f"Event weights: {event_weights}")

    agent = PPOLookaheadAgent(
        ppo_model_path=args.ppo_model,
        top_k=args.top_k,
        search_depth=args.depth,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        event_weights=event_weights,
    )

    results = {
        "agent": "PPOLookahead",
        "config": {
            "depth": args.depth,
            "top_k": args.top_k,
            "alpha": args.alpha,
            "beta": args.beta,
            "gamma": args.gamma,
            "event_weights": args.event_weights,
            "ppo_model": args.ppo_model,
        },
    }

    # vs Random
    print("\n--- vs RandomAgent ---")
    random_agent = RandomAgent(distribution="uniform")
    results["vs_random"] = run_duel(agent, random_agent, args.games, "Random")
    print(f"  Final: {results['vs_random']['win_rate']:.1%} [{results['vs_random']['ci_95_low']:.1%}, {results['vs_random']['ci_95_high']:.1%}]")

    # vs Greedy
    print("\n--- vs GreedyAgentBoost ---")
    greedy_agent = GreedyAgentBoost(name="Greedy", mode="value")
    results["vs_greedy"] = run_duel(agent, greedy_agent, args.games, "Greedy")
    print(f"  Final: {results['vs_greedy']['win_rate']:.1%} [{results['vs_greedy']['ci_95_low']:.1%}, {results['vs_greedy']['ci_95_high']:.1%}]")

    # Save
    tag = args.tag or f"d{args.depth}_k{args.top_k}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("project/experiments/evaluation/robust/ppo_lookahead") / args.bucket
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"ppo_lookahead_{tag}_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    md_path = out_dir / f"ppo_lookahead_{tag}_{ts}.md"
    with open(md_path, "w") as f:
        f.write(f"# PPO+Lookahead Evaluation — {tag}\n\n")
        f.write(f"- depth={args.depth}, top_k={args.top_k}\n")
        f.write(f"- α={args.alpha}, β={args.beta}, γ={args.gamma}\n")
        f.write(f"- Games: {args.games}\n\n")
        f.write("| Opponent | Win Rate | 95% CI | Avg Pts | Avg Opp Pts | sec/game |\n")
        f.write("|----------|----------|--------|---------|-------------|----------|\n")
        for name, key in [("Random", "vs_random"), ("Greedy", "vs_greedy")]:
            r = results[key]
            f.write(f"| {name} | {r['win_rate']:.1%} | [{r['ci_95_low']:.1%},{r['ci_95_high']:.1%}] | {r['avg_agent_pts']} | {r['avg_opp_pts']} | {r['sec_per_game']} |\n")

    print(f"\nSaved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
