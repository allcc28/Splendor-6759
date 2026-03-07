"""
Robust evaluation for MaskablePPO agent.

Runs multiple independent batches of games (each batch uses a different
random seed) and reports:
  - Mean ± std win-rate across batches  (variance estimate)
  - Wilson score 95% CI on the pooled total  (frequency CI)

Usage:
    python project/scripts/evaluate_robust.py
    python project/scripts/evaluate_robust.py --model <path> --games 500 --batches 5

Defaults:
    --games   500   (total games per opponent, split evenly across batches)
    --batches   5   (independent batches → variance estimate across seeds)
    --model       project/logs/maskable_ppo_score_v3_20260303_183435/final_model
"""

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "modules")

import argparse
import json
import random
from math import sqrt
from pathlib import Path
from datetime import datetime

import numpy as np
from tqdm import tqdm
from sb3_contrib import MaskablePPO

from agents.random_agent import RandomAgent
from agents.greedy_agent_boost import GreedyAgentBoost
from project.src.utils.splendor_gym_wrapper import SplendorGymWrapper


# ---------------------------------------------------------------------------
# Confidence interval utilities
# ---------------------------------------------------------------------------

def wilson_ci(wins: int, n: int, z: float = 1.96):
    """Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - margin) * 100, min(1.0, centre + margin) * 100


# ---------------------------------------------------------------------------
# Single-batch evaluation (mirrors evaluate_maskable_ppo.py logic)
# ---------------------------------------------------------------------------

def _run_batch(model, opponent_agent, n_games: int, seed: int, desc: str):
    """Run n_games games with a given numpy/random seed and return raw stats."""
    np.random.seed(seed)
    random.seed(seed)

    wins = losses = draws = 0
    agent_scores_list = []
    opp_scores_list = []
    game_lengths_list = []
    rewards_list = []

    for game_idx in tqdm(range(n_games), desc=desc, leave=False):
        player_id = game_idx % 2
        env = SplendorGymWrapper(
            opponent_agent=opponent_agent,
            reward_mode="score_progress",
            max_turns=200,
            player_id=player_id,
        )
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            n_legal = len(env.cached_legal_actions)
            if n_legal == 0:
                obs, reward, terminated, truncated, info = env.step(0)
                ep_reward += reward
                done = terminated or truncated
                continue

            action, _ = model.predict(
                obs,
                action_masks=env.action_masks(),
                deterministic=True,
            )
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += reward
            done = terminated or truncated

        agent_score = info.get("player_score", 0)
        opp_score   = info.get("opponent_score", 0)
        agent_won   = info.get("agent_won", False)
        agent_lost  = info.get("agent_lost", False)

        agent_scores_list.append(agent_score)
        opp_scores_list.append(opp_score)
        game_lengths_list.append(info.get("turn", 0))
        rewards_list.append(ep_reward)

        if agent_won:
            wins += 1
        elif agent_lost:
            losses += 1
        else:
            if agent_score > opp_score:
                wins += 1
            elif agent_score < opp_score:
                losses += 1
            else:
                draws += 1

    total = n_games
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / total * 100,
        "agent_score_mean": float(np.mean(agent_scores_list)),
        "opp_score_mean": float(np.mean(opp_scores_list)),
        "game_length_mean": float(np.mean(game_lengths_list)),
        "reward_mean": float(np.mean(rewards_list)),
    }


# ---------------------------------------------------------------------------
# Main evaluation loop: multiple batches per opponent
# ---------------------------------------------------------------------------

def evaluate_robust(model, opponent_agent, total_games: int, n_batches: int, label: str):
    """
    Run `n_batches` independent batches of size `total_games // n_batches`.
    Returns aggregated stats including Wilson CI on pooled results.
    """
    batch_size = total_games // n_batches
    base_seeds = [42 + i * 17 for i in range(n_batches)]

    batch_results = []
    for b_idx, seed in enumerate(base_seeds):
        desc = f"{label} batch {b_idx + 1}/{n_batches} (seed={seed})"
        result = _run_batch(model, opponent_agent, batch_size, seed, desc)
        batch_results.append(result)
        print(
            f"  Batch {b_idx + 1}: {result['win_rate']:.1f}% win  "
            f"Agent {result['agent_score_mean']:.1f}  Opp {result['opp_score_mean']:.1f}"
        )

    # Aggregate
    win_rates = [r["win_rate"] for r in batch_results]
    total_wins = sum(r["wins"] for r in batch_results)
    pooled_total = batch_size * n_batches

    lo, hi = wilson_ci(total_wins, pooled_total)

    return {
        "opponent": label,
        "total_games": pooled_total,
        "total_wins": total_wins,
        "total_losses": sum(r["losses"] for r in batch_results),
        "total_draws": sum(r["draws"] for r in batch_results),
        "win_rate_pooled": total_wins / pooled_total * 100,
        "win_rate_mean_across_batches": float(np.mean(win_rates)),
        "win_rate_std_across_batches": float(np.std(win_rates)),
        "wilson_ci_95_lo": lo,
        "wilson_ci_95_hi": hi,
        "agent_score_mean": float(np.mean([r["agent_score_mean"] for r in batch_results])),
        "opp_score_mean": float(np.mean([r["opp_score_mean"] for r in batch_results])),
        "n_batches": n_batches,
        "batch_size": batch_size,
        "batch_results": batch_results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Robust evaluation for MaskablePPO v3 agent")
    parser.add_argument(
        "--model",
        type=str,
        default="project/logs/maskable_ppo_score_v3_20260303_183435/final_model",
        help="Path to trained MaskablePPO model (without .zip)",
    )
    parser.add_argument("--games",   type=int, default=500, help="Total games per opponent")
    parser.add_argument("--batches", type=int, default=5,   help="Number of independent batches (seeds)")
    parser.add_argument(
        "--output",
        type=str,
        default="project/experiments/evaluation/robust",
    )
    args = parser.parse_args()

    games_per_batch = args.games // args.batches
    print("=" * 80)
    print("MaskablePPO v3 — Robust Evaluation")
    print("=" * 80)
    print(f"Model:   {args.model}")
    print(f"Games:   {args.games} per opponent  ({args.batches} batches × {games_per_batch})")
    print(f"Output:  {args.output}")
    print()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    print("Loading model …")
    model = MaskablePPO.load(args.model)
    print("✓ Model loaded\n")

    opponents = [
        ("Random (wrapper)", None),
        ("RandomAgent",      RandomAgent(distribution="uniform_on_types")),
        ("GreedyAgent",      GreedyAgentBoost(name="Greedy", mode="value")),
    ]

    all_results = {}
    for label, agent in opponents:
        print(f"\n{'─' * 60}")
        print(f"  Evaluating vs {label}")
        print(f"{'─' * 60}")
        result = evaluate_robust(model, agent, args.games, args.batches, label)
        all_results[label] = result

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("Robust Evaluation Summary")
    print(f"{'=' * 80}")
    header = f"  {'Opponent':<22}  {'Win%':>6}  {'±Batch':>7}  {'95% CI':>18}  {'Agent':>6}  {'Opp':>6}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for label, r in all_results.items():
        ci_str = f"[{r['wilson_ci_95_lo']:.1f}%, {r['wilson_ci_95_hi']:.1f}%]"
        print(
            f"  {label:<22}  {r['win_rate_pooled']:>5.1f}%"
            f"  ±{r['win_rate_std_across_batches']:>5.1f}%"
            f"  {ci_str:>18}"
            f"  {r['agent_score_mean']:>5.1f}"
            f"  {r['opp_score_mean']:>5.1f}"
        )

    print(f"\n  Notes:")
    print(f"  • Win% = pooled win rate over all {args.games} games × 3 opponents")
    print(f"  • ±Batch = std of win rate across {args.batches} independent seeds")
    print(f"  • 95% CI = Wilson score confidence interval (frequentist)")

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = Path(args.output) / f"robust_eval_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n✓ Saved → {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
