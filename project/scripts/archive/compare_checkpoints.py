"""
Compare two MaskablePPO checkpoints head-to-head.

Evaluates both models against all three opponents and prints a side-by-side
comparison table. Useful for answering: does the best checkpoint (820K steps)
actually outperform the final model (1M steps) vs GreedyAgent?

Usage:
    python project/scripts/compare_checkpoints.py             # defaults
    python project/scripts/compare_checkpoints.py \\
        --model-a project/logs/.../eval/best_model \\
        --model-b project/logs/.../final_model \\
        --games 100
"""

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "modules")

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from tqdm import tqdm
from sb3_contrib import MaskablePPO

from agents.random_agent import RandomAgent
from agents.greedy_agent_boost import GreedyAgentBoost
from project.src.utils.splendor_gym_wrapper import SplendorGymWrapper


# ---------------------------------------------------------------------------
# Default paths  (V3 training run — update here if you retrain)
# ---------------------------------------------------------------------------
_LOG_ROOT = "project/logs/maskable_ppo_score_v3_20260303_183435"
_DEFAULT_A = f"{_LOG_ROOT}/eval/best_model"      # 820K step best checkpoint
_DEFAULT_B = f"{_LOG_ROOT}/final_model"          # 1M step final model


# ---------------------------------------------------------------------------
# Evaluation helper (shared with evaluate_maskable_ppo.py pattern)
# ---------------------------------------------------------------------------

def _evaluate(model, opponent_agent, n_games: int, desc: str):
    """Run n_games and return aggregated stats dict."""
    wins = losses = draws = 0
    agent_scores = []
    opp_scores = []
    game_lengths = []
    rewards = []

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
            if len(env.cached_legal_actions) == 0:
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

        ag = info.get("player_score", 0)
        op = info.get("opponent_score", 0)
        agent_scores.append(ag)
        opp_scores.append(op)
        game_lengths.append(info.get("turn", 0))
        rewards.append(ep_reward)

        won  = info.get("agent_won", False)
        lost = info.get("agent_lost", False)
        if won:
            wins += 1
        elif lost:
            losses += 1
        else:
            if ag > op:
                wins += 1
            elif ag < op:
                losses += 1
            else:
                draws += 1

    total = n_games
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / total * 100,
        "agent_score": float(np.mean(agent_scores)),
        "opp_score": float(np.mean(opp_scores)),
        "game_length": float(np.mean(game_lengths)),
        "reward": float(np.mean(rewards)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare two MaskablePPO checkpoints")
    parser.add_argument("--model-a",  type=str, default=_DEFAULT_A,
                        help="Path to model A (without .zip); default = best_model @ 820K")
    parser.add_argument("--model-b",  type=str, default=_DEFAULT_B,
                        help="Path to model B (without .zip); default = final_model @ 1M")
    parser.add_argument("--label-a",  type=str, default="best_model (~820K steps)")
    parser.add_argument("--label-b",  type=str, default="final_model (1M steps)")
    parser.add_argument("--games",    type=int, default=100, help="Games per opponent per model")
    parser.add_argument(
        "--output",
        type=str,
        default="project/experiments/evaluation/checkpoint_compare",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("MaskablePPO Checkpoint Comparison")
    print("=" * 80)
    print(f"  Model A: {args.label_a}")
    print(f"           {args.model_a}")
    print(f"  Model B: {args.label_b}")
    print(f"           {args.model_b}")
    print(f"  Games:   {args.games} per opponent per model")
    print()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    print("Loading models …")
    model_a = MaskablePPO.load(args.model_a)
    print(f"  ✓ A: {args.label_a}")
    model_b = MaskablePPO.load(args.model_b)
    print(f"  ✓ B: {args.label_b}")
    print()

    opponents = [
        ("Random (wrapper)", None),
        ("RandomAgent",      RandomAgent(distribution="uniform_on_types")),
        ("GreedyAgent",      GreedyAgentBoost(name="Greedy", mode="value")),
    ]

    results = {}
    for opp_name, opp_agent in opponents:
        print(f"  {'─' * 60}")
        print(f"  vs {opp_name}")
        print(f"  {'─' * 60}")

        res_a = _evaluate(model_a, opp_agent, args.games, f"A vs {opp_name}")
        print(f"    A) {args.label_a:<28} {res_a['win_rate']:>5.1f}%  "
              f"Agent {res_a['agent_score']:.1f}  Opp {res_a['opp_score']:.1f}")

        res_b = _evaluate(model_b, opp_agent, args.games, f"B vs {opp_name}")
        print(f"    B) {args.label_b:<28} {res_b['win_rate']:>5.1f}%  "
              f"Agent {res_b['agent_score']:.1f}  Opp {res_b['opp_score']:.1f}")

        delta = res_b["win_rate"] - res_a["win_rate"]
        sign = "+" if delta >= 0 else ""
        print(f"    Δ (B − A):                          {sign}{delta:.1f} pp")

        results[opp_name] = {"A": res_a, "B": res_b, "delta_B_minus_A": delta}

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("Checkpoint Comparison Summary  (B − A = final_model − best_model)")
    print(f"{'=' * 80}")
    col = 28
    print(f"  {'Opponent':<22}  {'A Win%':>7}  {'B Win%':>7}  {'Δ (pp)':>8}")
    print("  " + "─" * 54)
    for opp_name, r in results.items():
        delta = r["delta_B_minus_A"]
        sign = "+" if delta >= 0 else ""
        print(f"  {opp_name:<22}  {r['A']['win_rate']:>6.1f}%  {r['B']['win_rate']:>6.1f}%  {sign}{delta:>6.1f}")

    print()
    print(f"  A = {args.label_a}")
    print(f"  B = {args.label_b}")
    print(f"\n  Interpretation:")
    print(f"  • Positive Δ means the final model (B) outperforms the best checkpoint (A)")
    print(f"  • Negative Δ means over-training occurred — best checkpoint is preferred")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = Path(args.output) / f"checkpoint_compare_{timestamp}.json"
    payload = {
        "model_a": {"path": args.model_a, "label": args.label_a},
        "model_b": {"path": args.model_b, "label": args.label_b},
        "games_per_opponent": args.games,
        "results": results,
    }
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"\n✓ Saved → {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
