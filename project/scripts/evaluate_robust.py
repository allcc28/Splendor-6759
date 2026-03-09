"""
Robust evaluation for MaskablePPO agent.

Runs multiple independent batches of games (each batch uses a different
random seed) and reports:
  - Mean ± std win-rate across batches  (variance estimate)
  - Wilson score 95% CI on the pooled total  (frequentist CI)
  - Auto-generated Markdown report alongside the JSON

Usage:
    python project/scripts/evaluate_robust.py
    python project/scripts/evaluate_robust.py \\
        --model project/logs/maskable_ppo_v4a_ent_lr_20260306_213530/eval/best_model \\
        --tag v4a --games 1000 --batches 10

Defaults:
    --model   project/logs/maskable_ppo_v4a_ent_lr_20260306_213530/eval/best_model
    --tag     v4a
    --games   1000  (total games per opponent, split evenly across batches)
    --batches   10  (independent batches × 100 games each → variance estimate across seeds)
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
import yaml
from tqdm import tqdm
from sb3_contrib import MaskablePPO

from agents.random_agent import RandomAgent
from agents.greedy_agent_boost import GreedyAgentBoost
from project.src.utils.splendor_gym_wrapper import SplendorGymWrapper
from project.src.utils.event_reward_wrapper import maybe_wrap_with_event_shaping


def load_eval_config(model_path: str, config_path: str | None) -> tuple[dict, str]:
    """Load eval config explicitly or infer it from the saved training run."""
    if config_path:
        with open(config_path, "r") as f:
            return yaml.safe_load(f), str(Path(config_path).resolve())

    model = Path(model_path).resolve()
    for parent in [model.parent, *model.parents]:
        candidate = parent / "config.yaml"
        if candidate.exists():
            with open(candidate, "r") as f:
                return yaml.safe_load(f), str(candidate)

    return {"environment": {"reward_mode": "score_progress"}}, "(default score-only)"


def create_eval_env(config: dict, opponent_agent, max_turns: int, player_id: int):
    """Build the eval env with the same observation shaping used during training."""
    env_cfg = config.get("environment", {})
    env = SplendorGymWrapper(
        opponent_agent=opponent_agent,
        reward_mode=env_cfg.get("reward_mode", "score_progress"),
        max_turns=max_turns,
        player_id=player_id,
    )
    return maybe_wrap_with_event_shaping(env, config)


def validate_observation_shape(model, env) -> None:
    expected = tuple(model.observation_space.shape)
    actual = tuple(env.observation_space.shape)
    if expected != actual:
        raise ValueError(
            f"Model expects observation shape {expected}, but eval env exposes {actual}. "
            "Use the training config via --config or keep event_shaping settings aligned."
        )


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

def _run_batch(model, config, opponent_agent, n_games: int, seed: int, desc: str):
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
        env = create_eval_env(
            config=config,
            opponent_agent=opponent_agent,
            max_turns=200,
            player_id=player_id,
        )
        if game_idx == 0:
            validate_observation_shape(model, env)
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

def evaluate_robust(model, config, opponent_agent, total_games: int, n_batches: int, label: str):
    """
    Run `n_batches` independent batches of size `total_games // n_batches`.
    Returns aggregated stats including Wilson CI on pooled results.
    """
    batch_size = total_games // n_batches
    base_seeds = [42 + i * 17 for i in range(n_batches)]

    batch_results = []
    for b_idx, seed in enumerate(base_seeds):
        desc = f"{label} batch {b_idx + 1}/{n_batches} (seed={seed})"
        result = _run_batch(model, config, opponent_agent, batch_size, seed, desc)
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
    parser = argparse.ArgumentParser(description="Robust evaluation for MaskablePPO agent")
    parser.add_argument(
        "--model",
        type=str,
        default="project/logs/maskable_ppo_v4a_ent_lr_20260306_213530/eval/best_model",
        help="Path to trained MaskablePPO model (without .zip)",
    )
    parser.add_argument("--games",   type=int, default=1000, help="Total games per opponent")
    parser.add_argument("--batches", type=int, default=10,   help="Number of independent batches (seeds)")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional training config. If omitted, inferred from model directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="project/experiments/evaluation/robust",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="v4a",
        help="Short label for this run (e.g. v3, v4a). Embedded in filenames and JSON.",
    )
    args = parser.parse_args()

    games_per_batch = args.games // args.batches
    tag_label = args.tag if args.tag else "(untagged)"
    print("=" * 80)
    print(f"MaskablePPO Robust Evaluation  [{tag_label}]")
    print("=" * 80)
    print(f"Model:   {args.model}")
    print(f"Tag:     {tag_label}")
    print(f"Games:   {args.games} per opponent  ({args.batches} batches × {games_per_batch})")
    print(f"Output:  {args.output}")
    print()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    config, config_source = load_eval_config(args.model, args.config)

    print("Loading model …")
    model = MaskablePPO.load(args.model)
    print(f"Config:  {config_source}")
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
        result = evaluate_robust(model, config, agent, args.games, args.batches, label)
        all_results[label] = result

    # -----------------------------------------------------------------------
    # Console summary table
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print(f"Robust Evaluation Summary  [{tag_label}]")
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
    # Save JSON with provenance metadata
    # -----------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_part  = f"_{args.tag}" if args.tag else ""
    out_dir   = Path(args.output)

    json_file = out_dir / f"robust_eval{tag_part}_{timestamp}.json"
    payload = {
        "_meta": {
            "model_path":     args.model,
            "model_path_abs": str(Path(args.model).resolve()),
            "eval_tag":       tag_label,
            "timestamp":      timestamp,
            "games_per_opponent": args.games,
            "batches":        args.batches,
            "games_per_batch": games_per_batch,
            "config_source": config_source,
            "event_shaping_enabled": bool(config.get("event_shaping", {}).get("enabled", False)),
        },
        **all_results,
    }
    with open(json_file, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n✓ JSON  → {json_file}")

    # -----------------------------------------------------------------------
    # Generate Markdown report
    # -----------------------------------------------------------------------
    report_file = out_dir / f"robust_eval{tag_part}_{timestamp}_report.md"
    _write_markdown_report(report_file, all_results, args, tag_label, timestamp, games_per_batch)
    print(f"✓ Report → {report_file}")
    print("=" * 80)


def _write_markdown_report(path, results, args, tag_label, timestamp, games_per_batch):
    """Write a self-contained Markdown evaluation report."""
    lines = []

    lines += [
        f"# Robust Evaluation Report — {tag_label}",
        "",
        f"**Date**: {timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}  ",
        f"**Model**: `{args.model}`  ",
        f"**Protocol**: {args.games} games per opponent, {args.batches} independent batches "
        f"({games_per_batch} games/batch), alternating first-mover  ",
        f"**Action masking**: ON — no invalid actions sampled  ",
        "",
        "---",
        "",
        "## Win Rate Summary",
        "",
        "| Opponent | Win% (pooled) | ±std (batches) | 95% Wilson CI | W / L / D | Agent avg score | Opp avg score |",
        "|----------|:-------------:|:--------------:|:-------------:|:---------:|:---------------:|:-------------:|",
    ]

    for label, r in results.items():
        ci = f"[{r['wilson_ci_95_lo']:.1f}%, {r['wilson_ci_95_hi']:.1f}%]"
        wld = f"{r['total_wins']} / {r['total_losses']} / {r['total_draws']}"
        lines.append(
            f"| {label} | **{r['win_rate_pooled']:.1f}%** | "
            f"±{r['win_rate_std_across_batches']:.1f}% | {ci} | {wld} | "
            f"{r['agent_score_mean']:.1f} | {r['opp_score_mean']:.1f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Batch-Level Breakdown",
        "",
    ]

    for label, r in results.items():
        lines += [
            f"### vs {label}",
            "",
            f"| Batch | Win% | Agent avg | Opp avg |",
            f"|-------|:----:|:---------:|:-------:|",
        ]
        for i, b in enumerate(r["batch_results"]):
            lines.append(
                f"| {i+1} | {b['win_rate']:.1f}% | {b['agent_score_mean']:.1f} | {b['opp_score_mean']:.1f} |"
            )
        lines += [
            f"| **Pooled** | **{r['win_rate_pooled']:.1f}%** | "
            f"**{r['agent_score_mean']:.1f}** | **{r['opp_score_mean']:.1f}** |",
            "",
        ]

    lines += [
        "---",
        "",
        "## Statistical Notes",
        "",
        f"- **n = {args.games}** games per opponent. "
        "At this sample size the Wilson 95% CI half-width is roughly ±1.6 pp for win rates near 80%.",
        "- **±std (batches)** measures run-to-run variance across independent random seeds — "
        "a proxy for how stable the win rate is across differently-seeded game sequences.",
        "- Results are reported as pooled win rates (all batches combined) and independently "
        "per batch to show consistency.",
        "- A game is a **win** when `info['agent_won'] == True` or (if not set) `agent_score > opp_score`; "
        "**draw** when scores are equal; **loss** otherwise.",
        "",
        "## Comparison to Previous Benchmarks",
        "",
        "| Model | Eval protocol | vs Random | vs RandomAgent | vs Greedy |",
        "|-------|--------------|:---------:|:--------------:|:---------:|",
        "| V3 MaskablePPO | n=100, single run | 95% | 91% | 78% |",
        "| V4a MaskablePPO (n=100) | n=100, single run | 90% | 89% | 82% |",
        f"| **{tag_label} MaskablePPO (n={args.games})** | n={args.games}, {args.batches}-batch Wilson CI | "
        f"**{results.get('Random (wrapper)', {}).get('win_rate_pooled', 0):.1f}%** | "
        f"**{results.get('RandomAgent', {}).get('win_rate_pooled', 0):.1f}%** | "
        f"**{results.get('GreedyAgent', {}).get('win_rate_pooled', 0):.1f}%** |",
        "",
        "---",
        "",
        f"*Generated by `project/scripts/evaluate_robust.py` — {timestamp}*",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
