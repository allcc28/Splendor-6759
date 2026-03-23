"""
Robust evaluation for MaskablePPO checkpoints.

The script runs multiple independent batches of games against three classic
opponents and writes both JSON and Markdown outputs. New results are routed to
the organized robust-evaluation tree by default:

  project/experiments/evaluation/robust/ppo_robust/<family>/

Examples:
  python project/scripts/evaluate_robust.py
  python project/scripts/evaluate_robust.py \
      --model project/logs/maskable_ppo_event_v1_20260309_110155/eval/best_model \
      --tag v5_event
  python project/scripts/evaluate_robust.py \
      --model project/logs/maskable_ppo_v4a_ent_lr_20260306_213530/eval/best_model \
      --tag v4a --family score_based
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from math import sqrt
from pathlib import Path

import numpy as np
import yaml
from sb3_contrib import MaskablePPO
from tqdm import tqdm


sys.path.insert(0, ".")
sys.path.insert(0, "modules")
sys.path.append("project/src")

from agents.greedy_agent_boost import GreedyAgentBoost
from agents.random_agent import RandomAgent
from project.src.utils.event_reward_wrapper import maybe_wrap_with_event_shaping
from project.src.utils.splendor_gym_wrapper import SplendorGymWrapper


def load_eval_config(model_path: str, config_path: str | None) -> tuple[dict, str]:
    """Load eval config explicitly or infer it from the saved training run."""
    if config_path:
        with open(config_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle), str(Path(config_path).resolve())

    model = Path(model_path).resolve()
    for parent in [model.parent, *model.parents]:
        candidate = parent / "config.yaml"
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as handle:
                return yaml.safe_load(handle), str(candidate)

    return {"environment": {"reward_mode": "score_progress"}}, "(default score-only)"


def detect_robust_family(tag: str, model_path: str, config: dict) -> str:
    tag_lower = (tag or "").lower()
    model_lower = (model_path or "").lower()
    event_shaping_enabled = bool(config.get("event_shaping", {}).get("enabled", False))

    if event_shaping_enabled or "event" in tag_lower or "event" in model_lower:
        return "event_based"
    if "score" in tag_lower or "score" in model_lower:
        return "score_based"
    return "other"


def resolve_output_dir(output_arg: str, family: str) -> Path:
    """Keep compatibility with old CLI while defaulting to the organized tree."""
    output_dir = Path(output_arg)

    if output_dir.name in {"score_based", "event_based", "other"} and output_dir.parent.name == "ppo_robust":
        return output_dir
    if output_dir.name == "ppo_robust":
        return output_dir / family
    if output_dir.name == "robust":
        return output_dir / "ppo_robust" / family
    return output_dir


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


def get_legal_actions(env) -> list[int]:
    """Walk the wrapper chain to find cached legal actions."""
    current = env
    while current is not None:
        if hasattr(current, "cached_legal_actions"):
            return current.cached_legal_actions
        current = getattr(current, "env", None)
    return []


def wilson_ci(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion."""
    if total == 0:
        return 0.0, 0.0
    proportion = wins / total
    denom = 1.0 + z * z / total
    center = (proportion + z * z / (2 * total)) / denom
    margin = z * sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denom
    return max(0.0, center - margin) * 100, min(1.0, center + margin) * 100


def run_single_batch(model, config, opponent_agent, n_games: int, seed: int, desc: str) -> dict:
    """Run one seeded batch and return raw stats."""
    np.random.seed(seed)
    random.seed(seed)

    wins = 0
    losses = 0
    draws = 0
    agent_scores: list[int] = []
    opponent_scores: list[int] = []
    turns: list[int] = []
    rewards: list[float] = []

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
        episode_reward = 0.0

        while not done:
            if not get_legal_actions(env):
                obs, reward, terminated, truncated, info = env.step(0)
                episode_reward += reward
                done = terminated or truncated
                continue

            action, _ = model.predict(
                obs,
                action_masks=env.action_masks(),
                deterministic=True,
            )
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_reward += reward
            done = terminated or truncated

        agent_score = int(info.get("player_score", 0))
        opponent_score = int(info.get("opponent_score", 0))

        agent_scores.append(agent_score)
        opponent_scores.append(opponent_score)
        turns.append(int(info.get("turn", 0)))
        rewards.append(episode_reward)

        if info.get("agent_won", False):
            wins += 1
        elif info.get("agent_lost", False):
            losses += 1
        elif agent_score > opponent_score:
            wins += 1
        elif agent_score < opponent_score:
            losses += 1
        else:
            draws += 1

    total = n_games
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / total * 100 if total else 0.0,
        "agent_score_mean": float(np.mean(agent_scores)) if agent_scores else 0.0,
        "opp_score_mean": float(np.mean(opponent_scores)) if opponent_scores else 0.0,
        "game_length_mean": float(np.mean(turns)) if turns else 0.0,
        "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
    }


def evaluate_robust(model, config, opponent_agent, total_games: int, n_batches: int, label: str) -> dict:
    """Run multiple independent batches and aggregate the results."""
    batch_size = total_games // n_batches
    base_seeds = [42 + index * 17 for index in range(n_batches)]

    batch_results = []
    for batch_index, seed in enumerate(base_seeds):
        desc = f"{label} batch {batch_index + 1}/{n_batches} seed={seed}"
        result = run_single_batch(model, config, opponent_agent, batch_size, seed, desc)
        batch_results.append(result)
        print(
            f"  Batch {batch_index + 1}: {result['win_rate']:.1f}% win  "
            f"Agent {result['agent_score_mean']:.1f}  Opp {result['opp_score_mean']:.1f}"
        )

    win_rates = [result["win_rate"] for result in batch_results]
    total_wins = sum(result["wins"] for result in batch_results)
    pooled_total = batch_size * n_batches
    ci_lo, ci_hi = wilson_ci(total_wins, pooled_total)

    return {
        "opponent": label,
        "total_games": pooled_total,
        "total_wins": total_wins,
        "total_losses": sum(result["losses"] for result in batch_results),
        "total_draws": sum(result["draws"] for result in batch_results),
        "win_rate_pooled": total_wins / pooled_total * 100 if pooled_total else 0.0,
        "win_rate_mean_across_batches": float(np.mean(win_rates)) if win_rates else 0.0,
        "win_rate_std_across_batches": float(np.std(win_rates)) if win_rates else 0.0,
        "wilson_ci_95_lo": ci_lo,
        "wilson_ci_95_hi": ci_hi,
        "agent_score_mean": float(np.mean([result["agent_score_mean"] for result in batch_results])),
        "opp_score_mean": float(np.mean([result["opp_score_mean"] for result in batch_results])),
        "n_batches": n_batches,
        "batch_size": batch_size,
        "batch_results": batch_results,
    }


def write_markdown_report(
    path: Path,
    results: dict,
    args: argparse.Namespace,
    tag_label: str,
    timestamp: str,
    games_per_batch: int,
    family: str,
) -> None:
    """Write a self-contained Markdown evaluation report."""
    lines: list[str] = []
    lines.extend(
        [
            f"# Robust Evaluation Report - {tag_label}",
            "",
            f"**Date**: {timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}  ",
            f"**Family**: `{family}`  ",
            f"**Model**: `{args.model}`  ",
            (
                f"**Protocol**: {args.games} games per opponent, {args.batches} independent batches "
                f"({games_per_batch} games/batch), alternating first-mover  "
            ),
            "**Action masking**: ON  ",
            "",
            "---",
            "",
            "## Win Rate Summary",
            "",
            "| Opponent | Win% (pooled) | Batch std | 95% Wilson CI | W / L / D | Agent avg score | Opp avg score |",
            "|----------|:-------------:|:---------:|:-------------:|:---------:|:---------------:|:-------------:|",
        ]
    )

    for label, result in results.items():
        ci = f"[{result['wilson_ci_95_lo']:.1f}%, {result['wilson_ci_95_hi']:.1f}%]"
        wld = f"{result['total_wins']} / {result['total_losses']} / {result['total_draws']}"
        lines.append(
            f"| {label} | **{result['win_rate_pooled']:.1f}%** | "
            f"{result['win_rate_std_across_batches']:.1f}% | {ci} | {wld} | "
            f"{result['agent_score_mean']:.1f} | {result['opp_score_mean']:.1f} |"
        )

    lines.extend(["", "---", "", "## Batch-Level Breakdown", ""])

    for label, result in results.items():
        lines.extend(
            [
                f"### vs {label}",
                "",
                "| Batch | Win% | Agent avg | Opp avg |",
                "|-------|:----:|:---------:|:-------:|",
            ]
        )
        for index, batch in enumerate(result["batch_results"], start=1):
            lines.append(
                f"| {index} | {batch['win_rate']:.1f}% | {batch['agent_score_mean']:.1f} | {batch['opp_score_mean']:.1f} |"
            )
        lines.extend(
            [
                (
                    f"| **Pooled** | **{result['win_rate_pooled']:.1f}%** | "
                    f"**{result['agent_score_mean']:.1f}** | **{result['opp_score_mean']:.1f}** |"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "---",
            "",
            "## Notes",
            "",
            (
                f"- n = {args.games} games per opponent. At this sample size the Wilson 95% CI half-width is "
                "roughly 1.6 percentage points for win rates near 80%."
            ),
            "- Batch std measures run-to-run variance across independent random seeds.",
            "- Wins use info['agent_won'] when present, otherwise final score comparison.",
            "",
            f"*Generated by `project/scripts/evaluate_robust.py` on {timestamp}*",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robust evaluation for a MaskablePPO checkpoint")
    parser.add_argument(
        "--model",
        type=str,
        default="project/logs/maskable_ppo_v4a_ent_lr_20260306_213530/eval/best_model",
        help="Path to trained MaskablePPO model (without .zip).",
    )
    parser.add_argument("--games", type=int, default=1000, help="Total games per opponent.")
    parser.add_argument("--batches", type=int, default=10, help="Number of independent batches.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional training config. If omitted, infer it from the model directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="project/experiments/evaluation/robust",
        help="Base output directory. By default results are routed into ppo_robust/<family>/.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="v4a",
        help="Short label for this evaluation run, embedded in filenames.",
    )
    parser.add_argument(
        "--family",
        type=str,
        choices=["score_based", "event_based", "other"],
        default=None,
        help="Optional explicit robust-eval family. If omitted, infer from tag/model/config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    games_per_batch = args.games // args.batches
    tag_label = args.tag or "(untagged)"

    config, config_source = load_eval_config(args.model, args.config)
    family = args.family or detect_robust_family(args.tag, args.model, config)
    output_dir = resolve_output_dir(args.output, family)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"MaskablePPO Robust Evaluation [{tag_label}]")
    print("=" * 80)
    print(f"Model:         {args.model}")
    print(f"Family:        {family}")
    print(f"Config source: {config_source}")
    print(f"Games:         {args.games} per opponent")
    print(f"Batches:       {args.batches} x {games_per_batch}")
    print(f"Output dir:    {output_dir}")
    print()

    print("Loading model...")
    model = MaskablePPO.load(args.model)
    print("Model loaded.")

    opponents = [
        ("Random (wrapper)", None),
        ("RandomAgent", RandomAgent(distribution="uniform_on_types")),
        ("GreedyAgent", GreedyAgentBoost(name="Greedy", mode="value")),
    ]

    all_results = {}
    for label, agent in opponents:
        print()
        print("-" * 60)
        print(f"Evaluating vs {label}")
        print("-" * 60)
        all_results[label] = evaluate_robust(model, config, agent, args.games, args.batches, label)

    print()
    print("=" * 80)
    print(f"Robust Evaluation Summary [{tag_label}]")
    print("=" * 80)
    header = f"  {'Opponent':<22}  {'Win%':>6}  {'BatchStd':>8}  {'95% CI':>18}  {'Agent':>6}  {'Opp':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, result in all_results.items():
        ci_text = f"[{result['wilson_ci_95_lo']:.1f}%, {result['wilson_ci_95_hi']:.1f}%]"
        print(
            f"  {label:<22}  {result['win_rate_pooled']:>5.1f}%"
            f"  {result['win_rate_std_across_batches']:>7.1f}%"
            f"  {ci_text:>18}"
            f"  {result['agent_score_mean']:>5.1f}"
            f"  {result['opp_score_mean']:>5.1f}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_part = f"_{args.tag}" if args.tag else ""
    json_path = output_dir / f"robust_eval{tag_part}_{timestamp}.json"
    report_path = output_dir / f"robust_eval{tag_part}_{timestamp}_report.md"

    payload = {
        "_meta": {
            "model_path": args.model,
            "model_path_abs": str(Path(args.model).resolve()),
            "eval_tag": tag_label,
            "timestamp": timestamp,
            "games_per_opponent": args.games,
            "batches": args.batches,
            "games_per_batch": games_per_batch,
            "config_source": config_source,
            "family": family,
            "event_shaping_enabled": bool(config.get("event_shaping", {}).get("enabled", False)),
        },
        **all_results,
    }

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)

    write_markdown_report(
        path=report_path,
        results=all_results,
        args=args,
        tag_label=tag_label,
        timestamp=timestamp,
        games_per_batch=games_per_batch,
        family=family,
    )

    print()
    print(f"Saved JSON:   {json_path}")
    print(f"Saved report: {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
