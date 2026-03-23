"""
Benchmark pure/guided MCTS against classic opponents and PPO checkpoints.

This script intentionally uses two evaluation paths:
1) Arena API for MCTS vs Random/Greedy (native Agent-vs-Agent interface)
2) Gym wrapper for PPO vs MCTS (keeps PPO action-index mapping correct)

Outputs are routed into the organized MCTS tree by default:
  project/experiments/evaluation/robust/mcts/<bucket>/

Usage examples:
  python project/scripts/benchmark_mcts.py --games 20 --iterations 100
  python project/scripts/benchmark_mcts.py --games 100 --iterations 100 300 --bucket canonical
  python project/scripts/benchmark_mcts.py --skip-ppo
"""

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "modules")
sys.path.append("project/src")

import argparse
import json
import random
import types
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from math import sqrt

import numpy as np
import yaml

try:
    import neptune  # noqa: F401
except ModuleNotFoundError:
    neptune_stub = types.ModuleType("neptune")

    def _noop(*args, **kwargs):
        return None

    neptune_stub.log_metric = _noop
    sys.modules["neptune"] = neptune_stub

from agents.random_agent import RandomAgent
from agents.greedy_agent_boost import GreedyAgentBoost
from agents.single_mcts_agent import SingleMCTSAgent
from monte_carlo_tree_search.evaluation_policies.dummy_eval import DummyEval
from gym_splendor_code.envs.splendor import SplendorEnv
from gym_splendor_code.envs.mechanics.game_settings import MAX_NUMBER_OF_MOVES

from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO

from project.src.utils.splendor_gym_wrapper import SplendorGymWrapper
from project.src.utils.event_reward_wrapper import maybe_wrap_with_event_shaping


@dataclass
class MCTSPreset:
    label: str
    make_eval_policy: callable


class MCTSWrapperOpponent:
    """Adapter for using SingleMCTSAgent in SplendorGymWrapper opponent API."""

    def __init__(self, mcts_agent):
        self.mcts_agent = mcts_agent
        self.name = f"{mcts_agent.name}-wrapper"

    def choose_action(self, observation, previous_actions):
        normalized_previous = previous_actions if previous_actions else [None]
        return self.mcts_agent.choose_action(observation, normalized_previous)


def make_heura_evaluator():
    from monte_carlo_tree_search.evaluation_policies.heura_val import HeuraEvaluator

    return HeuraEvaluator()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MCTS in Splendor")
    parser.add_argument("--games", type=int, default=20, help="Games per matchup")
    parser.add_argument("--iterations", type=int, nargs="+", default=[100], help="MCTS iteration limits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--exploration", type=float, default=0.4)
    parser.add_argument(
        "--ppo-model",
        type=str,
        default="",
        help="Path to PPO/MaskablePPO checkpoint. Empty means auto-detect latest event model.",
    )
    parser.add_argument("--skip-ppo", action="store_true", help="Skip PPO vs MCTS benchmarks")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="project/experiments/evaluation/robust",
        help="Base output directory. Results are routed into mcts/<bucket>/ when possible.",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        choices=["archive", "canonical"],
        default="archive",
        help="Use archive for smoke/debug runs and canonical for citation-worthy runs.",
    )
    parser.add_argument("--tag", type=str, default="")
    return parser.parse_args()


def _resolve_model_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path

    zip_path = Path(f"{path_text}.zip")
    if zip_path.exists():
        return zip_path

    raise FileNotFoundError(f"Model checkpoint not found: {path_text}")


def find_latest_event_model() -> Path:
    candidates = list(Path("project/logs").glob("maskable_ppo_event_*/final_model.zip"))
    if not candidates:
        raise FileNotFoundError("No event checkpoint found under project/logs/maskable_ppo_event_*/final_model.zip")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_config_for_model(model_path: Path) -> dict:
    for parent in [model_path.parent, *model_path.parents]:
        candidate = parent / "config.yaml"
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    return {"environment": {"reward_mode": "score_progress"}}


def wilson_interval_95(wins: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p_hat = wins / total
    denom = 1.0 + (z * z) / total
    center = (p_hat + (z * z) / (2.0 * total)) / denom
    margin = (z / denom) * sqrt((p_hat * (1.0 - p_hat) / total) + ((z * z) / (4.0 * total * total)))
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return 100.0 * lo, 100.0 * hi


def run_local_duel(list_of_agents, starting_agent_id: int) -> dict:
    env = SplendorEnv()
    env.reset()
    env.set_active_player(starting_agent_id)
    env.set_players_names([agent.name for agent in list_of_agents])

    is_done = False
    action_count = 0
    active_agent_id = starting_agent_id
    observation = env.show_observation("deterministic")
    previous_actions = [None]
    winner_id = None

    while action_count < MAX_NUMBER_OF_MOVES and not is_done:
        active_agent = list_of_agents[active_agent_id]
        action = active_agent.choose_action(observation, previous_actions)
        previous_actions = [action]

        observation, _, is_done, info = env.step("deterministic", action)
        if is_done:
            winner_id = info.get("winner_id", None)
        active_agent_id = (active_agent_id + 1) % len(list_of_agents)
        action_count += 1

    return {
        "winner_id": winner_id,
        "points_agent0": int(env.points_of_player_by_id(0)),
        "points_agent1": int(env.points_of_player_by_id(1)),
        "turns": action_count,
        "is_done": bool(is_done),
    }


def run_arena_matchup(mcts_label: str, mcts_agent, opponent_agent, games: int) -> dict:
    wins = 0
    losses = 0
    draws = 0
    mcts_points = []
    opp_points = []
    mcts_first_games = 0
    mcts_first_wins = 0
    mcts_second_games = 0
    mcts_second_wins = 0
    terminal_games = 0
    non_terminal_games = 0
    winner_below_15_count = 0

    start = time.time()
    for _ in range(games):
        starting_agent_id = random.choice([0, 1])
        duel = run_local_duel([mcts_agent, opponent_agent], starting_agent_id=starting_agent_id)
        mcts_started = starting_agent_id == 0

        mcts_points.append(duel["points_agent0"])
        opp_points.append(duel["points_agent1"])

        if duel["is_done"]:
            terminal_games += 1
        else:
            non_terminal_games += 1

        if duel["winner_id"] == 0:
            wins += 1
            if duel["points_agent0"] < 15:
                winner_below_15_count += 1
            if mcts_started:
                mcts_first_wins += 1
            else:
                mcts_second_wins += 1
        elif duel["winner_id"] == 1:
            losses += 1
            if duel["points_agent1"] < 15:
                winner_below_15_count += 1
        else:
            draws += 1

        if mcts_started:
            mcts_first_games += 1
        else:
            mcts_second_games += 1

        mcts_agent.finish_game()
        opponent_agent.finish_game()

    elapsed = time.time() - start
    ci_lo, ci_hi = wilson_interval_95(wins, games)
    first_lo, first_hi = wilson_interval_95(mcts_first_wins, mcts_first_games)
    second_lo, second_hi = wilson_interval_95(mcts_second_wins, mcts_second_games)

    first_wr = 100.0 * mcts_first_wins / mcts_first_games if mcts_first_games > 0 else None
    second_wr = 100.0 * mcts_second_wins / mcts_second_games if mcts_second_games > 0 else None

    return {
        "path": "local-duel",
        "mcts": mcts_label,
        "opponent": opponent_agent.name,
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": 100.0 * wins / games,
        "win_rate_ci95": [ci_lo, ci_hi],
        "mcts_first": {
            "games": mcts_first_games,
            "wins": mcts_first_wins,
            "win_rate": first_wr,
            "win_rate_ci95": [first_lo, first_hi],
        },
        "mcts_second": {
            "games": mcts_second_games,
            "wins": mcts_second_wins,
            "win_rate": second_wr,
            "win_rate_ci95": [second_lo, second_hi],
        },
        "agent_avg_points": float(np.mean(mcts_points)) if mcts_points else 0.0,
        "opponent_avg_points": float(np.mean(opp_points)) if opp_points else 0.0,
        "agent_avg_reward": None,
        "opponent_avg_reward": None,
        "terminal_games": terminal_games,
        "non_terminal_games": non_terminal_games,
        "winner_below_15_count": winner_below_15_count,
        "wall_time_sec": elapsed,
        "sec_per_game": elapsed / games,
    }


def _get_legal_actions(env):
    current = env
    while current is not None:
        if hasattr(current, "cached_legal_actions"):
            return current.cached_legal_actions
        current = getattr(current, "env", None)
    return []


def _build_eval_env(config: dict, opponent_agent, max_turns: int, player_id: int):
    env_cfg = config.get("environment", {})
    base = SplendorGymWrapper(
        opponent_agent=opponent_agent,
        reward_mode=env_cfg.get("reward_mode", "score_progress"),
        max_turns=max_turns,
        player_id=player_id,
    )
    return maybe_wrap_with_event_shaping(base, config)


def _predict_action(model, obs, action_mask):
    try:
        action, _ = model.predict(obs, action_masks=action_mask, deterministic=True)
    except TypeError:
        action, _ = model.predict(obs, deterministic=True)
    return int(action)


def run_ppo_vs_mcts(model, config: dict, mcts_agent, games: int, max_turns: int) -> dict:
    wins = 0
    losses = 0
    draws = 0
    ppo_scores = []
    mcts_scores = []
    game_lengths = []
    mcts_first_games = 0
    mcts_first_wins = 0
    mcts_second_games = 0
    mcts_second_wins = 0
    winner_below_15_count = 0

    start = time.time()
    for game_idx in range(games):
        player_id = game_idx % 2
        mcts_opponent = MCTSWrapperOpponent(mcts_agent)
        env = _build_eval_env(config, opponent_agent=mcts_opponent, max_turns=max_turns, player_id=player_id)
        obs, info = env.reset()
        done = False

        while not done:
            legal_actions = _get_legal_actions(env)
            if len(legal_actions) == 0:
                obs, _, terminated, truncated, info = env.step(0)
                done = terminated or truncated
                continue

            action = _predict_action(model, obs, env.action_masks())
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        mcts_agent.finish_game()

        ppo_score = int(info.get("player_score", 0))
        opp_score = int(info.get("opponent_score", 0))
        mcts_started = (player_id == 1)

        ppo_scores.append(ppo_score)
        mcts_scores.append(opp_score)
        game_lengths.append(int(info.get("turn", 0)))

        if info.get("agent_won", False):
            wins += 1
        elif info.get("agent_lost", False):
            losses += 1
        else:
            if ppo_score > opp_score:
                wins += 1
            elif ppo_score < opp_score:
                losses += 1
            else:
                draws += 1

        if mcts_started:
            mcts_first_games += 1
            if ppo_score < opp_score:
                mcts_first_wins += 1
        else:
            mcts_second_games += 1
            if ppo_score < opp_score:
                mcts_second_wins += 1

        if ppo_score < opp_score and opp_score < 15:
            winner_below_15_count += 1
        if ppo_score > opp_score and ppo_score < 15:
            winner_below_15_count += 1

    elapsed = time.time() - start
    total = wins + losses + draws
    mcts_wins = losses
    mcts_losses = wins
    ci_lo, ci_hi = wilson_interval_95(mcts_wins, total)
    first_lo, first_hi = wilson_interval_95(mcts_first_wins, mcts_first_games)
    second_lo, second_hi = wilson_interval_95(mcts_second_wins, mcts_second_games)

    first_wr = 100.0 * mcts_first_wins / mcts_first_games if mcts_first_games > 0 else None
    second_wr = 100.0 * mcts_second_wins / mcts_second_games if mcts_second_games > 0 else None

    return {
        "path": "gym-wrapper",
        "games": total,
        "wins": mcts_wins,
        "losses": mcts_losses,
        "draws": draws,
        "win_rate": 100.0 * mcts_wins / total if total > 0 else 0.0,
        "win_rate_ci95": [ci_lo, ci_hi],
        "mcts_first": {
            "games": mcts_first_games,
            "wins": mcts_first_wins,
            "win_rate": first_wr,
            "win_rate_ci95": [first_lo, first_hi],
        },
        "mcts_second": {
            "games": mcts_second_games,
            "wins": mcts_second_wins,
            "win_rate": second_wr,
            "win_rate_ci95": [second_lo, second_hi],
        },
        "agent_avg_points": float(np.mean(mcts_scores)) if mcts_scores else 0.0,
        "opponent_avg_points": float(np.mean(ppo_scores)) if ppo_scores else 0.0,
        "game_length_mean": float(np.mean(game_lengths)) if game_lengths else 0.0,
        "mcts_name": mcts_agent.name,
        "opponent": "PPO/MaskablePPO",
        "winner_below_15_count": winner_below_15_count,
        "wall_time_sec": elapsed,
        "sec_per_game": elapsed / total if total > 0 else None,
        "note": "MCTS win/loss is derived by inverting PPO perspective from wrapper eval.",
    }


def load_ppo_or_maskable(model_path: Path):
    errors = []
    try:
        return MaskablePPO.load(str(model_path)), "MaskablePPO"
    except Exception as exc:  # pragma: no cover
        errors.append(f"MaskablePPO.load failed: {exc}")

    try:
        return PPO.load(str(model_path)), "PPO"
    except Exception as exc:  # pragma: no cover
        errors.append(f"PPO.load failed: {exc}")

    raise RuntimeError("Could not load PPO checkpoint. " + " | ".join(errors))


def resolve_output_dir(output_arg: str, bucket: str) -> Path:
    """Keep compatibility with the old CLI while defaulting to robust/mcts/<bucket>/."""
    output_dir = Path(output_arg)
    if output_dir.name in {"archive", "canonical"} and output_dir.parent.name == "mcts":
        return output_dir
    if output_dir.name == "mcts":
        return output_dir / bucket
    if output_dir.name == "robust":
        return output_dir / "mcts" / bucket
    return output_dir


def make_mcts(preset: MCTSPreset, iteration_limit: int, exploration: float):
    return SingleMCTSAgent(
        iteration_limit=iteration_limit,
        evaluation_policy=preset.make_eval_policy(),
        exploration_parameter=exploration,
        create_visualizer=False,
    )


def save_outputs(output_dir: Path, tag: str, payload: dict) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_part = f"_{tag}" if tag else ""
    json_path = output_dir / f"mcts_benchmark{tag_part}_{timestamp}.json"
    md_path = output_dir / f"mcts_benchmark{tag_part}_{timestamp}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    lines = []
    lines.append("# MCTS Benchmark Report")
    lines.append("")
    lines.append(f"- Timestamp: {payload['meta']['timestamp']}")
    lines.append(f"- Games per matchup: {payload['meta']['games_per_matchup']}")
    lines.append(f"- Iterations tested: {payload['meta']['iterations']}")
    lines.append(f"- PPO model: {payload['meta'].get('ppo_model', '(skipped)')}")
    lines.append(f"- Bucket: {payload['meta'].get('bucket', 'archive')}")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Matchup | Path | Win rate | 95% CI | MCTS pts | Opp pts | Sec/game | Sanity |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    for row in payload["results"]:
        ci = row.get("win_rate_ci95", [0.0, 0.0])
        sanity = row.get("winner_below_15_count", 0)
        lines.append(
            "| {matchup} | {path} | {win_rate:.1f}% | [{ci0:.1f}, {ci1:.1f}] | {agent_avg_points:.2f} | {opponent_avg_points:.2f} | {sec_per_game:.3f} | {sanity} |".format(
                matchup=row["matchup"],
                path=row["path"],
                win_rate=row["win_rate"],
                ci0=ci[0],
                ci1=ci[1],
                agent_avg_points=row["agent_avg_points"],
                opponent_avg_points=row["opponent_avg_points"],
                sec_per_game=row["sec_per_game"],
                sanity=sanity,
            )
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- local-duel path is used for MCTS vs classic agents (Random/Greedy).")
    lines.append("- Gym-wrapper path is used for MCTS vs PPO to preserve PPO action-index semantics.")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return json_path, md_path


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    ppo_model_path = None
    ppo_model = None
    ppo_model_type = None
    ppo_config = None

    if not args.skip_ppo:
        if args.ppo_model.strip():
            ppo_model_path = _resolve_model_path(args.ppo_model.strip())
        else:
            ppo_model_path = find_latest_event_model()

        ppo_model, ppo_model_type = load_ppo_or_maskable(ppo_model_path)
        ppo_config = load_config_for_model(ppo_model_path)

    presets = [
        MCTSPreset(label="pure_dummy", make_eval_policy=DummyEval),
        MCTSPreset(label="guided_heura", make_eval_policy=make_heura_evaluator),
    ]

    rows = []
    skipped = []
    for iteration_limit in args.iterations:
        for preset in presets:
            try:
                _ = preset.make_eval_policy()
            except Exception as exc:
                skipped.append(
                    {
                        "preset": preset.label,
                        "iteration": iteration_limit,
                        "reason": str(exc),
                    }
                )
                continue

            for opponent_label, opponent_factory in [
                ("random_uniform_on_types", lambda: RandomAgent(distribution="uniform_on_types")),
                ("greedy_value", lambda: GreedyAgentBoost(name="Greedy", mode="value")),
            ]:
                mcts_agent = make_mcts(preset, iteration_limit, args.exploration)
                opponent = opponent_factory()
                arena_stats = run_arena_matchup(
                    mcts_label=f"{preset.label}@{iteration_limit}",
                    mcts_agent=mcts_agent,
                    opponent_agent=opponent,
                    games=args.games,
                )
                arena_stats["matchup"] = f"MCTS[{preset.label},iter={iteration_limit}] vs {opponent_label}"
                rows.append(arena_stats)

            if not args.skip_ppo:
                mcts_agent = make_mcts(preset, iteration_limit, args.exploration)
                ppo_stats = run_ppo_vs_mcts(
                    model=ppo_model,
                    config=ppo_config,
                    mcts_agent=mcts_agent,
                    games=args.games,
                    max_turns=args.max_turns,
                )
                ppo_stats["mcts"] = f"{preset.label}@{iteration_limit}"
                ppo_stats["matchup"] = f"MCTS[{preset.label},iter={iteration_limit}] vs PPO"
                ppo_stats["ppo_model_type"] = ppo_model_type
                rows.append(ppo_stats)

    payload = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "games_per_matchup": args.games,
            "iterations": args.iterations,
            "seed": args.seed,
            "exploration": args.exploration,
            "ppo_model": str(ppo_model_path) if ppo_model_path else None,
            "bucket": args.bucket,
            "skipped": skipped,
        },
        "results": rows,
    }

    output_dir = resolve_output_dir(args.output_dir, args.bucket)
    json_path, md_path = save_outputs(output_dir, args.tag, payload)

    print("=" * 88)
    print("MCTS Benchmark Summary")
    print("=" * 88)
    for row in rows:
        ci = row.get("win_rate_ci95", [0.0, 0.0])
        first_wr = row.get("mcts_first", {}).get("win_rate", None)
        second_wr = row.get("mcts_second", {}).get("win_rate", None)
        first_text = f"{first_wr:.1f}%" if first_wr is not None else "NA"
        second_text = f"{second_wr:.1f}%" if second_wr is not None else "NA"
        print(
            f"{row['matchup']:<52} win={row['win_rate']:5.1f}% "
            f"CI95=[{ci[0]:.1f},{ci[1]:.1f}] "
            f"first={first_text} second={second_text} "
            f"mcts_pts={row['agent_avg_points']:.2f} opp_pts={row['opponent_avg_points']:.2f} "
            f"sec/game={row['sec_per_game']:.3f} sanity={row.get('winner_below_15_count', 0)}"
        )
    print("=" * 88)
    print(f"Bucket: {args.bucket}")
    if skipped:
        print("Skipped presets:")
        for item in skipped:
            print(f"  {item['preset']} @ {item['iteration']}: {item['reason']}")
        print("=" * 88)
    print(f"Saved JSON: {json_path}")
    print(f"Saved MD:   {md_path}")


if __name__ == "__main__":
    main()
