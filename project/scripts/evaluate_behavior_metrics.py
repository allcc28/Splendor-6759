"""
Behavior-metrics evaluation for event-based Splendor agents.

Adds metrics not covered by evaluate_maskable_ppo.py:
- reservation frequency
- buy_reserved frequency
- noble acquisition rate
- average score by turn 20 / 40 / 60
- average purchased cards by color
- average game length

Usage example:
python project/scripts/evaluate_behavior_metrics.py \
  --model project/logs/<run>/eval/best_model \
  --config project/configs/training/<cfg>.yaml \
  --games 200 \
  --tag e4_behavior
"""

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "modules")
sys.path.append("project/src")

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from sb3_contrib import MaskablePPO
from tqdm import tqdm

from agents.greedy_agent_boost import GreedyAgentBoost
from agents.random_agent import RandomAgent
from gym_splendor_code.envs.mechanics.enums import GemColor
from project.src.reward.event_based_reward import EVENT_NAMES
from project.src.utils.event_detector import capture_state_snapshot, detect_events
from project.src.utils.event_reward_wrapper import maybe_wrap_with_event_shaping
from project.src.utils.splendor_gym_wrapper import SplendorGymWrapper

TURN_CHECKPOINTS = (20, 40, 60)
COLOR_ORDER = (GemColor.RED, GemColor.GREEN, GemColor.BLUE, GemColor.WHITE, GemColor.BLACK)


def load_eval_config(model_path: str, config_path: str | None) -> tuple[dict, str]:
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f), str(Path(config_path).resolve())

    model = Path(model_path).resolve()
    for parent in [model.parent, *model.parents]:
        candidate = parent / "config.yaml"
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as f:
                return yaml.safe_load(f), str(candidate)

    return {"environment": {"reward_mode": "score_progress"}}, "(default score-only)"


def create_eval_env(config: dict, opponent_agent, max_turns: int, player_id: int):
    env_cfg = config.get("environment", {})
    env = SplendorGymWrapper(
        opponent_agent=opponent_agent,
        reward_mode=env_cfg.get("reward_mode", "score_progress"),
        max_turns=max_turns,
        player_id=player_id,
    )
    return maybe_wrap_with_event_shaping(env, config)


def find_base_env(env):
    current = env
    while current is not None:
        if hasattr(current, "cached_legal_actions") and hasattr(current, "player_id"):
            return current
        current = getattr(current, "env", None)
    raise RuntimeError("Could not locate SplendorGymWrapper in wrapper chain")


def _score_at_turn(score_trace: list[tuple[int, int]], checkpoint: int) -> int:
    if not score_trace:
        return 0
    for turn, score in score_trace:
        if turn >= checkpoint:
            return score
    return score_trace[-1][1]


def _color_counts(players_hand) -> dict[str, int]:
    discount = players_hand.discount()
    return {
        "red": int(discount.value(GemColor.RED)),
        "green": int(discount.value(GemColor.GREEN)),
        "blue": int(discount.value(GemColor.BLUE)),
        "white": int(discount.value(GemColor.WHITE)),
        "black": int(discount.value(GemColor.BLACK)),
    }


def evaluate_behavior_vs_opponent(model, config, opponent_agent, games: int, max_turns: int, desc: str):
    wins = 0
    losses = 0
    draws = 0

    game_lengths = []
    final_agent_scores = []
    final_opp_scores = []

    event_totals = np.zeros(len(EVENT_NAMES), dtype=np.int64)
    total_agent_steps = 0

    checkpoint_scores = {cp: [] for cp in TURN_CHECKPOINTS}
    noble_counts = []
    color_totals = {"red": 0, "green": 0, "blue": 0, "white": 0, "black": 0}

    for game_idx in tqdm(range(games), desc=desc):
        player_id = game_idx % 2
        env = create_eval_env(config, opponent_agent, max_turns=max_turns, player_id=player_id)
        obs, info = env.reset()
        base_env = find_base_env(env)

        done = False
        score_trace: list[tuple[int, int]] = []

        while not done:
            n_legal = len(base_env.cached_legal_actions)
            if n_legal == 0:
                obs, reward, terminated, truncated, info = env.step(0)
                done = terminated or truncated
                score_trace.append((int(info.get("turn", 0)), int(info.get("player_score", 0))))
                continue

            prev_snapshot = capture_state_snapshot(
                base_env.env.current_state_of_the_game,
                base_env.player_id,
            )
            action_idx, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
            action_idx = int(action_idx)
            action_obj = base_env.cached_legal_actions[action_idx]

            obs, reward, terminated, truncated, info = env.step(action_idx)
            done = terminated or truncated

            next_snapshot = base_env.last_post_agent_snapshot
            if next_snapshot is None:
                next_snapshot = capture_state_snapshot(
                    base_env.env.current_state_of_the_game,
                    base_env.player_id,
                )
            events = detect_events(prev_snapshot, action_obj, next_snapshot)
            event_totals += events.astype(np.int64)
            total_agent_steps += 1

            score_trace.append((int(info.get("turn", 0)), int(info.get("player_score", 0))))

        agent_score = int(info.get("player_score", 0))
        opp_score = int(info.get("opponent_score", 0))
        agent_won = bool(info.get("agent_won", False))
        agent_lost = bool(info.get("agent_lost", False))

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

        game_lengths.append(int(info.get("turn", 0)))
        final_agent_scores.append(agent_score)
        final_opp_scores.append(opp_score)

        for cp in TURN_CHECKPOINTS:
            checkpoint_scores[cp].append(_score_at_turn(score_trace, cp))

        hand = base_env.env.current_state_of_the_game.list_of_players_hands[base_env.player_id]
        noble_counts.append(len(hand.nobles_possessed))
        counts = _color_counts(hand)
        for c in color_totals:
            color_totals[c] += counts[c]

    total_games = wins + losses + draws
    event_rates_per_step = {
        EVENT_NAMES[i]: (float(event_totals[i]) / total_agent_steps) if total_agent_steps > 0 else 0.0
        for i in range(len(EVENT_NAMES))
    }

    return {
        "games": total_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": (wins / total_games * 100.0) if total_games > 0 else 0.0,
        "agent_score_mean": float(np.mean(final_agent_scores)) if final_agent_scores else 0.0,
        "opponent_score_mean": float(np.mean(final_opp_scores)) if final_opp_scores else 0.0,
        "average_game_length": float(np.mean(game_lengths)) if game_lengths else 0.0,
        "event_rates_per_step": event_rates_per_step,
        "reservation_frequency": event_rates_per_step.get("reserve_card", 0.0),
        "buy_reserved_frequency": event_rates_per_step.get("buy_reserved", 0.0),
        "noble_acquisition_rate": float(np.mean(noble_counts)) if noble_counts else 0.0,
        "avg_score_by_turn": {
            str(cp): float(np.mean(checkpoint_scores[cp])) if checkpoint_scores[cp] else 0.0
            for cp in TURN_CHECKPOINTS
        },
        "avg_purchased_cards_by_color": {
            c: (float(color_totals[c]) / total_games) if total_games > 0 else 0.0
            for c in color_totals
        },
    }


def print_behavior_summary(label: str, metrics: dict):
    print("\n" + "=" * 72)
    print(f"Behavior metrics vs {label}")
    print("=" * 72)
    print(f"Win rate: {metrics['win_rate']:.1f}% ({metrics['wins']}/{metrics['games']})")
    print(f"Score: agent {metrics['agent_score_mean']:.2f} / opp {metrics['opponent_score_mean']:.2f}")
    print(f"Avg game length: {metrics['average_game_length']:.2f}")
    print(
        "Event rates/step: "
        f"reserve_card={metrics['reservation_frequency']:.4f}, "
        f"buy_reserved={metrics['buy_reserved_frequency']:.4f}"
    )
    print(f"Noble acquisition rate (avg nobles/game): {metrics['noble_acquisition_rate']:.4f}")
    print(
        "Avg score by turn: "
        f"T20={metrics['avg_score_by_turn']['20']:.2f}, "
        f"T40={metrics['avg_score_by_turn']['40']:.2f}, "
        f"T60={metrics['avg_score_by_turn']['60']:.2f}"
    )
    colors = metrics["avg_purchased_cards_by_color"]
    print(
        "Avg purchased cards by color: "
        f"R={colors['red']:.2f}, G={colors['green']:.2f}, B={colors['blue']:.2f}, "
        f"W={colors['white']:.2f}, K={colors['black']:.2f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate behavior metrics for MaskablePPO Splendor agents")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--tag", type=str, default="behavior")
    parser.add_argument(
        "--output",
        type=str,
        default="project/experiments/evaluation/behavior_metrics",
        help="Output directory for JSON report",
    )
    args = parser.parse_args()

    config, config_source = load_eval_config(args.model, args.config)
    model = MaskablePPO.load(args.model)

    opponents = {
        "random_wrapper": None,
        "random_agent": RandomAgent(distribution="uniform_on_types"),
        "greedy_agent": GreedyAgentBoost(name="Greedy", mode="value"),
    }

    all_results = {}
    for label, opponent in opponents.items():
        metrics = evaluate_behavior_vs_opponent(
            model,
            config,
            opponent_agent=opponent,
            games=args.games,
            max_turns=args.max_turns,
            desc=f"{args.tag}:{label}",
        )
        all_results[label] = metrics
        print_behavior_summary(label, metrics)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"behavior_metrics_{args.tag}_{ts}.json"

    payload = {
        "_meta": {
            "tag": args.tag,
            "timestamp": ts,
            "model_path": args.model,
            "model_path_abs": str(Path(args.model).resolve()),
            "config_source": config_source,
            "games_per_opponent": args.games,
            "max_turns": args.max_turns,
        },
        "results": all_results,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\nSaved behavior metrics ->", out_file)


if __name__ == "__main__":
    main()
