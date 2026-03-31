"""
Evaluation script for MaskablePPO v3 agent.

Key differences from evaluate_score_based_v3.py:
  - Loads model with MaskablePPO.load() (not PPO.load)
  - Passes action_masks to model.predict() at each step
  - No need for "fallback mode" — masking ensures only legal actions are sampled
  - Still reports strict mode stats for apples-to-apples comparison with V1

Usage:
    python project/scripts/evaluate_maskable_ppo.py
    python project/scripts/evaluate_maskable_ppo.py --model path/to/model --games 100
"""

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "modules")
sys.path.append("project/src")  # for reward.* and utils.* sub-modules

import argparse
import json
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


def _get_legal_actions(env):
    """Walk wrapper chain to find cached_legal_actions on SplendorGymWrapper."""
    current = env
    while current is not None:
        if hasattr(current, "cached_legal_actions"):
            return current.cached_legal_actions
        current = getattr(current, "env", None)
    return []


def evaluate_vs_opponent(
    model,
    config,
    opponent_agent,
    num_games: int,
    max_turns: int = 200,
    desc: str = "Evaluating",
):
    """
    Evaluate MaskablePPO model vs an opponent.

    At each step, `action_masks=env.action_masks()` is passed to predict()
    so the model only picks legal actions — no fallback needed.
    """
    wins = 0
    losses = 0
    draws = 0
    agent_scores = []
    opponent_scores = []
    game_lengths = []
    total_rewards = []
    zero_actions_count = 0

    for game_idx in tqdm(range(num_games), desc=desc):
        # Alternate which player moves first to remove first-mover bias.
        # Even games: agent is player 0 (moves first).
        # Odd  games: agent is player 1 (opponent moves first).
        player_id = game_idx % 2
        env = create_eval_env(
            config=config,
            opponent_agent=opponent_agent,
            max_turns=max_turns,
            player_id=player_id,
        )
        if game_idx == 0:
            validate_observation_shape(model, env)
        obs, info = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            n_legal = len(_get_legal_actions(env))

            if n_legal == 0:
                zero_actions_count += 1
                # Truncated inside wrapper, step with any action to get info
                obs, reward, terminated, truncated, info = env.step(0)
                ep_reward += reward
                done = terminated or truncated
                continue

            # MaskablePPO enforces the mask internally — no invalid actions possible
            action, _ = model.predict(
                obs,
                action_masks=env.action_masks(),
                deterministic=True,
            )
            action = int(action)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        agent_score = info.get("player_score", 0)
        opp_score = info.get("opponent_score", 0)
        agent_won = info.get("agent_won", False)
        agent_lost = info.get("agent_lost", False)

        agent_scores.append(agent_score)
        opponent_scores.append(opp_score)
        game_lengths.append(info.get("turn", 0))
        total_rewards.append(ep_reward)

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

    total = wins + losses + draws
    return {
        "total_games": total,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": (wins / total * 100) if total > 0 else 0,
        "loss_rate": (losses / total * 100) if total > 0 else 0,
        "agent_scores": {
            "mean": float(np.mean(agent_scores)),
            "std": float(np.std(agent_scores)),
            "min": int(np.min(agent_scores)),
            "max": int(np.max(agent_scores)),
            "median": float(np.median(agent_scores)),
        },
        "opponent_scores": {
            "mean": float(np.mean(opponent_scores)),
            "std": float(np.std(opponent_scores)),
            "min": int(np.min(opponent_scores)),
            "max": int(np.max(opponent_scores)),
        },
        "game_lengths": {
            "mean": float(np.mean(game_lengths)),
            "std": float(np.std(game_lengths)),
            "min": int(np.min(game_lengths)),
            "max": int(np.max(game_lengths)),
        },
        "rewards": {
            "mean": float(np.mean(total_rewards)),
            "std": float(np.std(total_rewards)),
        },
        "invalid_actions": {
            "mean_per_game": 0.0,
            "total": 0,
            "games_with_invalid": 0,
            "note": "MaskablePPO: impossible by design",
        },
        "zero_actions_episodes": zero_actions_count,
    }


def print_results(label: str, stats: dict):
    print(f"\n{'=' * 60}")
    print(f"Results vs {label} ({stats['total_games']} games)")
    print(f"{'=' * 60}")
    print(f"  Win/Loss/Draw: {stats['wins']}/{stats['losses']}/{stats['draws']}")
    print(f"  Win Rate:      {stats['win_rate']:.1f}%")
    print()
    print(f"  Agent Scores:    {stats['agent_scores']['mean']:.1f} ± {stats['agent_scores']['std']:.1f}"
          f"  (range: {stats['agent_scores']['min']}–{stats['agent_scores']['max']})")
    print(f"  Opponent Scores: {stats['opponent_scores']['mean']:.1f} ± {stats['opponent_scores']['std']:.1f}"
          f"  (range: {stats['opponent_scores']['min']}–{stats['opponent_scores']['max']})")
    print(f"  Game Length:     {stats['game_lengths']['mean']:.1f} ± {stats['game_lengths']['std']:.1f} turns")
    print(f"  Avg Reward:      {stats['rewards']['mean']:.2f}")
    print(f"  Invalid Actions: 0 (masked by design)")
    if stats["zero_actions_episodes"] > 0:
        print(f"  Zero-legal-actions episodes: {stats['zero_actions_episodes']}")
    if stats["agent_scores"]["mean"] < 3:
        print(f"\n  ⚠️  WARNING: Very low scores — check game loop!")
    if stats["agent_scores"]["max"] >= 15:
        print(f"  ✅ Agent reached 15+ points in at least one game")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MaskablePPO v3 agent")
    parser.add_argument(
        "--model",
        type=str,
        default="project/logs/maskable_ppo_score_v3_20260303_183435/final_model",
        help="Path to trained MaskablePPO model (without .zip)",
    )
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional training config. If omitted, inferred from model directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="project/experiments/evaluation/maskable_ppo_eval",
        help="Directory to write eval JSON into",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="Short label for this run (e.g. v3, v4a). Embedded in filename and JSON.",
    )
    args = parser.parse_args()

    tag_label = args.tag if args.tag else "(untagged)"
    print("=" * 80)
    print(f"MaskablePPO Agent Evaluation  [{tag_label}]")
    print("=" * 80)
    print(f"Model:   {args.model}")
    print(f"Tag:     {tag_label}")
    print(f"Games:   {args.games} per opponent")
    print(f"Masking: ON (no invalid actions possible)")
    print()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config, config_source = load_eval_config(args.model, args.config)

    print("Loading MaskablePPO model...")
    model = MaskablePPO.load(args.model)
    print(f"Config:  {config_source}")
    print("✓ Model loaded\n")

    all_results = {}

    # vs Random (wrapper: opponent_agent=None → built-in fast uniform random)
    # NOTE: Training uses _make_opponent('random') → RandomAgent() object with
    # full choose_action() API. These are slightly different random strategies.
    # Label: "Random (wrapper)" to distinguish from "RandomAgent" below.
    stats = evaluate_vs_opponent(
        model, config, opponent_agent=None, num_games=args.games,
        max_turns=args.max_turns, desc=f"{tag_label} vs Random",
    )
    all_results["vs_random_wrapper"] = stats
    print_results("Random (wrapper)", stats)

    # vs RandomAgent object
    random_agent = RandomAgent(distribution="uniform_on_types")
    stats = evaluate_vs_opponent(
        model, config, opponent_agent=random_agent, num_games=args.games,
        max_turns=args.max_turns, desc=f"{tag_label} vs RandomAgent",
    )
    all_results["vs_random_agent"] = stats
    print_results("RandomAgent", stats)

    # vs GreedyAgent
    greedy_agent = GreedyAgentBoost(name="Greedy", mode="value")
    stats = evaluate_vs_opponent(
        model, config, opponent_agent=greedy_agent, num_games=args.games,
        max_turns=args.max_turns, desc=f"{tag_label} vs GreedyAgent",
    )
    all_results["vs_greedy"] = stats
    print_results("GreedyAgent", stats)

    # Save — embed provenance so results can always be traced back to a model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_part = f"_{args.tag}" if args.tag else ""
    results_file = output_dir / f"eval_maskable{tag_part}_{timestamp}.json"
    model_path_abs = str(Path(args.model).resolve())
    output_payload = {
        "_meta": {
            "model_path": args.model,
            "model_path_abs": model_path_abs,
            "eval_tag": args.tag or "(unset)",
            "timestamp": timestamp,
            "games_per_opponent": args.games,
            "max_turns": args.max_turns,
            "config_source": config_source,
            "event_shaping_enabled": bool(config.get("event_shaping", {}).get("enabled", False)),
        },
        **all_results,
    }
    with open(results_file, "w") as f:
        json.dump(output_payload, f, indent=2, default=str)

    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")
    for label, key in [("Random (wrapper)", "vs_random_wrapper"),
                        ("RandomAgent", "vs_random_agent"),
                        ("GreedyAgent", "vs_greedy")]:
        r = all_results[key]
        print(f"  {label:<22} {r['win_rate']:>5.1f}% win  "
              f"Agent {r['agent_scores']['mean']:.1f} pts  "
              f"Opp {r['opponent_scores']['mean']:.1f} pts")
    print(f"\n✓ Saved → {results_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
