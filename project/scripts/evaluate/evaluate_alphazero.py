"""Robust evaluation for AlphaZero-style Splendor checkpoints.

This script evaluates an AlphaZero MCTS policy against:
1) RandomAgent and GreedyAgent via local Splendor duel API
2) PPO/MaskablePPO via SplendorGymWrapper (for action-index correctness)

Outputs:
  project/experiments/evaluation/robust/mcts/<bucket>/
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
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO

sys.path.insert(0, ".")
sys.path.insert(0, "modules")
sys.path.append("project/src")

from agents.greedy_agent_boost import GreedyAgentBoost
from agents.random_agent import RandomAgent
from gym_splendor_code.envs.mechanics.game_settings import MAX_NUMBER_OF_MOVES
from gym_splendor_code.envs.splendor import SplendorEnv
from mcts.action_indexer import StableActionIndexer
from mcts.alphazero_mcts import AlphaZeroMCTS, TorchPolicyValueFunction
from nn.policy_value_net import SplendorPolicyValueNet
from nn.tensor_encoder import SplendorTensorEncoder
from project.src.utils.event_reward_wrapper import maybe_wrap_with_event_shaping
from project.src.utils.splendor_gym_wrapper import SplendorGymWrapper


@dataclass
class AlphaZeroRuntime:
    """Pack encoder/model/searcher runtime for reuse across moves."""

    encoder: SplendorTensorEncoder
    model: SplendorPolicyValueNet
    mcts: AlphaZeroMCTS
    device: str


class AlphaZeroAgentAdapter:
    """Adapter exposing choose_action() API compatible with legacy arenas/wrappers."""

    def __init__(self, runtime: AlphaZeroRuntime, name: str = "AlphaZeroMCTS") -> None:
        self.runtime = runtime
        self.name = name

    def choose_action(self, observation, previous_actions):
        state = observation.recreate_state()
        try:
            result = self.runtime.mcts.search(state, temperature=1e-8)
            return result.selected_action
        except RuntimeError as exc:
            if "No legal actions available at root" in str(exc):
                return None
            raise

    def finish_game(self) -> None:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AlphaZero-style Splendor checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to AlphaZero checkpoint (.pt). Empty means auto-detect latest final_model.pt.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config path. If omitted, infer from checkpoint folder or payload config.",
    )
    parser.add_argument("--games", type=int, default=100, help="Games per matchup")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--simulations", type=int, default=0, help="Override MCTS simulations if > 0")
    parser.add_argument("--c-puct", type=float, default=0.0, help="Override PUCT coefficient if > 0")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.0, help="Override dirichlet alpha if > 0")
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.0, help="Override dirichlet epsilon if >= 0")
    parser.add_argument(
        "--ppo-model",
        type=str,
        default="",
        help="Path to PPO/MaskablePPO checkpoint. Empty means auto-detect latest event model.",
    )
    parser.add_argument("--skip-ppo", action="store_true", help="Skip PPO-vs-AlphaZero matchup")
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


def find_latest_alphazero_checkpoint() -> Path:
    candidates = list(Path("project/logs").glob("alphazero_*/final_model.pt"))
    candidates.extend(list(Path("project/logs").glob("alphazero_*/alphazero_iter_*.pt")))
    if not candidates:
        raise FileNotFoundError("No AlphaZero checkpoint found under project/logs/alphazero_*/")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def find_latest_event_model() -> Path:
    candidates = list(Path("project/logs").glob("maskable_ppo_event_*/final_model.zip"))
    if not candidates:
        raise FileNotFoundError("No event checkpoint found under project/logs/maskable_ppo_event_*/final_model.zip")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_checkpoint(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path
    raise FileNotFoundError(f"Checkpoint not found: {path_text}")


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_alphazero_config(checkpoint: Path, explicit_config: str | None, payload: dict) -> tuple[dict, str]:
    if explicit_config:
        cfg = load_yaml(Path(explicit_config))
        return cfg, str(Path(explicit_config).resolve())

    for parent in [checkpoint.parent, *checkpoint.parents]:
        candidate = parent / "config.yaml"
        if candidate.exists():
            return load_yaml(candidate), str(candidate)

    payload_cfg = payload.get("config")
    if isinstance(payload_cfg, dict):
        return payload_cfg, "(from checkpoint payload)"

    raise ValueError("Could not infer AlphaZero config. Pass --config explicitly.")


def create_runtime(checkpoint_path: Path, config: dict, args: argparse.Namespace) -> AlphaZeroRuntime:
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    net_cfg = config.get("network", {})
    mcts_cfg = config.get("mcts", {})
    train_cfg = config.get("training", {})

    encoder = SplendorTensorEncoder()
    policy_size = int(net_cfg.get("policy_size", 256))
    device = str(train_cfg.get("device", "cpu"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    model = SplendorPolicyValueNet(
        input_channels=encoder.spec.channels,
        policy_size=policy_size,
        trunk_channels=int(net_cfg.get("trunk_channels", 128)),
        num_res_blocks=int(net_cfg.get("num_res_blocks", 3)),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    indexer = StableActionIndexer(policy_size=policy_size)
    indexer_state = payload.get("action_indexer_state")
    if isinstance(indexer_state, dict):
        indexer.load_state_dict(indexer_state)

    policy_fn = TorchPolicyValueFunction(
        encoder=encoder,
        model=model,
        device=device,
        action_indexer=indexer,
    )

    num_simulations = int(args.simulations) if args.simulations > 0 else int(mcts_cfg.get("num_simulations", 100))
    c_puct = float(args.c_puct) if args.c_puct > 0 else float(mcts_cfg.get("c_puct", 1.5))
    dirichlet_alpha = float(args.dirichlet_alpha) if args.dirichlet_alpha > 0 else float(mcts_cfg.get("dirichlet_alpha", 0.3))
    if args.dirichlet_epsilon >= 0:
        dirichlet_epsilon = float(args.dirichlet_epsilon)
    else:
        dirichlet_epsilon = float(mcts_cfg.get("dirichlet_epsilon", 0.25))

    mcts = AlphaZeroMCTS(
        policy_value_fn=policy_fn,
        num_simulations=num_simulations,
        c_puct=c_puct,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        max_depth=min(int(args.max_turns), MAX_NUMBER_OF_MOVES),
    )
    return AlphaZeroRuntime(encoder=encoder, model=model, mcts=mcts, device=device)


def _resolve_model_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists():
        return path

    zip_path = Path(f"{path_text}.zip")
    if zip_path.exists():
        return zip_path

    raise FileNotFoundError(f"Model checkpoint not found: {path_text}")


def load_config_for_model(model_path: Path) -> dict:
    for parent in [model_path.parent, *model_path.parents]:
        candidate = parent / "config.yaml"
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    return {"environment": {"reward_mode": "score_progress"}}


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


def run_arena_matchup(alpha_agent, opponent_agent, games: int, seed: int) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    wins = 0
    losses = 0
    draws = 0
    alpha_points = []
    opp_points = []

    start = time.time()
    for _ in range(games):
        starting_agent_id = random.choice([0, 1])
        duel = run_local_duel([alpha_agent, opponent_agent], starting_agent_id=starting_agent_id)

        alpha_points.append(duel["points_agent0"])
        opp_points.append(duel["points_agent1"])

        if duel["winner_id"] == 0:
            wins += 1
        elif duel["winner_id"] == 1:
            losses += 1
        else:
            draws += 1

        alpha_agent.finish_game()
        opponent_agent.finish_game()

    elapsed = time.time() - start
    ci_lo, ci_hi = wilson_interval_95(wins, games)

    return {
        "path": "local-duel",
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": 100.0 * wins / games,
        "win_rate_ci95": [ci_lo, ci_hi],
        "agent_avg_points": float(np.mean(alpha_points)) if alpha_points else 0.0,
        "opponent_avg_points": float(np.mean(opp_points)) if opp_points else 0.0,
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


def run_ppo_vs_alpha(model, ppo_config: dict, alpha_agent, games: int, max_turns: int, seed: int) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    wins = 0
    losses = 0
    draws = 0
    ppo_scores = []
    alpha_scores = []

    start = time.time()
    for game_idx in range(games):
        player_id = game_idx % 2
        env = _build_eval_env(
            config=ppo_config,
            opponent_agent=alpha_agent,
            max_turns=max_turns,
            player_id=player_id,
        )

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

        alpha_agent.finish_game()

        ppo_score = int(info.get("player_score", 0))
        alpha_score = int(info.get("opponent_score", 0))

        ppo_scores.append(ppo_score)
        alpha_scores.append(alpha_score)

        if info.get("agent_won", False):
            losses += 1  # PPO win => alpha loss
        elif info.get("agent_lost", False):
            wins += 1    # PPO loss => alpha win
        else:
            if alpha_score > ppo_score:
                wins += 1
            elif alpha_score < ppo_score:
                losses += 1
            else:
                draws += 1

    elapsed = time.time() - start
    ci_lo, ci_hi = wilson_interval_95(wins, games)
    return {
        "path": "gym-wrapper",
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": 100.0 * wins / games if games > 0 else 0.0,
        "win_rate_ci95": [ci_lo, ci_hi],
        "agent_avg_points": float(np.mean(alpha_scores)) if alpha_scores else 0.0,
        "opponent_avg_points": float(np.mean(ppo_scores)) if ppo_scores else 0.0,
        "wall_time_sec": elapsed,
        "sec_per_game": elapsed / games if games > 0 else None,
        "note": "Alpha win/loss is derived by inverting PPO perspective from wrapper eval.",
    }


def resolve_output_dir(output_arg: str, bucket: str) -> Path:
    output_dir = Path(output_arg)
    if output_dir.name in {"archive", "canonical"} and output_dir.parent.name == "mcts":
        return output_dir
    if output_dir.name == "mcts":
        return output_dir / bucket
    if output_dir.name == "robust":
        return output_dir / "mcts" / bucket
    return output_dir


def save_outputs(output_dir: Path, tag: str, payload: dict) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_part = f"_{tag}" if tag else ""
    json_path = output_dir / f"alphazero_eval{tag_part}_{timestamp}.json"
    md_path = output_dir / f"alphazero_eval{tag_part}_{timestamp}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    lines = []
    lines.append("# AlphaZero Robust Evaluation Report")
    lines.append("")
    lines.append(f"- Timestamp: {payload['meta']['timestamp']}")
    lines.append(f"- Games per matchup: {payload['meta']['games_per_matchup']}")
    lines.append(f"- Alpha checkpoint: {payload['meta']['alpha_checkpoint']}")
    lines.append(f"- Bucket: {payload['meta']['bucket']}")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Matchup | Path | Win rate | 95% CI | Alpha pts | Opp pts | Sec/game |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for row in payload["results"]:
        ci = row.get("win_rate_ci95", [0.0, 0.0])
        lines.append(
            "| {matchup} | {path} | {win_rate:.1f}% | [{ci0:.1f}, {ci1:.1f}] | {agent_avg_points:.2f} | {opponent_avg_points:.2f} | {sec_per_game:.3f} |".format(
                matchup=row["matchup"],
                path=row["path"],
                win_rate=row["win_rate"],
                ci0=ci[0],
                ci1=ci[1],
                agent_avg_points=row["agent_avg_points"],
                opponent_avg_points=row["opponent_avg_points"],
                sec_per_game=row["sec_per_game"],
            )
        )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return json_path, md_path


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    checkpoint_path = resolve_checkpoint(args.checkpoint.strip()) if args.checkpoint.strip() else find_latest_alphazero_checkpoint()
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    alpha_config, alpha_config_source = load_alphazero_config(checkpoint_path, args.config, payload)
    runtime = create_runtime(checkpoint_path, alpha_config, args)
    alpha_agent = AlphaZeroAgentAdapter(runtime=runtime)

    rows = []
    for opponent_label, opponent_factory in [
        ("random_uniform_on_types", lambda: RandomAgent(distribution="uniform_on_types")),
        ("greedy_value", lambda: GreedyAgentBoost(name="Greedy", mode="value")),
    ]:
        opponent = opponent_factory()
        result = run_arena_matchup(alpha_agent=alpha_agent, opponent_agent=opponent, games=args.games, seed=args.seed)
        result["matchup"] = f"AlphaZeroMCTS vs {opponent_label}"
        rows.append(result)

    ppo_model_path = None
    ppo_model_type = None
    if not args.skip_ppo:
        ppo_model_path = _resolve_model_path(args.ppo_model.strip()) if args.ppo_model.strip() else find_latest_event_model()
        ppo_model, ppo_model_type = load_ppo_or_maskable(ppo_model_path)
        ppo_config = load_config_for_model(ppo_model_path)
        result = run_ppo_vs_alpha(
            model=ppo_model,
            ppo_config=ppo_config,
            alpha_agent=alpha_agent,
            games=args.games,
            max_turns=args.max_turns,
            seed=args.seed,
        )
        result["matchup"] = "AlphaZeroMCTS vs PPO"
        result["ppo_model_type"] = ppo_model_type
        rows.append(result)

    output_dir = resolve_output_dir(args.output_dir, args.bucket)
    payload_out = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "games_per_matchup": args.games,
            "seed": args.seed,
            "max_turns": args.max_turns,
            "alpha_checkpoint": str(checkpoint_path),
            "alpha_config_source": alpha_config_source,
            "alpha_device": runtime.device,
            "ppo_model": str(ppo_model_path) if ppo_model_path else None,
            "ppo_model_type": ppo_model_type,
            "bucket": args.bucket,
            "tag": args.tag,
        },
        "results": rows,
    }
    json_path, md_path = save_outputs(output_dir, args.tag, payload_out)

    print("=" * 88)
    print("AlphaZero Robust Evaluation Summary")
    print("=" * 88)
    for row in rows:
        ci = row.get("win_rate_ci95", [0.0, 0.0])
        print(
            f"{row['matchup']:<44} win={row['win_rate']:5.1f}% "
            f"CI95=[{ci[0]:.1f},{ci[1]:.1f}] "
            f"alpha_pts={row['agent_avg_points']:.2f} opp_pts={row['opponent_avg_points']:.2f} "
            f"sec/game={row['sec_per_game']:.3f}"
        )
    print("=" * 88)
    print(f"Saved JSON: {json_path}")
    print(f"Saved MD:   {md_path}")


if __name__ == "__main__":
    main()
