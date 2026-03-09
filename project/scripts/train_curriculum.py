"""
MaskablePPO Curriculum Training Script

Reads a config YAML with a `curriculum.stages` block and trains the model
across multiple stages, hot-swapping the opponent environment between stages.
Each stage can also change the learning rate.

Supports two config patterns:
  1. Full stages list (v4c):
       curriculum:
         stages:
           - {opponent, timesteps, lr, label}
           - ...

  2. Two-phase shorthand (v4b):
       curriculum:
         phase2_opponent: "greedy"
         phase2_timesteps: 300000
         phase2_lr: 0.0001
     (Phase 1 comes from environment.opponent + training.total_timesteps)

Usage:
    python project/scripts/train_curriculum.py
    python project/scripts/train_curriculum.py --config project/configs/training/maskable_ppo_v4c_curriculum.yaml
    python project/scripts/train_curriculum.py --config project/configs/training/maskable_ppo_v4b_rollout_curriculum.yaml
    python project/scripts/train_curriculum.py --resume path/to/model.zip --config ...
"""

import sys
import os
sys.path.insert(0, "project/src")  # lower priority: utils/
sys.path.insert(0, "modules")      # higher priority: agents/

import argparse
import yaml
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

try:
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback as EvalCallback
    USING_MASKABLE_EVAL = True
except ImportError:
    from stable_baselines3.common.callbacks import EvalCallback
    USING_MASKABLE_EVAL = False

from utils.event_reward_wrapper import event_shaping_enabled, maybe_wrap_with_event_shaping
from utils.event_stats_callback import EventStatsCallback
from utils.splendor_gym_wrapper import make_splendor_env
from agents.random_agent import RandomAgent
from agents.greedy_agent_boost import GreedyAgentBoost


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _mask_fn(env) -> np.ndarray:
    return env.action_masks()


def _make_opponent(opp_str):
    """Instantiate opponent from string label."""
    if opp_str is None:
        return None
    s = str(opp_str).lower().strip()
    if s in ("none", ""):
        return None
    if s in ("random", "random_agent"):
        return RandomAgent()
    if s in ("greedy", "greedy_agent"):
        return GreedyAgentBoost(name="Greedy", mode="value")
    raise ValueError(f"Unknown opponent: '{opp_str}'. Use: none, random, greedy.")


def _make_env(config: dict, opponent_str: str, monitor_dir: str = None):
    """Create a fresh ActionMasker-wrapped SplendorGymWrapper."""
    env_cfg = config["environment"]
    opp = _make_opponent(opponent_str)
    env = make_splendor_env(
        reward_mode=env_cfg["reward_mode"],
        opponent_agent=opp,
        max_turns=env_cfg["max_turns"],
    )
    env = maybe_wrap_with_event_shaping(env, config)
    env = ActionMasker(env, _mask_fn)
    if monitor_dir:
        os.makedirs(monitor_dir, exist_ok=True)
        env = Monitor(env, monitor_dir)
    return env


def _build_stages(config: dict) -> list[dict]:
    """
    Extract the list of curriculum stages from config.
    Returns a list of dicts: {opponent, timesteps, lr, label}
    """
    curr = config.get("curriculum", {})

    # Full stages list (v4c pattern)
    if "stages" in curr:
        stages = []
        for s in curr["stages"]:
            stages.append({
                "opponent":   s.get("opponent", config["environment"]["opponent"]),
                "timesteps":  s["timesteps"],
                "lr":         s.get("lr", config["ppo"]["learning_rate"]),
                "label":      s.get("label", f"stage_{len(stages)+1}"),
            })
        return stages

    # Two-phase shorthand (v4b pattern)
    if "phase2_opponent" in curr:
        stage1 = {
            "opponent":  config["environment"]["opponent"],
            "timesteps": config["training"]["total_timesteps"],
            "lr":        config["ppo"]["learning_rate"],
            "label":     "Phase1_Random",
        }
        stage2 = {
            "opponent":  curr["phase2_opponent"],
            "timesteps": curr["phase2_timesteps"],
            "lr":        curr.get("phase2_lr", config["ppo"]["learning_rate"]),
            "label":     f"Phase2_{curr['phase2_opponent'].capitalize()}",
        }
        return [stage1, stage2]

    # No curriculum block → single-stage (identical to train_maskable_ppo.py)
    return [{
        "opponent":  config["environment"].get("opponent", "random"),
        "timesteps": config["training"]["total_timesteps"],
        "lr":        config["ppo"]["learning_rate"],
        "label":     "SingleStage",
    }]


def _make_model(env, config: dict, tensorboard_log: str, lr_override: float = None) -> MaskablePPO:
    ppo = config["ppo"]
    pk = dict(ppo.get("policy_kwargs", {}))
    act = pk.get("activation_fn", "relu")
    pk["activation_fn"] = torch.nn.ReLU if act == "relu" else torch.nn.Tanh

    lr = lr_override if lr_override is not None else ppo["learning_rate"]

    return MaskablePPO(
        policy=ppo["policy"],
        env=env,
        learning_rate=lr,
        n_steps=ppo["n_steps"],
        batch_size=ppo["batch_size"],
        n_epochs=ppo["n_epochs"],
        gamma=ppo["gamma"],
        gae_lambda=ppo["gae_lambda"],
        clip_range=ppo["clip_range"],
        clip_range_vf=ppo.get("clip_range_vf"),
        normalize_advantage=ppo["normalize_advantage"],
        ent_coef=ppo["ent_coef"],
        vf_coef=ppo["vf_coef"],
        max_grad_norm=ppo["max_grad_norm"],
        policy_kwargs=pk,
        verbose=config["training"]["verbose"],
        tensorboard_log=tensorboard_log,
    )


def _make_callbacks(config, log_path, eval_env, stage_label) -> CallbackList:
    callbacks = []

    ckpt_path = os.path.join(log_path, "logs/checkpoints", stage_label)
    os.makedirs(ckpt_path, exist_ok=True)
    callbacks.append(CheckpointCallback(
        save_freq=config["training"]["save_freq"],
        save_path=ckpt_path,
        name_prefix=config["checkpoints"]["name_prefix"] + f"_{stage_label}",
    ))

    eval_path = os.path.join(log_path, "eval", stage_label)
    os.makedirs(eval_path, exist_ok=True)

    eval_kwargs = dict(
        eval_env=eval_env,
        best_model_save_path=eval_path,
        log_path=eval_path,
        eval_freq=config["training"]["eval_freq"],
        n_eval_episodes=config["training"]["eval_episodes"],
        deterministic=config["evaluation"]["deterministic"],
    )
    if USING_MASKABLE_EVAL:
        eval_kwargs["use_masking"] = True
    callbacks.append(EvalCallback(**eval_kwargs))
    if event_shaping_enabled(config):
        callbacks.append(
            EventStatsCallback(
                log_freq=config.get("event_shaping", {}).get(
                    "log_freq",
                    config["training"]["eval_freq"],
                )
            )
        )

    return CallbackList(callbacks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Curriculum MaskablePPO training")
    parser.add_argument(
        "--config",
        default="project/configs/training/maskable_ppo_v4c_curriculum.yaml",
    )
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    stages = _build_stages(config)

    # Output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = config["experiment"]["name"]
    log_path = f"project/logs/{exp_name}_{ts}"
    os.makedirs(log_path, exist_ok=True)

    # Save config snapshot
    with open(os.path.join(log_path, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    tboard_root = os.path.join(log_path, config["training"]["tensorboard_log"])
    os.makedirs(tboard_root, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Experiment : {exp_name}")
    print(f"  Log path   : {log_path}")
    print(f"  Stages     : {len(stages)}")
    for i, s in enumerate(stages, 1):
        print(f"    {i}. {s['label']:<30} opp={s['opponent']:<10} "
              f"steps={s['timesteps']:>7,}  lr={s['lr']}")
    print(f"  Total steps: {sum(s['timesteps'] for s in stages):,}")
    print(f"{'='*70}\n")

    model = None
    cumulative_steps = 0

    for stage_idx, stage in enumerate(stages, 1):
        print(f"\n{'─'*70}")
        print(f"  Starting stage {stage_idx}/{len(stages)}: {stage['label']}")
        print(f"  Opponent : {stage['opponent']}")
        print(f"  Steps    : {stage['timesteps']:,}  (total so far: {cumulative_steps:,})")
        print(f"  LR       : {stage['lr']}")
        print(f"{'─'*70}\n")

        monitor_dir = os.path.join(log_path, "monitor", stage["label"])
        train_env = _make_env(config, stage["opponent"], monitor_dir=monitor_dir)
        eval_env  = _make_env(config, stage["opponent"])

        if model is None:
            # First stage — create model or resume
            if args.resume:
                print(f"Resuming from: {args.resume}")
                model = MaskablePPO.load(args.resume, env=train_env)
                model.tensorboard_log = tboard_root
                # Apply stage LR
                model.learning_rate = stage["lr"]
            else:
                model = _make_model(train_env, config, tboard_root, lr_override=stage["lr"])
        else:
            # Subsequent stage — swap env and adjust LR
            model.set_env(train_env)
            model.learning_rate = stage["lr"]
            model.lr_schedule = lambda _: stage["lr"]
            print(f"  ↳ Swapped environment to: opp={stage['opponent']}, lr={stage['lr']}")

        callbacks = _make_callbacks(config, log_path, eval_env, stage["label"])

        try:
            model.learn(
                total_timesteps=stage["timesteps"],
                callback=callbacks,
                log_interval=config["training"]["log_interval"],
                tb_log_name=f"{exp_name}_{stage['label']}",
                reset_num_timesteps=(stage_idx == 1 and not args.resume),
            )
        except KeyboardInterrupt:
            interrupted_path = os.path.join(log_path, f"interrupted_{stage['label']}")
            model.save(interrupted_path)
            print(f"\n⚠️  Interrupted at stage {stage_idx}. Saved → {interrupted_path}.zip")
            train_env.close(); eval_env.close()
            return

        # Save stage checkpoint
        stage_ckpt = os.path.join(log_path, f"model_after_{stage['label']}")
        model.save(stage_ckpt)
        print(f"\n  ✓ Stage {stage_idx} complete → {stage_ckpt}.zip")

        cumulative_steps += stage["timesteps"]
        train_env.close()
        eval_env.close()

    # Final save
    final_path = os.path.join(log_path, "final_model")
    model.save(final_path)

    print(f"\n{'='*70}")
    print(f"  ✅ All {len(stages)} stages complete!")
    print(f"  Final model → {final_path}.zip")
    print(f"  Total steps : {cumulative_steps:,}")
    print(f"\n  Evaluate with:")
    print(f"    python project/scripts/evaluate_maskable_ppo.py \\")
    print(f"      --model {final_path} --games 100")
    print(f"\n  Compare checkpoints:")
    print(f"    python project/scripts/compare_checkpoints.py \\")
    print(f"      --model-a project/logs/maskable_ppo_score_v3.../final_model \\")
    print(f"      --model-b {final_path}")
    print(f"\n  TensorBoard:")
    print(f"    tensorboard --logdir {tboard_root} --host 0.0.0.0 --port 6006")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
