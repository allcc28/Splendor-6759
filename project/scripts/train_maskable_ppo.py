"""
MaskablePPO Training Script — Score-based Splendor Agent (v3)

Phase 9: Architecture Pivot — Action Masking
Upgrade from plain PPO to MaskablePPO (sb3-contrib).
The key change: the policy can only sample from *legal* actions at each step,
eliminating the -10 invalid-action penalty that dominated v1/v2 training.

Usage:
    python project/scripts/train_maskable_ppo.py
    python project/scripts/train_maskable_ppo.py --config path/to/config.yaml
    python project/scripts/train_maskable_ppo.py --resume path/to/model.zip

Author: AI Agent
Date: 2026-03-03
"""

import sys
import os
sys.path.insert(0, "modules")
sys.path.insert(0, "project/src")

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
from stable_baselines3.common.vec_env import DummyVecEnv

# Try to import the maskable eval callback; fall back to standard if not available
try:
    from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback as EvalCallback
    USING_MASKABLE_EVAL = True
except ImportError:
    from stable_baselines3.common.callbacks import EvalCallback
    USING_MASKABLE_EVAL = False

from utils.splendor_gym_wrapper import make_splendor_env
from agents.random_agent import RandomAgent


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _mask_fn(env) -> np.ndarray:
    """Callback for ActionMasker: return the current legal-action mask."""
    return env.action_masks()


def _make_opponent(opponent_cfg):
    """
    Instantiate an opponent agent from the config string.

    Supported values:
      'random' / 'random_agent' → RandomAgent()
      null / None / 'none'      → None  (wrapper uses built-in fast-random)

    Raises ValueError for unrecognised strings so misconfiguration is
    caught immediately rather than silently falling back to random.
    """
    if opponent_cfg is None:
        return None
    opp = str(opponent_cfg).lower().strip()
    if opp in ("random", "random_agent", "none", ""):
        return RandomAgent() if opp in ("random", "random_agent") else None
    raise ValueError(
        f"Unknown opponent config value: '{opponent_cfg}'. "
        "Supported: 'random', null/None."
    )


def create_env(config: dict, monitor_dir: str = None):
    """
    Create a single training environment wrapped with ActionMasker so that
    MaskablePPO can find the mask via env.action_masks().
    """
    env_config = config["environment"]
    opponent = _make_opponent(env_config.get("opponent"))
    env = make_splendor_env(
        reward_mode=env_config["reward_mode"],
        opponent_agent=opponent,
        max_turns=env_config["max_turns"],
    )
    env = ActionMasker(env, _mask_fn)       # ← exposes action_masks() to SB3
    if monitor_dir:
        os.makedirs(monitor_dir, exist_ok=True)
        env = Monitor(env, monitor_dir)
    return env


def create_eval_env(config: dict):
    """Create a separate eval environment (also masked)."""
    env_config = config["environment"]
    opponent = _make_opponent(env_config.get("opponent"))
    env = make_splendor_env(
        reward_mode=env_config["reward_mode"],
        opponent_agent=opponent,
        max_turns=env_config["max_turns"],
    )
    return ActionMasker(env, _mask_fn)


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

def create_model(env, config: dict, tensorboard_log: str) -> MaskablePPO:
    """Instantiate MaskablePPO with config hyperparameters."""
    ppo_cfg = config["ppo"]

    policy_kwargs = dict(ppo_cfg.get("policy_kwargs", {}))
    if policy_kwargs.get("activation_fn") == "relu":
        policy_kwargs["activation_fn"] = torch.nn.ReLU
    elif policy_kwargs.get("activation_fn") == "tanh":
        policy_kwargs["activation_fn"] = torch.nn.Tanh

    return MaskablePPO(
        policy=ppo_cfg["policy"],
        env=env,
        learning_rate=ppo_cfg["learning_rate"],
        n_steps=ppo_cfg["n_steps"],
        batch_size=ppo_cfg["batch_size"],
        n_epochs=ppo_cfg["n_epochs"],
        gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"],
        clip_range=ppo_cfg["clip_range"],
        clip_range_vf=ppo_cfg.get("clip_range_vf"),
        normalize_advantage=ppo_cfg["normalize_advantage"],
        ent_coef=ppo_cfg["ent_coef"],
        vf_coef=ppo_cfg["vf_coef"],
        max_grad_norm=ppo_cfg["max_grad_norm"],
        policy_kwargs=policy_kwargs,
        verbose=config["training"]["verbose"],
        tensorboard_log=tensorboard_log,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────────────────────────────────────

def setup_callbacks(config: dict, log_path: str, eval_env) -> CallbackList:
    callbacks = []

    # Checkpoint every N steps
    ckpt_cfg = config["checkpoints"]
    ckpt_path = os.path.join(log_path, ckpt_cfg["save_path"])
    os.makedirs(ckpt_path, exist_ok=True)
    callbacks.append(CheckpointCallback(
        save_freq=config["training"]["save_freq"],
        save_path=ckpt_path,
        name_prefix=ckpt_cfg["name_prefix"],
    ))

    # Evaluation callback
    eval_path = os.path.join(log_path, "eval")
    os.makedirs(eval_path, exist_ok=True)

    if USING_MASKABLE_EVAL:
        # MaskableEvalCallback is aware of action masks
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=eval_path,
            log_path=eval_path,
            eval_freq=config["training"]["eval_freq"],
            n_eval_episodes=config["training"]["eval_episodes"],
            deterministic=config["evaluation"]["deterministic"],
            use_masking=True,
        )
    else:
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=eval_path,
            log_path=eval_path,
            eval_freq=config["training"]["eval_freq"],
            n_eval_episodes=config["training"]["eval_episodes"],
            deterministic=config["evaluation"]["deterministic"],
        )

    callbacks.append(eval_cb)
    return CallbackList(callbacks)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train MaskablePPO on Splendor")
    parser.add_argument(
        "--config",
        type=str,
        default="project/configs/training/maskable_ppo_score_based.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint zip to resume from",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = config["experiment"]["name"]
    log_path = f"project/logs/{exp_name}_{timestamp}"
    os.makedirs(log_path, exist_ok=True)

    # Persist config snapshot
    with open(os.path.join(log_path, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    print(f"\n{'='*60}")
    print(f"  Experiment : {exp_name}")
    print(f"  Log path   : {log_path}")
    print(f"  Masking    : {'MaskableEvalCallback' if USING_MASKABLE_EVAL else 'EvalCallback (fallback)'}")
    print(f"{'='*60}\n")

    # Environments
    monitor_dir = os.path.join(log_path, "monitor")
    train_env = create_env(config, monitor_dir=monitor_dir)
    eval_env  = create_eval_env(config)

    print(f"Observation space : {train_env.observation_space}")
    print(f"Action space      : {train_env.action_space}")

    # TensorBoard
    tensorboard_log = os.path.join(log_path, config["training"]["tensorboard_log"])
    os.makedirs(tensorboard_log, exist_ok=True)

    # Model
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        model = MaskablePPO.load(args.resume, env=train_env)
        model.tensorboard_log = tensorboard_log
    else:
        print("\nCreating new MaskablePPO model...")
        model = create_model(train_env, config, tensorboard_log)

    print(f"\nPolicy         : {model.policy.__class__.__name__}")
    print(f"Learning rate  : {model.learning_rate}")
    print(f"Batch size     : {model.batch_size}")
    print(f"Device         : {model.device}\n")

    # Callbacks
    callbacks = setup_callbacks(config, log_path, eval_env)

    print(f"{'='*60}")
    print(f"Training for {config['training']['total_timesteps']:,} timesteps …")
    print(f"TensorBoard : {tensorboard_log}")
    print(f"{'='*60}\n")

    try:
        model.learn(
            total_timesteps=config["training"]["total_timesteps"],
            callback=callbacks,
            log_interval=config["training"]["log_interval"],
            tb_log_name=exp_name,
            reset_num_timesteps=not bool(args.resume),
        )

        final_path = os.path.join(log_path, "final_model")
        model.save(final_path)
        print(f"\n✅ Training done. Model saved → {final_path}.zip")

    except KeyboardInterrupt:
        interrupted_path = os.path.join(log_path, "interrupted_model")
        model.save(interrupted_path)
        print(f"\n⚠️  Interrupted. Model saved → {interrupted_path}.zip")

    finally:
        train_env.close()
        eval_env.close()

    print(f"\nTensorBoard:\n  tensorboard --logdir {tensorboard_log} --host 0.0.0.0 --port 6006")


if __name__ == "__main__":
    main()
