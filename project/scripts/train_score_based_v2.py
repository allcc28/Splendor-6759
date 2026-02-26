"""
PPO v2 Training Script — Score-Based with Greedy Opponent

Key difference from v1: trains against GreedyAgentBoost instead of random opponent.
This is Experiment 1: testing whether a stronger training opponent improves policy quality.

Hypothesis:
    Training vs. Greedy opponent forces the agent to learn more structured, long-term
    strategies, resulting in a policy that beats v1 (trained vs. random) in head-to-head.

Usage:
    python project/scripts/train_score_based_v2.py
    python project/scripts/train_score_based_v2.py --config project/configs/training/ppo_score_based_v2_greedy_opp.yaml

Evaluation (after training):
    python project/scripts/evaluate_v1_vs_v2.py

Author: Yehao Yan
Date: 2026-02-25
"""

import sys
import os

# Step 1: add modules/ FIRST so agents.* resolves from modules/agents/, not project/src/agents/
sys.path.insert(0, "modules")

# Import legacy agents NOW, before project/src is added (which would shadow modules/agents/)
from agents.greedy_agent_boost import GreedyAgentBoost

# Step 2: add project/src for utils.* imports
sys.path.insert(0, "project/src")

import argparse
import yaml
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList, CheckpointCallback, EvalCallback
)
from stable_baselines3.common.monitor import Monitor

from utils.splendor_gym_wrapper import make_splendor_env
# GreedyAgentBoost already imported at the top before project/src was added to sys.path


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_env(config: dict, opponent, monitor_dir: str = None):
    env_config = config['environment']
    env = make_splendor_env(
        reward_mode=env_config['reward_mode'],
        opponent_agent=opponent,
        max_turns=env_config['max_turns']
    )
    if monitor_dir:
        os.makedirs(monitor_dir, exist_ok=True)
        env = Monitor(env, monitor_dir)
    return env


def setup_callbacks(config: dict, log_path: str, eval_env):
    callbacks = []

    checkpoint_config = config['checkpoints']
    checkpoint_path = os.path.join(log_path, checkpoint_config['save_path'])
    os.makedirs(checkpoint_path, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=config['training']['save_freq'],
        save_path=checkpoint_path,
        name_prefix=checkpoint_config['name_prefix'],
        save_replay_buffer=checkpoint_config['save_replay_buffer'],
        save_vecnormalize=checkpoint_config['save_vecnormalize']
    )
    callbacks.append(checkpoint_callback)

    eval_path = os.path.join(log_path, "eval")
    os.makedirs(eval_path, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=eval_path,
        log_path=eval_path,
        eval_freq=config['training']['eval_freq'],
        n_eval_episodes=config['training']['eval_episodes'],
        deterministic=config['evaluation']['deterministic'],
        render=config['evaluation']['render']
    )
    callbacks.append(eval_callback)

    return CallbackList(callbacks)


def create_ppo_model(env, config: dict, tensorboard_log: str):
    ppo_config = config['ppo']
    policy_kwargs = ppo_config.get('policy_kwargs', {})
    if 'activation_fn' in policy_kwargs:
        if policy_kwargs['activation_fn'] == 'relu':
            policy_kwargs['activation_fn'] = torch.nn.ReLU
        elif policy_kwargs['activation_fn'] == 'tanh':
            policy_kwargs['activation_fn'] = torch.nn.Tanh

    model = PPO(
        policy=ppo_config['policy'],
        env=env,
        learning_rate=ppo_config['learning_rate'],
        n_steps=ppo_config['n_steps'],
        batch_size=ppo_config['batch_size'],
        n_epochs=ppo_config['n_epochs'],
        gamma=ppo_config['gamma'],
        gae_lambda=ppo_config['gae_lambda'],
        clip_range=ppo_config['clip_range'],
        clip_range_vf=ppo_config.get('clip_range_vf'),
        normalize_advantage=ppo_config['normalize_advantage'],
        ent_coef=ppo_config['ent_coef'],
        vf_coef=ppo_config['vf_coef'],
        max_grad_norm=ppo_config['max_grad_norm'],
        policy_kwargs=policy_kwargs,
        verbose=config['training']['verbose'],
        tensorboard_log=tensorboard_log,
        device=config['device'],
        seed=config.get('seed')
    )
    return model


def main():
    parser = argparse.ArgumentParser(description='Train PPO v2 vs Greedy opponent')
    parser.add_argument(
        '--config',
        type=str,
        default='project/configs/training/ppo_score_based_v2_greedy_opp.yaml',
        help='Path to config file'
    )
    args = parser.parse_args()

    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Setup paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = config['experiment']['name']
    log_path = f"project/logs/{exp_name}_{timestamp}"
    os.makedirs(log_path, exist_ok=True)

    # Save config snapshot
    with open(os.path.join(log_path, "config.yaml"), 'w') as f:
        yaml.dump(config, f)
    print(f"Config saved to: {log_path}/config.yaml")

    # Seed
    if 'seed' in config:
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])

    # Create GREEDY opponent — key difference vs v1
    # mode='event' uses action.evaluate() (fast); mode='value' deep-copies the full state per action (too slow).
    print("\n[EXPERIMENT 1] Opponent: GreedyAgentBoost(mode='event') (vs v1's RandomAgent)")
    greedy_opponent = GreedyAgentBoost(mode="event")

    # Create environments
    print("Creating environments...")
    monitor_dir = os.path.join(log_path, "monitor")
    train_env = create_env(config, opponent=greedy_opponent, monitor_dir=monitor_dir)

    # Eval env also uses greedy opponent — consistent evaluation
    eval_env = create_env(config, opponent=GreedyAgentBoost(mode="value"))

    print(f"  Observation space: {train_env.observation_space}")
    print(f"  Action space: {train_env.action_space}")

    # Create model
    tensorboard_log = os.path.join(log_path, config['training']['tensorboard_log'])
    os.makedirs(tensorboard_log, exist_ok=True)

    print("Creating new PPO model...")
    model = create_ppo_model(train_env, config, tensorboard_log)

    print(f"\n  Policy:        {model.policy.__class__.__name__}")
    print(f"  Learning rate: {model.learning_rate}")
    print(f"  Batch size:    {model.batch_size}")
    print(f"  Device:        {model.device}")

    # Callbacks
    callbacks = setup_callbacks(config, log_path, eval_env)

    # Train
    print(f"\n{'='*60}")
    print(f"EXPERIMENT 1 — PPO v2 (Greedy Opponent)")
    print(f"Total timesteps: {config['training']['total_timesteps']:,}")
    print(f"TensorBoard:     {tensorboard_log}")
    print(f"Checkpoints:     {os.path.join(log_path, config['checkpoints']['save_path'])}")
    print(f"{'='*60}\n")

    try:
        model.learn(
            total_timesteps=config['training']['total_timesteps'],
            callback=callbacks,
            log_interval=config['training']['log_interval'],
            tb_log_name=exp_name,
            reset_num_timesteps=True
        )

        # Save final model
        final_model_path = os.path.join(log_path, "final_model")
        model.save(final_model_path)
        print(f"\nTraining complete. Final model: {final_model_path}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        interrupt_path = os.path.join(log_path, "interrupted_model")
        model.save(interrupt_path)
        print(f"Saved to: {interrupt_path}")

    finally:
        train_env.close()
        eval_env.close()

    print(f"\nNext step — run head-to-head evaluation:")
    print(f"  python project/scripts/evaluate_v1_vs_v2.py --v2_model {log_path}/final_model")
    print(f"\nTo monitor training:")
    print(f"  tensorboard --logdir {tensorboard_log}")


if __name__ == "__main__":
    main()
