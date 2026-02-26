"""
PPO Training Script for Score-based Splendor Agent

Trains a PPO agent to play Splendor using score-based rewards.
Supports TensorBoard logging, checkpointing, and evaluation.

Usage:
    python project/scripts/train_score_based.py [--config CONFIG_PATH]

Author: AI Agent
Date: 2026-02-24
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
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList, CheckpointCallback, EvalCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from utils.splendor_gym_wrapper import make_splendor_env


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_env(config: dict, monitor_dir: str = None):
    """Create and wrap Splendor environment."""
    env_config = config['environment']
    
    env = make_splendor_env(
        reward_mode=env_config['reward_mode'],
        opponent_agent=None,  # Random opponent for now
        max_turns=env_config['max_turns']
    )
    
    if monitor_dir:
        os.makedirs(monitor_dir, exist_ok=True)
        env = Monitor(env, monitor_dir)
    
    return env


def create_eval_env(config: dict):
    """Create evaluation environment."""
    env_config = config['environment']
    env = make_splendor_env(
        reward_mode=env_config['reward_mode'],
        opponent_agent=None,
        max_turns=env_config['max_turns']
    )
    return env


def setup_callbacks(config: dict, log_path: str, eval_env):
    """Setup training callbacks."""
    callbacks = []
    
    # Checkpoint callback - save model periodically
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
    
    # Evaluation callback - evaluate during training
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
    """Create PPO model with configured hyperparameters."""
    ppo_config = config['ppo']
    
    # Parse policy kwargs
    policy_kwargs = ppo_config.get('policy_kwargs', {})
    if 'activation_fn' in policy_kwargs:
        # Convert string to actual function
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
    """Main training loop."""
    parser = argparse.ArgumentParser(description='Train PPO agent on Splendor')
    parser.add_argument(
        '--config',
        type=str,
        default='project/configs/training/ppo_score_based.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    # Setup paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = config['experiment']['name']
    log_path = f"project/logs/{exp_name}_{timestamp}"
    os.makedirs(log_path, exist_ok=True)
    
    # Save config to log directory
    config_save_path = os.path.join(log_path, "config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    print(f"Config saved to: {config_save_path}")
    
    # Set seed
    if 'seed' in config:
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
    
    # Create environments
    print("Creating environments...")
    monitor_dir = os.path.join(log_path, "monitor")
    train_env = create_env(config, monitor_dir=monitor_dir)
    eval_env = create_eval_env(config)
    
    print(f"Environment: {train_env}")
    print(f"  Observation space: {train_env.observation_space}")
    print(f"  Action space: {train_env.action_space}")
    
    # Create or load model
    tensorboard_log = os.path.join(log_path, config['training']['tensorboard_log'])
    os.makedirs(tensorboard_log, exist_ok=True)
    
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        model = PPO.load(args.resume, env=train_env)
        model.tensorboard_log = tensorboard_log
    else:
        print("Creating new PPO model...")
        model = create_ppo_model(train_env, config, tensorboard_log)
    
    # Print model info
    print("\nModel Configuration:")
    print(f"  Policy: {model.policy.__class__.__name__}")
    print(f"  Learning rate: {model.learning_rate}")
    print(f"  Batch size: {model.batch_size}")
    print(f"  Device: {model.device}")
    
    # Setup callbacks
    print("\nSetting up callbacks...")
    callbacks = setup_callbacks(config, log_path, eval_env)
    
    # Train
    print(f"\n{'='*60}")
    print(f"Starting training for {config['training']['total_timesteps']:,} timesteps")
    print(f"TensorBoard logs: {tensorboard_log}")
    print(f"Checkpoints: {os.path.join(log_path, config['checkpoints']['save_path'])}")
    print(f"{'='*60}\n")
    
    try:
        model.learn(
            total_timesteps=config['training']['total_timesteps'],
            callback=callbacks,
            log_interval=config['training']['log_interval'],
            tb_log_name=exp_name,
            reset_num_timesteps=not bool(args.resume)
        )
        
        # Save final model
        final_model_path = os.path.join(log_path, "final_model")
        model.save(final_model_path)
        print(f"\nTraining completed! Final model saved to: {final_model_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        interrupt_path = os.path.join(log_path, "interrupted_model")
        model.save(interrupt_path)
        print(f"Model saved to: {interrupt_path}")
    
    finally:
        train_env.close()
        eval_env.close()
    
    print("\nTo view training progress:")
    print(f"  tensorboard --logdir {tensorboard_log}")


if __name__ == "__main__":
    main()
