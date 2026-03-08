"""
Event-based training script (isolated copy).
"""
import sys
import os
from pathlib import Path

# Fix numpy/tensorboard compatibility: numpy >= 1.24 removed np.bool8, but old tensorboard uses it
import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = bool
# Fix protobuf compatibility issue with tensorboard
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Ensure correct path setup regardless of where script is called from
script_dir = Path(__file__).resolve().parent  # .../project_event_based/scripts
root_dir = script_dir.parent.parent  # .../Splendor-6759
sys.path.insert(0, str(root_dir / 'project_event_based' / 'src'))
sys.path.insert(0, str(root_dir / 'modules'))

import argparse
import yaml
from datetime import datetime

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from utils.splendor_gym_wrapper import make_splendor_env
from utils.state_vectorizer_event import vectorize_state_event
from utils.event_detector import detect_events
from reward.event_based_reward import compute_event_reward

# Optional maskable policy support (sb3-contrib)
try:
    from sb3_contrib.maskable import MaskablePPO
    from sb3_contrib.common.maskable import ActionMasker
    HAS_MASKABLE = True
except Exception:
    MaskablePPO = None
    ActionMasker = None
    HAS_MASKABLE = False


class EventRewardWrapper(gym.Wrapper):
    def __init__(self, env, combine_with_score: bool = False, event_weights=None):
        super().__init__(env)
        self.combine_with_score = combine_with_score
        self.event_weights = event_weights
        self.last_event = np.zeros(9, dtype=np.int32)
        
        # 核心修复：在这里先给一个初始值，防止某些调用在 reset 前访问 step
        self.last_obs_raw = np.zeros(135, dtype=np.float32) 
        
        self.event_totals = np.zeros(9, dtype=np.float32)
        self.remapped_count = 0
        self.total_steps = 0
        
        # 观测空间：40维状态 + 9维事件位
        low = np.concatenate([np.zeros(40, dtype=np.float32), np.zeros(9, dtype=np.float32)])
        high = np.concatenate([np.ones(40, dtype=np.float32) * 999.0, np.ones(9, dtype=np.float32)])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, **kwargs):
        # 1. 获取底层 135 维向量
        obs, info = self.env.reset(**kwargs)
        
        # 2. 必须在这里保存初始观测值
        self.last_obs_raw = obs.copy()
        
        self.last_event = np.zeros(9, dtype=np.int32)
        state40 = obs[:40]
        
        # 返回拼接后的 49 维向量
        return np.concatenate([state40, self.last_event]).astype(np.float32), info

    def step(self, action):
        # 使用上一步存储的原始向量作为 prev_vec
        prev_vec = self.last_obs_raw[:40]
        
        # 执行动作，获取新的 135 维向量
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        next_vec = obs[:40]
        
        # 获取动作对象用于检测
        try:
            actual_action = self.unwrapped.cached_legal_actions[action]
        except Exception:
            actual_action = None

        # 调用你修好的检测器
        ev = detect_events(prev_vec, actual_action, next_vec)
        
        # 计算奖励
        ev_reward = compute_event_reward(ev, weights=self.event_weights)
        reward = float(ev_reward + base_reward) if self.combine_with_score else float(ev_reward)
        
        # 更新状态缓存和统计
        self.last_obs_raw = obs.copy()
        self.last_event = ev.astype(np.int32)
        self.event_totals += ev.astype(np.float32)
        self.total_steps += 1

        # 拼接 49 维观测值返回
        new_obs = np.concatenate([next_vec, self.last_event]).astype(np.float32)
        return new_obs, reward, terminated, truncated, info

class DynamicRewardCallback(BaseCallback):
    def __init__(self, switch_step=300000, verbose=1):
        super().__init__(verbose)
        self.switch_step = switch_step
        self.has_switched = False

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.switch_step and not self.has_switched:
            # 穿透 VecEnv 找到你的 Wrapper
            curr_env = self.training_env.envs[0]
            # 循环解包直到找到 EventRewardWrapper
            while not hasattr(curr_env, 'combine_with_score'):
                if hasattr(curr_env, 'env'):
                    curr_env = curr_env.env
                else:
                    break
            
            if hasattr(curr_env, 'combine_with_score'):
                curr_env.combine_with_score = True
                self.has_switched = True
                print(f"\n [Step {self.num_timesteps}] turn on: combine_event_and_score")
        return True


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description='Train PPO event-based agent on Splendor (isolated)')
    parser.add_argument(
        '--config',
        type=str,
        default='project_event_based/configs/training/ppo_event_based.yaml',
        help='Path to config file'
    )
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = config['experiment']['name'] + '_event_based'
    log_path = f"project_event_based/logs/{exp_name}_{timestamp}"
    os.makedirs(log_path, exist_ok=True)
    with open(os.path.join(log_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    if 'seed' in config:
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
    monitor_dir = os.path.join(log_path, 'monitor')
    base_env = make_splendor_env(
        reward_mode='score_progress',
        opponent_agent=None,
        max_turns=config['environment']['max_turns']
    )
    os.makedirs(monitor_dir, exist_ok=True)
    base_env = Monitor(base_env, monitor_dir)
    combine = config.get('environment', {}).get('combine_event_and_score', False)
    event_weights = None
    if 'reward' in config and 'event_weights' in config['reward']:
        event_weights = config['reward']['event_weights']
    # Protect training from illegal action indices by remapping them to legal ones
    from utils.safe_action_wrapper import SafeActionWrapper
    safe_base = SafeActionWrapper(base_env, seed=config.get('seed'))

    # If MaskablePPO is available, wrap the environment with ActionMasker so the policy
    # will receive action masks and avoid illegal actions. Otherwise use SafeActionWrapper.
    if HAS_MASKABLE:
        try:
            masked = ActionMasker(safe_base, lambda e: e.get_action_mask())
            env = EventRewardWrapper(masked, combine_with_score=combine, event_weights=event_weights)
            use_maskable = True
        except Exception:
            env = EventRewardWrapper(safe_base, combine_with_score=combine, event_weights=event_weights)
            use_maskable = False
    else:
        env = EventRewardWrapper(safe_base, combine_with_score=combine, event_weights=event_weights)
        use_maskable = False
    # Vectorize and normalize observations/rewards for stable training
    vec_env = DummyVecEnv([lambda: env])
    use_vecnorm = config.get('training', {}).get('use_vecnormalize', True)
    if use_vecnorm:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_base = make_splendor_env(reward_mode='score_progress', opponent_agent=None, max_turns=config['environment']['max_turns'])
    eval_safe = SafeActionWrapper(Monitor(eval_base, None), seed=config.get('seed'))
    if HAS_MASKABLE:
        try:
            eval_masked = ActionMasker(eval_safe, lambda e: e.get_action_mask())
            eval_env = EventRewardWrapper(eval_masked, combine_with_score=combine, event_weights=event_weights)
        except Exception:
            eval_env = EventRewardWrapper(eval_safe, combine_with_score=combine, event_weights=event_weights)
    else:
        eval_env = EventRewardWrapper(eval_safe, combine_with_score=combine, event_weights=event_weights)
    # eval env vectorize/normalize
    eval_vec = DummyVecEnv([lambda: eval_env])
    if use_vecnorm:
        # separate VecNormalize for eval (training=False)
        eval_vec = VecNormalize(eval_vec, norm_obs=True, norm_reward=False)
    # Disable tensorboard log due to numpy/tensorboard compatibility issues
    tensorboard_log = None
    # tensorboard_log = os.path.join(log_path, config['training']['tensorboard_log'])
    # os.makedirs(tensorboard_log, exist_ok=True)
    # choose model class (MaskablePPO if available and used)
    ModelClass = MaskablePPO if (HAS_MASKABLE and 'use_maskable' in locals() and use_maskable) else PPO

    if args.resume:
        model = ModelClass.load(args.resume, env=vec_env)
        model.tensorboard_log = tensorboard_log
    else:
        ppo_config = config['ppo']
        policy_kwargs = ppo_config.get('policy_kwargs', {})
        if 'activation_fn' in policy_kwargs:
            if policy_kwargs['activation_fn'] == 'relu':
                policy_kwargs['activation_fn'] = torch.nn.ReLU
            elif policy_kwargs['activation_fn'] == 'tanh':
                policy_kwargs['activation_fn'] = torch.nn.Tanh
        model = ModelClass(
            policy=ppo_config['policy'],
            env=vec_env,
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
    eval_path = os.path.join(log_path, 'eval')
    os.makedirs(eval_path, exist_ok=True)
    eval_callback = EvalCallback(
        eval_vec,
        best_model_save_path=eval_path,
        log_path=eval_path,
        eval_freq=config['training']['eval_freq'],
        n_eval_episodes=config['training']['eval_episodes'],
        deterministic=config['evaluation']['deterministic'],
        render=config['evaluation']['render']
    )
    # Add event stats callback
    switch_step = config['training'].get('switch_step', 300000)
    dynamic_reward_cb = DynamicRewardCallback(switch_step=switch_step, verbose=1)
    try:
        from utils.event_stats_callback import EventStatsCallback
        event_cb = EventStatsCallback(log_freq=config['training'].get('log_interval', 1000))
        callbacks = CallbackList([
            checkpoint_callback, 
            eval_callback, 
            event_cb, 
            dynamic_reward_cb
        ])
    except Exception:
        callbacks = CallbackList([checkpoint_callback, eval_callback, dynamic_reward_cb])
    try:
        model.learn(
            total_timesteps=config['training']['total_timesteps'],
            callback=callbacks,
            log_interval=config['training']['log_interval'],
            tb_log_name=exp_name,
            reset_num_timesteps=True
        )
        model.save(os.path.join(log_path, 'final_model'))
        # Save VecNormalize statistics if used
        try:
            if use_vecnorm and isinstance(vec_env, VecNormalize):
                vecnorm_path = os.path.join(log_path, 'vecnormalize.pkl')
                vec_env.save(vecnorm_path)
        except Exception:
            pass
        # Write a simple summary file for quick discovery
        try:
            import json
            summary = {
                'experiment': exp_name,
                'timestamp': timestamp,
                'total_timesteps': config['training']['total_timesteps'],
                'model_path': os.path.join(log_path, 'final_model'),
                'checkpoint_path': checkpoint_path,
                'tensorboard_log': tensorboard_log
            }
            with open(os.path.join(log_path, 'summary.json'), 'w') as sf:
                json.dump(summary, sf, indent=2)
        except Exception:
            pass
        # Update a 'latest' symlink for convenience
        try:
            latest_link = os.path.join('project_event_based', 'logs', exp_name + '_latest')
            if os.path.islink(latest_link) or os.path.exists(latest_link):
                try:
                    os.remove(latest_link)
                except Exception:
                    pass
            os.symlink(os.path.abspath(log_path), latest_link)
        except Exception:
            pass
    except KeyboardInterrupt:
        model.save(os.path.join(log_path, 'interrupted_model'))


if __name__ == '__main__':
    main()
