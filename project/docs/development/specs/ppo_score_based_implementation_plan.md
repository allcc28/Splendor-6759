# Implementation Plan: PPO Score-Based RL Agent

**Date**: 2026-02-24  
**Goal**: Build a working PPO-based Score-only RL agent for Splendor  
**Status**: Planning Phase  
**Estimated Time**: 6-8 hours of implementation + overnight training

---

## Overview

We will build the agent incrementally, testing each component before moving to the next. This ensures we catch bugs early and have a working system at each stage.

**High-Level Flow**:
```
Environment Setup → State Representation → Reward Wrapper → 
PPO Integration → Training Script → Test Training Run
```

---

## Phase 1: Environment Setup & Verification (30 min)

### ✅ Task 1.1: Verify GPU Environment
**Goal**: Confirm PyTorch, CUDA, and dependencies are working  
**Files**: None (terminal commands)  
**Test**: Run check script, confirm GPU is visible

```bash
python check_env.py
python -c "import torch; print(torch.cuda.is_available())"
```

**Success Criteria**: 
- ✅ Python can import torch, stable_baselines3, gym
- ✅ CUDA is available and GPU is detected
- ✅ No import errors

---

### ✅ Task 1.2: Test Existing Splendor Environment
**Goal**: Verify the gym_splendor environment works  
**Files**: Create `project/tests/test_env_basic.py`  
**Test**: Run a random game

```python
# Test script
import gym_open_ai
from modules.agents.random_agent import RandomAgent

env = gym_open_ai.make('gym_splendor_code:splendor-v0')
agent = RandomAgent()
agent.env = env

obs = env.reset()
done = False
steps = 0

while not done and steps < 500:
    action = agent.choose_act('stochastic')
    obs, reward, done, info = env.step('stochastic', action)
    steps += 1

print(f"Game finished in {steps} steps")
print(f"Winner: {env.first_winner}")
```

**Success Criteria**:
- ✅ Environment loads without errors
- ✅ Game completes successfully
- ✅ Winner is determined

**Estimated Time**: 15 min

---

## Phase 2: State Representation Module (1 hour)

### ✅ Task 2.1: Design State Vector Specification
**Goal**: Define exactly what goes into the state vector  
**Files**: `project/docs/development/specs/state_representation_spec.md`  
**Test**: Document review

**State Vector Components** (target: ~150 dimensions):
```
1. My Resources (6): [white, blue, green, red, black, gold]
2. My Cards (5): [white_cards, blue_cards, green_cards, red_cards, black_cards]
3. My Points (1): [current_points]
4. Board Gems (5): [white_avail, blue_avail, green_avail, red_avail, black_avail]
5. Visible Cards (84): 12 cards × 7 features (5 cost + 1 bonus + 1 points)
6. Reserved Cards (21): 3 cards × 7 features
7. Opponent State (4): [opp_points, opp_cards, opp_gems, opp_reserved]
8. Game State (3): [turn_number, my_reserves_count, nobles_available]

Total: ~130 dimensions (may adjust)
```

**Success Criteria**:
- ✅ State vector fully specified
- ✅ All game-relevant information captured
- ✅ Fixed size (no variable dimensions)

**Estimated Time**: 20 min

---

### ✅ Task 2.2: Implement State Vectorizer
**Goal**: Convert Splendor game state to numpy array  
**Files**: `project/src/utils/state_vectorizer.py`  
**Test**: Unit test with known game state

```python
class SplendorStateVectorizer:
    def __init__(self):
        self.state_dim = 130  # Adjust based on spec
    
    def vectorize(self, env, player_id=0):
        """Convert Splendor state to fixed-size numpy array."""
        state = env.current_state_of_the_game
        vector = []
        
        # My resources
        my_hand = state.list_of_players_hands[player_id]
        vector.extend([
            my_hand.gems_possessed.get('white', 0),
            my_hand.gems_possessed.get('blue', 0),
            # ... etc
        ])
        
        # My cards
        # ... 
        
        return np.array(vector, dtype=np.float32)
    
    def get_state_dim(self):
        return self.state_dim
```

**Success Criteria**:
- ✅ Function returns numpy array
- ✅ Array shape is always (state_dim,)
- ✅ Values are normalized (0-1 range recommended)
- ✅ No NaN or Inf values

**Estimated Time**: 40 min

---

### ✅ Task 2.3: Test State Vectorizer
**Goal**: Verify vectorizer works on real game states  
**Files**: `project/tests/test_state_vectorizer.py`

```python
def test_vectorizer():
    env = gym_open_ai.make('gym_splendor_code:splendor-v0')
    vectorizer = SplendorStateVectorizer()
    
    # Test initial state
    env.reset()
    vec = vectorizer.vectorize(env, player_id=0)
    assert vec.shape == (vectorizer.state_dim,)
    assert not np.any(np.isnan(vec))
    
    # Test mid-game state
    for _ in range(10):
        action = env.action_space.sample()
        env.step('stochastic', action)
    
    vec2 = vectorizer.vectorize(env, player_id=0)
    assert vec2.shape == (vectorizer.state_dim,)
    assert not np.array_equal(vec, vec2)  # State should have changed
```

**Success Criteria**:
- ✅ All assertions pass
- ✅ Vectorizer handles edge cases (no cards, no gems, etc.)

**Estimated Time**: 20 min

---

## Phase 3: Gym Wrapper for Stable-Baselines3 (1.5 hours)

### ✅ Task 3.1: Create Gym-Compatible Wrapper
**Goal**: Make Splendor env compatible with SB3  
**Files**: `project/src/utils/splendor_gym_wrapper.py`  
**Test**: Check with SB3's check_env

Stable-Baselines3 expects:
- `reset()` returns observation only
- `step(action)` returns `(obs, reward, done, info)`
- `observation_space` is gym.Space
- `action_space` is gym.Space

```python
import gym
from gym import spaces

class SplendorGymWrapper(gym.Env):
    """Wraps Splendor environment for Stable-Baselines3."""
    
    def __init__(self, reward_mode='score'):
        super().__init__()
        self.env = gym_open_ai.make('gym_splendor_code:splendor-v0')
        self.vectorizer = SplendorStateVectorizer()
        self.reward_mode = reward_mode
        self.current_player = 0
        
        # Define spaces
        self.observation_space = spaces.Box(
            low=-1.0, high=100.0,
            shape=(self.vectorizer.state_dim,),
            dtype=np.float32
        )
        
        # Discrete action space (we'll handle action masking separately)
        self.action_space = spaces.Discrete(200)  # Max possible actions
        
        self.previous_points = 0
    
    def reset(self):
        self.env.reset()
        self.env.update_actions()
        self.previous_points = 0
        return self.vectorizer.vectorize(self.env, self.current_player)
    
    def step(self, action_idx):
        """Execute action and return (obs, reward, done, info)."""
        # Map action_idx to actual Splendor action
        legal_actions = self.env.action_space.list_of_actions
        
        if action_idx >= len(legal_actions):
            # Invalid action - choose random legal action
            action_idx = np.random.randint(len(legal_actions))
        
        action = legal_actions[action_idx]
        
        # Execute action
        _, _, done, info = self.env.step('stochastic', action)
        
        # Compute reward
        current_points = self.env.current_state_of_the_game.list_of_players_hands[
            self.current_player
        ].number_of_my_points()
        
        reward = self._compute_reward(current_points, done)
        
        # Get next observation
        obs = self.vectorizer.vectorize(self.env, self.current_player)
        
        # Update state
        self.previous_points = current_points
        self.env.update_actions()
        
        return obs, reward, done, info
    
    def _compute_reward(self, current_points, done):
        """Score-based reward function."""
        if self.reward_mode == 'score_naive':
            return current_points - self.previous_points
        
        elif self.reward_mode == 'score_win_bonus':
            reward = current_points - self.previous_points
            if done:
                if self.env.first_winner == self.current_player:
                    reward += 10.0
                else:
                    reward -= 5.0
            return reward
        
        elif self.reward_mode == 'score_progress':
            reward = current_points - self.previous_points
            reward += 0.01  # Tiny progress bonus
            if done:
                if self.env.first_winner == self.current_player:
                    reward += 10.0
                else:
                    reward -= 5.0
            return reward
        
        else:
            raise ValueError(f"Unknown reward mode: {self.reward_mode}")
```

**Success Criteria**:
- ✅ Wrapper follows gym.Env interface
- ✅ reset() returns correct observation shape
- ✅ step() returns correct tuple format
- ✅ Rewards are computed correctly

**Estimated Time**: 60 min

---

### ✅ Task 3.2: Test Wrapper with SB3
**Goal**: Verify wrapper works with Stable-Baselines3  
**Files**: `project/tests/test_wrapper.py`

```python
from stable_baselines3.common.env_checker import check_env

def test_wrapper():
    env = SplendorGymWrapper(reward_mode='score_progress')
    
    # SB3's built-in checker
    check_env(env, warn=True)
    
    # Manual test
    obs = env.reset()
    assert obs.shape == (env.observation_space.shape[0],)
    
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        
        if done:
            break
    
    print("✅ Wrapper test passed!")
```

**Success Criteria**:
- ✅ check_env() passes with no errors
- ✅ Manual test completes successfully
- ✅ Observations, rewards, dones are correct types

**Estimated Time**: 30 min

---

## Phase 4: PPO Integration (1 hour)

### ✅ Task 4.1: Create Training Configuration
**Goal**: Define hyperparameters and settings  
**Files**: `project/configs/training/ppo_score_based.yaml`

```yaml
experiment:
  name: "score_based_naive_v1"
  description: "PPO with score-only reward (naive)"
  seed: 42

environment:
  reward_mode: "score_naive"  # or score_win_bonus, score_progress
  max_steps: 500

ppo:
  policy: "MlpPolicy"
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5

network:
  net_arch:
    - 256
    - 128
    - 64

training:
  total_timesteps: 1000000
  eval_freq: 10000
  n_eval_episodes: 50
  save_freq: 50000
  
logging:
  tensorboard_log: "./logs/tensorboard/"
  model_save_path: "./models/"
  verbose: 1
```

**Success Criteria**:
- ✅ Config is valid YAML
- ✅ All necessary parameters included
- ✅ Values are reasonable

**Estimated Time**: 20 min

---

### ✅ Task 4.2: Create Training Script
**Goal**: Main script to train the agent  
**Files**: `project/scripts/train_score_based.py`

```python
#!/usr/bin/env python3
"""
Train PPO Score-based RL agent for Splendor.
"""
import argparse
import yaml
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.splendor_gym_wrapper import SplendorGymWrapper

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main(args):
    # Load config
    config = load_config(args.config)
    print(f"Training: {config['experiment']['name']}")
    
    # Create environment
    env = SplendorGymWrapper(reward_mode=config['environment']['reward_mode'])
    env = Monitor(env, filename=None)  # For automatic logging
    
    # Create eval environment
    eval_env = SplendorGymWrapper(reward_mode=config['environment']['reward_mode'])
    eval_env = Monitor(eval_env, filename=None)
    
    # Create model
    model = PPO(
        policy=config['ppo']['policy'],
        env=env,
        learning_rate=config['ppo']['learning_rate'],
        n_steps=config['ppo']['n_steps'],
        batch_size=config['ppo']['batch_size'],
        n_epochs=config['ppo']['n_epochs'],
        gamma=config['ppo']['gamma'],
        gae_lambda=config['ppo']['gae_lambda'],
        clip_range=config['ppo']['clip_range'],
        ent_coef=config['ppo']['ent_coef'],
        vf_coef=config['ppo']['vf_coef'],
        max_grad_norm=config['ppo']['max_grad_norm'],
        policy_kwargs={
            "net_arch": config['network']['net_arch']
        },
        tensorboard_log=config['logging']['tensorboard_log'],
        verbose=config['logging']['verbose'],
        seed=config['experiment']['seed']
    )
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{config['logging']['model_save_path']}/best",
        log_path=f"{config['logging']['model_save_path']}/eval",
        eval_freq=config['training']['eval_freq'],
        n_eval_episodes=config['training']['n_eval_episodes'],
        deterministic=True
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=config['training']['save_freq'],
        save_path=f"{config['logging']['model_save_path']}/checkpoints",
        name_prefix=config['experiment']['name']
    )
    
    # Train
    print("Starting training...")
    model.learn(
        total_timesteps=config['training']['total_timesteps'],
        callback=[eval_callback, checkpoint_callback],
        tb_log_name=config['experiment']['name']
    )
    
    # Save final model
    final_path = f"{config['logging']['model_save_path']}/final/{config['experiment']['name']}"
    model.save(final_path)
    print(f"Training complete! Model saved to {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/ppo_score_based.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()
    main(args)
```

**Success Criteria**:
- ✅ Script loads config correctly
- ✅ Creates PPO model without errors
- ✅ Can be run from command line

**Estimated Time**: 40 min

---

## Phase 5: Quick Test Run (30 min)

### ✅ Task 5.1: Run Short Training Test
**Goal**: Verify everything works end-to-end  
**Files**: `project/configs/training/ppo_score_test.yaml` (reduced timesteps)  
**Test**: Run training for 10k steps (~5-10 min)

```yaml
# Minimal config for testing
training:
  total_timesteps: 10000  # Just 10k for testing
  eval_freq: 5000
  n_eval_episodes: 10
```

```bash
cd project
python scripts/train_score_based.py --config configs/training/ppo_score_test.yaml
```

**Success Criteria**:
- ✅ Training starts without errors
- ✅ TensorBoard logs are created
- ✅ Model checkpoints are saved
- ✅ Training completes successfully
- ✅ No crashes or exceptions

**Estimated Time**: 30 min (including small training run)

---

### ✅ Task 5.2: Check TensorBoard Logs
**Goal**: Verify metrics are being logged  
**Files**: None (view in browser)

```bash
tensorboard --logdir ./logs/tensorboard/
```

Open `http://localhost:6006`

**Success Criteria**:
- ✅ Reward curves appear
- ✅ Loss curves appear
- ✅ Values are reasonable (not NaN, not exploding)

**Estimated Time**: 10 min

---

## Phase 6: Full Training Run (Overnight)

### ✅ Task 6.1: Launch Full Training
**Goal**: Train agent for 1M steps  
**Files**: Use main config  
**Test**: Monitor overnight

```bash
# Start training (will run for hours)
python scripts/train_score_based.py

# In another terminal, monitor progress
tensorboard --logdir ./logs/tensorboard/
```

**Success Criteria**:
- ✅ Training runs overnight without crashes
- ✅ Reward increases over time (shows learning)
- ✅ Model checkpoints saved regularly

**Estimated Time**: 12-24 hours (mostly unattended)

---

### ✅ Task 6.2: Evaluate Trained Agent
**Goal**: Test final model against RandomAgent  
**Files**: `project/scripts/evaluate_agent.py`

```python
def evaluate_agent(model_path, n_games=100):
    """Evaluate trained agent vs RandomAgent."""
    from modules.agents.random_agent import RandomAgent
    
    model = PPO.load(model_path)
    random_agent = RandomAgent()
    
    wins = 0
    for game in range(n_games):
        # Play game
        # ... (similar to legacy/run_match.py logic)
        if winner == 0:  # Agent is player 0
            wins += 1
    
    win_rate = wins / n_games
    print(f"Win rate vs Random: {win_rate:.2%}")
    return win_rate
```

**Success Criteria**:
- ✅ Agent wins >60% against RandomAgent (shows learning)
- ✅ If using naive score-only: May only be ~50-55% (demonstrates problem)
- ✅ If using progress hints: Should be ~70%+

**Estimated Time**: 30 min

---

## Summary Checklist

### Phase 1: Setup (30 min)
- [ ] 1.1: Verify GPU environment
- [ ] 1.2: Test existing Splendor environment

### Phase 2: State Representation (1 hour)
- [ ] 2.1: Design state vector specification
- [ ] 2.2: Implement state vectorizer
- [ ] 2.3: Test state vectorizer

### Phase 3: Gym Wrapper (1.5 hours)
- [ ] 3.1: Create gym-compatible wrapper
- [ ] 3.2: Test wrapper with SB3

### Phase 4: PPO Integration (1 hour)
- [ ] 4.1: Create training configuration
- [ ] 4.2: Create training script

### Phase 5: Quick Test (30 min)
- [ ] 5.1: Run short training test (10k steps)
- [ ] 5.2: Check TensorBoard logs

### Phase 6: Full Training (Overnight)
- [ ] 6.1: Launch full training (1M steps)
- [ ] 6.2: Evaluate trained agent

**Total Estimated Time**: 4.5 hours implementation + overnight training

---

## Risk Mitigation

### Common Issues & Solutions

**Issue**: Action space mismatch  
**Solution**: Use action masking or random fallback for invalid actions

**Issue**: Reward is always zero  
**Solution**: Add debug prints in reward function, verify points are changing

**Issue**: Training crashes after N steps  
**Solution**: Add exception handling, save checkpoints frequently

**Issue**: GPU out of memory  
**Solution**: Reduce batch_size or n_steps in config

**Issue**: Agent doesn't learn (flat reward curve)  
**Solution**: This may be expected for naive score-only! Try score_progress mode.

---

## Next Steps After Training

1. Train Event-based agent (use same pipeline, different reward mode)
2. Run tournament comparison (Score vs Event)
3. Generate plots for paper
4. Document results in dev log

---

**Status**: Ready to start implementation  
**First Task**: Task 1.1 - Verify GPU Environment
