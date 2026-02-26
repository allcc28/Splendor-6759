# Training Monitoring & Experiment Tracking Guide

**Date**: 2026-02-24  
**Purpose**: Choose and configure tools for tracking training curves, metrics, and experiments  
**Status**: Research Phase → Recommendation

---

## The Problem

During RL training, we need to monitor:
- **Training curves**: reward, loss, win rate over time
- **Episode statistics**: game length, final score, actions taken
- **Hyperparameters**: learning rate, batch size, exploration settings
- **Comparison**: Score-based vs Event-based performance
- **System metrics**: GPU utilization, memory usage, training speed

Without proper tracking, we can't:
- Debug failed training runs
- Identify convergence issues
- Compare different approaches
- Generate figures for the final report

---

## Three Main Options

### Option 1: TensorBoard (Recommended ✅)

**What it is**: Google's visualization tool for ML experiments

**Pros**:
- ✅ **Free and local** - no cloud dependency
- ✅ **Built into Stable-Baselines3** - zero extra code
- ✅ **Lightweight** - minimal overhead during training
- ✅ **Real-time visualization** - watch training live in browser
- ✅ **Standard format** - widely recognized, easy to share
- ✅ **Offline capable** - works without internet

**Cons**:
- ⚠️ Limited comparison features (manual work to overlay curves)
- ⚠️ No hyperparameter optimization built-in
- ⚠️ Storage grows large over time (GB for long runs)

**Setup Complexity**: ⭐☆☆☆☆ (1/5) - Literally one line of code

**Use Case**: Perfect for our project - single-machine training, clear visualization needs, no team collaboration issues.

---

### Option 2: Weights & Biases (W&B)

**What it is**: Cloud-based experiment tracking platform

**Pros**:
- ✅ Beautiful dashboards with comparison features
- ✅ Automatic hyperparameter tracking
- ✅ Easy collaboration (team can view remotely)
- ✅ Experiment comparison tools (side-by-side plots)
- ✅ Report generation for papers

**Cons**:
- ⚠️ Requires cloud account (free tier available)
- ⚠️ Needs internet connection during training
- ⚠️ Data stored externally (privacy concerns for some)
- ⚠️ Slight overhead from uploading metrics

**Setup Complexity**: ⭐⭐☆☆☆ (2/5) - Account + API key

**Use Case**: Great if team is distributed or you want fancy reports, but overkill for our project.

---

### Option 3: MLflow

**What it is**: Open-source ML lifecycle management platform

**Pros**:
- ✅ Self-hosted (privacy control)
- ✅ Experiment tracking + model versioning
- ✅ Hyperparameter comparison
- ✅ Model registry for deployment

**Cons**:
- ⚠️ More complex setup (server + database)
- ⚠️ Heavier weight than TensorBoard
- ⚠️ UI less polished than W&B
- ⚠️ Overkill for small projects

**Setup Complexity**: ⭐⭐⭐☆☆ (3/5) - Server setup required

**Use Case**: Production ML pipelines with many team members and models.

---

## Recommendation: TensorBoard + Simple Logging

### Why TensorBoard?

For your IFT6759 course project:
1. **Zero setup**: Already integrated with Stable-Baselines3
2. **Local control**: All data stays on your machine
3. **Sufficient features**: Has everything you need for the paper
4. **Standard**: Professors/reviewers familiar with it
5. **No distractions**: Just train and visualize

### What You'll Track

```python
# Automatically logged by Stable-Baselines3:
- rollout/ep_rew_mean      # Average reward per episode
- rollout/ep_len_mean      # Average episode length
- train/value_loss         # Critic loss
- train/policy_loss        # Actor loss
- train/entropy_loss       # Exploration bonus
- train/learning_rate      # LR (if using schedule)
- time/fps                 # Training speed

# Custom metrics (we'll add):
- eval/win_rate           # Win rate vs RandomAgent
- eval/score_diff         # Average point margin
- eval/game_length        # Turns per game
- eval/buy_actions        # Action distribution
```

### Implementation

```python
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

# Wrap env with Monitor for automatic logging
env = Monitor(env, "./logs/train")
eval_env = Monitor(eval_env, "./logs/eval")

# Create model with TensorBoard logging
model = PPO(
    "MlpPolicy",
    env,
    tensorboard_log="./logs/tensorboard/",
    verbose=1
)

# Evaluation callback for win rate tracking
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/best",
    log_path="./logs/eval",
    eval_freq=10000,  # Evaluate every 10k steps
    n_eval_episodes=100,
    deterministic=True
)

# Train with logging
model.learn(
    total_timesteps=1_000_000,
    callback=eval_callback,
    tb_log_name="score_based_v1"
)
```

### Viewing Results

```bash
# Start TensorBoard server
tensorboard --logdir ./logs/tensorboard/

# Open browser to http://localhost:6006
# You'll see:
# - Scalars: training curves
# - Graphs: network architecture
# - Distributions: weight histograms
# - Text/Images: (if we add them)
```

### Comparing Experiments

```bash
# Compare Score-based vs Event-based
tensorboard --logdir ./logs/tensorboard/ \
  --port 6006

# TensorBoard automatically overlays runs with different names:
# - score_based_v1
# - event_based_v1
# - score_based_v2_tuned
# etc.
```

---

## Enhanced Logging for Project Needs

### Custom Metrics Callback

```python
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class SplendorMetricsCallback(BaseCallback):
    """Custom callback to log Splendor-specific metrics."""
    
    def __init__(self, eval_env, n_eval_episodes=100, eval_freq=10000):
        super().__init__()
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.evaluations_timesteps = []
        self.evaluations_results = []
    
    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            # Run evaluation games
            wins = 0
            total_score_diff = 0
            game_lengths = []
            
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = False
                steps = 0
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, info = self.eval_env.step(action)
                    steps += 1
                
                # Extract game results
                if info.get('winner_id') == 0:  # Assuming agent is player 0
                    wins += 1
                total_score_diff += info.get('score_diff', 0)
                game_lengths.append(steps)
            
            # Log to TensorBoard
            win_rate = wins / self.n_eval_episodes
            avg_score_diff = total_score_diff / self.n_eval_episodes
            avg_game_length = np.mean(game_lengths)
            
            self.logger.record('eval/win_rate', win_rate)
            self.logger.record('eval/avg_score_diff', avg_score_diff)
            self.logger.record('eval/avg_game_length', avg_game_length)
            
            # Store for later analysis
            self.evaluations_timesteps.append(self.num_timesteps)
            self.evaluations_results.append(win_rate)
        
        return True
```

---

## File Organization

```
project/
├── logs/
│   ├── tensorboard/           # TensorBoard event files
│   │   ├── score_based_v1/
│   │   ├── event_based_v1/
│   │   └── ...
│   ├── train/                 # Training monitor logs
│   └── eval/                  # Evaluation logs
├── models/
│   ├── best/                  # Best models by eval win rate
│   ├── checkpoints/           # Regular checkpoints
│   └── final/                 # Final trained models
└── outputs/
    ├── figures/               # Exported plots for paper
    │   ├── score_vs_event_learning_curves.png
    │   ├── win_rate_comparison.png
    │   └── action_distribution.png
    └── results/               # CSV exports of metrics
        ├── score_based_metrics.csv
        └── event_based_metrics.csv
```

---

## Exporting Figures for Paper

### Method 1: TensorBoard Export (Manual)
1. Open TensorBoard in browser
2. Click "Download CSV" for each metric
3. Plot in matplotlib/seaborn for publication-quality figures

### Method 2: Automated Export Script

```python
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def export_learning_curves(logdir, output_path):
    """Export TensorBoard data to publication-ready plots."""
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()
    
    # Extract reward data
    reward_data = pd.DataFrame(ea.Scalars('rollout/ep_rew_mean'))
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(reward_data['step'], reward_data['value'])
    plt.xlabel('Training Steps')
    plt.ylabel('Average Reward')
    plt.title('Learning Curve')
    plt.grid(True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# Use after training
export_learning_curves(
    './logs/tensorboard/score_based_v1',
    './outputs/figures/score_based_learning_curve.png'
)
```

---

## Comparison Dashboard Script

For quick comparison during development:

```python
def compare_agents(score_logdir, event_logdir):
    """Create side-by-side comparison of Score vs Event agents."""
    # Load both runs
    score_ea = event_accumulator.EventAccumulator(score_logdir)
    event_ea = event_accumulator.EventAccumulator(event_logdir)
    score_ea.Reload()
    event_ea.Reload()
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Reward
    axes[0, 0].plot(score_reward['step'], score_reward['value'], label='Score-based')
    axes[0, 0].plot(event_reward['step'], event_reward['value'], label='Event-based')
    axes[0, 0].set_title('Average Reward')
    axes[0, 0].legend()
    
    # Plot 2: Win Rate
    # Plot 3: Episode Length
    # Plot 4: Loss
    
    plt.tight_layout()
    plt.savefig('./outputs/figures/comparison_dashboard.png', dpi=300)
```

---

## Alternative: Hybrid Approach (Advanced)

If you want the best of both worlds:

```python
# Use TensorBoard for real-time monitoring
# + Save important milestones to W&B for nice reports

import wandb

# Initialize W&B (optional, for final report)
wandb.init(
    project="splendor-rl-ift6759",
    name="score_based_v1",
    config={
        "algorithm": "PPO",
        "reward_type": "score_only",
        "learning_rate": 3e-4,
        # ... other hyperparameters
    },
    mode="offline"  # Sync manually later
)

# During training, log to both
def combined_callback(locals, globals):
    # TensorBoard (automatic)
    # + W&B (manual)
    if timestep % 10000 == 0:
        wandb.log({
            "reward": ep_reward_mean,
            "win_rate": win_rate,
            # ...
        })
```

**Verdict**: Not necessary for your project, but good to know it exists.

---

## My Recommendation: Start Simple

### Phase 1: TensorBoard Only
- Use built-in Stable-Baselines3 logging
- Add custom SplendorMetricsCallback for win rate
- View live training in browser

### Phase 2: Export for Paper
- After training completes, export CSVs
- Create publication-quality matplotlib plots
- Include in final report

### Phase 3 (Optional): Fancy Tools
- If you present at a conference, consider W&B for interactive demos
- Otherwise, TensorBoard is sufficient

---

## Quick Start Commands

```bash
# Terminal 1: Start training
cd project
python scripts/train_score_based.py --use-tensorboard

# Terminal 2: Watch training live
tensorboard --logdir ./logs/tensorboard/

# Browser: Open http://localhost:6006
```

---

## Next Steps

1. ✅ Decision made: Use TensorBoard
2. ⬜ Implement SplendorMetricsCallback
3. ⬜ Set up logging directory structure
4. ⬜ Test with small training run
5. ⬜ Create export script for paper figures

---

**Status**: Ready to implement  
**Integrated with**: ADR-001 (PPO), Stable-Baselines3 setup
