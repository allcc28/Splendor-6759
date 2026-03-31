"""
Extract TensorBoard data and generate plots for the v2 (Greedy Opponent) training run.
"""
import sys
sys.path.insert(0, ".")

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("ERROR: tensorboard not installed. Run: pip install tensorboard")
    sys.exit(1)

# Config
TB_DIR = "project/logs/ppo_score_based_v2_greedy_opp_20260225_205855/logs/tensorboard"
OUTPUT_DIR = "project/experiments/reports/v2_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_events_dir(base_dir):
    """Find the directory containing tfevents files."""
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if "tfevents" in f:
                return root
    return base_dir

def extract_scalar(ea, tag):
    """Extract (steps, values) for a scalar tag."""
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return np.array(steps), np.array(values)

def main():
    events_dir = find_events_dir(TB_DIR)
    print(f"Loading TensorBoard events from: {events_dir}")
    
    ea = EventAccumulator(events_dir)
    ea.Reload()
    
    tags = ea.Tags()
    print(f"Available scalar tags: {tags.get('scalars', [])}")
    
    # --- Episode Reward Mean ---
    if 'eval/mean_reward' in tags.get('scalars', []):
        steps, values = extract_scalar(ea, 'eval/mean_reward')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(steps, values, color='#e74c3c', alpha=0.4, linewidth=0.8, label='Raw')
        
        # Smoothed line (rolling average)
        if len(values) > 5:
            window = min(10, len(values) // 3)
            smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
            smooth_steps = steps[window-1:]
            ax.plot(smooth_steps, smoothed, color='#e74c3c', linewidth=2.5, label=f'Smoothed (window={window})')
        
        ax.set_xlabel('Timesteps', fontsize=12)
        ax.set_ylabel('Episode Reward Mean', fontsize=12)
        ax.set_title('V2 (Greedy Opponent) — Episode Reward During Training', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        fig.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/v2_episode_reward_mean.png', dpi=150)
        plt.close(fig)
        print(f"Saved: {OUTPUT_DIR}/v2_episode_reward_mean.png")
        
        # Print key milestones
        print(f"\nReward milestones:")
        print(f"  Initial (10K):  {values[0]:.2f}")
        if len(values) > 10:
            print(f"  100K:           {values[min(9, len(values)-1)]:.2f}")
        print(f"  Best:           {values.max():.2f} (at step {steps[np.argmax(values)]})")
        print(f"  Final:          {values[-1]:.2f}")
    else:
        print("WARNING: 'eval/mean_reward' tag not found, trying 'rollout/ep_rew_mean'")
        if 'rollout/ep_rew_mean' in tags.get('scalars', []):
            steps, values = extract_scalar(ea, 'rollout/ep_rew_mean')
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(steps, values, color='#e74c3c', alpha=0.4, linewidth=0.8, label='Raw')
            if len(values) > 5:
                window = min(10, len(values) // 3)
                smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
                smooth_steps = steps[window-1:]
                ax.plot(smooth_steps, smoothed, color='#e74c3c', linewidth=2.5, label=f'Smoothed (window={window})')
            ax.set_xlabel('Timesteps', fontsize=12)
            ax.set_ylabel('Episode Reward Mean', fontsize=12)
            ax.set_title('V2 (Greedy Opponent) — Episode Reward During Training', fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            fig.tight_layout()
            fig.savefig(f'{OUTPUT_DIR}/v2_episode_reward_mean.png', dpi=150)
            plt.close(fig)
            print(f"Saved: {OUTPUT_DIR}/v2_episode_reward_mean.png")
    
    # --- Episode Length Mean ---
    ep_len_tag = None
    for tag_name in ['eval/mean_ep_length', 'rollout/ep_len_mean']:
        if tag_name in tags.get('scalars', []):
            ep_len_tag = tag_name
            break
    
    if ep_len_tag:
        steps, values = extract_scalar(ea, ep_len_tag)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(steps, values, color='#3498db', alpha=0.4, linewidth=0.8, label='Raw')
        if len(values) > 5:
            window = min(10, len(values) // 3)
            smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
            smooth_steps = steps[window-1:]
            ax.plot(smooth_steps, smoothed, color='#3498db', linewidth=2.5, label=f'Smoothed (window={window})')
        ax.set_xlabel('Timesteps', fontsize=12)
        ax.set_ylabel('Episode Length Mean', fontsize=12)
        ax.set_title('V2 (Greedy Opponent) — Episode Length During Training', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/v2_episode_length_mean.png', dpi=150)
        plt.close(fig)
        print(f"Saved: {OUTPUT_DIR}/v2_episode_length_mean.png")
    
    # --- Comparison: v1 vs v2 training curves ---
    # Try to load v1 data for overlay plot
    v1_tb_dir = "project/logs/ppo_score_based_v1_20260224_113524/logs/tensorboard"
    v1_events_dir = find_events_dir(v1_tb_dir)
    
    try:
        ea_v1 = EventAccumulator(v1_events_dir)
        ea_v1.Reload()
        
        v1_tags = ea_v1.Tags()
        v1_reward_tag = None
        for tag_name in ['eval/mean_reward', 'rollout/ep_rew_mean']:
            if tag_name in v1_tags.get('scalars', []):
                v1_reward_tag = tag_name
                break
        
        v2_reward_tag = None
        for tag_name in ['eval/mean_reward', 'rollout/ep_rew_mean']:
            if tag_name in tags.get('scalars', []):
                v2_reward_tag = tag_name
                break
        
        if v1_reward_tag and v2_reward_tag:
            v1_steps, v1_values = extract_scalar(ea_v1, v1_reward_tag)
            v2_steps, v2_values = extract_scalar(ea, v2_reward_tag)
            
            fig, ax = plt.subplots(figsize=(14, 7))
            
            # V1 (Random Opponent)
            ax.plot(v1_steps, v1_values, color='#2ecc71', alpha=0.3, linewidth=0.5)
            if len(v1_values) > 5:
                w = min(10, len(v1_values) // 3)
                sm = np.convolve(v1_values, np.ones(w)/w, mode='valid')
                ax.plot(v1_steps[w-1:], sm, color='#2ecc71', linewidth=2.5, label='V1 (vs Random)')
            
            # V2 (Greedy Opponent)
            ax.plot(v2_steps, v2_values, color='#e74c3c', alpha=0.3, linewidth=0.5)
            if len(v2_values) > 5:
                w = min(10, len(v2_values) // 3)
                sm = np.convolve(v2_values, np.ones(w)/w, mode='valid')
                ax.plot(v2_steps[w-1:], sm, color='#e74c3c', linewidth=2.5, label='V2 (vs Greedy)')
            
            ax.set_xlabel('Timesteps', fontsize=12)
            ax.set_ylabel('Episode Reward Mean', fontsize=12)
            ax.set_title('Training Reward Comparison: V1 (Random) vs V2 (Greedy)', fontsize=14)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            fig.tight_layout()
            fig.savefig(f'{OUTPUT_DIR}/v1_vs_v2_training_comparison.png', dpi=150)
            plt.close(fig)
            print(f"Saved: {OUTPUT_DIR}/v1_vs_v2_training_comparison.png")
    except Exception as e:
        print(f"Could not load v1 data for comparison: {e}")
    
    print("\nDone! All plots saved to:", OUTPUT_DIR)

if __name__ == '__main__':
    main()
