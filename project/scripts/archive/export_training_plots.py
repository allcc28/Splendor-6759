import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def plot_tensorboard_data(log_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load event data
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # Extract tags
    tags = event_acc.Tags()['scalars']
    print(f"Found scalar tags: {tags}")
    
    # Plot Episode Reward
    if 'rollout/ep_rew_mean' in tags:
        events = event_acc.Scalars('rollout/ep_rew_mean')
        steps = [e.step for e in events]
        values = [e.value for e in events]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, values, label='Episode Reward Mean', color='blue')
        plt.title('Training Progress: Episode Reward Mean')
        plt.xlabel('Timesteps')
        plt.ylabel('Reward')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'episode_reward_mean.png'), dpi=300)
        plt.close()
        print("Saved episode_reward_mean.png")
        
    # Plot Episode Length
    if 'rollout/ep_len_mean' in tags:
        events = event_acc.Scalars('rollout/ep_len_mean')
        steps = [e.step for e in events]
        values = [e.value for e in events]
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, values, label='Episode Length Mean', color='green')
        plt.title('Training Progress: Episode Length Mean')
        plt.xlabel('Timesteps')
        plt.ylabel('Length')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'episode_length_mean.png'), dpi=300)
        plt.close()
        print("Saved episode_length_mean.png")

if __name__ == "__main__":
    log_dir = "project/logs/ppo_score_based_v1_20260224_113524/logs/tensorboard/ppo_score_based_v1_1"
    output_dir = "project/experiments/reports/figures"
    plot_tensorboard_data(log_dir, output_dir)
