"""
Deep diagnostic: Test PPO model inside the exact same wrapper used for training.

This eliminates all evaluation bugs by using the exact training interface.
"""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "modules")

import numpy as np
from stable_baselines3 import PPO
from project.src.utils.splendor_gym_wrapper import SplendorGymWrapper


def test_model_in_wrapper():
    """Test model using exact training configuration."""
    print("=" * 80)
    print("DIAGNOSTIC: PPO Model Inside Training Wrapper")
    print("=" * 80)
    
    # Load model
    model_path = 'project/logs/ppo_score_based_v1_20260224_113524/final_model'
    model = PPO.load(model_path)
    print(f"Model loaded from: {model_path}")
    print(f"Action space: {model.action_space}")
    print(f"Observation space: {model.observation_space}")
    print()
    
    # Test 1: Exact training config (opponent=None, max_turns=120)
    print("=" * 80)
    print("TEST 1: Exact training config (opponent=None, max_turns=120)")
    print("=" * 80)
    
    results = run_episodes(model, opponent=None, max_turns=120, num_episodes=10, verbose=True)
    print_summary("Training Config", results)
    
    # Test 2: Higher max_turns to see if games finish
    print()
    print("=" * 80)
    print("TEST 2: Extended game (opponent=None, max_turns=500)")
    print("=" * 80)
    
    results2 = run_episodes(model, opponent=None, max_turns=500, num_episodes=5, verbose=True)
    print_summary("Extended Config", results2)
    
    # Test 3: Check action distribution
    print()
    print("=" * 80)
    print("TEST 3: Action Distribution Analysis")
    print("=" * 80)
    
    analyze_actions(model)


def run_episodes(model, opponent, max_turns, num_episodes, verbose=False):
    """Run episodes and collect detailed stats."""
    all_stats = []
    
    for ep in range(num_episodes):
        env = SplendorGymWrapper(
            opponent_agent=opponent,
            reward_mode='score_progress',
            max_turns=max_turns
        )
        
        obs, info = env.reset()
        done = False
        ep_reward = 0
        ep_steps = 0
        invalid_actions = 0
        action_types = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            # Check if action is valid
            if action >= len(env.cached_legal_actions):
                invalid_actions += 1
            
            n_legal = len(env.cached_legal_actions)
            
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps += 1
            done = terminated or truncated
            
            if verbose and ep == 0 and ep_steps <= 10:
                print(f"  Step {ep_steps}: action={action}, legal={n_legal}, "
                      f"reward={reward:.3f}, score={info.get('player_score', 0)}, "
                      f"opp_score={info.get('opponent_score', 0)}")
        
        final_score = info.get('player_score', 0)
        opp_score = info.get('opponent_score', 0)
        won = info.get('agent_won', False)
        
        stats = {
            'steps': ep_steps,
            'reward': ep_reward,
            'player_score': final_score,
            'opp_score': opp_score,
            'won': won,
            'invalid_actions': invalid_actions,
            'truncated': truncated
        }
        all_stats.append(stats)
        
        if verbose:
            status = "WON" if won else ("TRUNCATED" if truncated else "LOST")
            print(f"  Episode {ep+1}: {status} | steps={ep_steps} | reward={ep_reward:.2f} | "
                  f"score={final_score} | opp={opp_score} | invalid={invalid_actions}")
    
    return all_stats


def print_summary(label, stats):
    """Print summary statistics."""
    print(f"\n--- {label} Summary ({len(stats)} episodes) ---")
    
    wins = sum(1 for s in stats if s['won'])
    avg_reward = np.mean([s['reward'] for s in stats])
    avg_steps = np.mean([s['steps'] for s in stats])
    avg_score = np.mean([s['player_score'] for s in stats])
    avg_opp = np.mean([s['opp_score'] for s in stats])
    avg_invalid = np.mean([s['invalid_actions'] for s in stats])
    truncated = sum(1 for s in stats if s['truncated'])
    
    print(f"  Win rate: {wins}/{len(stats)} ({100*wins/len(stats):.0f}%)")
    print(f"  Avg reward: {avg_reward:.2f}")
    print(f"  Avg steps: {avg_steps:.1f}")
    print(f"  Avg agent score: {avg_score:.1f}")
    print(f"  Avg opponent score: {avg_opp:.1f}")
    print(f"  Avg invalid actions/episode: {avg_invalid:.1f}")
    print(f"  Truncated (hit max turns): {truncated}/{len(stats)}")


def analyze_actions(model):
    """Analyze what actions the model actually produces."""
    env = SplendorGymWrapper(
        opponent_agent=None,
        reward_mode='score_progress',
        max_turns=120
    )
    
    obs, info = env.reset()
    action_indices = []
    legal_counts = []
    
    for step in range(50):
        action, _ = model.predict(obs, deterministic=True)
        n_legal = len(env.cached_legal_actions)
        
        action_indices.append(int(action))
        legal_counts.append(n_legal)
        
        if step < 20:
            valid = "VALID" if action < n_legal else "INVALID"
            print(f"  Step {step}: action_idx={action}, legal_count={n_legal} [{valid}]")
        
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"  Episode ended at step {step+1}")
            break
    
    print(f"\n  Action index range: [{min(action_indices)}, {max(action_indices)}]")
    print(f"  Legal action range: [{min(legal_counts)}, {max(legal_counts)}]")
    print(f"  Unique actions used: {len(set(action_indices))}")
    print(f"  Most common action: {max(set(action_indices), key=action_indices.count)}")
    
    invalid_rate = sum(1 for a, l in zip(action_indices, legal_counts) if a >= l) / len(action_indices)
    print(f"  Invalid action rate: {100*invalid_rate:.1f}%")


if __name__ == '__main__':
    test_model_in_wrapper()
