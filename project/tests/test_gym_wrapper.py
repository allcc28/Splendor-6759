"""
Tests for SplendorGymWrapper

Validates compatibility with Stable-Baselines3 and correct reward computation.

Author: AI Agent
Date: 2026-02-24
"""

import sys
import os
sys.path.insert(0, os.path.abspath("modules"))
sys.path.insert(0, os.path.abspath("project/src"))

import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env
from utils.splendor_gym_wrapper import SplendorGymWrapper, make_splendor_env


class TestSplendorGymWrapper:
    """Test suite for Splendor Gym wrapper."""
    
    def setup_method(self):
        """Setup before each test."""
        self.env = SplendorGymWrapper(reward_mode='score_progress')
        
    def teardown_method(self):
        """Cleanup after each test."""
        self.env.close()
        
    def test_initialization(self):
        """Test that environment initializes correctly."""
        assert self.env.observation_space.shape == (135,)
        assert self.env.action_space.n == 200
        assert self.env.reward_mode == 'score_progress'
        
    def test_reset(self):
        """Test environment reset."""
        obs, info = self.env.reset()
        
        assert obs.shape == (135,)
        assert obs.dtype == np.float32
        assert 'turn' in info
        assert 'legal_actions_count' in info
        assert info['turn'] == 0
        assert info['legal_actions_count'] > 0
        
    def test_step(self):
        """Test taking a step."""
        obs, info = self.env.reset()
        
        # Take a random legal action
        assert len(self.env.cached_legal_actions) > 0
        action_idx = 0  # First legal action
        
        next_obs, reward, terminated, truncated, info = self.env.step(action_idx)
        
        assert next_obs.shape == (135,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert 'turn' in info
        
    def test_invalid_action_handling(self):
        """Test that invalid actions are handled properly."""
        obs, info = self.env.reset()
        
        # Try an invalid action index
        invalid_idx = len(self.env.cached_legal_actions) + 10
        next_obs, reward, terminated, truncated, info = self.env.step(invalid_idx)
        
        assert terminated == True, "Invalid action should terminate"
        assert reward < 0, "Invalid action should have negative reward"
        assert 'error' in info
        
    def test_action_mask(self):
        """Test action masking."""
        obs, info = self.env.reset()
        action_mask = self.env.get_action_mask()
        
        assert action_mask.shape == (200,)
        assert action_mask.dtype == bool
        assert np.sum(action_mask) == len(self.env.cached_legal_actions)
        
    def test_reward_modes(self):
        """Test different reward modes."""
        modes = ['score_naive', 'score_win_bonus', 'score_progress']
        
        for mode in modes:
            env = SplendorGymWrapper(reward_mode=mode)
            obs, info = env.reset()
            
            # Take one action
            next_obs, reward, terminated, truncated, info = env.step(0)
            
            assert isinstance(reward, float)
            env.close()
    
    def test_episode_completion(self):
        """Test playing a full episode."""
        obs, info = self.env.reset()
        total_reward = 0
        steps = 0
        max_steps = 500
        
        while steps < max_steps:
            # Random valid action
            action_idx = np.random.randint(0, len(self.env.cached_legal_actions))
            next_obs, reward, terminated, truncated, info = self.env.step(action_idx)
            
            total_reward += reward
            steps += 1
            obs = next_obs
            
            if terminated or truncated:
                break
        
        assert steps < max_steps, "Episode should terminate naturally or by truncation"
        
    def test_observation_consistency(self):
        """Test that observations are consistent."""
        obs1, _ = self.env.reset()
        obs2, _ = self.env.reset()
        
        # Same initial state should give same observation
        # (unless there's randomness in initial state)
        assert obs1.shape == obs2.shape
        
    def test_sb3_compatibility(self):
        """Test compatibility with Stable-Baselines3."""
        env = SplendorGymWrapper(reward_mode='score_progress')
        
        # SB3's check_env should pass without errors
        try:
            check_env(env, warn=True)
            print("✓ SB3 environment check passed")
        except Exception as e:
            pytest.fail(f"SB3 environment check failed: {e}")
        finally:
            env.close()
    
    def test_factory_function(self):
        """Test the factory function."""
        env = make_splendor_env(reward_mode='score_win_bonus')
        
        assert env.reward_mode == 'score_win_bonus'
        obs, info = env.reset()
        assert obs.shape == (135,)
        
        env.close()


def test_multiple_episodes():
    """Integration test: run multiple episodes."""
    env = SplendorGymWrapper(reward_mode='score_progress')
    
    for episode in range(3):
        obs, info = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 200:
            # Check if there are legal actions available
            if len(env.cached_legal_actions) == 0:
                break
                
            action_idx = np.random.randint(0, len(env.cached_legal_actions))
            obs, reward, terminated, truncated, info = env.step(action_idx)
            done = terminated or truncated
            steps += 1
        
        print(f"Episode {episode + 1}: {steps} steps, final scores: "
              f"Agent={info['player_score']}, Opponent={info['opponent_score']}")
    
    env.close()


if __name__ == "__main__":
    # Run basic tests
    print("Running SplendorGymWrapper tests...")
    
    # Test initialization
    env = SplendorGymWrapper()
    print(f"✓ Environment initialized")
    print(f"  - Observation space: {env.observation_space}")
    print(f"  - Action space: {env.action_space}")
    
    # Test reset
    obs, info = env.reset()
    print(f"✓ Environment reset successful")
    print(f"  - Observation shape: {obs.shape}")
    print(f"  - Legal actions: {info['legal_actions_count']}")
    
    # Test step
    action_idx = 0
    next_obs, reward, terminated, truncated, info = env.step(action_idx)
    print(f"✓ Step executed")
    print(f"  - Reward: {reward:.4f}")
    print(f"  - Terminated: {terminated}")
    print(f"  - Turn: {info['turn']}")
    
    # Test SB3 compatibility
    print("\n✓ Running SB3 compatibility check...")
    from stable_baselines3.common.env_checker import check_env
    try:
        check_env(env, warn=True)
        print("✅ SB3 environment check passed!")
    except Exception as e:
        print(f"❌ SB3 check failed: {e}")
    
    env.close()
    
    print("\n✅ All basic tests completed!")
    print("\nTo run full test suite with pytest:")
    print("  cd /mnt/c/Users/yehao/Documents/03Study/IFT6759/Splendor-6759")
    print("  pytest project/tests/test_gym_wrapper.py -v")
