"""
PPO Agent Wrapper for Splendor Arena

Wraps a trained Stable-Baselines3 PPO model to implement the Agent interface
for compatibility with legacy arena evaluation system.

Author: AI Agent
Date: 2026-02-25
Version: 1.0
"""

import sys
sys.path.insert(0, "modules")

import numpy as np
from stable_baselines3 import PPO

from agents.abstract_agent import Agent
from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.abstract_observation import SplendorObservation
from project.src.utils.state_vectorizer import SplendorStateVectorizer


class PPOAgent(Agent):
    """
    Wrapper for trained PPO model to play in Splendor arena.
    
    Implements the Agent interface required by legacy evaluation code.
    """
    
    def __init__(self, model_path: str, name: str = "PPO-ScoreBased", deterministic: bool = True):
        """
        Initialize PPO agent.
        
        Args:
            model_path: Path to trained PPO model (.zip file)
            name: Agent name for display
            deterministic: Whether to use deterministic policy (True for evaluation)
        """
        super().__init__()
        self.name = name
        self.deterministic = deterministic
        
        # Load trained model
        self.model = PPO.load(model_path)
        
        # Initialize state vectorizer
        self.vectorizer = SplendorStateVectorizer()
        
        # Track player ID (set during game)
        self.player_id = 0
        self.turn_count = 0
        
    def choose_act(self, mode, info=False):
        """
        Choose action using PPO policy.
        
        Args:
            mode: 'deterministic' or 'stochastic' (ignored - we use self.deterministic)
            info: Whether to return additional info (not used)
            
        Returns:
            Action object selected by PPO policy
        """
        # Get current state from environment
        state = self.env.current_state_of_the_game
        legal_actions = self.env.action_space.list_of_actions
        
        if not legal_actions:
            return None
            
        # Vectorize state
        obs = self.vectorizer.vectorize(state, self.player_id, self.turn_count)
        
        # Get action from PPO model
        action_idx, _states = self.model.predict(obs, deterministic=self.deterministic)
        
        # Map action index to legal action
        if action_idx >= len(legal_actions):
            # Fallback: choose random legal action if model outputs invalid index
            print(f"Warning: PPO output invalid action index {action_idx} (max: {len(legal_actions)-1})")
            return np.random.choice(legal_actions)
        
        self.turn_count += 1
        return legal_actions[action_idx]
    
    def deterministic_choose_action(self, observation, previous_actions):
        """Override to track player ID."""
        # Infer player ID from observation
        self.env.load_observation(observation)
        self.env.update_actions_light()
        
        # Get active player ID
        state = self.env.current_state_of_the_game
        if hasattr(state, 'active_player_id'):
            self.player_id = state.active_player_id
        else:
            # Fallback: assume player 0 if not available
            self.player_id = 0
            
        return self.choose_act(mode="deterministic")
    
    def finish_game(self):
        """Reset turn counter after game ends."""
        self.turn_count = 0
        super().finish_game()
