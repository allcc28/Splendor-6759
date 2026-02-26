"""
Gymnasium-compatible wrapper for Splendor environment.

Adapted for Stable-Baselines3 PPO training with Score-based rewards.

Author: AI Agent
Date: 2026-02-24
Version: 1.0
"""

import sys
sys.path.insert(0, "modules")

import numpy as np
import gymnasium  as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional

from gym_splendor_code.envs.splendor import SplendorEnv
from gym_splendor_code.envs.mechanics.action import Action
from .state_vectorizer import SplendorStateVectorizer


class SplendorGymWrapper(gym.Env):
    """
    Gymnasium wrapper for Splendor environment compatible with Stable-Baselines3.
    
    Features:
    - Fixed-size observation space (135-dim vector)
    - Discrete action space (variable size based on legal moves)
    - Score-based reward with configurable modes
    - Single-agent perspective (opponent uses configured policy)
    
    Reward modes:
    - 'score_naive': +1 for win, 0 otherwise (sparse)
    - 'score_win_bonus': score_diff + 100 for win
    - 'score_progress': score_diff + 0.01 per valid action (dense)
    """
    
    metadata = {'render_modes': []}
    
    def __init__(
        self,
        opponent_agent=None,
        reward_mode: str = 'score_progress',
        max_turns: int = 120,
        player_id: int = 0
    ):
        """
        Initialize the Splendor Gym wrapper.
        
        Args:
            opponent_agent: Agent to play as opponent (None = random)
            reward_mode: One of ['score_naive', 'score_win_bonus', 'score_progress']
            max_turns: Maximum turns before draw
            player_id: Which player ID the learning agent controls (0 or 1)
        """
        super().__init__()
        
        # Core environment
        self.env = SplendorEnv()
        self.vectorizer = SplendorStateVectorizer()
        
        # Configuration
        self.opponent_agent = opponent_agent
        self.reward_mode = reward_mode
        self.max_turns = max_turns
        self.player_id = player_id
        self.opponent_id = 1 - player_id
        
        # State tracking
        self.turn_count = 0
        self.prev_score = 0
        
        # Define observation space: 135-dim continuous vector
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(135,),
            dtype=np.float32
        )
        
        # Define action space: discrete (size will vary based on legal moves)
        # Max actions: ~100 (take gems, reserve, buy from board/reserved)
        # We'll use action masking during training
        self.action_space = spaces.Discrete(200)  # Upper bound
        
        # Action caching
        self.cached_legal_actions = []
        self.action_mask = np.zeros(200, dtype=bool)
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Returns:
            observation: 135-dim state vector
            info: Dictionary with metadata
        """
        super().reset(seed=seed)
        
        # Reset Splendor environment
        observation_obj = self.env.reset()
        
        # Reset tracking
        self.turn_count = 0
        self.prev_score = 0
        
        # If learning agent is not player 0, let opponent move first
        if self.player_id == 1:
            self._opponent_move()
        
        # Update legal actions and get observation
        self._update_legal_actions()
        obs = self._get_observation()
        
        info = {
            'turn': self.turn_count,
            'legal_actions_count': len(self.cached_legal_actions),
            'player_score': self.env.current_state_of_the_game.list_of_players_hands[self.player_id].number_of_my_points(),
            'opponent_score': self.env.current_state_of_the_game.list_of_players_hands[self.opponent_id].number_of_my_points()
        }
        
        return obs, info
    
    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action_idx: Index into cached_legal_actions list
            
        Returns:
            observation: Next state vector
            reward: Scalar reward
            terminated: Whether episode ended naturally
            truncated: Whether episode was cut off
            info: Metadata dictionary
        """
        # Validate action
        if action_idx >= len(self.cached_legal_actions):
            # Invalid action - heavily penalize and terminate
            obs = self._get_observation()
            reward = -10.0
            terminated = True
            truncated = False
            info = {
                'turn': self.turn_count,
                'error': 'invalid_action',
                'legal_actions_count': len(self.cached_legal_actions)
            }
            return obs, reward, terminated, truncated, info
        
        # Execute action
        action = self.cached_legal_actions[action_idx]
        observation_obj, _, done, info_dict = self.env.step('deterministic', action)
        
        # Track score before opponent move
        current_score = self.env.current_state_of_the_game.list_of_players_hands[self.player_id].number_of_my_points()
        score_diff = current_score - self.prev_score
        
        # Check if agent won
        agent_won = done and info_dict.get('winner_id') == self.player_id
        agent_lost = done and info_dict.get('winner_id') == self.opponent_id
        
        # Compute reward
        reward = self._compute_reward(score_diff, agent_won, agent_lost)
        
        # Update turn counter
        self.turn_count += 1
        
        # Check for max turns (draw)
        truncated = self.turn_count >= self.max_turns
        terminated = done or agent_lost
        
        # If not done and not truncated, opponent moves
        if not terminated and not truncated:
            self._opponent_move()
            
            # Check if opponent won
            if self.env.is_done:
                terminated = True
                agent_lost = True
                reward = -1.0  # Penalty for letting opponent win
        
        # Update state for next turn
        self.prev_score = self.env.current_state_of_the_game.list_of_players_hands[self.player_id].number_of_my_points()
        self._update_legal_actions()
        obs = self._get_observation()
        
        # Prepare info
        info = {
            'turn': self.turn_count,
            'legal_actions_count': len(self.cached_legal_actions),
            'player_score': self.prev_score,
            'opponent_score': self.env.current_state_of_the_game.list_of_players_hands[self.opponent_id].number_of_my_points(),
            'agent_won': agent_won,
            'agent_lost': agent_lost,
            'score_diff': score_diff
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation as state vector."""
        state = self.env.current_state_of_the_game
        return self.vectorizer.vectorize(state, self.player_id, self.turn_count)
    
    def _update_legal_actions(self):
        """Update list of legal actions for current state."""
        self.env.update_actions_light()
        self.cached_legal_actions = self.env.action_space.list_of_actions
        
        # Update action mask
        self.action_mask = np.zeros(self.action_space.n, dtype=bool)
        self.action_mask[:len(self.cached_legal_actions)] = True
    
    def get_action_mask(self) -> np.ndarray:
        """
        Get mask of legal actions.
        
        Returns:
            np.ndarray: Boolean array where True = legal action
        """
        return self.action_mask.copy()
    
    def _opponent_move(self):
        """Execute opponent's move."""
        if self.opponent_agent is None:
            # Random opponent - update actions and sample
            self.env.update_actions_light()
            legal_actions = self.env.action_space.list_of_actions
            if legal_actions:
                action = np.random.choice(legal_actions)
                self.env.step('deterministic', action)
        else:
            # Use opponent agent
            observation_obj = self.env.current_state_of_the_game
            action = self.opponent_agent.choose_action(observation_obj)
            self.env.step('deterministic', action)
    
    def _compute_reward(self, score_diff: int, won: bool, lost: bool) -> float:
        """
        Compute reward based on configured mode.
        
        Args:
            score_diff: Change in player's score this turn
            won: Whether player won the game
            lost: Whether player lost the game
            
        Returns:
            float: Reward value
        """
        if self.reward_mode == 'score_naive':
            # Sparse: +1 for win, -1 for loss, 0 otherwise
            if won:
                return 1.0
            elif lost:
                return -1.0
            else:
                return 0.0
                
        elif self.reward_mode == 'score_win_bonus':
            # Score difference + large bonus for win
            reward = float(score_diff)
            if won:
                reward += 100.0
            elif lost:
                reward -= 100.0
            return reward
            
        elif self.reward_mode == 'score_progress':
            # Dense: score difference + small progress reward + win bonus
            reward = float(score_diff) + 0.01  # Progress reward
            if won:
                reward += 50.0
            elif lost:
                reward -= 50.0
            return reward
            
        else:
            raise ValueError(f"Unknown reward mode: {self.reward_mode}")
    
    def render(self):
        """Render the current state (not implemented)."""
        pass
    
    def close(self):
        """Clean up resources."""
        pass


# Utility functions
def make_splendor_env(
    reward_mode: str = 'score_progress',
    opponent_agent=None,
    **kwargs
) -> SplendorGymWrapper:
    """
    Factory function to create Splendor environment.
    
    Args:
        reward_mode: Reward shaping mode
        opponent_agent: Opponent agent (None for random)
        **kwargs: Additional arguments for wrapper
        
    Returns:
        SplendorGymWrapper: Wrapped environment
    """
    return SplendorGymWrapper(
        opponent_agent=opponent_agent,
        reward_mode=reward_mode,
        **kwargs
    )
