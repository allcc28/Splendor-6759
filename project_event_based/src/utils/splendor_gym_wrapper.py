"""
Gym-compatible wrapper for Splendor environment (event-based isolated copy).

This file is a local copy of the project's `splendor_gym_wrapper.py` adapted
to prefer the event-based state vectorizer when available, so
`project_event_based/` can run without importing from `project/src`.
"""

import sys
sys.path.insert(0, "modules")

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional

from gym_splendor_code.envs.splendor import SplendorEnv
from gym_splendor_code.envs.mechanics.action import Action

# Prefer local event vectorizer when present; otherwise try to import the
# original SplendorStateVectorizer. If neither is available create a
# minimal fallback that returns zeros.
try:
    # If a legacy class is present in this folder
    from .state_vectorizer import SplendorStateVectorizer
except Exception:
    try:
        # Use event-based vectorizer to build a 40-d prefix and pad to 135
        from .state_vectorizer_event import vectorize_state_event

        class SplendorStateVectorizer:
            def vectorize(self, state_obj, player_id, turn_count=0):
                # Convert the game state object into the expected dict shape
                board_gems = [0] * 6
                try:
                    bg = getattr(state_obj.board, 'gems_on_board', None)
                    if bg is None:
                        bg = getattr(state_obj.board, 'tokens_on_board', None)
                    if bg is not None:
                        board_gems = list(bg)[:6]
                except Exception:
                    pass

                players = []
                try:
                    hands = list(state_obj.list_of_players_hands)
                except Exception:
                    hands = []
                for p in hands:
                    try:
                        score = p.number_of_my_points()
                    except Exception:
                        score = getattr(p, 'score', 0)
                    try:
                        gems = list(getattr(p, 'gems_on_hand', getattr(p, 'tokens', [0]*6)))[:6]
                    except Exception:
                        gems = [0]*6
                    try:
                        discounts = list(getattr(p, 'discounts', getattr(p, 'permanent_discounts', [0]*6)))[:6]
                    except Exception:
                        discounts = [0]*6
                    try:
                        reserved = int(len(getattr(p, 'reserved_cards', [])))
                    except Exception:
                        reserved = int(getattr(p, 'reserved_count', 0))
                    players.append({'score': score, 'gems': gems, 'discounts': discounts, 'reserved_count': reserved})
                while len(players) < 2:
                    players.append({'score': 0, 'gems': [0]*6, 'discounts': [0]*6, 'reserved_count': 0})

                state_dict = {'board': {'gems': board_gems}, 'players': players}
                vec40 = vectorize_state_event(state_dict, active_player_index=player_id)
                vec = np.zeros(135, dtype=np.float32)
                vec[:40] = vec40
                return vec

    except Exception:
        class SplendorStateVectorizer:
            def vectorize(self, state_obj, player_id, turn_count=0):
                return np.zeros(135, dtype=np.float32)


class SplendorGymWrapper(gym.Env):
    """
    Gym wrapper for Splendor environment compatible with Stable-Baselines3.
    
    This copy is intentionally self-contained under `project_event_based/src/utils`.
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        opponent_agent=None,
        reward_mode: str = 'score_progress',
        max_turns: int = 120,
        player_id: int = 0
    ):
        super().__init__()
        self.env = SplendorEnv()
        self.vectorizer = SplendorStateVectorizer()
        self.opponent_agent = opponent_agent
        self.reward_mode = reward_mode
        self.max_turns = max_turns
        self.player_id = player_id
        self.opponent_id = 1 - player_id
        self.turn_count = 0
        self.prev_score = 0
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(135,), dtype=np.float32)
        self.action_space = spaces.Discrete(200)
        self.cached_legal_actions = []
        self.action_mask = np.zeros(200, dtype=bool)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        observation_obj = self.env.reset()
        self.turn_count = 0
        self.prev_score = 0
        if self.player_id == 1:
            self._opponent_move()
        self._update_legal_actions()
        obs = self._get_observation()
        info = {
            'turn': self.turn_count,
            'legal_actions_count': len(self.cached_legal_actions),
            'player_score': self.env.current_state_of_the_game.list_of_players_hands[self.player_id].number_of_my_points(),
            'opponent_score': self.env.current_state_of_the_game.list_of_players_hands[self.opponent_id].number_of_my_points()
        }
        return obs, info

    def step(self, action_idx: int):
        if action_idx >= len(self.cached_legal_actions):
            obs = self._get_observation()
            reward = -1.0
            terminated = True
            truncated = False
            info = {
                'turn': self.turn_count,
                'error': 'invalid_action',
                'legal_actions_count': len(self.cached_legal_actions)
            }
            return obs, reward, terminated, truncated, info

        action = self.cached_legal_actions[action_idx]
        observation_obj, _, done, info_dict = self.env.step('deterministic', action)
        current_score = self.env.current_state_of_the_game.list_of_players_hands[self.player_id].number_of_my_points()
        score_diff = current_score - self.prev_score
        agent_won = done and info_dict.get('winner_id') == self.player_id
        agent_lost = done and info_dict.get('winner_id') == self.opponent_id
        reward = self._compute_reward(score_diff, agent_won, agent_lost)
        self.turn_count += 1
        truncated = self.turn_count >= self.max_turns
        terminated = done or agent_lost
        if not terminated and not truncated:
            self._opponent_move()
            if self.env.is_done:
                terminated = True
                agent_lost = True
                reward = -1.0
        self.prev_score = self.env.current_state_of_the_game.list_of_players_hands[self.player_id].number_of_my_points()
        self._update_legal_actions()
        obs = self._get_observation()
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
        state = self.env.current_state_of_the_game
        return self.vectorizer.vectorize(state, self.player_id, self.turn_count)

    def _update_legal_actions(self):
        self.env.update_actions_light()
        self.cached_legal_actions = self.env.action_space.list_of_actions
        self.action_mask = np.zeros(self.action_space.n, dtype=bool)
        self.action_mask[:len(self.cached_legal_actions)] = True

    def get_action_mask(self) -> np.ndarray:
        return self.action_mask.copy()

    def _opponent_move(self):
        if self.opponent_agent is None:
            self.env.update_actions_light()
            legal_actions = self.env.action_space.list_of_actions
            if legal_actions:
                action = np.random.choice(legal_actions)
                self.env.step('deterministic', action)
        else:
            observation = self.env.show_observation('deterministic')
            action = self.opponent_agent.choose_action(observation, [])
            if action is not None:
                self.env.step('deterministic', action)

    def _compute_reward(self, score_diff: int, won: bool, lost: bool) -> float:
        if self.reward_mode == 'score_naive':
            if won:
                return 1.0
            elif lost:
                return -1.0
            else:
                return 0.0
        elif self.reward_mode == 'score_win_bonus':
            reward = float(score_diff)
            if won:
                reward += 100.0
            elif lost:
                reward -= 100.0
            return reward
        elif self.reward_mode == 'score_progress':
            reward = float(score_diff) + 0.01
            if won:
                reward += 50.0
            elif lost:
                reward -= 50.0
            return reward
        else:
            raise ValueError(f"Unknown reward mode: {self.reward_mode}")

    def render(self):
        pass

    def close(self):
        pass


def make_splendor_env(reward_mode: str = 'score_progress', opponent_agent=None, **kwargs) -> SplendorGymWrapper:
    return SplendorGymWrapper(opponent_agent=opponent_agent, reward_mode=reward_mode, **kwargs)
