"""Wrapper that augments observations and rewards with event-based shaping."""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    from ..reward.event_based_reward import DEFAULT_EVENT_WEIGHTS, EVENT_NAMES, compute_event_reward
except (ImportError, ValueError):
    from reward.event_based_reward import DEFAULT_EVENT_WEIGHTS, EVENT_NAMES, compute_event_reward

try:
    from .event_detector import capture_state_snapshot, detect_events
    from .state_vectorizer import SplendorStateVectorizer
except (ImportError, ValueError):
    from event_detector import capture_state_snapshot, detect_events
    from state_vectorizer import SplendorStateVectorizer


class EventRewardWrapper(gym.Wrapper):
    """Apply event-based reward shaping on top of the current Splendor wrapper."""

    def __init__(self, env: gym.Env, event_config: Optional[dict[str, Any]] = None):
        super().__init__(env)
        cfg = event_config or {}

        self.combine_with_score = bool(cfg.get("combine_with_score", True))
        self.include_gem_gaps = bool(cfg.get("include_gem_gaps", True))
        self.append_last_event = bool(cfg.get("append_last_event", True))

        weights = cfg.get("weights")
        self.event_weights = (
            np.asarray(weights, dtype=np.float32)
            if weights is not None
            else DEFAULT_EVENT_WEIGHTS.copy()
        )
        if self.event_weights.shape != (len(EVENT_NAMES),):
            raise ValueError(f"event_shaping.weights must be length {len(EVENT_NAMES)}")

        self.vectorizer = SplendorStateVectorizer()
        self.last_event = np.zeros(len(EVENT_NAMES), dtype=np.float32)
        self.event_totals = np.zeros(len(EVENT_NAMES), dtype=np.float32)
        self.remapped_count = 0
        self.total_steps = 0
        self.event_reward_total = 0.0
        self.base_reward_total = 0.0
        self._base_env = None

        self._base_obs_dim = int(np.prod(self.env.observation_space.shape))
        self._gap_dim = 60 if self.include_gem_gaps else 0
        self._event_dim = len(EVENT_NAMES) if self.append_last_event else 0
        total_dim = self._base_obs_dim + self._gap_dim + self._event_dim
        self.observation_space = spaces.Box(
            low=np.zeros(total_dim, dtype=np.float32),
            high=np.ones(total_dim, dtype=np.float32),
            dtype=np.float32,
        )

    def reset_event_stats(self) -> None:
        self.event_totals = np.zeros_like(self.event_totals)
        self.remapped_count = 0
        self.total_steps = 0
        self.event_reward_total = 0.0
        self.base_reward_total = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_event.fill(0.0)
        base_env = self._find_splendor_wrapper()
        return self._build_observation(obs, base_env), info

    def step(self, action):
        base_env = self._find_splendor_wrapper()
        prev_snapshot = capture_state_snapshot(
            base_env.env.current_state_of_the_game,
            base_env.player_id,
        )
        action_obj = self._resolve_action(action, base_env)

        obs, base_reward, terminated, truncated, info = self.env.step(action)

        next_snapshot = getattr(base_env, "last_post_agent_snapshot", None)
        if next_snapshot is None:
            next_snapshot = capture_state_snapshot(
                base_env.env.current_state_of_the_game,
                base_env.player_id,
            )
        event_vec = detect_events(prev_snapshot, action_obj, next_snapshot)
        event_reward = compute_event_reward(event_vec, weights=self.event_weights)
        reward = base_reward + event_reward if self.combine_with_score else event_reward

        self.last_event = event_vec.astype(np.float32)
        self.event_totals += self.last_event
        self.total_steps += 1
        self.event_reward_total += float(event_reward)
        self.base_reward_total += float(base_reward)

        info = dict(info) if info is not None else {}
        if info.get("remapped_action"):
            self.remapped_count += 1
        info["last_event"] = self.last_event.copy()
        info["event_reward"] = float(event_reward)
        info["base_reward"] = float(base_reward)

        return self._build_observation(obs, base_env), float(reward), terminated, truncated, info

    def action_masks(self):
        return self.env.action_masks()

    def get_action_mask(self):
        return self.env.get_action_mask()

    def _find_splendor_wrapper(self):
        if self._base_env is not None:
            return self._base_env
        current = self.env
        visited = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            if hasattr(current, "cached_legal_actions") and hasattr(current, "player_id"):
                self._base_env = current
                return current
            current = getattr(current, "env", None)
        raise RuntimeError("Could not locate SplendorGymWrapper under EventRewardWrapper")

    def _resolve_action(self, action_idx: int, base_env) -> Optional[Any]:
        if not isinstance(action_idx, (int, np.integer)):
            return None
        if 0 <= int(action_idx) < len(base_env.cached_legal_actions):
            return base_env.cached_legal_actions[int(action_idx)]
        return None

    def _build_observation(self, base_obs: np.ndarray, base_env=None) -> np.ndarray:
        obs_parts = [np.asarray(base_obs, dtype=np.float32).reshape(-1)]
        if self.include_gem_gaps:
            if base_env is None:
                base_env = self._find_splendor_wrapper()
            gap_features = self.vectorizer.get_gem_gap_features(
                base_env.env.current_state_of_the_game,
                base_env.player_id,
            )
            obs_parts.append(gap_features)
        if self.append_last_event:
            obs_parts.append(self.last_event.astype(np.float32))
        return np.concatenate(obs_parts).astype(np.float32)


def event_shaping_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("event_shaping", {}).get("enabled", False))


def maybe_wrap_with_event_shaping(env: gym.Env, config: dict[str, Any]) -> gym.Env:
    if not event_shaping_enabled(config):
        return env
    return EventRewardWrapper(env, config.get("event_shaping", {}))
