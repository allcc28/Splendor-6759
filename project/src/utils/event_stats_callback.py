"""TensorBoard logging for event-shaped training."""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from ..reward.event_based_reward import EVENT_NAMES


class EventStatsCallback(BaseCallback):
    """Log aggregated event statistics from EventRewardWrapper."""

    def __init__(self, log_freq: int = 10000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = int(log_freq)

    def _find_event_wrapper(self, env):
        current = env
        if hasattr(current, "envs") and current.envs:
            current = current.envs[0]

        visited = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            if hasattr(current, "event_totals") and hasattr(current, "total_steps"):
                return current
            current = getattr(current, "env", None)
        return None

    def _on_step(self) -> bool:
        if self.log_freq <= 0 or self.num_timesteps % self.log_freq != 0:
            return True

        wrapper = self._find_event_wrapper(self.training_env)
        if wrapper is None:
            return True

        steps = max(1, int(getattr(wrapper, "total_steps", 0)))
        event_totals = np.asarray(getattr(wrapper, "event_totals", None), dtype=np.float32)
        if event_totals.shape != (len(EVENT_NAMES),):
            return True

        for idx, value in enumerate(event_totals / steps):
            self.logger.record(f"events/event_{idx}_rate", float(value))

        self.logger.record(
            "events/remap_rate",
            float(getattr(wrapper, "remapped_count", 0)) / steps,
        )
        self.logger.record(
            "events/event_reward_mean",
            float(getattr(wrapper, "event_reward_total", 0.0)) / steps,
        )
        self.logger.record(
            "events/base_reward_mean",
            float(getattr(wrapper, "base_reward_total", 0.0)) / steps,
        )

        wrapper.reset_event_stats()
        return True
