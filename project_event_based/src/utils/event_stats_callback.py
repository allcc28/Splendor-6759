from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class EventStatsCallback(BaseCallback):
    """Callback to log aggregated event statistics from EventRewardWrapper.

    It looks for an environment wrapper that has attributes `event_totals`,
    `remapped_count`, and `total_steps` (provided by EventRewardWrapper).
    Logs per-event rates and remap rate to TensorBoard via `self.logger`.
    """

    def __init__(self, log_freq: int = 1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = int(log_freq)

    def _find_wrapper(self, env):
        # env may be VecEnv; try to get first inner env
        try:
            v = env
            if hasattr(v, 'envs'):
                v = v.envs[0]
        except Exception:
            v = env

        # drill down through wrappers
        cur = v
        visited = set()
        while cur is not None and id(cur) not in visited:
            visited.add(id(cur))
            if hasattr(cur, 'event_totals') and hasattr(cur, 'total_steps'):
                return cur
            cur = getattr(cur, 'env', None)
        return None

    def _on_step(self) -> bool:
        # only log at multiples of log_freq
        if self.num_timesteps % self.log_freq != 0:
            return True

        wrapper = self._find_wrapper(self.training_env)
        if wrapper is None:
            return True

        steps = float(max(1, getattr(wrapper, 'total_steps', 0)))
        totals = getattr(wrapper, 'event_totals', None)
        remapped = getattr(wrapper, 'remapped_count', 0)

        if totals is None:
            return True

        rates = (np.array(totals, dtype=np.float32) / steps).tolist()
        for i, r in enumerate(rates):
            self.logger.record(f'events/event_{i}_rate', float(r))

        self.logger.record('events/remap_rate', float(remapped) / steps)

        # reset counters after logging
        try:
            wrapper.event_totals = np.zeros_like(wrapper.event_totals)
            wrapper.remapped_count = 0
            wrapper.total_steps = 0
        except Exception:
            pass

        return True
