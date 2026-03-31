import numpy as np
import gymnasium as gym
from typing import Optional


class SafeActionWrapper(gym.Wrapper):
    """Wrap an environment to avoid executing illegal/disallowed actions.

    Behavior:
      - If the incoming action index is legal (according to env.get_action_mask() or
        env.cached_legal_actions length), it is passed through.
      - If illegal, it is replaced with a uniformly random legal action index.
      - Adds `info['remapped_action'] = True` when remapping occurs.

    This is a pragmatic safety-layer when MaskablePPO is not available.
    """

    def __init__(self, env: gym.Env, seed: Optional[int] = None):
        super().__init__(env)
        self.rng = np.random.RandomState(seed)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action_idx: int):
        # try to get mask
        try:
            mask = self.env.get_action_mask()
        except Exception:
            mask = None

        legal_n = None
        try:
            legal_n = len(self.env.cached_legal_actions)
        except Exception:
            legal_n = None

        remapped = False
        # If mask exists, check it
        if mask is not None:
            mask = mask.astype(bool)
            if action_idx < 0 or action_idx >= mask.shape[0] or not mask[action_idx]:
                # pick a legal index
                legal_indices = np.nonzero(mask)[0]
                if len(legal_indices) == 0:
                    # no legal actions: pass through to let env handle
                    chosen = action_idx
                else:
                    chosen = int(self.rng.choice(legal_indices))
                    remapped = True
            else:
                chosen = int(action_idx)

        elif legal_n is not None:
            if action_idx < 0 or action_idx >= legal_n:
                # remap to random legal in range
                chosen = int(self.rng.randint(0, legal_n))
                remapped = True
            else:
                chosen = int(action_idx)
        else:
            # no info available; pass through
            chosen = int(action_idx)

        obs, reward, terminated, truncated, info = self.env.step(chosen)
        if remapped:
            info = dict(info) if info is not None else {}
            info['remapped_action'] = True
            info['remapped_from'] = int(action_idx)
            info['remapped_to'] = int(chosen)

        return obs, reward, terminated, truncated, info
