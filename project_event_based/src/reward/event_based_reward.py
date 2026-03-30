import numpy as np
from typing import Sequence, Optional


DEFAULT_WEIGHTS = np.array([
    0.01,  # Is_Take_Gems
    5.0,   # Is_Buy_Card
    -0.5,  # Is_Reserve
    8.0,   # Is_Score_Up
    50.0,  # Is_Lethal
    0.25,  # Scarcity_Take
    0.5,   # Block_Reserve
    0.4,   # Buy_Reserved
    2.0,   # Engine_Spike
], dtype=np.float32)


def compute_event_reward(event_vec: Sequence[int], weights: Optional[Sequence[float]] = None) -> float:
    ev = np.asarray(event_vec, dtype=np.float32)
    if ev.shape != (9,):
        raise ValueError('event_vec must be length 9')
    w = np.asarray(weights, dtype=np.float32) if weights is not None else DEFAULT_WEIGHTS
    if w.shape != (9,):
        raise ValueError('weights must be length 9')
    return float(np.dot(ev, w))


if __name__ == '__main__':
    ev = np.array([1, 1, 0, 1, 0, 0, 0, 0, 1], dtype=np.int32)
    print('reward =', compute_event_reward(ev))
