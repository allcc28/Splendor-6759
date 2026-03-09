"""Event-based reward utilities."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


EVENT_NAMES = (
    "take_gems",
    "buy_card",
    "reserve_card",
    "score_up",
    "reach_15",
    "scarcity_take",
    "block_reserve",
    "buy_reserved",
    "engine_spike",
)


DEFAULT_EVENT_WEIGHTS = np.array(
    [
        0.01,
        1.00,
        0.05,
        0.80,
        25.0,
        0.25,
        0.50,
        0.40,
        2.00,
    ],
    dtype=np.float32,
)


def compute_event_reward(
    event_vec: Sequence[int],
    weights: Optional[Sequence[float]] = None,
) -> float:
    """Return the scalar reward induced by a 9-d event vector."""
    events = np.asarray(event_vec, dtype=np.float32)
    if events.shape != (len(EVENT_NAMES),):
        raise ValueError(f"event_vec must be length {len(EVENT_NAMES)}")

    reward_weights = (
        np.asarray(weights, dtype=np.float32)
        if weights is not None
        else DEFAULT_EVENT_WEIGHTS
    )
    if reward_weights.shape != (len(EVENT_NAMES),):
        raise ValueError(f"weights must be length {len(EVENT_NAMES)}")

    return float(np.dot(events, reward_weights))
