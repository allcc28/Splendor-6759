"""Reward shaping utilities for Splendor RL."""

from .event_based_reward import DEFAULT_EVENT_WEIGHTS, EVENT_NAMES, compute_event_reward

__all__ = [
    "DEFAULT_EVENT_WEIGHTS",
    "EVENT_NAMES",
    "compute_event_reward",
]
