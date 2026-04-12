"""Utility modules for Splendor RL agents."""

try:
    from .event_reward_wrapper import EventRewardWrapper, event_shaping_enabled, maybe_wrap_with_event_shaping
    from .state_vectorizer import SplendorStateVectorizer, vectorize_state
    from .splendor_gym_wrapper import SplendorGymWrapper, make_splendor_env

    __all__ = [
        'EventRewardWrapper',
        'SplendorStateVectorizer',
        'event_shaping_enabled',
        'vectorize_state',
        'maybe_wrap_with_event_shaping',
        'SplendorGymWrapper',
        'make_splendor_env'
    ]
except ImportError:
    # When imported from a flat PYTHONPATH (e.g., AlphaZero trainer),
    # relative imports may fail. Individual modules can still be imported directly.
    pass
