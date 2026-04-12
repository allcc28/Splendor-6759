"""PPO agents for Splendor RL project."""

try:
    from .ppo_agent import PPOAgent
    __all__ = ['PPOAgent']
except ImportError:
    # When imported from a flat PYTHONPATH, relative imports may fail.
    __all__ = []
