#!/usr/bin/env python3
"""Run a single random action and print vectorizer/detection results."""
import sys
from pathlib import Path
import numpy as np

# ensure project paths
root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root / 'project_event_based' / 'src'))
sys.path.insert(0, str(root / 'modules'))

from utils.splendor_gym_wrapper import make_splendor_env
from utils.state_vectorizer_event import vectorize_state_event
from utils.event_detector import detect_events
from reward.event_based_reward import compute_event_reward
from utils.safe_action_wrapper import SafeActionWrapper


from utils.splendor_gym_wrapper import SplendorGymWrapper

def unwrap(env):
    cur = env
    # unwrap until we reach the SplendorGymWrapper itself
    while hasattr(cur, 'env') and not isinstance(cur, SplendorGymWrapper):
        cur = cur.env
    return cur


def main():
    print('cwd', Path.cwd())
    env = make_splendor_env(reward_mode='score_progress', opponent_agent=None, max_turns=50)
    env = SafeActionWrapper(env)
    obs, info = env.reset()
    print('reset info', info)

    inner = unwrap(env)
    inner._update_legal_actions()
    print('legal count', len(inner.cached_legal_actions))
    action_idx = np.random.randint(0, len(inner.cached_legal_actions))
    action_obj = inner.cached_legal_actions[action_idx]
    action_meta = {'type': getattr(action_obj, 'action_type', action_obj.__class__.__name__)}

    # use wrapper to get observation and then slice first 40 dims for event-based vector
    prev_vec_full = inner._get_observation()
    prev_vec = prev_vec_full[:40]
    obs, base_reward, done, trunc, info2 = env.step(action_idx)
    next_vec_full = inner._get_observation()
    next_vec = next_vec_full[:40]

    ev = detect_events(prev_vec, action_meta, next_vec)
    print('action_meta', action_meta)
    print('prev_vec', prev_vec)
    print('next_vec', next_vec)
    print('ev', ev)
    print('ev_reward', compute_event_reward(ev))


if __name__ == '__main__':
    main()
