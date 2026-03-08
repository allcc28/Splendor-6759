"""Debug smoke-test (isolated)"""
import sys
import os
# Ensure local event-based src is preferred on import path, then modules/
sys.path.insert(0, "project_event_based/src")
sys.path.insert(0, "modules")

import numpy as np
from utils.splendor_gym_wrapper import make_splendor_env
from utils.state_vectorizer_event import vectorize_state_event
from utils.event_detector import detect_events
from reward.event_based_reward import compute_event_reward


def main(steps=5):
    env = make_splendor_env(reward_mode='score_progress', opponent_agent=None, max_turns=50)
    # wrap with SafeActionWrapper so debug run won't error on invalid indices
    from utils.safe_action_wrapper import SafeActionWrapper
    env = SafeActionWrapper(env)
    obs, info = env.reset()
    print('Reset info:', info)

    # helper to unwrap all wrappers to original SplendorGymWrapper
    def unwrap(obj):
        cur = obj
        while hasattr(cur, 'env'):
            cur = cur.env
        return cur

    for t in range(steps):
        # refresh legal actions on inner env
        try:
            unwrap(env)._update_legal_actions()
        except Exception:
            pass
        inner = unwrap(env)
        n_legal = len(getattr(inner, 'cached_legal_actions', []))
        if n_legal == 0:
            print('No legal actions, ending')
            break
        action_idx = np.random.randint(0, n_legal)

        prev_state = unwrap(env).current_state_of_the_game
        prev_vec = vectorize_state_event({'board': {'gems': list(prev_state.board.gems_on_board)},
                                          'players': [
                                              {'score': prev_state.list_of_players_hands[env.player_id].number_of_my_points(),
                                               'gems': list(getattr(prev_state.list_of_players_hands[env.player_id], 'gems_on_hand', [0]*6)),
                                               'discounts': list(getattr(prev_state.list_of_players_hands[env.player_id], 'discounts', [0]*6)),
                                               'reserved_count': len(getattr(prev_state.list_of_players_hands[env.player_id], 'reserved_cards', []))},
                                              {'score': prev_state.list_of_players_hands[env.opponent_id].number_of_my_points(),
                                               'gems': list(getattr(prev_state.list_of_players_hands[env.opponent_id], 'gems_on_hand', [0]*6)),
                                               'discounts': list(getattr(prev_state.list_of_players_hands[env.opponent_id], 'discounts', [0]*6)),
                                               'reserved_count': len(getattr(prev_state.list_of_players_hands[env.opponent_id], 'reserved_cards', []))}
                                          ]}, env.player_id)

        try:
            action_obj = env.cached_legal_actions[action_idx]
            action_meta = {'type': getattr(action_obj, 'action_type', action_obj.__class__.__name__)}
        except Exception:
            action_meta = {'type': 'unknown'}

        obs, reward, terminated, truncated, info = env.step(action_idx)

        next_state = env.env.current_state_of_the_game
        next_vec = vectorize_state_event({'board': {'gems': list(next_state.board.gems_on_board)},
                                          'players': [
                                              {'score': next_state.list_of_players_hands[env.player_id].number_of_my_points(),
                                               'gems': list(getattr(next_state.list_of_players_hands[env.player_id], 'gems_on_hand', [0]*6)),
                                               'discounts': list(getattr(next_state.list_of_players_hands[env.player_id], 'discounts', [0]*6)),
                                               'reserved_count': len(getattr(next_state.list_of_players_hands[env.player_id], 'reserved_cards', []))},
                                              {'score': next_state.list_of_players_hands[env.opponent_id].number_of_my_points(),
                                               'gems': list(getattr(next_state.list_of_players_hands[env.opponent_id], 'gems_on_hand', [0]*6)),
                                               'discounts': list(getattr(next_state.list_of_players_hands[env.opponent_id], 'discounts', [0]*6)),
                                               'reserved_count': len(getattr(next_state.list_of_players_hands[env.opponent_id], 'reserved_cards', []))}
                                          ]}, env.player_id)

        ev = detect_events(prev_vec, action_meta, next_vec)
        ev_reward = compute_event_reward(ev)

        print(f"Step {t}: action_idx={action_idx}, action_meta={action_meta['type']}")
        print('  prev_score=', prev_vec[6], ' next_score=', next_vec[6], ' score_diff=', next_vec[6]-prev_vec[6])
        print('  event_vec=', ev.tolist())
        print('  event_reward=', ev_reward, ' env_reward(returned)=', reward)

        if terminated or truncated:
            print('Episode ended:', info)
            break

    env.close()


if __name__ == '__main__':
    main(steps=10)
