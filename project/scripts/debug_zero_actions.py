"""Debug script to investigate why cached_legal_actions becomes empty."""

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "modules")

import numpy as np
from stable_baselines3 import PPO
from project.src.utils.splendor_gym_wrapper import SplendorGymWrapper


def diagnose_zero_actions():
    model = PPO.load("project/logs/ppo_score_based_v1_20260224_113524/final_model")
    
    zero_action_count = 0
    normal_end_count = 0
    total_episodes = 20
    
    for ep in range(total_episodes):
        env = SplendorGymWrapper(opponent_agent=None, reward_mode='score_progress', max_turns=200)
        obs, info = env.reset()
        
        for step in range(200):
            n_legal = len(env.cached_legal_actions)
            
            if n_legal == 0:
                state = env.env.current_state_of_the_game
                print(f"\nEp {ep}, Step {step}: 0 legal actions!")
                print(f"  is_done={env.env.is_done}, turn={env.turn_count}")
                print(f"  active_player={state.active_player_id}, agent_id={env.player_id}")
                print(f"  P0 score={state.list_of_players_hands[0].number_of_my_points()}")
                print(f"  P1 score={state.list_of_players_hands[1].number_of_my_points()}")
                
                # Try full action update
                env.env.update_actions()
                full = env.env.action_space.list_of_actions
                print(f"  After update_actions(): {len(full)} actions")
                
                # Check board state
                board = state.board
                print(f"  Board gems: {dict(board.gems_on_board)}")
                print(f"  Cards on board: {[len(board.cards_on_board.get(r, [])) for r in board.cards_on_board]}")
                
                zero_action_count += 1
                break
            
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(int(action))
            
            if term or trunc:
                agent_score = info.get('player_score', 0)
                opp_score = info.get('opponent_score', 0)
                won = info.get('agent_won', False)
                err = info.get('error', '')
                if err == 'invalid_action':
                    # This shouldn't happen with n_legal > 0
                    print(f"Ep {ep}: Invalid action at step {step}! n_legal was {n_legal}, action was {action}")
                else:
                    normal_end_count += 1
                    status = "WON" if won else "LOST"
                    print(f"Ep {ep}: {status} at step {step}, score={agent_score} vs {opp_score}")
                break
    
    print(f"\n=== Summary ({total_episodes} episodes) ===")
    print(f"Zero-action terminations: {zero_action_count}")
    print(f"Normal game endings: {normal_end_count}")
    print(f"Rate: {zero_action_count/total_episodes*100:.0f}% games had 0 legal actions")


if __name__ == "__main__":
    diagnose_zero_actions()
