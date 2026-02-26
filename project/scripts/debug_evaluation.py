"""Debug evaluation to check game logic."""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "modules")

from gym_splendor_code.envs.splendor import SplendorEnv
from agents.random_agent import RandomAgent
from stable_baselines3 import PPO
from project.src.utils.state_vectorizer import SplendorStateVectorizer
import numpy as np


def debug_one_game():
    """Play one game with detailed logging."""
    print("=" * 80)
    print("DEBUG: Single Game PPO vs RandomAgent")
    print("=" * 80)
    
    # Load model
    model_path = 'project/logs/ppo_score_based_v1_20260224_113524/final_model'
    ppo_model = PPO.load(model_path)
    vectorizer = SplendorStateVectorizer()
    
    # Create environment
    env = SplendorEnv()
    env.reset()
    
    # Create opponent
    opponent = RandomAgent(distribution='uniform_on_types')
    opponent.env = env
    
    # PPO is player 0, Random is player 1
    ppo_id = 0
    random_id = 1
    env.set_active_player(0)
    
    print(f"PPO is Player {ppo_id}, Random is Player {random_id}")
    print(f"Initial state: Turn 0")
    print()
    
    done = False
    turn = 0
    max_turns = 200
    
    while not done and turn < max_turns:
        current_player = env.active_player_id()
        state = env.current_state_of_the_game
        
        # Get current scores
        ppo_score = state.list_of_players_hands[ppo_id].number_of_my_points()
        random_score = state.list_of_players_hands[random_id].number_of_my_points()
        
        if turn % 20 == 0 or done:
            print(f"Turn {turn}: Active Player = {current_player}")
            print(f"  PPO Score: {ppo_score}, Random Score: {random_score}")
        
        if current_player == ppo_id:
            # PPO's turn
            obs_vector = vectorizer.vectorize(state, ppo_id, turn)
            legal_actions = env.action_space.list_of_actions
            
            if not legal_actions:
                print(f"  ERROR: No legal actions for PPO at turn {turn}!")
                break
            
            action_idx, _ = ppo_model.predict(obs_vector, deterministic=True)
            
            if action_idx >= len(legal_actions):
                print(f"  WARNING: Invalid action_idx={action_idx}, legal={len(legal_actions)}")
                action_idx = 0
            
            action = legal_actions[action_idx]
            
            if turn % 20 == 0:
                print(f"  PPO chooses: {action.action_type}")
        else:
            # Random's turn
            env.update_actions_light()
            action = opponent.choose_act('deterministic')
            
            if turn % 20 == 0 and action:
                print(f"  Random chooses: {action.action_type}")
        
        if action is None:
            print(f"  ERROR: None action at turn {turn}!")
            break
        
        _, reward, done, info = env.step('deterministic', action)
        
        if done:
            print(f"\nGame ended at turn {turn + 1}")
            print(f"  Done flag: {done}")
            print(f"  Winner ID from info: {info.get('winner_id')}")
            print(f"  Final PPO Score: {ppo_score}")
            print(f"  Final Random Score: {random_score}")
        
        turn += 1
    
    if turn >= max_turns:
        print(f"\nGame hit max turns ({max_turns})")
    
    # Final scores
    final_state = env.current_state_of_the_game
    ppo_final = final_state.list_of_players_hands[ppo_id].number_of_my_points()
    random_final = final_state.list_of_players_hands[random_id].number_of_my_points()
    
    print()
    print("=" * 80)
    print(f"FINAL RESULTS:")
    print(f"  Total turns: {turn}")
    print(f"  PPO final score: {ppo_final}")
    print(f"  Random final score: {random_final}")
    print(f"  Winner: {'PPO' if info.get('winner_id') == ppo_id else 'Random'}")
    print(f"  Done status: {done}")
    print("=" * 80)


if __name__ == '__main__':
    debug_one_game()
