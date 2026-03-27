import os
import sys
import numpy as np
from pathlib import Path
from splendor_mcts import SplendorMCTS

# 1. Base interception (Numpy 2.0 & Gym)
if not hasattr(np, 'bool8'): np.bool8 = bool
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# --- Import SB3 and environment ---
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO

# --- Path Injection ---
ROOT_PATH = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = str(ROOT_PATH)

sys.path.insert(0, os.path.join(PROJECT_ROOT, "project_event_based", "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "modules"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "project_event_based", "scripts"))

from utils.splendor_gym_wrapper import make_splendor_env
from train_maskable import EventRewardWrapper

class ScoreBasedOpponent:
    def __init__(self, path):
        self.model = MaskablePPO.load(path, device='cpu')
        self.env_ref = None 

    def choose_action(self, observation, action_mask=[]):
        try:
            # A. Extract observation value (handle DeterministicObservation)
            obs_data = observation.observation if hasattr(observation, 'observation') else observation
            
            # B. Force alignment to 135 dimensions
            obs_array = np.array(obs_data).astype(np.float32).reshape(135,)
            
            # C. Predict index
            action_idx, _ = self.model.predict(obs_array, deterministic=True)
            action_idx = int(action_idx)
            
            # D. [Core Fix] Force fetch underlying object
            actual_env = self.env_ref.unwrapped.env
            legal_actions = actual_env.current_state_of_the_game.get_out_possible_actions()
            
            # Must return an object, absolutely cannot return an int
            selected_action = legal_actions[action_idx] if action_idx < len(legal_actions) else legal_actions[0]
            
            if isinstance(selected_action, int):
                raise TypeError("Still an int!")
                
            return selected_action

        except Exception:
            # Ultimate fallback: If anything breaks above, just grab a random object from current legal actions
            try:
                return self.env_ref.unwrapped.env.current_state_of_the_game.get_out_possible_actions()[0]
            except:
                return None 

def arena_match():
    # Use dynamically generated paths
    NEW_PATH = str(ROOT_PATH / "project_event_based" / "notebooks" / "models" / "v3_1m_3800000_steps.zip")
    OLD_PATH = str(ROOT_PATH / "project" / "logs" / "ppo_score_based_v1_20260224_113524" / "final_model.zip")
    
    opp = ScoreBasedOpponent(OLD_PATH)
    # Create environment
    base_env = make_splendor_env(reward_mode='score_progress', opponent_agent=opp, max_turns=200)
    opp.env_ref = base_env # Establish vital link
    
    
    # Wrap new model
    config = {
        'reward': {
            'event_weights': np.array([
                0.5,  # 1. Buy card (Base reward)
                0.1,  # 2. Take gem (Small reward, encourage action)
                0.05, # 3. Reserve card (Strategy reward)
                1.0,  # 4. Key: Reduce gem gap (The soul of your 109-dim feature!)
                -0.1, # 5. Increase gem gap (Penalize random spending)
                0.2,  # 6. Activate noble progress
                0.0,  # 7. Other events...
                0.0,
                0.0
            ])
        },
        'environment': {
            'combine_event_and_score': True # Must be set to True, otherwise it only looks at Score
        }
    }
    env = EventRewardWrapper(base_env, config=config)
    
    my_model = MaskablePPO.load(NEW_PATH, device='cpu')
    mcts = SplendorMCTS(model=my_model, num_simulations=50)
    
    print(" Arena officially open: New vs Old model deathmatch...")
    for i in range(30):
        obs, _ = env.reset()
        done = False
        while not done:
            masks = env.action_masks()
            action = mcts.search(env, current_obs=obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        print(f"Game {i+1} Finished! Winner Info: {info.get('agent_won', 'Unknown')}")

if __name__ == "__main__":
    arena_match()