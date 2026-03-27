import sys
import numpy as np

# 1. Fix Numpy 2.0 Compatibility
if not hasattr(np, 'bool8'):
    np.bool8 = bool

# 2. [Core Fix] Force intercept shimmy error logic via sys.modules
# We fake a shimmy.GymV26CompatibilityV0 that directly returns the environment
class FakeShimmy:
    @staticmethod
    def GymV26CompatibilityV0(env, **kwargs):
        return env

# Create fake shimmy module structure
import types
fake_shimmy = types.ModuleType("shimmy")
fake_shimmy.GymV26CompatibilityV0 = FakeShimmy.GymV26CompatibilityV0
sys.modules["shimmy"] = fake_shimmy

# 3. Force hijack SB3's patch_gym module (double insurance)
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Now proceed with normal imports - SB3 will use our fake shimmy
import gymnasium as gym
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

print("Low-level interception activated: Fake shimmy module created to bypass Discrete(200) conversion error")



import yaml
from pathlib import Path


# --- 0. Path Mounting ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
modules_dir = project_root / "modules"
vectorizer_dir = modules_dir / "nn_models" / "utils"
scripts_dir = current_dir.parent / "scripts" # project_event_based/scripts
for p in [modules_dir, vectorizer_dir, scripts_dir]:
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)
# --- 1. Reinforcement Learning Core Library Imports ---
# Note: Need to run pip install sb3-contrib
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# --- 2. Module Imports (paths are now mounted) ---
from gym_splendor_code.envs.mechanics.enums import GemColor
from vectorizer import Vectorizer 
# Assume SplendorGymWrapper is in the same directory as your script or mounted path
try:
    from utils.splendor_gym_wrapper import SplendorGymWrapper
except ImportError:
    # If not found, adjust import or sys.path according to actual location
    print("Tip: Ensure SplendorGymWrapper path is correct")

# --- 3. Define Integrated EventRewardWrapper ---
import gymnasium as gym
from gymnasium import spaces
class EventRewardWrapper(gym.Wrapper):
    def __init__(self, env, config):
        super().__init__(env)
        self.config = config
        # Feature dimensions: 40(base) + 60(gem gaps) + 9(event history) = 109
        self.observation_space = spaces.Box(
            low=0, high=999, shape=(109,), dtype=np.float32
        )
        self.vectorizer = Vectorizer()
        self.event_weights = np.array(config['reward']['event_weights'])
        self.combine_with_score = config['environment'].get('combine_event_and_score', False)
        self.last_event = np.zeros(9)

    def _get_gem_gap_features(self):
        """
        Calculate the number of missing gems for the current player to buy each of the 12 cards on the field (60 dimensions).
        Directly tell the Agent: 'How many more gems you need to buy this card'.
        """
        obs = self.last_obs_raw # 135-dimensional raw vector
        
        # 1. Extract current player assets (gem holdings + discounts from purchased cards)
        # In standard Splendor Vectorizer: [0:5] are gems, [5:10] are discounts
        player_gems = obs[0:5] 
        player_discounts = obs[5:10]
        total_assets = player_gems + player_discounts
        
        gaps = []
        # 2. Extract cost information of 12 cards on the field
        # Index 40 onwards is usually detailed data of field cards
        cards_offset = 40 
        
        for i in range(12): # Iterate over 12 slots on the field
            card_start = cards_offset + (i * 5)
            # Ensure no out-of-bounds extraction for 135-dimensional vector
            if card_start + 5 <= len(obs):
                card_cost = obs[card_start : card_start + 5]
                # Core logic: gap = max(0, cost - existing assets)
                # This directly reduces the difficulty of model learning "purchase conditions"
                gap = np.maximum(0, card_cost - total_assets)
                gaps.extend(gap.tolist())
            else:
                gaps.extend([0.0] * 5)
        
        # 3. Strictly pad to 60 dimensions to prevent ValueError caused by dimension collapse
        if len(gaps) < 60:
            gaps.extend([0.0] * (60 - len(gaps)))
            
        return np.array(gaps[:60], dtype=np.float32)

    def _get_obs(self, obs_40, events_9):
        """Ensure definition receives self + 2 logical parameters"""
        # Get 60-dimensional gem gap features
        gap_features = self._get_gem_gap_features() 
        # Concatenate: 40 + 60 + 9 = 109
        return np.concatenate([obs_40, gap_features, events_9]).astype(np.float32)

    def reset(self, seed=None, options=None):
        # 1. Explicitly handle seed (required by Gymnasium)
        if seed is not None:
            # If underlying environment supports seed (usually SplendorGymWrapper needs this)
            # Alternatively, simply call self.env.reset(seed=seed, options=options)
            pass 
        
        # 2. Call reset of underlying environment
        # Note: Gymnasium's reset returns (obs, info)
        obs, info = self.env.reset(seed=seed, options=options)
        
        # 3. Your logic: initialize event recording
        self.last_event = np.zeros(9, dtype=np.int32)
        self.last_obs_raw = obs.copy()
        
        # 4. Concatenate your 109-dimensional features (40 base + 60 gaps + 9 events)
        # Ensure _get_obs logic is correct here
        full_obs = self._get_obs(obs[:40], self.last_event)
        return full_obs, info

    def step(self, action):
        # 1. Lock raw state before action (must be before env.step)
        prev_vec = self.last_obs_raw[:40].copy() 

        # 2. Execute action and get 5 return values from Gymnasium
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        next_vec = obs[:40]

        # 3. Core: If state hasn't changed at all, skip detection to prevent 1.0 false positive
        if np.array_equal(prev_vec, next_vec):
            ev = np.zeros(9, dtype=np.int32)
        else:
            # Core Fix 2: Only run your trusted detect_events when state changes
            try:
                actual_action = self.unwrapped.cached_legal_actions[action]
            except:
                actual_action = None
            from utils.event_detector import detect_events
            ev = detect_events(prev_vec, actual_action, next_vec)

        # 4. Calculate reward and update cache
        from reward.event_based_reward import compute_event_reward
        ev_reward = compute_event_reward(ev, weights=self.event_weights)
        reward = float(ev_reward + base_reward) if self.combine_with_score else float(ev_reward)
        
        self.last_obs_raw = obs.copy()
        self.last_event = ev.astype(np.int32)
        
        # 5. Ensure last_event is added to info for console table printing
        info['last_event'] = ev.astype(np.int32)
        
        # 6. Generate enhanced 109-dimensional vector containing 60-dimensional gem gaps
        new_obs = self._get_obs(next_vec, self.last_event)
        
        return new_obs, reward, terminated, truncated, info

    def action_masks(self):
        """Action mask interface for MaskablePPO"""
        return self.unwrapped.get_action_mask()

# --- 4. Curriculum Learning Control ---
class AdvancedCurriculumCallback(BaseCallback):
    def __init__(self, total_timesteps,switch_step=300000, check_freq=1000, entropy_threshold=0.5, verbose=1):
        super().__init__(verbose)
        self.switch_step = switch_step
        self.has_switched = False
        self.total_timesteps = total_timesteps
        # Now these two lines won't throw NameError
        self.check_freq = check_freq           
        self.entropy_threshold = entropy_threshold

    def _on_step(self) -> bool:
        curr_env = self.training_env.envs[0]
        while not hasattr(curr_env, 'combine_with_score') and hasattr(curr_env, 'env'):
            curr_env = curr_env.env
        # 1. Check if we should switch to combined reward
        if self.num_timesteps >= self.switch_step and not self.has_switched:
            curr_env = self.training_env.envs[0]
            # Traverse wrappers to find variable
            while not hasattr(curr_env, 'combine_with_score') and hasattr(curr_env, 'env'):
                curr_env = curr_env.env
            if hasattr(curr_env, 'combine_with_score'):
                curr_env.combine_with_score = True
                self.has_switched = True
                print(f"\n [Step {self.num_timesteps}] Trigger switch: Introduce original score for elite training")
        #
        if hasattr(curr_env, 'event_weights'):
            # calculate the progress (0.0 to 1.0)
            progress = min(1.0, self.num_timesteps / self.total_timesteps)
            
            # dynamically adjust weights: as training progresses, we can slowly shift focus from Engine_Spike to Is_Score_Up
            new_engine_weight = 12.0 - (10.0 * progress)  # 12.0 slowly decreases to 2.0
            new_score_weight = 4.0 + (4.0 * progress)     # 4.0 slowly increases to 10.0
            
            # Modify the weights in the environment
            curr_env.event_weights[8] = new_engine_weight
            curr_env.event_weights[3] = new_score_weight

            # Record to TensorBoard, so we can visualize the dynamic curriculum in the logs
            self.logger.record("dynamic_weights/engine_spike", new_engine_weight)
            self.logger.record("dynamic_weights/is_score_up", new_score_weight)
        return True
    

from stable_baselines3.common.monitor import Monitor



# --- 1. Core: Force replicate Callback with all fields from the image ---
class EventStatsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(EventStatsCallback, self).__init__(verbose)
        self.episode_rewards = []
        
        #  Core Fix: Add accumulators for episode events and steps
        self.rollout_events = np.zeros(9)
        self.rollout_steps = 0

    def _on_step(self) -> bool:
        # 1. Accumulate rewards per step
        if "rewards" in self.locals:
            self.episode_rewards.append(np.mean(self.locals["rewards"]))
            
        # 2. Accumulate events per step (completely fix bug of only counting last frame)
        infos = self.locals.get("infos", [])
        for info in infos:
            if "last_event" in info:
                self.rollout_events += info["last_event"]
                self.rollout_steps += 1 # Record total steps
                
        return True

    def _on_rollout_end(self) -> None:
        """Calculate true average occurrence rate at the end of rollout"""
        if self.rollout_steps > 0:
            # True frequency = total triggers / e.g., 2048 steps
            avg_rates = self.rollout_events / self.rollout_steps
            for i, rate in enumerate(avg_rates):
                self.logger.record(f"events/event_{i}_rate", rate)
        else:
            for i in range(9):
                self.logger.record(f"events/event_{i}_rate", 0.0)
                
        self.logger.record("events/remap_rate", 0.0)

        # Print rewards
        if self.episode_rewards:
            self.logger.record("rollout/ep_rew_mean", np.mean(self.episode_rewards))
            self.episode_rewards = [] # Clear cache
            
        # 💡 Must clear accumulators for next rollout
        self.rollout_events = np.zeros(9)
        self.rollout_steps = 0

def main():
    import yaml
    # Path resolution: point to project_event_based/configs/
    root_path = Path(__file__).resolve().parent.parent
    config_path = root_path / 'configs' / 'training' / 'ppo_event_based.yaml'
    
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return

    with open(config_path, 'r') as f:
        # Define cfg here to ensure it's a local variable
        cfg = yaml.safe_load(f)

    # 💡 Core Fix: Explicitly pass cfg to make_env
    def make_env(config):
        from utils.splendor_gym_wrapper import SplendorGymWrapper
        env = SplendorGymWrapper()
        
        # Adapt to Gymnasium spaces
        env.action_space = gym.spaces.Discrete(200)
        env.observation_space = gym.spaces.Box(low=0, high=999, shape=(109,), dtype=np.float32)
        
        # Monitor is required for automatic reward recording
        env = Monitor(env) 
        
        # Pass config to wrapper
        env = EventRewardWrapper(env, config) 
        return env

    # Use lambda closure to pass cfg
    venv = DummyVecEnv([lambda: make_env(cfg)])

    # 4. Initialize model
    model = MaskablePPO(
        "MlpPolicy",
        venv,
        verbose=1,
        ent_coef=0.05, # Maintain high exploration to trigger card purchase events
        tensorboard_log="./logs/maskable_v3_1m/"
    )
    TOTAL_STEPS = 4_000_000
    # 5. Assemble Callback list matching image requirements
    callbacks = CallbackList([
        AdvancedCurriculumCallback(total_timesteps=TOTAL_STEPS, switch_step=300000), 
        EventStatsCallback(), 
        CheckpointCallback(save_freq=50000, save_path='./models/', name_prefix='v3_1m')
    ])

    print(" Start 1M step experiment: Console will display episode_reward and event_rates tables synchronously...")
    
    # log_interval=1 ensures table refresh every round
    model.learn(total_timesteps=TOTAL_STEPS, callback=callbacks, log_interval=1)

if __name__ == '__main__':
    main()