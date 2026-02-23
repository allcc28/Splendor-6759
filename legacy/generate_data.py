import sys
import os
import random
import pickle
import tqdm
import numpy as np
import re

# ==========================================
#  Environment Setup & Imports
def mount_package_root():
    """
    Dynamically add the package root directory to sys.path
    to ensure smooth importing of the gym_splendor_code package.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_pkg = 'gym_splendor_code'
    check_dir = current_dir
    for _ in range(3):
        if os.path.exists(os.path.join(check_dir, target_pkg)):
            if check_dir not in sys.path:
                sys.path.insert(0, check_dir)
            return True
        check_dir = os.path.dirname(check_dir)
    for root, dirs, files in os.walk(current_dir):
        if target_pkg in dirs:
            if root not in sys.path:
                sys.path.insert(0, root)
            return True
    return False

if not mount_package_root():
    print("!! Cannot find package 'gym_splendor_code' ...")

try:
    from gym_splendor_code.envs.splendor import SplendorEnv
except ImportError:
    pass

# ==========================================
# Global Hyperparameters

NUM_GAMES = 500
OUTPUT_FILE = "expert_data_vectorized.pkl"
GREEDY_PROB = 0.8 

class MixedPolicyAgent:
    """
    A simple heuristic agent used to generate expert-like trajectories.
    Prefers buying cards and taking gems over random valid actions.
    """
    def __init__(self, env):
        self.env = env

    def select_action(self, valid_actions):
        if not valid_actions: 
            return None
        if random.random() > GREEDY_PROB:
            return random.choice(valid_actions)
        
        buy_actions = [a for a in valid_actions if 'buy' in str(a).lower() or 'purchase' in str(a).lower()]
        take_actions = [a for a in valid_actions if 'take' in str(a).lower() or 'trade' in str(a).lower()]
        
        if buy_actions: 
            return random.choice(buy_actions)
        elif take_actions: 
            return random.choice(take_actions)
        return random.choice(valid_actions)

# ==========================================
# Utility: Robust Gems Extractor

def get_gems_list(obj):
    """
    Extracts gem counts into a standard 6-dim list: [Red, Blue, Green, White, Black, Gold].
    Uses Regex to bypass highly encapsulated Enum classes (e.g., GemsCollection with <RED: 1> keys).
    """
    colors = ['red', 'blue', 'green', 'white', 'black', 'gold']
    values = [0] * 6
    if obj is None: 
        return values

    # Ultimate Regex Parsing for Enum strings (e.g., <RED: 1>: 2)
    obj_str = str(obj)
    if '<' in obj_str and '>' in obj_str and ':' in obj_str:
        try:
            for i, c in enumerate(colors):
                # Matches patterns like "<RED: 1>: 2" and extracts the '2'
                pattern = re.compile(rf'<{c}[^>]*>:\s*(\d+)', re.IGNORECASE)
                match = pattern.search(obj_str)
                if match:
                    values[i] = int(match.group(1))
            
            # Return if any valid extraction occurred
            if sum(values) > 0 or "0" in obj_str: 
                return values
        except Exception as e:
            print(f"[Warning] Regex parsing failed: {e}")

    # Fallback for standard dictionary objects
    if isinstance(obj, dict):
        for k, v in obj.items():
            k_str = str(k).lower()
            for i, c in enumerate(colors):
                if c in k_str:
                    values[i] = int(v.value) if hasattr(v, 'value') else int(v)
        return values

    # Fallback for standard object attributes
    try:
        for i, c in enumerate(colors):
            if hasattr(obj, c): values[i] = getattr(obj, c)
            elif hasattr(obj, c + '_token'): values[i] = getattr(obj, c + '_token')
    except:
        pass
            
    return values

# ==========================================
# State Vectorization
def vectorize_state(state):
    """
    Flattens the SplendorState object into a 40-dim canonical Numpy array.
    
    Dimension Dictionary:
    ---------------------------------------------------
    [Public Board] (0-5)
        [0-5]: Available Gems (Red, Blue, Green, White, Black, Gold)
    
    [Active Player] (6-19) - Always positioned first (Canonical perspective)
        [6]: Score / Victory Points
        [7-12]: Gems Possessed
        [13-18]: Card Discounts / Engines
        [19]: Number of Reserved Cards
    
    [Opponent] (20-33)
        [20]: Score / Victory Points
        [21-26]: Gems Possessed
        [27-32]: Card Discounts / Engines
        [33]: Number of Reserved Cards
        
    [Padding] (34-39)
        [34-39]: Padding zeros for future feature expansion (e.g., Nobles)
    ---------------------------------------------------
    """
    features = []
    
    # --- 1. Board Gems [Index 0-5] ---
    try:
        board_gems = getattr(state.board, 'gems_on_board', None)
        features.extend(get_gems_list(board_gems))
    except:
        features.extend([0]*6)

    # --- 2. Players Info [Index 6-33] ---
    try:
        me = state.active_players_hand() if callable(state.active_players_hand) else getattr(state, 'active_players_hand', None)
        others = state.other_players_hand() if callable(state.other_players_hand) else getattr(state, 'other_players_hand', None)
        opp = others[0] if isinstance(others, list) and len(others) > 0 else others
        
        for p in [me, opp]:
            if p is None:
                features.extend([0]*14)
                continue
                
            # A. Score (1 dim)
            score = 0
            if hasattr(p, 'number_of_my_points'):
                score = p.number_of_my_points() if callable(p.number_of_my_points) else p.number_of_my_points
            elif hasattr(p, 'victory_points'):
                val = p.victory_points
                score = val.value if hasattr(val, 'value') else val
            features.append(score)
            
            # B. Gems Possessed (6 dims)
            gems_obj = getattr(p, 'gems_possessed', getattr(p, 'gems', None))
            features.extend(get_gems_list(gems_obj))
            
            # C. Card Discounts (6 dims)
            discount_vals = [0]*6
            if hasattr(p, 'discount'):
                discount_vals = get_gems_list(p.discount)
            features.extend(discount_vals)
            
            # D. Reserved Cards Count (1 dim)
            res_cards = getattr(p, 'cards_reserved', getattr(p, 'reserved_cards', []))
            features.append(len(res_cards) if res_cards else 0)
            
    except Exception as e:
        print(f"[Warning] Vectorize Error: {e}")
        current_len = len(features)
        features.extend([0] * (34 - current_len))

    # Pad to a fixed 40 dimensions
    vec = np.array(features, dtype=np.float32)
    vec = np.pad(vec, (0, max(0, 40 - len(vec))), 'constant')[:40]
    return vec

# ==========================================
# Tactical Event Vector

def extract_event_vector(action, vec_prev, vec_curr):
    """
    Extracts a 9-dim Event Vector capturing the tactical intent of the action.
    
    Dimension Dictionary:
    ---------------------------------------------------
    [Base Actions]
        [0]: Is_Take_Gems -> Player collected gems from the board
        [1]: Is_Buy_Card  -> Player purchased a card
        [2]: Is_Reserve   -> Player reserved a card
        
    [Goal Actions]
        [3]: Is_Score_Up  -> Action immediately increased Victory Points
        [4]: Is_Lethal    -> Action reached the winning threshold (>=15 VP)
        
    [Defensive Tactics]
        [5]: Scarcity_Take -> Monopolized a scarce resource (took from <=2 remaining)
        [6]: Block_Reserve -> Reserved a card while opponent is close to winning (>=10 VP)
        
    [Offensive Tactics]
        [7]: Buy_Reserved  -> Played a previously reserved card (cleared hand space)
        [8]: Engine_Spike  -> Huge single-turn score jump (>=3 VP, implies Nobles or high-tier cards)
    ---------------------------------------------------
    """
    event = np.zeros(9, dtype=np.float32)
    act_str = str(action).lower()

    # --- 1. Base Actions ---
    if 'trade' in act_str or 'take' in act_str: event[0] = 1.0
    elif 'buy' in act_str or 'purchase' in act_str: event[1] = 1.0
    elif 'reserve' in act_str: event[2] = 1.0

    if vec_prev is not None and vec_curr is not None:
        try:
            my_score_prev = vec_prev[6]
            my_score_curr = vec_curr[6]
            my_res_prev = vec_prev[19]
            my_res_curr = vec_curr[19]

            # --- 2. Goal Actions ---
            if my_score_curr > my_score_prev: event[3] = 1.0
            if my_score_curr >= 15: event[4] = 1.0

            # --- 3. Defensive Tactics ---
            if event[0] == 1.0: # Scarcity Take
                board_gems_prev = vec_prev[0:6]
                board_gems_curr = vec_curr[0:6]
                diff = board_gems_prev - board_gems_curr
                if np.any((diff > 0.1) & (board_gems_prev <= 2)):
                    event[5] = 1.0

            if event[2] == 1.0: # Block Reserve
                opp_score = vec_prev[20]
                if opp_score >= 10: 
                     event[6] = 1.0

            # --- 4. Offensive Tactics ---
            if event[1] == 1.0: # Buy Reserved
                if my_res_curr < my_res_prev:
                    event[7] = 1.0

            if my_score_curr - my_score_prev >= 3: # Engine Spike
                event[8] = 1.0

        except Exception:
            pass

    return event

# ==========================================
# Main Simulation Loop

def main():
    try:
        env = SplendorEnv()
        print("[System] Environment setup successful! Initiating data generation...")
    except Exception as e:
        print(f"[Error] Environment instantiation failed: {e}")
        return

    agent = MixedPolicyAgent(env)
    dataset = []
    success_count = 0

    for i in tqdm.tqdm(range(NUM_GAMES), desc="Simulating Games"):
        env.reset()
        state = env.current_state_of_the_game
        if hasattr(env.action_space, 'update'): env.action_space.update(state)

        done = False
        game_memory = []
        current_player_idx = 0 
        steps_this_game = 0
        
        vec_curr = vectorize_state(state)
        
        while not done:
            try: valid_actions = env.action_space.list_of_actions
            except: break
            if not valid_actions: break

            action = agent.select_action(valid_actions)
            if action is None: break

            vec_prev = vec_curr

            try:
                obs, reward, is_done, info = env.step(action=action, mode='instant_end')
                done = is_done
            except: break
            
            state = env.current_state_of_the_game
            if hasattr(env.action_space, 'update'): env.action_space.update(state)
            
            vec_curr = vectorize_state(state)
            event_vec = extract_event_vector(action, vec_prev, vec_curr)
            
            game_memory.append({
                's': vec_prev, 'e': event_vec, 'a': str(action), 'p': current_player_idx
            })
            
            current_player_idx = 1 - current_player_idx
            steps_this_game += 1
            if steps_this_game > 300: break

        # Resolve winner and append to dataset
        if len(game_memory) > 0:
            winner_id = -1
            if done:
                success_count += 1
                try:
                    scores = []
                    players_list = getattr(state, 'list_of_players_hands', [])
                    for p in players_list:
                        if hasattr(p, 'number_of_my_points'):
                            scores.append(p.number_of_my_points() if callable(p.number_of_my_points) else p.number_of_my_points)
                        elif hasattr(p, 'victory_points'):
                            val = p.victory_points
                            scores.append(val.value if hasattr(val, 'value') else val)
                        else: scores.append(0)
                            
                    if len(scores) >= 2: winner_id = 0 if scores[0] > scores[1] else 1
                except Exception:
                    winner_id = game_memory[-1]['p']
            
            if winner_id != -1:
                for step in game_memory:
                    # z = 1.0 (win) or -1.0 (loss) based on whose turn it was
                    z = 1.0 if step['p'] == winner_id else -1.0
                    dataset.append((step['s'], step['e'], step['a'], z))

    print(f"\n[Stats] Valid Games Completed (Done=True): {success_count}/{NUM_GAMES}")

    # ==========================================
    # Data Saving & Preview

    if len(dataset) > 0:
        with open(OUTPUT_FILE, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"[Success] Saved {len(dataset)} transitions to {OUTPUT_FILE}")
        
        print("\n" + "="*60)
        print("👀 PREVIEW: FIRST 5 SAMPLES")
        print("="*60)
        
        event_names = ["Take", "Buy", "Reserve", "Score+", "Win", 
                       "Scarcity", "Block", "Digest", "Spike"]
        
        for idx in range(min(5, len(dataset))):
            s_vec, e_vec, a_str, z_val = dataset[idx]
            print(f"\n[Sample {idx}]")
            print(f"  > Action: {a_str}")
            print(f"  > Result (z): {z_val} {'(WIN)' if z_val>0 else '(LOSE)'}")
            
            active_events = [name for i, name in enumerate(event_names) if e_vec[i] == 1.0]
            print(f"  > Events Triggered: {active_events if active_events else 'None'}")
            print(f"  > Event Vec (Raw): {e_vec}")
            print(f"  > State Vec (First 6 - Board Gems): {s_vec[:6]}")
        print("\n" + "="*60)

if __name__ == "__main__":
    main()