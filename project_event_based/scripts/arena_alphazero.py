import os
import sys
import numpy as np
import torch
from alphazero_general.SplendorNNet import SplendorNNet
from alphazero_general.SplendorGame import SplendorGame
from alphazero_general.SplendorLogic import list_different_gems_up_to_3

script_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.dirname(script_dir)

repo_root = os.path.dirname(project_root)

sys.path.insert(0, os.path.join(repo_root, 'modules'))

sys.path.insert(0, os.path.join(project_root, 'src'))

RL_MODEL_PATH = os.path.join(project_root, "notebooks/models/A_hybrid_start", "ppo_event_based_A_hybrid_start_s42_20260328_155415_1000000_steps.zip")

# AlphaZero path
AZ_WEIGHT_PATH = os.path.join(project_root, "notebooks/models/pretrained_models", "pretrained_2players.pt")


from utils.splendor_gym_wrapper_alphazero import make_splendor_env
from train_maskable import EventRewardWrapper
from sb3_contrib import MaskablePPO


class AlphaZeroOpponent:
    def __init__(self, weight_path):
        print("waking AlphaZero brain...")
        self.game = SplendorGame()
        
        # loading pretrained weights
        args = {'nn_version': 76, 'dropout': 0.0} 
        self.nnet = SplendorNNet(self.game, args)
        
        # loading checkpoint with proper handling of different PyTorch versions and potential extra metadata
        import sys
        sys.modules['splendor'] = sys.modules['alphazero_general']
        sys.modules['splendor.SplendorNNet'] = sys.modules['alphazero_general.SplendorNNet']
        checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
        # 2. dynamically determine the neural network version from the checkpoint (some checkpoints might have extra metadata, some might be raw state_dict)
        nn_ver = checkpoint.get('nn_version', 76)
        args = {'nn_version': nn_ver, 'dropout': 0.0} 
        
        # 3. instantiate the neural network with the correct version and load the weights
        self.nnet = SplendorNNet(self.game, args)
        
        # 4. open the checkpoint and load the state dict, with compatibility handling for different checkpoint formats
        if 'state_dict' in checkpoint:
            self.nnet.load_state_dict(checkpoint['state_dict'])
        else:
            self.nnet.load_state_dict(checkpoint)
            
        self.nnet.eval()
        self.env_ref = None 
        self.AZ_COLORS = ['White', 'Blue', 'Green', 'Red', 'Black', 'Gold']
        print("AlphaZero ready")

    def choose_action(self, observation, action_mask=[]):
        # 1. obtain the current Gym state (you might need to extract it from the observation if it's wrapped in a DeterministicObservation or similar)
        gym_state = self.env_ref.unwrapped.env.current_state_of_the_game
        # RL: P0，AlphaZero: P1
        ai_id = 1 
        opp_id = 0 
        
        # 2. translate the state to a 56x7 matrix
        az_matrix = self._translate_state_to_56x7(gym_state, opp_id, ai_id)
        #print(f"AZ (gem): {az_matrix[34][:6]}")
        #print(f"AZ (card): {az_matrix[35][:5]}")
        
        # 3. ask AlphaZero for legal actions
        # translate Gym legal actions to AZ valid move mask (length 80)
        az_matrix_numba = az_matrix.astype(np.int8)

        az_matrix = self._translate_state_to_56x7(gym_state, ai_id, opp_id)
        
        # try to get the valid moves directly from the AZ logic, which should be perfectly aligned with how we constructed the AZ representation
        valid_moves = self.game.getValidMoves(az_matrix, player=1)

        #if sum(az_matrix[34][:5]) >= 5: 
            #print("\n [judgement] card in Az (first):")
            #for i in range(1, 6, 2):
                #print(f"   card {i//2}: need {az_matrix[i][:5]}, ouput {az_matrix[i+1][:5]} (point {az_matrix[i+1][6]})")

        # 1. extract player and bank info from the Gym state to apply sanity checks and create the valid move mask for AZ
        p_hands = gym_state.list_of_players_hands
        me = p_hands[ai_id]
        bank = gym_state.board.gems_on_board
        
        # 2. reserved card limit protection 
        reserved_cards = getattr(me, 'cards_reserved', getattr(me, 'reserved_cards', []))
        if len(reserved_cards) >= 3:
            valid_moves[12:24] = 0  # already have 3 reserved cards, can't reserve more 
            
        # 3. gem limit protection 
        az_total_colored = sum(az_matrix[34][:5])
        az_gold = az_matrix[34][5]
        total_all_gems = az_total_colored + az_gold
        
        if total_all_gems >= 9:
            valid_moves[30:80] = 0  # already have 9 gems, can't take more
        if total_all_gems == 10:
            valid_moves[12:24] = 0  # already have 10 gems, can't reserve (because reserving would lead to 11)

        # 4. bank inventory protection (prevent AZ from taking empty promises)
        az_colors_lower = [c.lower() for c in self.AZ_COLORS]
        bank_counts = [self._get_gem(bank, c) for c in az_colors_lower[:5]]
        
        # filter out any gem-taking moves that would take more gems than the bank has (this is a sanity check, ideally the AZ logic should already handle this, but we add it here just in case)

        # if the AZ action is to take 2 gems of the same color, check if the bank has at least 4 of that color (because taking 2 requires at least 4 in the bank)
        if np.sum(valid_moves) == 0:
            valid_moves[:] = 0
            valid_moves[80] = 1
        
        # 1. if already have 9 gems, absolutely forbid taking more (mask out 30-79, which are the gem-taking actions)
        if total_all_gems >= 9:
            valid_moves[30:80] = 0
            
        # 2. if already have 10 gems, absolutely forbid reserving cards (mask out 12-23, because reserving would lead to 11)
        if total_all_gems == 10:
            valid_moves[12:24] = 0
            
        # 3. fallback: if all moves are blocked, force Pass (80)
        if np.sum(valid_moves) == 0:
            valid_moves[80] = 1
        
        # 4. use the AZ logic to get the valid move mask, which should be perfectly aligned with how we constructed the AZ representation (this is a sanity check to ensure our translation is correct)
        az_matrix_tensor = torch.FloatTensor(az_matrix.astype(np.float64))
        valid_moves_tensor = torch.BoolTensor(valid_moves)
        
        with torch.no_grad():
            pi, _ = self.nnet(az_matrix_tensor, valid_moves_tensor)
            
        # 5. apply the valid move mask to the AZ output probabilities, and select the action with the highest probability among the valid moves
        pi = torch.exp(pi) * valid_moves_tensor 
        if pi.sum() > 0:
            pi /= pi.sum()
        else:
            # extreme fallback: if AZ gives zero probability to all valid moves (which shouldn't happen, but just in case), then we can either choose randomly among valid moves or default to Pass
            pi = valid_moves_tensor / valid_moves_tensor.sum()
            
        best_az_action_idx = int(torch.argmax(pi).item())
        
        # 6. use the best AZ action index to determine the corresponding Gym action object (need to maintain a mapping between AZ action indices and Gym actions, which should be consistent with how you constructed the AZ valid move mask)
        if torch.is_tensor(pi):
            pi_np = pi.detach().cpu().numpy()
        else:
                pi_np = pi

        #print(f"DEBUG: pro_mix: {np.max(pi_np)}, min: {np.min(pi_np)}")
        #print(f"🤖 AZ is confident and choose: {best_az_action_idx}")
    
    # try to decode the AZ action index into a human-readable form for debugging
        if best_az_action_idx < 12:
            print(f"   target: buy {best_az_action_idx//4} level, card {best_az_action_idx%4}")
        elif 12 <= best_az_action_idx < 24:
            print(f"   target: reserve card at level {(best_az_action_idx-12)//4}, position {(best_az_action_idx-12)%4}")
        elif best_az_action_idx >= 30:
            print(f"   target: take gems logic branch")
        return self._translate_action_to_gym(best_az_action_idx, gym_state, ai_id)
    
    def _get_gem(self, gems_obj, color_name):
        """extract the number of gems of a specific color from various possible representations (dict, list, custom object)"""
        color_lower = color_name.lower()
        
        # 1. if it's a dict (like {'white': 2, 'blue': 1}), try to get the value using both the original color name and the lowercase version for robustness
        if isinstance(gems_obj, dict):
            return int(gems_obj.get(color_name, gems_obj.get(color_lower, 0)))
        
        # 2. Core Fix: if Gym passes an array/list [white, blue, green, red, black, gold]
        if isinstance(gems_obj, (list, tuple, np.ndarray)):
            color_idx = {'white': 0, 'blue': 1, 'green': 2, 'red': 3, 'black': 4, 'gold': 5}
            idx = color_idx.get(color_lower, -1)
            if 0 <= idx < len(gems_obj):
                return int(gems_obj[idx])
            return 0
            
        # 3. if it's a custom object with a gems_dict attribute
        if hasattr(gems_obj, 'gems_dict'):
            for k, v in gems_obj.gems_dict.items():
                if color_lower in str(k).lower():
                    return int(v)
            return 0
            
        # 4. if it's an object attribute (like obj.red, obj.RED)
        val = getattr(gems_obj, color_name, getattr(gems_obj, color_lower, getattr(gems_obj, color_name.upper(), 0)))
        return int(val)

    def _get_bonuses(self, player):
        """extract bonuses from a player object"""
        az_colors = ['white', 'blue', 'green', 'red', 'black']
        bonuses = {c: 0 for c in az_colors}
        
        # 1. if the environment directly provides a bonuses array (like player.bonuses = [1, 0, 2, 0, 0])
        direct_bonuses = getattr(player, 'bonuses', getattr(player, 'discounts', None))
        if isinstance(direct_bonuses, (list, tuple, np.ndarray)):
            for i, c in enumerate(az_colors):
                if i < len(direct_bonuses):
                    bonuses[c] = int(direct_bonuses[i])
            return bonuses
            
        # 2. if the environment only provides a list of cards, manually iterate through the cards to calculate discounts
        cards = getattr(player, 'cards_possessed', getattr(player, 'cards_won', []))
        for card in cards:
            c = str(getattr(card, 'gem_color', getattr(card, 'color', ''))).lower()
            for key in bonuses.keys():
                if key in c:
                    bonuses[key] += 1
        return bonuses
    
    def _get_aligned_cards(self, gym_state):
        """unify the card representation across different Gym versions by categorizing them into tiers and sorting by ID, ensuring a consistent order for the AZ representation"""
        cards_by_tier = [[], [], []]
        for c in gym_state.board.cards_on_board:
            t = getattr(c, 'tier', None)
            if t is None:
                t = getattr(c, 'level', 1) - 1 # 1,2,3 mapping 0,1,2
            t = max(0, min(2, int(t))) 
            cards_by_tier[t].append(c)
            
        for t in range(3):
            cards_by_tier[t].sort(key=lambda c: getattr(c, 'id', str(c)))
            while len(cards_by_tier[t]) < 4:
                cards_by_tier[t].append(None)
        return cards_by_tier

    def _translate_state_to_56x7(self, gym_state, ai_id, opp_id):
        matrix = np.zeros((56, 7), dtype=np.int8)
        p_hands = gym_state.list_of_players_hands
        me = p_hands[ai_id]
        it = p_hands[opp_id]
        bank = gym_state.board.gems_on_board
        az_colors_lower = [c.lower() for c in self.AZ_COLORS]

        # Row 0: bank
        matrix[0, :6] = [self._get_gem(bank, c) for c in self.AZ_COLORS]

        # Row 1-24: cards on the board (using the unified card initializer, absolutely synchronized!)
        cards_by_tier = self._get_aligned_cards(gym_state)
        row = 1
        for t in range(3):
            for card in cards_by_tier[t]:
                if card is not None:
                    # consume price/cost info in various possible formats (dict, list, custom object) and extract the number of gems of each color required for this card  
                    price = getattr(card, 'price', getattr(card, 'cost', {}))
                    matrix[row, :5] = [self._get_gem(price, c) for c in self.AZ_COLORS[:5]]
                    
                    # color/discount info can also be in various formats, we try to extract it robustly and then determine which color it corresponds to for the AZ representation
                    card_color_val = getattr(card, 'discount_profit', getattr(card, 'color', getattr(card, 'gem_color', None)))
                    c_idx = 0
                    if card_color_val is not None:
                        raw_str = str(card_color_val).lower()
                        for i, c_name in enumerate(az_colors_lower[:5]):
                            if c_name in raw_str:
                                c_idx = i
                                break
                    matrix[row+1, c_idx] = 1
                    matrix[row+1, 6] = getattr(card, 'points', getattr(card, 'victory_points', 0))
                row += 2

        # Row 31-33: noble
        nobles = list(gym_state.board.nobles_on_board)
        nobles.sort(key=lambda n: getattr(n, 'id', str(n)))
        row = 31
        for noble in nobles[:3]:
            req = getattr(noble, 'price', getattr(noble, 'requirements', getattr(noble, 'cost', {})))
            matrix[row, :5] = [self._get_gem(req, c) for c in self.AZ_COLORS[:5]]
            matrix[row, 6] = getattr(noble, 'points', getattr(noble, 'victory_points', 3))
            row += 1

        # Row 34-55: player states (first me, then opponent, same format)
        def fill_player_state(p, start_row):
            matrix[start_row, :5] = [self._get_gem(p.gems_possessed, c) for c in self.AZ_COLORS[:5]]
            matrix[start_row, 5] = self._get_gem(p.gems_possessed, 'gold')
            matrix[start_row, 6] = p.number_of_my_points()

            bonuses = self._get_bonuses(p)
            matrix[start_row+1, :5] = [bonuses.get(c, 0) for c in az_colors_lower[:5]]

            reserved_raw = getattr(p, 'cards_reserved', getattr(p, 'reserved_cards', []))
            reserved = list(reserved_raw)
            reserved.sort(key=lambda c: getattr(c, 'id', str(c)))
            r_row = start_row + 2
            for card in reserved[:3]:
                price = getattr(card, 'price', getattr(card, 'cost', {}))
                matrix[r_row, :5] = [self._get_gem(price, c) for c in self.AZ_COLORS[:5]]
                
                card_color_val = getattr(card, 'discount_profit', getattr(card, 'color', getattr(card, 'gem_color', None)))
                c_idx = 0
                if card_color_val is not None:
                    raw_str = str(card_color_val).lower()
                    for i, c_name in enumerate(az_colors_lower[:5]):
                        if c_name in raw_str:
                            c_idx = i
                            break
                matrix[r_row+1, c_idx] = 1
                matrix[r_row+1, 6] = getattr(card, 'points', getattr(card, 'victory_points', 0))
                r_row += 2

        fill_player_state(me, 34)
        fill_player_state(it, 45)
        return matrix
    
    def _is_same_card(self, c1, c2):
        if c1 is None or c2 is None: return False
        if c1 == c2: return True
        # use a more robust way to compare cards by their unique identifiers (like id or name), which should be consistent across different Gym versions, instead of relying on direct object comparison which might fail due to different instances or representations
        id1 = str(getattr(c1, 'id', getattr(c1, 'name', str(c1))))
        id2 = str(getattr(c2, 'id', getattr(c2, 'name', str(c2))))
        return id1 == id2

    def _translate_action_to_gym(self, az_idx, gym_state, ai_id):
        valid_gym_actions = self.env_ref.unwrapped.env.action_space.list_of_actions
        az_colors_lower = [c.lower() for c in self.AZ_COLORS]
        cards_by_tier = self._get_aligned_cards(gym_state)

        # 1 & 2: buy card (0-11) & reserve (12-23)
        if 0 <= az_idx < 24:
            is_buy = (az_idx < 12)
            offset = 0 if is_buy else 12
            t_tier = (az_idx - offset) // 4
            t_idx = (az_idx - offset) % 4
            
            target_card = cards_by_tier[t_tier][t_idx]
            if target_card is not None:
                for act in valid_gym_actions:
                    act_name = act.__class__.__name__
                    if ('Buy' in act_name if is_buy else 'Reserve' in act_name):
                        act_card = getattr(act, 'card', getattr(act, 'target_card', None))
                        # use the robust card comparison function to check if this action corresponds to the target card that AZ wants to buy or reserve
                        if self._is_same_card(act_card, target_card):
                            return act

        # 3: take gems (30 - 79)
        elif az_idx >= 30 and az_idx < 80:
            import itertools
            colors = [0, 1, 2, 3, 4]
            list_diff = []
            for k in [1, 2, 3]:
                list_diff.extend(list(itertools.combinations(colors, k)))
            
            i = az_idx - 30
            target_gems = {c: 0 for c in az_colors_lower[:5]}
            
            if i < len(list_diff):
                for c_idx in list_diff[i]:
                    target_gems[az_colors_lower[c_idx]] += 1
            else:
                color_idx = i - len(list_diff)
                if color_idx < 5:
                    target_gems[az_colors_lower[color_idx]] += 2

            best_act = None
            best_score = -9999
            
            for act in valid_gym_actions:
                if 'Trade' in act.__class__.__name__ or 'Gem' in act.__class__.__name__ or 'Take' in act.__class__.__name__:
                    # use a more robust way to determine which colors this action is trying to take by checking its attributes and converting them to a string for matching, instead of relying on the action's class name which might not be consistent across different Gym versions or custom implementations
                    act_str = ""
                    if hasattr(act, 'to_dict'): act_str += str(act.to_dict()).lower()
                    act_str += " " + str(act).lower() + " " + str(vars(act)).lower()
                    
                    score = 0
                    for c in az_colors_lower[:5]:
                        wanted = target_gems[c]
                        has_color = c in act_str
                        if wanted > 0:
                            if has_color: score += 10
                            else: score -= 50
                        else:
                            if has_color: score -= 1
                                
                    if score > best_score:
                        best_score = score
                        best_act = act
            
            if best_act is not None and best_score > 0:
                return best_act

        if az_idx == 80:
            pass_acts = [a for a in valid_gym_actions if 'Pass' in a.__class__.__name__]
            if pass_acts: return pass_acts[0]
        #else:
            #print(f"⚠️ action {az_idx} mapping failed, triggering 【safe fallback】。")
            
        # failback 1: if AZ wanted to buy/reserve but we couldn't find the exact card, try to find any buy/reserve action as a fallback (this is a bit risky but better than doing something completely unrelated, and it keeps the AZ logic somewhat alive instead of just defaulting to Pass)
        buy_acts = [a for a in valid_gym_actions if 'Buy' in a.__class__.__name__]
        if buy_acts: return buy_acts[0]
        
        pass_acts = [a for a in valid_gym_actions if 'Pass' in a.__class__.__name__]
        if pass_acts: return pass_acts[0]

        if not valid_gym_actions:
            #print(f"\n💥 [urgent] action {az_idx} is invalid!")
            return None 

        
        #print(f"⚠️ action {az_idx} mapping failed, triggering 【safe fallback】。")
        if not valid_gym_actions:
            #print(f"\n💥 [urgent] action {az_idx} is invalid!")
            return None 

        import random
        # print(f"⚠️ action {az_idx} mapping failed, randomly executing a valid action to survive.")
        return random.choice(valid_gym_actions)

def run_alphazero_arena(n_games=10):
    print(" Initiate Splendor battle ")
    
    # 1. Instantiate the AlphaZero opponent and the Gym environment, making sure to pass a reference
    az_opponent = AlphaZeroOpponent(AZ_WEIGHT_PATH)
    base_env = make_splendor_env(reward_mode='score_progress', opponent_agent=az_opponent)
    config = {'reward': {'event_weights': np.zeros(9)}, 'environment': {'combine_event_and_score': True}}
    env = EventRewardWrapper(base_env, config=config)
    az_opponent.env_ref = env
    
    print(f"waking RL model: {RL_MODEL_PATH}")
    my_model = MaskablePPO.load(RL_MODEL_PATH, device='cpu')
    
    total_wins = 0
    total_losses = 0
    total_draws = 0
    total_deadlocks = 0  
    
    print(f"\Start( {n_games} games)")
    
    for i in range(n_games):
        obs, _ = env.reset()
        done = False
        steps = 0
        info = {} 
        
        while not done:
            masks = env.action_masks()

            if not any(masks):
                print("\n warning:all deadlock")
                info['is_deadlock'] = True
                break
                
            try:
                action, _ = my_model.predict(obs, action_masks=masks, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
            except Exception as e:
                # capture any unexpected errors during action prediction or environment stepping, log the error type, and mark this game as a deadlock to be excluded from final statistics, ensuring that the arena can continue running smoothly without crashing due to unforeseen issues
                info['is_deadlock'] = True
                print(f"\n  capturing unusual error: {type(e).__name__}")
                break
        

        final_state = env.unwrapped.env.current_state_of_the_game
        my_score = final_state.list_of_players_hands[0].number_of_my_points() 
        az_score = final_state.list_of_players_hands[1].number_of_my_points()
        
        is_deadlock = info.get('is_deadlock', False)
        
        # 1. first check if this game ended in a deadlock due to environment issues or unexpected errors, and if so, mark it as a special type of draw that doesn't count towards wins/losses, ensuring that we maintain the integrity of our statistics by excluding these anomalous games from affecting the perceived performance of the RL agent against AlphaZero. This way, we can provide a "pure" win/draw/loss breakdown that reflects only the valid, complete games where both agents had a fair chance to compete, while still acknowledging the presence of any environmental issues that may have caused some games to be invalid without penalizing the RL agent's record for circumstances beyond its control.
        if is_deadlock:
            result_str = "deadlock"
            total_deadlocks += 1
            
        # 2. if not a deadlock, then we can determine the result based on the scores and any explicit win/loss flags in the info (some environments might provide these flags, but we also use score comparison as a fallback to ensure we can determine the result even if those flags are missing or unreliable)
        elif info.get('agent_won', False) or my_score > az_score:
            result_str = "win"
            total_wins += 1
            
        # 3. if not a deadlock and not a win, then check if it's a loss based on explicit flags or score comparison, which should be mutually exclusive with the win condition, ensuring that we correctly categorize the outcome of each game without overlap or ambiguity, and that every valid game is classified as either a win, loss, or draw based on the available information and score comparison
        elif info.get('agent_lost', False) or my_score < az_score:
            result_str = "loss"
            total_losses += 1
            
        # 4. if it's not a deadlock, win, or loss based on the above checks, then we can classify it as a draw (either because the scores are the same or because the environment explicitly marked it as a draw), which should be the default fallback category for any game that doesn't meet the criteria for win or loss, ensuring that we account for all possible outcomes and provide a complete breakdown of wins, losses, draws, and deadlocks in our final statistics
        else:
            result_str = "not win/loss"
            total_draws += 1
        
        print(f"Game {i+1}/{n_games} result : {result_str} | game point (event vs AZ): {my_score} - {az_score} | step: {steps}")

    valid_games = n_games - total_deadlocks
    print("\n" + "="*45)
    print(" final result")
    print("="*45)
    print(f"total games: {n_games}")
    print(f"deadlocks: {total_deadlocks} 局")
    print(f"valid games: {valid_games}")
    print("-" * 45)
    
    if valid_games > 0:
        win_rate = (total_wins / valid_games) * 100
        print(f"  win: {total_wins}")
        print(f"  ❌ loss: {total_losses}")
        print(f"  🤝 draw: {total_draws}")
        print(f"  🔥 RL real win rate: {win_rate:.2f}%")
    else:
        print("No valid games were completed, unable to calculate win rate.")
    print("="*45)

if __name__ == "__main__":
    run_alphazero_arena(n_games=50)