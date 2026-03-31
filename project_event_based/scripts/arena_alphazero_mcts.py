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

# Event-based opponent path
RL_MODEL_PATH = os.path.join(project_root, "notebooks/models/A_hybrid_start", "ppo_event_based_A_hybrid_start_s42_20260328_155415_1000000_steps.zip")

# AlphaZero pretrained model path
AZ_WEIGHT_PATH = os.path.join(project_root, "notebooks/models/pretrained_models", "pretrained_2players.pt")


from utils.splendor_gym_wrapper_alphazero import make_splendor_env
from train_maskable import EventRewardWrapper
from sb3_contrib import MaskablePPO

import math
import numpy as np
import copy


class MCTS:
    def __init__(self, model, c_puct=1.4, n_simulations=100):
        self.model = model  
        self.c_puct = c_puct
        self.n_simulations = n_simulations

    def search(self, state, env_wrapper):
        """
        state: real-time game state from Gym (not AZ matrix, but the actual game state object)
        env_wrapper: the Gym wrapper that can step through the game and provide legal actions
        """
        # Initialization: Create root node with the current state
        # predict the policy and value for the root node
        root_masks = env_wrapper.action_masks()
        # assumption:  model can take the raw Gym state and action mask, and output action probabilities and value
        probs, _ = self.model.predict_probs_and_value(state, root_masks)
        root = MCTSNode(prior_p=1.0)
        
        # pre-populate root children with the probabilities from the model (only for legal actions)
        legal_indices = np.where(root_masks == 1)[0]
        for idx in legal_indices:
            root.children[idx] = MCTSNode(prior_p=probs[idx], parent=root)

        for _ in range(self.n_simulations):
            node = root
            # clone a separate environment for simulation (make sure to reset it to the current state)
            sim_env = copy.deepcopy(env_wrapper.env) 
            
            # Selection
            search_path = [node]
            while node.children:
                action_idx, node = node.select(self.c_puct)
                # execute the action in the simulated environment
                real_action = env_wrapper.cached_legal_actions[action_idx]
                sim_env.step('deterministic', real_action)
                search_path.append(node)

            # Expansion & Evaluation
            # check if the simulated environment is in a terminal state
            if not sim_env.is_done:
                # translate the simulated environment's current state back to the model's expected input format
                sim_obs = env_wrapper._get_observation_from_state(sim_env.current_state_of_the_game)
                sim_masks = env_wrapper._get_masks_from_state(sim_env.current_state_of_the_game)
                
                probs, value = self.model.predict_probs_and_value(sim_obs, sim_masks)
                
                # expand the node with the new probabilities (only for legal actions)
                legal_indices = np.where(sim_masks == 1)[0]
                for idx in legal_indices:
                    node.children[idx] = MCTSNode(prior_p=probs[idx], parent=node)
            else:
                # if it's a terminal state, determine the value (win/loss) for the current player
                value = 1.0 if sim_env.winner == env_wrapper.player_id else -1.0

            # Backpropagation
            for node in reversed(search_path):
                node.update(value)
                value = -value # switch perspective for the opponent

        # return the action index with the most visits from the root node
        return max(root.children.items(), key=lambda node: node[1].n_visits)[0]
    
class MCTSNode:
    def __init__(self, prior_p, parent=None):
        self.parent = parent
        self.children = {}  # key: action_idx, value: MCTSNode
        self.n_visits = 0
        self.p = prior_p
        self.w = 0
        self.q = 0

    def select(self, c_puct):
        """use PUCT to select the child with the highest score"""
        return max(self.children.items(), 
                   key=lambda node: node[1].get_value(c_puct))

    def get_value(self, c_puct):
        """PUCT (Predictor + Upper Confidence Bound applied to Trees)"""
        u = c_puct * self.p * math.sqrt(self.parent.n_visits) / (1 + self.n_visits)
        return self.q + u

    def update(self, leaf_value):
        """backpropagate to update parameters"""
        self.n_visits += 1
        self.w += leaf_value
        self.q = self.w / self.n_visits

class AlphaZeroOpponent:
    def __init__(self, weight_path):
        print("waking AlphaZero brain...")
        self.game = SplendorGame()
        
        # loading checkpoint
        args = {'nn_version': 76, 'dropout': 0.0} 
        self.nnet = SplendorNNet(self.game, args)
        
        # 1. loading weights with dynamic handling of different checkpoint formats
        import sys
        import alphazero_general.SplendorNNet
        sys.modules['splendor'] = sys.modules['alphazero_general']
        sys.modules['splendor.SplendorNNet'] = sys.modules['alphazero_general.SplendorNNet']
        checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
        # 2. dynamically determine the neural network version from the checkpoint (handle both old and new formats)
        nn_ver = checkpoint.get('nn_version', 76)
        args = {'nn_version': nn_ver, 'dropout': 0.0} 
        
        # 3. initialize the neural network with the correct version and load the weights
        self.nnet = SplendorNNet(self.game, args)
        
        # 4. open the checkpoint and load the state dict, handling both cases where 'state_dict' is a key or the checkpoint itself is the state dict
        if 'state_dict' in checkpoint:
            self.nnet.load_state_dict(checkpoint['state_dict'])
        else:
            self.nnet.load_state_dict(checkpoint)
            
        self.nnet.eval()
        self.env_ref = None 
        self.AZ_COLORS = ['White', 'Blue', 'Green', 'Red', 'Black', 'Gold']
        print("AlphaZero 就绪！")

    def get_az_representation(self, gym_state):
        """translate Gym state to AZ matrix"""
        ai_id, opp_id = 1, 0  # 保持你的逻辑
        return self._translate_state_to_56x7(gym_state, ai_id, opp_id)

    def get_az_valid_moves(self, az_matrix):
        """translate Gym legal actions to AZ valid move mask (length 80)"""
        return self.game.getValidMoves(az_matrix, player=1)
    
    def mcts_search(self, root_gym_state, n_simulations=50):

    # 1. Initialize MCTS with the root state and get the AZ representation for the root
    # translate the root Gym state to AZ matrix format for the MCTS to use
        root_matrix = self._translate_state_to_56x7(root_gym_state, 1, 0)
        
        # 2. Simulate n_simulations games from the root state, using the MCTS logic to explore the game tree
        for _ in range(n_simulations):
            # clone a separate environment for simulation (make sure to reset it to the current state)
            sim_state = copy.deepcopy(root_gym_state)
            
        # 3. search the MCTS tree to find the best action index (0-79) based on visit counts or Q values
        return best_az_idx

    def choose_action(self, observation, action_mask=[]):
        # 1. obtain the current Gym state (you might need to extract it from the observation if it's wrapped in a DeterministicObservation or similar)
        current_gym_state = self.env_ref.unwrapped.env.current_state_of_the_game
        
        # 2. translate the current Gym state to the AZ matrix representation and get the valid move mask for MCTS
        best_az_idx = self.mcts.search(current_gym_state) 
        
        # 3. use the best AZ action index to determine the corresponding Gym action object (need to maintain a mapping between AZ action indices and Gym actions, which should be consistent with how you constructed the AZ valid move mask)
        return self._translate_action_to_gym(best_az_idx, current_gym_state, 1)

    
    def _get_gem(self, gems_obj, color_name):
        """超级提取器：支持字典、数组、枚举、对象属性"""
        color_lower = color_name.lower()
        
        # 1. 如果 Gym 传的是普通字典
        if isinstance(gems_obj, dict):
            return int(gems_obj.get(color_name, gems_obj.get(color_lower, 0)))
        
        # 2. 💡 核心修复：如果 Gym 传的是数组/列表 [白, 蓝, 绿, 红, 黑, 黄金]
        if isinstance(gems_obj, (list, tuple, np.ndarray)):
            color_idx = {'white': 0, 'blue': 1, 'green': 2, 'red': 3, 'black': 4, 'gold': 5}
            idx = color_idx.get(color_lower, -1)
            if 0 <= idx < len(gems_obj):
                return int(gems_obj[idx])
            return 0
            
        # 3. 如果是带 gems_dict 的定制对象
        if hasattr(gems_obj, 'gems_dict'):
            for k, v in gems_obj.gems_dict.items():
                if color_lower in str(k).lower():
                    return int(v)
            return 0
            
        # 4. 如果是对象属性 (如 obj.red, obj.RED)
        val = getattr(gems_obj, color_name, getattr(gems_obj, color_lower, getattr(gems_obj, color_name.upper(), 0)))
        return int(val)

    def _get_bonuses(self, player):
        """超级折扣提取器：同时兼容卡牌列表和直接的折扣数组"""
        az_colors = ['white', 'blue', 'green', 'red', 'black']
        bonuses = {c: 0 for c in az_colors}
        
        # 1. 如果环境直接提供了折扣数组 (如 player.bonuses = [1, 0, 2, 0, 0])
        direct_bonuses = getattr(player, 'bonuses', getattr(player, 'discounts', None))
        if isinstance(direct_bonuses, (list, tuple, np.ndarray)):
            for i, c in enumerate(az_colors):
                if i < len(direct_bonuses):
                    bonuses[c] = int(direct_bonuses[i])
            return bonuses
            
        # 2. 如果环境只有卡牌列表，手动遍历卡牌计算折扣
        cards = getattr(player, 'cards_possessed', getattr(player, 'cards_won', []))
        for card in cards:
            c = str(getattr(card, 'gem_color', getattr(card, 'color', ''))).lower()
            for key in bonuses.keys():
                if key in c:
                    bonuses[key] += 1
        return bonuses
    
    def _get_aligned_cards(self, gym_state):
        """统一发牌器：确保矩阵和动作翻译器看到绝对一致的卡牌顺序"""
        cards_by_tier = [[], [], []]
        for c in gym_state.board.cards_on_board:
            t = getattr(c, 'tier', None)
            if t is None:
                t = getattr(c, 'level', 1) - 1 # 1,2,3 映射到 0,1,2
            t = max(0, min(2, int(t))) # 安全兜底
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

        # Row 0: 银行
        matrix[0, :6] = [self._get_gem(bank, c) for c in self.AZ_COLORS]

        # Row 1-24: 场上卡牌 (使用统一发牌器，绝对同步！)
        cards_by_tier = self._get_aligned_cards(gym_state)
        row = 1
        for t in range(3):
            for card in cards_by_tier[t]:
                if card is not None:
                    # 消耗
                    price = getattr(card, 'price', getattr(card, 'cost', {}))
                    matrix[row, :5] = [self._get_gem(price, c) for c in self.AZ_COLORS[:5]]
                    
                    # 产出颜色 (精准使用 discount_profit)
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

        # Row 31-33: 贵族
        nobles = list(gym_state.board.nobles_on_board)
        nobles.sort(key=lambda n: getattr(n, 'id', str(n)))
        row = 31
        for noble in nobles[:3]:
            req = getattr(noble, 'price', getattr(noble, 'requirements', getattr(noble, 'cost', {})))
            matrix[row, :5] = [self._get_gem(req, c) for c in self.AZ_COLORS[:5]]
            matrix[row, 6] = getattr(noble, 'points', getattr(noble, 'victory_points', 3))
            row += 1

        # Row 34-55: 玩家区域
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
        # Gym 喜欢深拷贝对象，用 ID 或名字比对最安全
        id1 = str(getattr(c1, 'id', getattr(c1, 'name', str(c1))))
        id2 = str(getattr(c2, 'id', getattr(c2, 'name', str(c2))))
        return id1 == id2

    def _translate_action_to_gym(self, az_idx, gym_state, ai_id):
        valid_gym_actions = self.env_ref.unwrapped.env.action_space.list_of_actions
        az_colors_lower = [c.lower() for c in self.AZ_COLORS]
        cards_by_tier = self._get_aligned_cards(gym_state)

        # 1 & 2: 买牌 (0-11) & 预留 (12-23)
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
                        # 💡 核心修复：使用强化版的卡牌比对！
                        if self._is_same_card(act_card, target_card):
                            return act

        # 3: 拿取宝石 (30 - 79)
# ==========================================
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
                    # 序列化所有内部数据为字符串
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

        # ==========================================
        # 🚨 终极安全兜底：绝不随机拿宝石导致崩溃！
        # ==========================================
        if az_idx == 80:
            pass_acts = [a for a in valid_gym_actions if 'Pass' in a.__class__.__name__]
            if pass_acts: return pass_acts[0]
        else:
            print(f"⚠️ 动作 {az_idx} 映射失败，触发【安全兜底】。")
            
        # 哪怕映射失败，宁可买便宜牌或者干等，也绝不瞎拿宝石触发退还 Bug！
        buy_acts = [a for a in valid_gym_actions if 'Buy' in a.__class__.__name__]
        if buy_acts: return buy_acts[0]
        
        pass_acts = [a for a in valid_gym_actions if 'Pass' in a.__class__.__name__]
        if pass_acts: return pass_acts[0]

        # ==========================================
        # 🚨 终极防御：绝不信任 valid_gym_actions[0]
        # ==========================================
        if not valid_gym_actions:
            print(f"\n💥 [紧急] 环境在动作 {az_idx} 处未提供任何合法动作！")
            # 💡 物理破局：很多 Gym 实现里如果不走 ActionSpace，直接返回 None 可能会跳过回合
            # 或者抛出一个自定义的“无操作”，这里我们先返回 None 并在外层捕获
            return None 

        # 如果前面的精确匹配全失败了，且确实有合法动作，才敢用索引 0
        print(f"⚠️ 动作 {az_idx} 映射完全失败，强制执行环境首个可用动作。")
        # ==========================================
        # 🚨 终极安全兜底 (打破死循环版)
        # ==========================================
        if not valid_gym_actions:
            print(f"\n💥 [紧急] 环境在动作 {az_idx} 处未提供任何合法动作！")
            return None 

        # 💡 不要再用 valid_gym_actions[0] 了！
        # 引入随机性，强行把 AZ 踹出死循环泥潭
        import random
        # print(f"⚠️ 动作 {az_idx} 映射完全失败，随机执行一个合法动作保命。")
        return random.choice(valid_gym_actions)
# ==========================================
# 3. 竞技场主循环
# ==========================================
def run_alphazero_arena(n_games=10):
    print(" 初始化 Splendor 终极竞技场 ")
    
    # 1. 实例化与加载 (保持原样)
    az_opponent = AlphaZeroOpponent(AZ_WEIGHT_PATH)
    base_env = make_splendor_env(reward_mode='score_progress', opponent_agent=az_opponent)
    config = {'reward': {'event_weights': np.zeros(9)}, 'environment': {'combine_event_and_score': True}}
    env = EventRewardWrapper(base_env, config=config)
    az_opponent.env_ref = env
    
    print(f"正在唤醒你的 RL 模型: {RL_MODEL_PATH}")
    my_model = MaskablePPO.load(RL_MODEL_PATH, device='cpu')
    
    # 📊 重新定义更细致的计数器
    total_wins = 0
    total_losses = 0
    total_draws = 0
    total_deadlocks = 0  # 专门记录死锁局
    
    print(f"\n开始对战！(共 {n_games} 局)")
    
    for i in range(n_games):
        obs, _ = env.reset()
        done = False
        steps = 0
        info = {} # 确保 info 在每局开始时是空的
        
        while not done:
            masks = env.action_masks()
            
            # ==========================================
            # 🛡️ 异常拦截：死锁判定
            # ==========================================
            if not any(masks):
                print("\n💥 警告：环境全零死锁！")
                info['is_deadlock'] = True
                break
                
            try:
                action, _ = my_model.predict(obs, action_masks=masks, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
            except Exception as e:
                # 拦截 PyTorch 掩码崩溃或未知错误
                info['is_deadlock'] = True
                print(f"\n💥 捕获异常拦截: {type(e).__name__}")
                break
        
        # ==========================================
        # ⚖️ 结算逻辑：死锁局、赢、输、平
        # ==========================================
        # 获取最终分数用于判定
        final_state = env.unwrapped.env.current_state_of_the_game
        my_score = final_state.list_of_players_hands[0].number_of_my_points() 
        az_score = final_state.list_of_players_hands[1].number_of_my_points()
        
# ==========================================
        # ⚖️ 结算逻辑：严丝合缝版
        # ==========================================
        is_deadlock = info.get('is_deadlock', False)
        
        # 1. 优先判定死锁 (作废局)
        if is_deadlock:
            result_str = "平局 (环境死锁/作废)"
            total_deadlocks += 1
            # 🚨 注意：这里不需要执行 wins += 1，也不要执行 losses += 1
            
        # 2. 如果不是死锁，再看是不是赢了
        elif info.get('agent_won', False) or my_score > az_score:
            result_str = "赢"
            total_wins += 1
            
        # 3. 如果不是赢，再看是不是输了
        elif info.get('agent_lost', False) or my_score < az_score:
            result_str = "输"
            total_losses += 1
            
        # 4. 最后才是真正的平局 (分够了但分数相同)
        else:
            result_str = "平局 (分数相同)"
            total_draws += 1
        
        print(f"Game {i+1}/{n_games} 结束 | 胜负: {result_str} | 比分 (你 vs AZ): {my_score} - {az_score} | 步数: {steps}")

    # ==========================================
    # 🏆 最终战报统计 (剔除死锁局)
    # ==========================================
    valid_games = n_games - total_deadlocks
    print("\n" + "="*45)
    print("       🏁 竞技场最终战报 (纯净版) 🏁")
    print("="*45)
    print(f"总对战局数: {n_games}")
    print(f"因环境 Bug 作废: {total_deadlocks} 局")
    print(f"有效对战局数: {valid_games}")
    print("-" * 45)
    
    if valid_games > 0:
        win_rate = (total_wins / valid_games) * 100
        print(f"  ✅ 胜: {total_wins}")
        print(f"  ❌ 负: {total_losses}")
        print(f"  🤝 平: {total_draws}")
        print(f"  🔥 RL 真实有效胜率: {win_rate:.2f}%")
    else:
        print("⚠️ 警告：没有产生任何有效的完整对局！")
    print("="*45)

if __name__ == "__main__":
    # 为了测试有没有 Bug，我们先跑 5 局看看
    run_alphazero_arena(n_games=20)