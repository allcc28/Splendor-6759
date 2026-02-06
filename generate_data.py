import sys
import os
import random
import pickle
import tqdm
import numpy as np


def mount_package_root():
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
    print("❌ cannot find package...")

try:
    from gym_splendor_code.envs.splendor import SplendorEnv
except ImportError:
    pass


NUM_GAMES = 500
OUTPUT_FILE = "expert_data_mixed_policy.pkl"
GREEDY_PROB = 0.8 

class MixedPolicyAgent:
    def __init__(self, env):
        self.env = env

    def select_action(self, valid_actions):
        if not valid_actions: return None
        if random.random() > GREEDY_PROB:
            return random.choice(valid_actions)
        
        buy_actions = [a for a in valid_actions if 'buy' in str(a).lower()]
        take_actions = [a for a in valid_actions if 'take' in str(a).lower()]
        
        if buy_actions: return random.choice(buy_actions)
        elif take_actions: return random.choice(take_actions)
        return random.choice(valid_actions)

def vectorize_state(state):
    
    return np.zeros(200, dtype=np.float32)

def extract_event_vector(prev_state, curr_state, opponent_idx):
   
    return np.zeros(3, dtype=np.float32)



def main():
    try:
        env = SplendorEnv()
        print("🚀 Environment set up sucess！Start generating data...")
    except Exception as e:
        print(f"❌ Environment instanciez fail: {e}")
        return

    agent = MixedPolicyAgent(env)
    dataset = []
    
    print("processing...")
    
    success_count = 0
    total_steps_all_games = 0

    for i in range(NUM_GAMES):
        # 1. Reset
        env.reset()
        state = env.current_state_of_the_game
        
        # 🚑 fix:force reset
        if hasattr(env.action_space, 'update'):
            env.action_space.update(state)

        prev_state = None
        done = False
        game_memory = []
        current_player_idx = 0 
        
        # 记录这一局跑了多少步
        steps_this_game = 0
        
        while not done:
            # 2. 获取动作
            try:
                valid_actions = env.action_space.list_of_actions
            except:
                valid_actions = []

            # Debug: 如果第一步就没动作，打印出来
            if not valid_actions:
                if steps_this_game == 0 and i < 3: # 只报前3局的错
                     print(f"⚠️ Game {i}: 刚开局就没有合法动作！可能需要 update。")
                break

            action = agent.select_action(valid_actions)
            if action is None: break

            # 记录数据
            state_vec = vectorize_state(state)
            opponent_idx = 1 - current_player_idx
            event_vec = extract_event_vector(prev_state, state, opponent_idx)
            
            # 3. Step
            try:
                obs, reward, is_done, info = env.step(action=action, mode='instant_end')
                done = is_done
            except Exception as e:
                print(f"❌ Step 报错: {e}")
                break
            
            game_memory.append({
                's': state_vec, 'e': event_vec, 'a': action, 'p': current_player_idx
            })

            # 更新状态
            prev_state = state
            state = env.current_state_of_the_game

            # 🚑 关键修复 2: 每一步走完，再次强制刷新动作空间
            if hasattr(env.action_space, 'update'):
                env.action_space.update(state)
            
            # players ID
            try:
                if hasattr(state, 'active_player_id'):
                    current_player_idx = state.active_player_id
                elif hasattr(env, 'active_player_id'):
                     current_player_idx = env.active_player_id() if callable(env.active_player_id) else env.active_player_id
                else:
                    current_player_idx = 1 - current_player_idx
            except:
                 current_player_idx = 1 - current_player_idx
            
            steps_this_game += 1
            if steps_this_game > 500: # 防止死循环
                break

        # 累计步数
        total_steps_all_games += steps_this_game

        # 结算
        if len(game_memory) > 0:
            # 注意：即使 done 为 False (比如死局或者步数耗尽)，我们也尽量保存数据
            # 除非明确要求只存赢家数据。这里我们放宽条件，只要跑了就有分。
            
            winner_id = -1
            if done:
                success_count += 1
                try:
                    scores = []
                    for p in env.current_state_of_the_game.players:
                        val = p.victory_points.value if hasattr(p.victory_points, 'value') else p.victory_points
                        scores.append(val)
                    if len(scores) == 2:
                        winner_id = 0 if scores[0] > scores[1] else 1
                except:
                    winner_id = game_memory[-1]['p'] # 默认最后一步的人赢
            
            # 只有分出胜负才存，还是都存？
            # 如果没分出胜负，z 可以设为 0 或者根据当前分数差设定
            if winner_id != -1:
                for step in game_memory:
                    z = 1.0 if step['p'] == winner_id else -1.0
                    dataset.append((step['s'], step['e'], step['a'], z))
        
        # 每10局打印一次状态，确保不是在空跑
        if i % 10 == 0:
            print(f"Game {i}: Steps={steps_this_game}, Done={done}, DataLen={len(dataset)}")

    print(f"\n📊 统计: 总共跑了 {total_steps_all_games} 步。")
    print(f"📊 统计: 有效结束(Done=True)的局数: {success_count}/{NUM_GAMES}")

    if len(dataset) > 0:
        print("\n" + "="*50)
        print("🔍 数据预览 (First 5 Rows)")
        print("="*50)
        for idx, row in enumerate(dataset[:5]):
            s, e, a, z = row
            print(f"\n[Row {idx}]")
            print(f"  State Vec Shape: {s.shape if hasattr(s, 'shape') else 'N/A'}")
            print(f"  Event Vec: {e}")
            print(f"  Action: {a}")
            print(f"  Result (z): {z}")
        print("="*50 + "\n")
        with open(OUTPUT_FILE, 'wb') as f:
            pickle.dump(dataset, f)

        print(f"\n✅ 成功! 保存了 {len(dataset)} 条数据到 {OUTPUT_FILE}")
    else:
        print("\n❌ 依然失败: 数据集为空。")
        print("建议：检查 Game X: Steps=... 这一行。如果 Steps 都是 0，说明 update 也没用。")

if __name__ == "__main__":
    main()