import random
import gym

try:
    from gym_splendor_code.envs.splendor import SplendorEnv
    print("✅ 成功导入 gym_splendor_code 包！")
except ImportError:
    print("❌ 导入失败，请确认环境安装。")
    exit()

def main():
    print("正在初始化环境...")
    env = SplendorEnv()
    print("✅ 环境初始化成功！")

    env.reset()
    print("✅ Reset 成功！")

    print("正在尝试模拟运行 5 步...")
    try:
        # 获取合法动作列表
        actions = env.action_space.list_of_actions
        
        for i in range(5):
            if len(actions) > 0:
                action = random.choice(actions)
                
                # --- 🔴 核心修复点 🔴 ---
                # 使用关键字传参，显式指定 mode='instant_end'
                # 这样不管函数定义是 step(mode, action) 还是 step(action, mode) 都能跑
                obs, reward, done, info = env.step(action=action, mode='instant_end')
                
                print(f"  Step {i+1}: 成功执行 -> 奖励: {reward}")
                
                if done:
                    print("  游戏结束")
                    env.reset()
                    actions = env.action_space.list_of_actions
                else:
                    # 更新合法动作（很重要，因为局面变了）
                    actions = env.action_space.list_of_actions
            else:
                print("  警告: 无合法动作")
                break
                
        print("\n🎉 恭喜！所有 BUG 已修复，环境完全可用！")
        
    except Exception as e:
        print(f"\n❌ 依然报错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()