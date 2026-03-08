"""Evaluate a trained PPO model vs Random and Greedy opponents."""
import sys
import numpy as np

if not hasattr(np, 'bool8'):
    np.bool8 = bool

class FakeShimmy:
    @staticmethod
    def GymV26CompatibilityV0(env, **kwargs):
        return env

import types
fake_shimmy = types.ModuleType("shimmy")
fake_shimmy.GymV26CompatibilityV0 = FakeShimmy.GymV26CompatibilityV0
sys.modules["shimmy"] = fake_shimmy

import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'


import os
import sys
import argparse
import numpy as np
from pathlib import Path
import importlib.util  

# Dynamically locate the Splendor-6759 root directory
ROOT_PATH = Path(__file__).resolve().parent.parent.parent

MODULES_PATH = str(ROOT_PATH / "modules")
PROJECT_SRC_PATH = str(ROOT_PATH / "project_event_based" / "src")
PROJECT_SCRIPTS_PATH = str(ROOT_PATH / "project_event_based" / "scripts")

for p in [MODULES_PATH, PROJECT_SRC_PATH, PROJECT_SCRIPTS_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)

from sb3_contrib import MaskablePPO
from utils.splendor_gym_wrapper import make_splendor_env 
from train_maskable import EventRewardWrapper 

from agents.random_agent import RandomAgent

import importlib.util

greedy_file = ROOT_PATH / "modules" / "agents" / "greedy_agent_boost.py"
spec = importlib.util.spec_from_file_location("greedy_agent_boost", str(greedy_file))
greedy_module = importlib.util.module_from_spec(spec)
sys.modules["greedy_agent_boost"] = greedy_module
spec.loader.exec_module(greedy_module)
GreedyAgent = greedy_module.GreedyAgentBoost

def evaluate_policy(model_path: str, opponent, n_games: int = 200):
    model = MaskablePPO.load(model_path, device='cpu')
    base_env = make_splendor_env(reward_mode='score_progress', opponent_agent=opponent, max_turns=200)

    dummy_config = {
        'reward': {
            'event_weights': np.zeros(9)
        },
        'environment': {
            'combine_event_and_score': False
        }
    }
    env = EventRewardWrapper(base_env, config=dummy_config)

    wins = 0
    agent_scores = []
    opp_scores = []

    for i in range(n_games):
        obs, info = env.reset()
        done = False
        while not done:
            action_masks = env.action_masks() 
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            
            action_idx = int(np.asarray(action).squeeze())
            obs, reward, terminated, truncated, info = env.step(action_idx)
            done = terminated or truncated

        try:
            unwrapped_env = env.unwrapped
            agent_score = unwrapped_env.prev_score
            opp_score = unwrapped_env.env.current_state_of_the_game.list_of_players_hands[unwrapped_env.opponent_id].number_of_my_points()
        except Exception:
            agent_score = info.get('player_score', 0)
            opp_score = info.get('opponent_score', 0)

        agent_scores.append(int(agent_score))
        opp_scores.append(int(opp_score))

        agent_won = info.get('agent_won', False) if agent_score > opp_score else False
        if agent_score > opp_score: wins += 1

    return {
        'win_rate': wins / n_games,
        'avg_agent_score': np.mean(agent_scores),
        'avg_opp_score': np.mean(opp_scores),
        'avg_score_diff': np.mean(np.array(agent_scores) - np.array(opp_scores)),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--n_games', type=int, default=200)
    parser.add_argument('--opponent', type=str, default='greedy', choices=['greedy', 'random'])
    args = parser.parse_args()

    opp_type = args.opponent.lower()
    print(f'🚀 Start: {args.model} vs {opp_type}')

    if opp_type == 'random':
        opponent_agent = RandomAgent()
    else:
        if GreedyAgent is None:
            print(" Error: Cannot load GreedyAgent")
            return
        
        try:
            opponent_agent = GreedyAgent(mode="value") 
            print("✅ Success: Initialized GreedyAgent with 'value' mode")
        except TypeError:
            opponent_agent = GreedyAgent()

    results = evaluate_policy(args.model, opponent_agent, n_games=args.n_games)

    print('\n' + '='*40)
    print(f"🏆 Win Rate: {results['win_rate']:.2%}")
    print(f"📈 Player Score: {results['avg_agent_score']:.2f}")
    print(f"📉 Opponent: {results['avg_opp_score']:.2f}")
    print(f"⚖️ Average Score Difference: {results['avg_score_diff']:.2f}")
    print('='*40)

if __name__ == '__main__':
    main()