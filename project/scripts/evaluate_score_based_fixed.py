"""
Fixed evaluation script using Gym wrapper (consistent with training).

Author: AI Agent  
Date: 2026-02-25
Version: 2.0 (FIXED)
"""

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "modules")

import argparse
import json
from pathlib import Path
from typing import Dict
from datetime import datetime

import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO

from agents.random_agent import RandomAgent
from agents.greedy_agent_boost import GreedyAgentBoost
from project.src.utils.splendor_gym_wrapper import SplendorGymWrapper


def evaluate_model(model_path: str, num_games: int = 100, opponent_name: str = 'random'):
    """
    Evaluate PPO model using the same Gym wrapper as training.
    
    Args:
        model_path: Path to saved PPO model
        num_games: Number of games to play
        opponent_name: 'random' or 'greedy'
        
    Returns:
        dict: Evaluation statistics
    """
    # Load model
    model = PPO.load(model_path)
    
    # Create opponent
    if opponent_name == 'random':
        opponent = RandomAgent(distribution='uniform_on_types')
    elif opponent_name == 'greedy':
        opponent = GreedyAgentBoost(name="Greedy", mode="value")
    else:
        raise ValueError(f"Unknown opponent: {opponent_name}")
    
    # Statistics
    wins = 0
    losses = 0
    draws = 0
    agent_scores = []
    opponent_scores = []
    game_lengths = []
    
    # Run games
    for game_idx in tqdm(range(num_games), desc=f"PPO vs {opponent_name}"):
        # Create fresh environment with opponent
        env = SplendorGymWrapper(
            opponent_agent=opponent,
            reward_mode='score_progress',
            max_turns=500
        )
        
        obs, info = env.reset()
        done = False
        turn_count = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            turn_count += 1
        
        # Get final scores
        agent_score = info.get('player_score', 0)
        opp_score = info.get('opponent_score', 0)
        
        agent_scores.append(agent_score)
        opponent_scores.append(opp_score)
        game_lengths.append(turn_count)
        
        # Determine winner
        if agent_score > opp_score:
            wins += 1
        elif agent_score < opp_score:
            losses += 1
        else:
            draws += 1
    
    # Compile statistics
    total_games = wins + losses + draws
    stats = {
        'total_games': total_games,
        'agent': {
            'name': 'PPO-ScoreBased',
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': (wins / total_games * 100) if total_games > 0 else 0,
            'avg_score': np.mean(agent_scores),
            'std_score': np.std(agent_scores),
            'max_score': np.max(agent_scores),
            'min_score': np.min(agent_scores)
        },
        'opponent': {
            'name': opponent_name.capitalize(),
            'wins': losses,
            'losses': wins,
            'draws': draws,
            'win_rate': (losses / total_games * 100) if total_games > 0 else 0,
            'avg_score': np.mean(opponent_scores),
            'std_score': np.std(opponent_scores),
            'max_score': np.max(opponent_scores),
            'min_score': np.min(opponent_scores)
        },
        'game_stats': {
            'avg_length': np.mean(game_lengths),
            'std_length': np.std(game_lengths),
            'max_length': np.max(game_lengths),
            'min_length': np.min(game_lengths)
        }
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Evaluate PPO agent (FIXED with wrapper)')
    parser.add_argument('--model', type=str,
                       default='project/logs/ppo_score_based_v1_20260224_113524/final_model',
                       help='Path to trained PPO model')
    parser.add_argument('--games', type=int, default=100,
                       help='Number of games per opponent')
    parser.add_argument('--output', type=str,
                       default='project/experiments/evaluation/ppo_score_based_eval_fixed',
                       help='Output directory for results')
    args = parser.parse_args()
    
    print("=" * 80)
    print("PPO Score-Based Agent Evaluation (FIXED VERSION)")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Games per opponent: {args.games}")
    print(f"Using Gym wrapper (consistent with training)")
    print()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    # Evaluate vs RandomAgent
    print(f"Evaluating vs RandomAgent ({args.games} games)...")
    print("-" * 80)
    stats_random = evaluate_model(args.model, args.games, 'random')
    all_results['vs_random'] = stats_random
    
    print(f"\nResults vs RandomAgent:")
    print(f"  PPO Win Rate: {stats_random['agent']['win_rate']:.1f}%")
    print(f"  PPO Avg Score: {stats_random['agent']['avg_score']:.1f} ± {stats_random['agent']['std_score']:.1f}")
    print(f"  PPO Score Range: [{stats_random['agent']['min_score']}, {stats_random['agent']['max_score']}]")
    print(f"  Random Avg Score: {stats_random['opponent']['avg_score']:.1f} ± {stats_random['opponent']['std_score']:.1f}")
    print(f"  Avg Game Length: {stats_random['game_stats']['avg_length']:.1f} turns")
    print()
    
    # Evaluate vs GreedyAgent
    print(f"Evaluating vs GreedyAgent ({args.games} games)...")
    print("-" * 80)
    stats_greedy = evaluate_model(args.model, args.games, 'greedy')
    all_results['vs_greedy'] = stats_greedy
    
    print(f"\nResults vs GreedyAgent:")
    print(f"  PPO Win Rate: {stats_greedy['agent']['win_rate']:.1f}%")
    print(f"  PPO Avg Score: {stats_greedy['agent']['avg_score']:.1f} ± {stats_greedy['agent']['std_score']:.1f}")
    print(f"  PPO Score Range: [{stats_greedy['agent']['min_score']}, {stats_greedy['agent']['max_score']}]")
    print(f"  Greedy Avg Score: {stats_greedy['opponent']['avg_score']:.1f} ± {stats_greedy['opponent']['std_score']:.1f}")
    print(f"  Avg Game Length: {stats_greedy['game_stats']['avg_length']:.1f} turns")
    print()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f"evaluation_results_fixed_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("=" * 80)
    print(f"✓ Evaluation complete!")
    print(f"✓ Results saved to: {results_file}")
    print("=" * 80)
    
    return all_results


if __name__ == '__main__':
    main()
