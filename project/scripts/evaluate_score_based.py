"""
Evaluation script for PPO Score-based agent.

Runs evaluation against baseline agents (RandomAgent, GreedyAgent)
using direct Splendor environment interaction.

Author: AI Agent
Date: 2026-02-25
Version: 1.0
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

from gym_splendor_code.envs.splendor import SplendorEnv
from agents.random_agent import RandomAgent
from agents.greedy_agent_boost import GreedyAgentBoost
from project.src.utils.state_vectorizer import SplendorStateVectorizer


def play_game_ppo_vs_agent(ppo_model, opponent_agent, vectorizer, ppo_first: bool = True):
    """
    Play one game between PPO and opponent.
    
    Args:
        ppo_model: Trained PPO model
        opponent_agent: Opponent agent
        vectorizer: State vectorizer
        ppo_first: Whether PPO goes first
        
    Returns:
        tuple: (ppo_won, ppo_points, opponent_points, game_length)
    """
    # Create environment
    env = SplendorEnv()
    env.reset()
    
    # Assign player IDs
    ppo_player_id = 0 if ppo_first else 1
    opponent_player_id = 1 - ppo_player_id
    
    # Set opponent agent's environment
    opponent_agent.env = env
    
    # Set starting player
    env.set_active_player(0 if ppo_first else 1)
    
    done = False
    turn_count = 0
    max_turns = 200
    
    while not done and turn_count < max_turns:
        current_player = env.active_player_id()
        state = env.current_state_of_the_game
        
        if current_player == ppo_player_id:
            # PPO's turn
            obs_vector = vectorizer.vectorize(state, ppo_player_id, turn_count)
            legal_actions = env.action_space.list_of_actions
            
            if not legal_actions:
                break
                
            action_idx, _ = ppo_model.predict(obs_vector, deterministic=True)
            
            if action_idx >= len(legal_actions):
                action_idx = np.random.randint(len(legal_actions))
            
            action = legal_actions[action_idx]
        else:
            # Opponent's turn
            env.update_actions_light()
            action = opponent_agent.choose_act('deterministic')
        
        if action is None:
            break
            
        _, reward, done, info = env.step('deterministic', action)
        turn_count += 1
    
    # Get final scores
    ppo_points = env.current_state_of_the_game.list_of_players_hands[ppo_player_id].number_of_my_points()
    opponent_points = env.current_state_of_the_game.list_of_players_hands[opponent_player_id].number_of_my_points()
    
    # Determine winner
    ppo_won = info.get('winner_id') == ppo_player_id if done else (ppo_points >= opponent_points)
    
    return ppo_won, ppo_points, opponent_points, turn_count


def run_matches_ppo_vs_agent(ppo_model, opponent_agent, vectorizer, num_games: int, desc: str = ""):
    """
    Run matches between PPO agent and a baseline agent.
    
    Args:
        ppo_model: Trained PPO model
        opponent_agent: Opponent agent (RandomAgent or GreedyAgent)
        vectorizer: State vectorizer
        num_games: Number of games to play
        desc: Description for progress bar
        
    Returns:
        dict: Statistics including wins, points, and game lengths
    """
    results = {
        'ppo_wins': 0,
        'opponent_wins': 0,
        'ppo_points': [],
        'opponent_points': [],
        'game_lengths': [],
    }
    
    for i in tqdm(range(num_games), desc=desc):
        # Alternate who goes first
        ppo_first = (i % 2 == 0)
        
        ppo_won, ppo_pts, opp_pts, turns = play_game_ppo_vs_agent(
            ppo_model, opponent_agent, vectorizer, ppo_first
        )
        
        results['ppo_points'].append(ppo_pts)
        results['opponent_points'].append(opp_pts)
        results['game_lengths'].append(turns)
        
        if ppo_won:
            results['ppo_wins'] += 1
        else:
            results['opponent_wins'] += 1
    
    return results


def format_statistics(results: dict, ppo_name: str, opponent_name: str) -> Dict:
    """Format match statistics."""
    total_games = len(results['ppo_points'])
    
    stats = {
        'total_games': total_games,
        'ppo': {
            'name': ppo_name,
            'wins': results['ppo_wins'],
            'win_rate': results['ppo_wins'] / total_games * 100,
            'avg_points': np.mean(results['ppo_points']),
            'std_points': np.std(results['ppo_points']),
        },
        'opponent': {
            'name': opponent_name,
            'wins': results['opponent_wins'],
            'win_rate': results['opponent_wins'] / total_games * 100,
            'avg_points': np.mean(results['opponent_points']),
            'std_points': np.std(results['opponent_points']),
        },
        'avg_game_length': np.mean(results['game_lengths']),
        'std_game_length': np.std(results['game_lengths']),
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Evaluate PPO agent against baselines')
    parser.add_argument('--model', type=str, 
                       default='project/logs/ppo_score_based_v1_20260224_113524/final_model',
                       help='Path to trained PPO model')
    parser.add_argument('--games', type=int, default=100,
                       help='Number of games per opponent')
    parser.add_argument('--output', type=str, 
                       default='project/experiments/evaluation/ppo_score_based_eval',
                       help='Output directory for results')
    args = parser.parse_args()
    
    print("=" * 80)
    print("PPO Score-Based Agent Evaluation")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Games per opponent: {args.games}")
    print()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load PPO model
    print("Loading PPO model...")
    ppo_model = PPO.load(args.model)
    vectorizer = SplendorStateVectorizer()
    print(f"✓ Loaded model from {args.model}")
    print()
    
    # Initialize baseline agents
    random_agent = RandomAgent(distribution='uniform_on_types')
    greedy_value_agent = GreedyAgentBoost(name="Greedy", mode="value")
    
    all_results = {}
    
    # Evaluate vs RandomAgent
    print(f"Evaluating vs RandomAgent ({args.games} games)...")
    print("-" * 80)
    results_random = run_matches_ppo_vs_agent(
        ppo_model, random_agent, vectorizer, args.games,
        desc="PPO vs Random"
    )
    stats_random = format_statistics(results_random, "PPO-ScoreBased", "RandomAgent")
    all_results['vs_random'] = stats_random
    
    print(f"\nResults vs RandomAgent:")
    print(f"  PPO Win Rate: {stats_random['ppo']['win_rate']:.1f}%")
    print(f"  PPO Avg Points: {stats_random['ppo']['avg_points']:.1f} ± {stats_random['ppo']['std_points']:.1f}")
    print(f"  Random Avg Points: {stats_random['opponent']['avg_points']:.1f} ± {stats_random['opponent']['std_points']:.1f}")
    print(f"  Avg Game Length: {stats_random['avg_game_length']:.1f} turns")
    print()
    
    # Evaluate vs GreedyAgent
    print(f"Evaluating vs GreedyAgent-value ({args.games} games)...")
    print("-" * 80)
    results_greedy = run_matches_ppo_vs_agent(
        ppo_model, greedy_value_agent, vectorizer, args.games,
        desc="PPO vs Greedy"
    )
    stats_greedy = format_statistics(results_greedy, "PPO-ScoreBased", "GreedyAgent-value")
    all_results['vs_greedy'] = stats_greedy
    
    print(f"\nResults vs GreedyAgent-value:")
    print(f"  PPO Win Rate: {stats_greedy['ppo']['win_rate']:.1f}%")
    print(f"  PPO Avg Points: {stats_greedy['ppo']['avg_points']:.1f} ± {stats_greedy['ppo']['std_points']:.1f}")
    print(f"  Greedy Avg Points: {stats_greedy['opponent']['avg_points']:.1f} ± {stats_greedy['opponent']['std_points']:.1f}")
    print(f"  Avg Game Length: {stats_greedy['avg_game_length']:.1f} turns")
    print()
    
    # Save results
    results_file = output_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("=" * 80)
    print(f"✓ Evaluation complete!")
    print(f"✓ Results saved to: {results_file}")
    print("=" * 80)
    
    return all_results


if __name__ == '__main__':
    main()
