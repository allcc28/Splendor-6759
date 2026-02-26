"""
Evaluation script for PPO Score-based agent v3.

Adds a "fallback mode" for invalid actions: when the model picks an action
index that exceeds the number of legal actions, we fall back to a random legal
action instead of terminating with -10. This measures the model's actual game
strategy, separating it from the action masking problem.

Reports both strict (original) and forgiving (fallback) metrics.
"""

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "modules")

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO

from agents.random_agent import RandomAgent
from agents.greedy_agent_boost import GreedyAgentBoost
from gym_splendor_code.envs.splendor import SplendorEnv
from project.src.utils.splendor_gym_wrapper import SplendorGymWrapper
from project.src.utils.state_vectorizer import SplendorStateVectorizer


def evaluate_vs_opponent(model, opponent_agent, num_games: int, max_turns: int = 200,
                         fallback_on_invalid: bool = True, desc: str = "Evaluating"):
    """
    Evaluate PPO model vs an opponent using the Gym wrapper.
    
    Args:
        model: Loaded PPO model
        opponent_agent: Agent object or None (for random opponent)
        num_games: Number of games to play
        max_turns: Max turns per game
        fallback_on_invalid: If True, use random legal action when model picks invalid
        desc: Progress bar description
        
    Returns:
        dict: Detailed evaluation statistics
    """
    wins = 0
    losses = 0
    draws = 0
    agent_scores = []
    opponent_scores = []
    game_lengths = []
    total_rewards = []
    invalid_action_counts = []
    zero_actions_count = 0
    fallback_counts = []
    
    for game_idx in tqdm(range(num_games), desc=desc):
        env = SplendorGymWrapper(
            opponent_agent=opponent_agent,
            reward_mode='score_progress',
            max_turns=max_turns
        )
        
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        ep_invalid = 0
        ep_fallbacks = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            
            n_legal = len(env.cached_legal_actions)
            
            # Handle 0 legal actions (engine edge case)
            if n_legal == 0:
                zero_actions_count += 1
                # Nothing we can do - any action is invalid
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                done = terminated or truncated
                continue
            
            # Handle invalid action index
            if action >= n_legal:
                ep_invalid += 1
                if fallback_on_invalid:
                    # Fall back to random legal action
                    action = np.random.randint(0, n_legal)
                    ep_fallbacks += 1
                # else: let the wrapper terminate with -10
            
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        
        # Extract final state
        agent_score = info.get('player_score', 0)
        opp_score = info.get('opponent_score', 0)
        agent_won = info.get('agent_won', False)
        agent_lost = info.get('agent_lost', False)
        turn = info.get('turn', 0)
        
        agent_scores.append(agent_score)
        opponent_scores.append(opp_score)
        game_lengths.append(turn)
        total_rewards.append(ep_reward)
        invalid_action_counts.append(ep_invalid)
        fallback_counts.append(ep_fallbacks)
        
        if agent_won:
            wins += 1
        elif agent_lost:
            losses += 1
        else:
            if agent_score > opp_score:
                wins += 1
            elif agent_score < opp_score:
                losses += 1
            else:
                draws += 1
    
    total = wins + losses + draws
    stats = {
        'total_games': total,
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': (wins / total * 100) if total > 0 else 0,
        'loss_rate': (losses / total * 100) if total > 0 else 0,
        'agent_scores': {
            'mean': float(np.mean(agent_scores)),
            'std': float(np.std(agent_scores)),
            'min': int(np.min(agent_scores)),
            'max': int(np.max(agent_scores)),
            'median': float(np.median(agent_scores))
        },
        'opponent_scores': {
            'mean': float(np.mean(opponent_scores)),
            'std': float(np.std(opponent_scores)),
            'min': int(np.min(opponent_scores)),
            'max': int(np.max(opponent_scores)),
            'median': float(np.median(opponent_scores))
        },
        'game_lengths': {
            'mean': float(np.mean(game_lengths)),
            'std': float(np.std(game_lengths)),
            'min': int(np.min(game_lengths)),
            'max': int(np.max(game_lengths))
        },
        'rewards': {
            'mean': float(np.mean(total_rewards)),
            'std': float(np.std(total_rewards))
        },
        'invalid_actions': {
            'mean_per_game': float(np.mean(invalid_action_counts)),
            'total': int(np.sum(invalid_action_counts)),
            'games_with_invalid': int(sum(1 for x in invalid_action_counts if x > 0))
        },
        'fallbacks': {
            'mean_per_game': float(np.mean(fallback_counts)),
            'total': int(np.sum(fallback_counts)),
        },
        'zero_actions_episodes': zero_actions_count,
    }
    
    return stats


def print_results(label: str, stats: dict):
    """Pretty-print evaluation results."""
    print(f"\n{'=' * 60}")
    print(f"Results vs {label} ({stats['total_games']} games)")
    print(f"{'=' * 60}")
    print(f"  Win/Loss/Draw: {stats['wins']}/{stats['losses']}/{stats['draws']}")
    print(f"  Win Rate: {stats['win_rate']:.1f}%")
    print()
    print(f"  Agent Scores:    {stats['agent_scores']['mean']:.1f} ± {stats['agent_scores']['std']:.1f} "
          f"(range: {stats['agent_scores']['min']}-{stats['agent_scores']['max']})")
    print(f"  Opponent Scores: {stats['opponent_scores']['mean']:.1f} ± {stats['opponent_scores']['std']:.1f} "
          f"(range: {stats['opponent_scores']['min']}-{stats['opponent_scores']['max']})")
    print(f"  Game Length:     {stats['game_lengths']['mean']:.1f} ± {stats['game_lengths']['std']:.1f} turns")
    print(f"  Avg Reward:      {stats['rewards']['mean']:.2f}")
    print(f"  Invalid Actions: {stats['invalid_actions']['mean_per_game']:.1f}/game "
          f"({stats['invalid_actions']['games_with_invalid']}/{stats['total_games']} games affected)")
    if stats['fallbacks']['total'] > 0:
        print(f"  Fallback Actions: {stats['fallbacks']['mean_per_game']:.1f}/game "
              f"({stats['fallbacks']['total']} total)")
    if stats['zero_actions_episodes'] > 0:
        print(f"  Zero-legal-actions episodes: {stats['zero_actions_episodes']}")
    
    # Sanity check
    if stats['agent_scores']['mean'] < 3 and stats['opponent_scores']['mean'] < 3:
        print(f"\n  ⚠️  WARNING: Very low scores — games may not be completing properly!")
    if stats['agent_scores']['max'] >= 15:
        print(f"  ✅ Agent reached 15+ points in at least one game")
    if stats['opponent_scores']['max'] >= 15:
        print(f"  ℹ️  Opponent reached 15+ points in at least one game")


def main():
    parser = argparse.ArgumentParser(description='Evaluate PPO agent v3 (with fallback mode)')
    parser.add_argument('--model', type=str,
                       default='project/logs/ppo_score_based_v1_20260224_113524/final_model',
                       help='Path to trained PPO model')
    parser.add_argument('--games', type=int, default=100,
                       help='Number of games per opponent')
    parser.add_argument('--max-turns', type=int, default=200,
                       help='Max turns per game')
    parser.add_argument('--no-fallback', action='store_true',
                       help='Disable fallback (terminate on invalid action)')
    parser.add_argument('--output', type=str,
                       default='project/experiments/evaluation/ppo_score_based_eval_v3',
                       help='Output directory')
    args = parser.parse_args()
    
    fallback = not args.no_fallback
    mode_str = "WITH fallback (random on invalid)" if fallback else "STRICT (terminate on invalid)"
    
    print("=" * 80)
    print("PPO Score-Based Agent Evaluation v3")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Games per opponent: {args.games}")
    print(f"Max turns: {args.max_turns}")
    print(f"Invalid action mode: {mode_str}")
    print()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading PPO model...")
    model = PPO.load(args.model)
    print(f"✓ Model loaded\n")
    
    all_results = {}
    
    # ---- vs Random (opponent=None, same as training) ----
    print(f"Evaluating vs Random Opponent ({args.games} games)...")
    print(f"  Using opponent=None (same random policy as training)")
    print("-" * 60)
    
    stats_random = evaluate_vs_opponent(
        model, opponent_agent=None, num_games=args.games,
        max_turns=args.max_turns, fallback_on_invalid=fallback,
        desc="PPO vs Random"
    )
    all_results['vs_random_wrapper'] = stats_random
    print_results("Random (wrapper)", stats_random)
    
    # ---- vs RandomAgent object ----
    print(f"\nEvaluating vs RandomAgent object ({args.games} games)...")
    print(f"  Using RandomAgent(distribution='uniform_on_types')")
    print("-" * 60)
    
    random_agent = RandomAgent(distribution='uniform_on_types')
    stats_random_agent = evaluate_vs_opponent(
        model, opponent_agent=random_agent, num_games=args.games,
        max_turns=args.max_turns, fallback_on_invalid=fallback,
        desc="PPO vs RandomAgent"
    )
    all_results['vs_random_agent'] = stats_random_agent
    print_results("RandomAgent", stats_random_agent)
    
    # ---- vs GreedyAgent ----
    print(f"\nEvaluating vs GreedyAgent ({args.games} games)...")
    print(f"  Using GreedyAgentBoost(mode='value')")
    print("-" * 60)
    
    greedy_agent = GreedyAgentBoost(name="Greedy", mode="value")
    stats_greedy = evaluate_vs_opponent(
        model, opponent_agent=greedy_agent, num_games=args.games,
        max_turns=args.max_turns, fallback_on_invalid=fallback,
        desc="PPO vs Greedy"
    )
    all_results['vs_greedy'] = stats_greedy
    print_results("GreedyAgent", stats_greedy)
    
    # ---- Also run STRICT mode for comparison ----
    if fallback:
        print(f"\n{'=' * 80}")
        print("COMPARISON: Strict mode (no fallback) vs Random")
        print(f"{'=' * 80}")
        stats_strict = evaluate_vs_opponent(
            model, opponent_agent=None, num_games=args.games,
            max_turns=args.max_turns, fallback_on_invalid=False,
            desc="PPO vs Random (strict)"
        )
        all_results['vs_random_strict'] = stats_strict
        print_results("Random (strict)", stats_strict)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f"eval_v3_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'=' * 80}")
    print(f"✓ All evaluations complete!")
    print(f"✓ Results saved to: {results_file}")
    
    # Summary comparison
    if fallback and 'vs_random_strict' in all_results:
        print(f"\n--- Impact of fallback mode ---")
        print(f"  Strict:   {all_results['vs_random_strict']['win_rate']:.1f}% win rate")
        print(f"  Fallback: {all_results['vs_random_wrapper']['win_rate']:.1f}% win rate")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()
