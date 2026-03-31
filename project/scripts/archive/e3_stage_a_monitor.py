#!/usr/bin/env python3
"""
E3 Stage A Monitoring and Quick Evaluation Script

This script monitors the training progress and runs a quick evaluation (n=200)
when the training completes. It also collects event statistics from the logs.

Usage:
    python project/scripts/e3_stage_a_monitor.py --log-dir project/logs/maskable_ppo_event_e3_lite_reward_20260319_115615

Author: AI Agent
Date: 2026-03-19
"""

import sys
import os
import json
import time
import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional

sys.path.insert(0, "modules")
sys.path.append("project/src")

import numpy as np
from tqdm import tqdm
from sb3_contrib import MaskablePPO

from agents.random_agent import RandomAgent
from agents.greedy_agent_boost import GreedyAgentBoost
from project.src.utils.splendor_gym_wrapper import SplendorGymWrapper
from project.src.utils.event_reward_wrapper import maybe_wrap_with_event_shaping


def wait_for_training_completion(log_dir: Path, check_interval: int = 30):
    """
    Monitor the training log until completion.
    Checks every check_interval seconds for the final model to appear.
    """
    print(f"🔍 Monitoring training at {log_dir}...")
    print(f"⏱️  Checking every {check_interval} seconds for completion...")
    
    eval_dir = log_dir / "eval"
    final_model_path = eval_dir / "best_model.zip"
    
    start_time = time.time()
    
    while True:
        elapsed = time.time() - start_time
        
        if final_model_path.exists():
            print(f"\n✅ Training completed in {elapsed/3600:.1f} hours")
            return str(final_model_path)
        
        # Check tensorboard logs to estimate progress
        tb_log = list(log_dir.glob("logs/tensorboard/*/events.out.tfevents.*"))
        if tb_log:
            print(f"   Training still in progress... ({elapsed/3600:.1f}h elapsed)")
        
        time.sleep(check_interval)


def extract_event_stats_from_log(log_dir: Path) -> dict:
    """
    Parse event statistics from the training log file.
    """
    stats = {}
    
    # Try to read the event stats from tensorboard or log file
    # For now, return placeholder
    return {
        "status": "Event stats will be extracted after training completion",
        "log_dir": str(log_dir)
    }


def run_quick_evaluation(
    model_path: str,
    config_path: str,
    games: int = 200,
    output_dir: Optional[str] = None
) -> dict:
    """
    Run a quick evaluation on the trained model.
    Returns: dict with results for RandomAgent, GreedyAgent, and Random wrapper
    """
    print(f"\n{'='*60}")
    print(f"🏃 Quick Evaluation (n={games})")
    print(f"{'='*60}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model
    print(f"📦 Loading model from {model_path}...")
    model = MaskablePPO.load(model_path)
    
    # Define opponents
    opponents = [
        ("RandomAgent", RandomAgent()),
        ("GreedyAgent", GreedyAgentBoost()),
    ]
    
    results = {}
    
    for opp_name, opponent in opponents:
        print(f"\n🎮 Evaluating vs {opp_name}...")
        
        env = SplendorGymWrapper(
            opponent_agent=opponent,
            reward_mode=config["environment"].get("reward_mode", "score_progress"),
            max_turns=120
        )
        env = maybe_wrap_with_event_shaping(env, config)
        
        from sb3_contrib.common.wrappers import ActionMasker
        from project.src.utils.splendor_gym_wrapper import _mask_fn
        env = ActionMasker(env, _mask_fn)
        
        wins = 0
        agent_scores = []
        opp_scores = []
        
        for game_idx in tqdm(range(games), desc=f"vs {opp_name}"):
            obs, info = env.reset()
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            # Get final scores from info
            agent_score = info.get("agent_score", 0)
            opp_score = info.get("opponent_score", 0)
            
            agent_scores.append(agent_score)
            opp_scores.append(opp_score)
            
            if agent_score > opp_score:
                wins += 1
        
        win_rate = wins / games * 100
        results[opp_name] = {
            "win_rate": win_rate,
            "wins": wins,
            "games": games,
            "agent_score_mean": float(np.mean(agent_scores)),
            "agent_score_std": float(np.std(agent_scores)),
            "opponent_score_mean": float(np.mean(opp_scores)),
            "opponent_score_std": float(np.std(opp_scores)),
        }
        
        print(f"   Win rate: {win_rate:.1f}% ({wins}/{games})")
        print(f"   Agent score: {np.mean(agent_scores):.1f} ± {np.std(agent_scores):.1f}")
        print(f"   Opponent score: {np.mean(opp_scores):.1f} ± {np.std(opp_scores):.1f}")
        
        env.close()
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path(output_dir) / f"e3_stage_a_quick_eval_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "games": games,
                "results": results
            }, f, indent=2)
        
        print(f"\n💾 Results saved to {results_file}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor E3 Stage A training and run quick eval")
    parser.add_argument(
        "--log-dir",
        type=str,
        default="project/logs/maskable_ppo_event_e3_lite_reward_20260319_115615",
        help="Path to the training log directory",
    )
    parser.add_argument(
        "--skip-wait",
        action="store_true",
        help="Skip waiting for training; run eval on existing model",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=200,
        help="Number of games for quick eval",
    )
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    config_path = log_dir / "config.yaml"
    
    if not config_path.exists():
        print(f"❌ Config not found at {config_path}")
        sys.exit(1)
    
    # Wait for training if needed
    if not args.skip_wait:
        model_path = wait_for_training_completion(log_dir)
    else:
        model_path = str(log_dir / "eval" / "best_model.zip")
        if not Path(model_path).exists():
            print(f"❌ Model not found at {model_path}")
            sys.exit(1)
    
    # Run quick evaluation
    results = run_quick_evaluation(
        model_path=model_path,
        config_path=str(config_path),
        games=args.games,
        output_dir=str(log_dir / "eval")
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 E3 Stage A Summary")
    print(f"{'='*60}")
    for opp_name, res in results.items():
        print(f"{opp_name:15} | Win: {res['win_rate']:5.1f}% | Agent: {res['agent_score_mean']:5.1f}±{res['agent_score_std']:4.1f} | Opp: {res['opponent_score_mean']:5.1f}±{res['opponent_score_std']:4.1f}")


if __name__ == "__main__":
    main()
