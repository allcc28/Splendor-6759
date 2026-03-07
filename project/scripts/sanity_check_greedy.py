"""
Sanity check: verify GreedyAgent behavior and simulate_next_state perspective.

Tests:
1. Does action.execute() switch the active player in the state?
2. Does GreedyAgent score_next_state evaluate the correct player?
3. Manual game: GreedyAgent score distribution over N games
4. RandomAgent vs GreedyAgent head-to-head (no PPO) via the wrapper
"""

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "modules")

import numpy as np
from tqdm import tqdm

from gym_splendor_code.envs.splendor import SplendorEnv
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict
from agents.greedy_agent_boost import GreedyAgentBoost
from agents.random_agent import RandomAgent
from evaluators import simulate_next_state, ValueBasedEvaluator


def test_active_player_switch():
    """Check whether action.execute() changes state.active_player_id."""
    print("=" * 60)
    print("TEST 1: Does action.execute() switch active player?")
    print("=" * 60)

    env = SplendorEnv()
    env.reset()
    state = env.current_state_of_the_game

    # Get active player id before action
    pid_before = state.active_player_id if hasattr(state, 'active_player_id') else "N/A"
    active_hand_before = state.active_players_hand()
    points_before = active_hand_before.number_of_my_points()

    env.update_actions_light()
    actions = env.action_space.list_of_actions
    if not actions:
        print("No actions available. Skipping.")
        return

    action = actions[0]
    
    # Simulate (state copy)
    next_state = simulate_next_state(state, action)
    pid_after = next_state.active_player_id if hasattr(next_state, 'active_player_id') else "N/A"
    active_hand_after = next_state.active_players_hand()
    points_after = active_hand_after.number_of_my_points()

    print(f"  Active player BEFORE execute: {pid_before}")
    print(f"  Active player AFTER  execute: {pid_after}")
    print(f"  Points of active_players_hand() BEFORE: {points_before}")
    print(f"  Points of active_players_hand() AFTER:  {points_after}")

    # Also check player 0's hand specifically after execution
    hand_p0_after = next_state.list_of_players_hands[0]
    hand_p1_after = next_state.list_of_players_hands[1]
    print(f"  Player 0 points after execute: {hand_p0_after.number_of_my_points()}")
    print(f"  Player 1 points after execute: {hand_p1_after.number_of_my_points()}")

    if pid_before != pid_after:
        print("\n  ⚠️  action.execute() DOES switch active_player_id (game engine behavior).")
        print("     Evaluators must use get_actor_hand() (NOT active_players_hand())")
        print("     after simulate_next_state(). evaluators.py has been fixed.")
    else:
        print("\n  ✅ active player does NOT switch after execute() (unusual)")
        print("     Using get_active_hand() is safe here")


def test_greedy_random_headtohead(n_games=20):
    """Run N games of GreedyAgent vs RandomAgent using SplendorEnv directly."""
    print("\n" + "=" * 60)
    print(f"TEST 2: Greedy vs Random head-to-head ({n_games} games)")
    print("=" * 60)

    greedy = GreedyAgentBoost(name="Greedy", mode="value")
    random_ag = RandomAgent(distribution="uniform_on_types")

    greedy_scores = []
    random_scores = []
    greedy_wins = 0

    env = SplendorEnv()

    for game_i in tqdm(range(n_games), desc="Greedy vs Random"):
        obs = env.reset()
        done = False
        turn = 0
        max_turns = 200

        while not done and turn < max_turns:
            # Player 0 = greedy
            state_obs = env.show_observation('deterministic')
            action = greedy.choose_action(state_obs, [])
            if action is None:
                env.update_actions_light()
                acts = env.action_space.list_of_actions
                if acts:
                    action = np.random.choice(acts)
                else:
                    break
            _, _, done, info = env.step('deterministic', action)
            turn += 1
            if done:
                break

            # Player 1 = random
            state_obs = env.show_observation('deterministic')
            action = random_ag.choose_action(state_obs, [])
            if action is None:
                env.update_actions_light()
                acts = env.action_space.list_of_actions
                if acts:
                    action = np.random.choice(acts)
                else:
                    break
            _, _, done, info = env.step('deterministic', action)
            turn += 1

        state = env.current_state_of_the_game
        g_pts = state.list_of_players_hands[0].number_of_my_points()
        r_pts = state.list_of_players_hands[1].number_of_my_points()
        greedy_scores.append(g_pts)
        random_scores.append(r_pts)
        if g_pts > r_pts:
            greedy_wins += 1

    print(f"\n  Greedy avg score:  {np.mean(greedy_scores):.1f} ± {np.std(greedy_scores):.1f}")
    print(f"  Random avg score:  {np.mean(random_scores):.1f} ± {np.std(random_scores):.1f}")
    print(f"  Greedy win rate:   {greedy_wins}/{n_games} = {100*greedy_wins/n_games:.0f}%")

    if np.mean(greedy_scores) < np.mean(random_scores):
        print("\n  ❌ WARNING: Greedy scores LESS than Random — evaluator may be broken!")
    else:
        print("\n  ✅ Greedy outperforms Random as expected")


def test_greedy_score_distribution(n_games=30):
    """Use the gym wrapper to see raw scores of GreedyAgent as opponent."""
    print("\n" + "=" * 60)
    print(f"TEST 3: GreedyAgent opponent score distribution via wrapper ({n_games} games)")
    print("=" * 60)

    from project.src.utils.splendor_gym_wrapper import SplendorGymWrapper
    from sb3_contrib import MaskablePPO

    model_path = "project/logs/maskable_ppo_score_v3_20260303_183435/final_model"
    model = MaskablePPO.load(model_path)
    greedy_agent = GreedyAgentBoost(name="Greedy", mode="value")

    greedy_opp_scores = []
    agent_scores = []

    for _ in tqdm(range(n_games), desc="Quick eval vs GreedyAgent"):
        env = SplendorGymWrapper(opponent_agent=greedy_agent, reward_mode="score_progress", max_turns=200)
        obs, info = env.reset()
        done = False
        while not done:
            if len(env.cached_legal_actions) == 0:
                break
            action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

        agent_scores.append(info.get('player_score', 0))
        greedy_opp_scores.append(info.get('opponent_score', 0))

    print(f"  V3 agent avg score:    {np.mean(agent_scores):.1f} ± {np.std(agent_scores):.1f}")
    print(f"  GreedyAgent opp score: {np.mean(greedy_opp_scores):.1f} ± {np.std(greedy_opp_scores):.1f}")
    print(f"  (Compare: 100-game result was 15.3 vs 1.1)")


if __name__ == "__main__":
    test_active_player_switch()
    test_greedy_random_headtohead(n_games=20)
    test_greedy_score_distribution(n_games=20)
