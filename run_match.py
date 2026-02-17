from tqdm import tqdm
from gym_splendor_code.envs.splendor import SplendorEnv

from agents.greedy_agent_boost import GreedyAgentBoost  
NUM_GAMES = 100
MODE = "stochastic"   # 'deterministic' or 'stochastic'


def get_winner_id(env: SplendorEnv):
    
    if env.first_winner is not None:
        return int(env.first_winner)

    # fallback: compare final points (might be used if episode didn't end cleanly)
    p0 = env.points_of_player_by_id(0)

    p1 = env.points_of_player_by_id(1)
    print(f"Final points - Player 0: {p0}, Player 1: {p1}")
    if p0 > p1:
        return 0
    elif p1 > p0:
        return 1
    else:
        return -1


def run_one_game(env, agent_for_player0, agent_for_player1, max_steps=500):
    """
    Runs a single game. Each step chooses an action for the current active player.
    Returns winner_id in {0,1,-1} and number of steps.
    """
    env.reset()

    # Attach shared env
    agent_for_player0.env = env
    agent_for_player1.env = env

    done = False
    steps = 0

    # Make sure action space initialized
    env.update_actions()

    while not done and steps < max_steps:
        # active player id is stored in state
        pid = env.current_state_of_the_game.active_player_id

        if pid == 0:
            action = agent_for_player0.choose_act(MODE)
        else:
            action = agent_for_player1.choose_act(MODE)

        # If no action, end
        if action is None:
            # env.step handles action None, but we can break early too
            _, _, done, info = env.step(MODE, None)
            break

        # IMPORTANT: your env.step signature is step(mode, action, ...)
        obs, reward, done, info = env.step(MODE, action)

        # refresh legal actions (some agents rely on env.action_space.list_of_actions)
        env.update_actions_light()

        steps += 1

    winner = get_winner_id(env)
    return winner, steps


def main():
    env = SplendorEnv()

    # Build two agents
    value_agent = GreedyAgentBoost(mode="value")
    event_agent = GreedyAgentBoost(mode="event")

    wins = {"value": 0, "event": 0, "draw": 0}
    total_steps = 0

    for g in tqdm(range(NUM_GAMES)):
        # Swap seats every game to remove first-player advantage
        if g % 2 == 0:
            # player0=value, player1=event
            winner, steps = run_one_game(env, value_agent, event_agent)
            if winner == 0:
                wins["value"] += 1
            elif winner == 1:
                wins["event"] += 1
            else:
                wins["draw"] += 1
        else:
            # player0=event, player1=value
            winner, steps = run_one_game(env, event_agent, value_agent)
            if winner == 0:
                wins["event"] += 1
            elif winner == 1:
                wins["value"] += 1
            else:
                wins["draw"] += 1

        total_steps += steps

    print("\n====================")
    print("Match results (100 games)")
    print("====================")
    print(f"Value-based wins: {wins['value']}")
    print(f"Event-based wins: {wins['event']}")
    print(f"Draws          : {wins['draw']}")
    print(f"Avg steps/game : {total_steps / NUM_GAMES:.2f}")


if __name__ == "__main__":
    main()
