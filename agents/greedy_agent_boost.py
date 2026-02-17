import random
import numpy as np
from agents.abstract_agent import Agent
from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.game_settings import POINTS_TO_WIN
from evaluators import ValueBasedEvaluator, EventBasedEvaluator, simulate_next_state


class GreedyAgentBoost(Agent):

    def __init__(self, name="Greedy", mode=None, value_weights=None, event_weights=None):
        super().__init__()
        assert mode in ("value", "event")
        self.name = f"{name}-{mode}"
        self.mode = mode
        self.value_eval = ValueBasedEvaluator(value_weights)
        self.event_eval = EventBasedEvaluator(event_weights)

    def choose_act(self, mode):
        state = self.env.current_state_of_the_game
        actions = self.env.action_space.list_of_actions
        if not actions:
            return None

        current_points = state.active_players_hand().number_of_my_points()

        best_score = -float("inf")
        best_actions = []

        for action in actions:
            if self.mode == "value":
                next_state = simulate_next_state(state, action)
                s = self.value_eval.score_next_state(next_state)
            else:
                ae = action.evaluate(state)
                s = self.event_eval.score_event(current_points, ae)

            if s > best_score:
                best_score = s
                best_actions = [action]
            elif s == best_score:
                best_actions.append(action)

        return random.choice(best_actions)


    def normalize_weight(self):
        if np.linalg.norm(self.weight) > 0:
            self.weight = self.weight/np.linalg.norm(self.weight)

    def update_weight(self, list, lr, ratio):
        list = list/np.linalg.norm(list)
        lr = lr * ratio
        self.weight = [a + b *lr for a, b in zip(self.weight, list)]
        self.normalize_weight()
