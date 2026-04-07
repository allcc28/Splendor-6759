"""Greedy agent using heuristic card/gem scoring.

This is a minimal re-implementation of GreedyAgentBoost compatible with the
evaluate_alphazero.py arena interface (choose_action / choose_act).
"""

from __future__ import annotations

import random

import numpy as np

from agents.abstract_agent import Agent
from gym_splendor_code.envs.mechanics.game_settings import POINTS_TO_WIN


class GreedyAgentBoost(Agent):
    """Single-ply greedy agent with heuristic weights over action features.

    Parameters
    ----------
    name: str
        Agent display name.
    mode: str
        Evaluation mode.  Only "value" is supported; kept for API compatibility.
    """

    # Default weights: [win_bonus, vp_weight, noble_weight, card_weight, gem_weight]
    DEFAULT_WEIGHTS = np.array([100.0, 2.0, 2.0, 1.0, 0.1])

    def __init__(self, name: str = "GreedyBoost", mode: str = "value") -> None:
        super().__init__()
        self.name = name
        self.mode = mode
        self.weight = self.DEFAULT_WEIGHTS / np.linalg.norm(self.DEFAULT_WEIGHTS)

    def choose_act(self, mode, info=False):
        actions = self.env.action_space.list_of_actions
        if not actions:
            return None

        current_points = float(
            self.env.current_state_of_the_game.active_players_hand().number_of_my_points()
        )
        best_score = -1e9
        best_actions = []

        for action in actions:
            ae = action.evaluate(self.env.current_state_of_the_game)
            score = (
                np.floor((current_points + ae["card"][2]) / POINTS_TO_WIN) * self.weight[0]
                + self.weight[1] * ae["card"][2]
                + self.weight[2] * ae["nobles"]
                + self.weight[3] * ae["card"][0]
                + self.weight[4] * sum(ae["gems_flow"])
            )
            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)

        return random.choice(best_actions)
