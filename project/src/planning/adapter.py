"""Shared Splendor planning adapter for MCTS-based algorithms.

This adapter wraps the native game State and provides a clean interface for
AlphaZero and MuZero search/training code. It implements correct terminal
logic (let_all_move: finish the round when someone reaches 15 points) and
public-information observation encoding.

Reference: cestpasphoto/alpha-zero-general SplendorGame.py for game interface pattern.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from gym_splendor_code.envs.mechanics.action import Action
from gym_splendor_code.envs.mechanics.action_space_generator import (
    generate_all_legal_buys,
    generate_all_legal_reservations,
    generate_all_legal_trades,
)
from gym_splendor_code.envs.mechanics.game_settings import (
    MAX_NUMBER_OF_MOVES,
    POINTS_TO_WIN,
)
from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict

from nn.tensor_encoder import SplendorTensorEncoder


class SplendorPlanningAdapter:
    """Unified game-logic wrapper for planning algorithms.

    Terminal semantics: ``let_all_move`` — once any player reaches
    ``POINTS_TO_WIN`` (15), the current *round* finishes so every player
    gets an equal number of turns, then the winner is resolved.

    Tie-break: highest points wins.  If tied, fewest purchased cards wins.
    If still tied, the result is a draw.
    """

    def __init__(
        self,
        state: Optional[State] = None,
        encoder: Optional[SplendorTensorEncoder] = None,
        include_opponent_reserved_identity: bool = False,
    ) -> None:
        self._state = state if state is not None else State()
        self._encoder = encoder if encoder is not None else SplendorTensorEncoder()
        self._include_opp_reserved = include_opponent_reserved_identity

        # Track turn counts per player for let_all_move round completion.
        self._turn_counts: List[int] = [0, 0]
        # Set when any player first reaches POINTS_TO_WIN.
        self._final_round_triggered: bool = False
        # The turn count of the *first* player at the moment the trigger fires.
        # We finish the round when all players have taken this many turns.
        self._final_round_target: Optional[int] = None
        # Total moves for MAX_NUMBER_OF_MOVES safety cap.
        self._total_moves: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def state(self) -> State:
        """Direct access to the underlying State (read-only intent)."""
        return self._state

    @property
    def current_player(self) -> int:
        return self._state.active_player_id

    @property
    def num_players(self) -> int:
        return len(self._state.list_of_players_hands)

    @property
    def legal_actions(self) -> List[Action]:
        """All legal actions for the current player."""
        trades = generate_all_legal_trades(self._state)
        buys = generate_all_legal_buys(self._state)
        reserves = generate_all_legal_reservations(self._state)
        return trades + buys + reserves

    def step(self, action: Action) -> None:
        """Execute *action* and advance to the next player.

        The action's ``execute()`` already changes the active player.
        """
        player_before = self._state.active_player_id
        action.execute(self._state)
        self._turn_counts[player_before] += 1
        self._total_moves += 1

        # Check if any player just reached POINTS_TO_WIN.
        if not self._final_round_triggered:
            for hand in self._state.list_of_players_hands:
                if hand.number_of_my_points() >= POINTS_TO_WIN:
                    self._final_round_triggered = True
                    # The round ends when all players have taken the same
                    # number of turns as the maximum turn count at trigger time.
                    self._final_round_target = max(self._turn_counts)
                    break

    def is_terminal(self) -> bool:
        """True when the game is over.

        Terminal conditions (in priority order):
        1. Safety cap: total moves >= MAX_NUMBER_OF_MOVES.
        2. No legal actions for the current player.
        3. ``let_all_move``: final round was triggered AND all players
           have taken ``_final_round_target`` turns.
        """
        if self._total_moves >= MAX_NUMBER_OF_MOVES:
            return True

        if self._final_round_triggered:
            # All players must have taken at least _final_round_target turns.
            return all(
                tc >= self._final_round_target for tc in self._turn_counts
            )

        # Also terminal if current player has no legal actions (rare edge case).
        if not self.legal_actions:
            return True

        return False

    def terminal_value(self, player_id: int) -> float:
        """Return +1 / -1 / 0 from *player_id*'s perspective.

        Tie-break: highest points → fewest purchased cards → draw.
        """
        points = [
            hand.number_of_my_points()
            for hand in self._state.list_of_players_hands
        ]
        cards_count = [
            len(hand.cards_possessed)
            for hand in self._state.list_of_players_hands
        ]

        my_pts = points[player_id]
        opp_pts = points[1 - player_id]

        if my_pts > opp_pts:
            return 1.0
        if my_pts < opp_pts:
            return -1.0

        # Tie on points — fewest cards wins.
        my_cards = cards_count[player_id]
        opp_cards = cards_count[1 - player_id]
        if my_cards < opp_cards:
            return 1.0
        if my_cards > opp_cards:
            return -1.0

        return 0.0

    def winner_id(self) -> Optional[int]:
        """Return the winner's player_id, or None for a draw."""
        points = [
            hand.number_of_my_points()
            for hand in self._state.list_of_players_hands
        ]
        cards_count = [
            len(hand.cards_possessed)
            for hand in self._state.list_of_players_hands
        ]

        if points[0] > points[1]:
            return 0
        if points[1] > points[0]:
            return 1

        # Tie on points — fewest cards wins.
        if cards_count[0] < cards_count[1]:
            return 0
        if cards_count[1] < cards_count[0]:
            return 1

        return None

    def encode_observation(self, player_id: int, turn_count: int = 0) -> np.ndarray:
        """Public-information tensor encoding by default.

        If ``include_opponent_reserved_identity`` was set to True at
        construction, opponent reserved card channels are kept.
        """
        tensor = self._encoder.encode(
            self._state,
            player_id=player_id,
            turn_count=turn_count,
        )
        if not self._include_opp_reserved:
            # Zero out opponent reserved identity channels 65-79.
            tensor[65:80, :, :] = 0.0
        return tensor

    def clone(self) -> SplendorPlanningAdapter:
        """Deep-copy for MCTS rollouts.

        Uses StateAsDict round-trip for safe state cloning.
        """
        state_dict = StateAsDict(self._state)
        cloned_state = state_dict.to_state()
        adapter = SplendorPlanningAdapter(
            state=cloned_state,
            encoder=self._encoder,
            include_opponent_reserved_identity=self._include_opp_reserved,
        )
        adapter._turn_counts = list(self._turn_counts)
        adapter._final_round_triggered = self._final_round_triggered
        adapter._final_round_target = self._final_round_target
        adapter._total_moves = self._total_moves
        return adapter

    def reset(self) -> None:
        """Reset to a fresh game."""
        self._state = State()
        self._turn_counts = [0, 0]
        self._final_round_triggered = False
        self._final_round_target = None
        self._total_moves = 0

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def points(self) -> List[int]:
        """Return [player_0_points, player_1_points]."""
        return [
            hand.number_of_my_points()
            for hand in self._state.list_of_players_hands
        ]

    def __repr__(self) -> str:
        pts = self.points()
        return (
            f"SplendorPlanningAdapter(player={self.current_player}, "
            f"points={pts}, moves={self._total_moves}, "
            f"final_round={self._final_round_triggered})"
        )
