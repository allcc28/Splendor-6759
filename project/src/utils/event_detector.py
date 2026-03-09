"""Heuristic event detection for event-based reward shaping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet

import numpy as np

from gym_splendor_code.envs.mechanics.action import (
    ActionBuyCard,
    ActionReserveCard,
    ActionTradeGems,
)
from gym_splendor_code.envs.mechanics.enums import GemColor
from gym_splendor_code.envs.mechanics.state import State
from reward.event_based_reward import EVENT_NAMES


GEM_ORDER = (
    GemColor.GOLD,
    GemColor.RED,
    GemColor.GREEN,
    GemColor.BLUE,
    GemColor.WHITE,
    GemColor.BLACK,
)
NON_GOLD_COLORS = GEM_ORDER[1:]


@dataclass(frozen=True)
class PlayerSnapshot:
    score: int
    gems: tuple[int, ...]
    discounts: tuple[int, ...]
    reserved_card_ids: FrozenSet[int]
    can_afford_board_card_ids: FrozenSet[int]
    can_afford_board_count: int
    purchased_card_count: int
    discount_total: int


@dataclass(frozen=True)
class StateSnapshot:
    player: PlayerSnapshot
    opponent: PlayerSnapshot
    board_gems: tuple[int, ...]


def capture_state_snapshot(state: State, player_id: int) -> StateSnapshot:
    """Capture the minimal state needed for event detection."""
    player_hand = state.list_of_players_hands[player_id]
    opponent_hand = state.list_of_players_hands[1 - player_id]
    board_cards = list(state.board.cards_on_board)

    def _player_snapshot(hand) -> PlayerSnapshot:
        discount = hand.discount()
        affordable_ids = frozenset(
            card.id for card in board_cards if hand.can_afford_card(card, discount=discount)
        )
        discounts = tuple(discount.value(color) for color in NON_GOLD_COLORS)
        return PlayerSnapshot(
            score=hand.number_of_my_points(),
            gems=tuple(hand.gems_possessed.value(color) for color in GEM_ORDER),
            discounts=discounts,
            reserved_card_ids=frozenset(card.id for card in hand.cards_reserved),
            can_afford_board_card_ids=affordable_ids,
            can_afford_board_count=len(affordable_ids),
            purchased_card_count=len(hand.cards_possessed),
            discount_total=sum(discounts),
        )

    return StateSnapshot(
        player=_player_snapshot(player_hand),
        opponent=_player_snapshot(opponent_hand),
        board_gems=tuple(state.board.gems_on_board.value(color) for color in GEM_ORDER),
    )


def detect_events(prev: StateSnapshot, action, next_state: StateSnapshot) -> np.ndarray:
    """
    Detect the 9 event features used by the event-based reward.

    These events are heuristic, not ground-truth semantic labels.
    """
    events = np.zeros(len(EVENT_NAMES), dtype=np.int32)

    is_take = isinstance(action, ActionTradeGems)
    is_buy = isinstance(action, ActionBuyCard)
    is_reserve = isinstance(action, ActionReserveCard)

    gem_gain = sum(next_state.player.gems) - sum(prev.player.gems)
    score_diff = next_state.player.score - prev.player.score
    bought_card = next_state.player.purchased_card_count > prev.player.purchased_card_count
    reserved_delta = len(next_state.player.reserved_card_ids) - len(prev.player.reserved_card_ids)

    if is_take or gem_gain > 0:
        events[0] = 1

    if is_buy or bought_card:
        events[1] = 1

    if is_reserve or reserved_delta > 0:
        events[2] = 1

    if score_diff > 0:
        events[3] = 1

    if next_state.player.score >= 15 and score_diff > 0:
        events[4] = 1

    if is_take and isinstance(action, ActionTradeGems):
        for index, color in enumerate(NON_GOLD_COLORS, start=1):
            taken = action.gems_from_board_to_player.value(color)
            if taken > 0 and prev.board_gems[index] <= 2:
                events[5] = 1
                break

    if is_reserve and isinstance(action, ActionReserveCard):
        reserved_card = getattr(action, "card", None)
        if reserved_card is not None:
            opponent_can_buy = reserved_card.id in prev.opponent.can_afford_board_card_ids
            late_game_denial = prev.opponent.score >= 12 and reserved_card.victory_points > 0
            if opponent_can_buy or late_game_denial:
                events[6] = 1

    if is_buy and isinstance(action, ActionBuyCard):
        bought_reserved = action.card.id in prev.player.reserved_card_ids
        if bought_reserved:
            events[7] = 1

        can_afford_gain = (
            next_state.player.can_afford_board_count - prev.player.can_afford_board_count
        )
        discount_gain = next_state.player.discount_total - prev.player.discount_total
        if score_diff >= 3 or (discount_gain > 0 and can_afford_gain > 0):
            events[8] = 1

    return events


__all__ = [
    "EVENT_NAMES",
    "PlayerSnapshot",
    "StateSnapshot",
    "capture_state_snapshot",
    "detect_events",
]
