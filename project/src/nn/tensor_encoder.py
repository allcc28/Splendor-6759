"""Tensor encoder for AlphaZero-style Splendor models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gym_splendor_code.envs.mechanics.enums import GemColor, Row
from gym_splendor_code.envs.mechanics.state import State


@dataclass(frozen=True)
class TensorEncodingSpec:
    """Static metadata for the fixed tensor encoding."""

    channels: int = 80
    height: int = 3
    width: int = 4


class SplendorTensorEncoder:
    """Encode a Splendor state into a compact board-centric tensor.

    Output shape is ``(50, 3, 4)``:
    - Spatial dimensions are board card slots arranged as rows x columns.
    - Global features are broadcast across the board plane.
    - Card-local features are placed at each occupied slot.
    """

    GEM_COLORS = [
        GemColor.RED,
        GemColor.GREEN,
        GemColor.BLUE,
        GemColor.WHITE,
        GemColor.BLACK,
    ]
    CARD_ROWS = [Row.CHEAP, Row.MEDIUM, Row.EXPENSIVE]

    MAX_GEMS_ON_BOARD = 7.0
    MAX_GEMS_ON_HAND = 10.0
    MAX_DISCOUNT = 10.0
    MAX_POINTS = 20.0
    MAX_CARD_COST = 7.0
    MAX_CARD_VP = 5.0
    MAX_CARD_ID = 90.0
    MAX_TURNS = 120.0

    def __init__(self) -> None:
        self.spec = TensorEncodingSpec()
        self.output_shape = (self.spec.channels, self.spec.height, self.spec.width)

    def encode(self, state: State, player_id: int, turn_count: int = 0) -> np.ndarray:
        """Encode state from the requested player's perspective."""
        tensor = np.zeros(self.output_shape, dtype=np.float32)

        active_hand = state.list_of_players_hands[player_id]
        opponent_hand = state.list_of_players_hands[1 - player_id]

        # Global planes: board gems, both players' gems/discounts, scores, progress.
        self._fill_global_planes(tensor, state, active_hand, opponent_hand, turn_count)

        # Board card slots (3x4) with deterministic ordering by card id in each row.
        sorted_cards = self._sorted_board_cards(state)
        self._fill_card_planes(tensor, sorted_cards, active_hand)

        # Noble pressure + reserve summary planes.
        self._fill_summary_planes(tensor, state, active_hand, opponent_hand)

        # Reserved card identity planes (prevents aliasing distinct reserved sets).
        self._fill_reserved_identity_planes(tensor, active_hand, opponent_hand)

        return np.clip(tensor, 0.0, 1.0)

    def _fill_global_planes(self, tensor: np.ndarray, state: State, active_hand, opponent_hand, turn_count: int) -> None:
        for i, color in enumerate(self.GEM_COLORS):
            tensor[i, :, :] = state.board.gems_on_board.value(color) / self.MAX_GEMS_ON_BOARD
            tensor[5 + i, :, :] = active_hand.gems_possessed.value(color) / self.MAX_GEMS_ON_HAND
            tensor[10 + i, :, :] = opponent_hand.gems_possessed.value(color) / self.MAX_GEMS_ON_HAND

        active_discount = active_hand.discount()
        opp_discount = opponent_hand.discount()
        for i, color in enumerate(self.GEM_COLORS):
            tensor[15 + i, :, :] = active_discount.value(color) / self.MAX_DISCOUNT
            tensor[20 + i, :, :] = opp_discount.value(color) / self.MAX_DISCOUNT

        tensor[25, :, :] = active_hand.number_of_my_points() / self.MAX_POINTS
        tensor[26, :, :] = opponent_hand.number_of_my_points() / self.MAX_POINTS
        tensor[27, :, :] = min(turn_count, self.MAX_TURNS) / self.MAX_TURNS

    def _fill_card_planes(self, tensor: np.ndarray, sorted_cards, active_hand) -> None:
        for row_idx, row_cards in enumerate(sorted_cards):
            for col_idx, card in enumerate(row_cards):
                if card is None:
                    continue

                if card.discount_profit in self.GEM_COLORS:
                    channel = 28 + self.GEM_COLORS.index(card.discount_profit)
                    tensor[channel, row_idx, col_idx] = 1.0

                tensor[33, row_idx, col_idx] = card.victory_points / self.MAX_CARD_VP
                for i, color in enumerate(self.GEM_COLORS):
                    tensor[34 + i, row_idx, col_idx] = card.price.value(color) / self.MAX_CARD_COST

                tensor[39, row_idx, col_idx] = 1.0 if active_hand.can_afford_card(card) else 0.0
                tensor[40, row_idx, col_idx] = 1.0

    def _fill_summary_planes(self, tensor: np.ndarray, state: State, active_hand, opponent_hand) -> None:
        noble_pressure = [0.0] * len(self.GEM_COLORS)
        nobles = sorted(state.board.nobles_on_board, key=lambda noble: noble.id)
        if nobles:
            for noble in nobles:
                for i, color in enumerate(self.GEM_COLORS):
                    noble_pressure[i] += noble.price.value(color)
            noble_pressure = [min(value / (len(nobles) * 4.0), 1.0) for value in noble_pressure]

        for i, value in enumerate(noble_pressure):
            tensor[41 + i, :, :] = value

        tensor[46, :, :] = len(state.board.nobles_on_board) / 3.0
        tensor[47, :, :] = len(active_hand.cards_reserved) / 3.0
        tensor[48, :, :] = len(opponent_hand.cards_reserved) / 3.0
        tensor[49, :, :] = active_hand.gems_possessed.value(GemColor.GOLD) / self.MAX_GEMS_ON_HAND

    def _fill_reserved_identity_planes(self, tensor: np.ndarray, active_hand, opponent_hand) -> None:
        active_reserved = sorted(active_hand.cards_reserved, key=lambda card: card.id)[:3]
        opp_reserved = sorted(opponent_hand.cards_reserved, key=lambda card: card.id)[:3]

        # 5 planes per reserved slot: presence, row, discount, vp, card_id.
        # Active slots: channels 50..64 (3 * 5), Opponent slots: 65..79 (3 * 5).
        self._write_reserved_slots(tensor, base_channel=50, reserved_cards=active_reserved)
        self._write_reserved_slots(tensor, base_channel=65, reserved_cards=opp_reserved)

    def _write_reserved_slots(self, tensor: np.ndarray, base_channel: int, reserved_cards) -> None:
        for slot in range(3):
            channel = base_channel + slot * 5
            if slot >= len(reserved_cards):
                continue

            card = reserved_cards[slot]
            tensor[channel + 0, :, :] = 1.0
            tensor[channel + 1, :, :] = (self.CARD_ROWS.index(card.row) + 1) / 3.0

            if card.discount_profit in self.GEM_COLORS:
                discount_norm = (self.GEM_COLORS.index(card.discount_profit) + 1) / len(self.GEM_COLORS)
            else:
                discount_norm = 0.0
            tensor[channel + 2, :, :] = discount_norm

            tensor[channel + 3, :, :] = card.victory_points / self.MAX_CARD_VP
            tensor[channel + 4, :, :] = card.id / self.MAX_CARD_ID

    def _sorted_board_cards(self, state: State):
        cards_by_row = {row: [] for row in self.CARD_ROWS}
        for card in state.board.cards_on_board:
            cards_by_row[card.row].append(card)

        sorted_cards = []
        for row in self.CARD_ROWS:
            cards = sorted(cards_by_row[row], key=lambda card: card.id)[:4]
            cards.extend([None] * (4 - len(cards)))
            sorted_cards.append(cards)
        return sorted_cards
