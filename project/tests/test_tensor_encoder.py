"""Tests for SplendorTensorEncoder."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath("modules"))
sys.path.insert(0, os.path.abspath("project/src"))

from gym_splendor_code.envs.mechanics.state import State
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict

from nn.tensor_encoder import SplendorTensorEncoder


def test_tensor_encoder_shape_and_range() -> None:
    encoder = SplendorTensorEncoder()
    state = State()
    tensor = encoder.encode(state, player_id=0, turn_count=0)

    assert tensor.shape == (encoder.spec.channels, encoder.spec.height, encoder.spec.width)
    assert tensor.dtype == np.float32
    assert np.all(tensor >= 0.0)
    assert np.all(tensor <= 1.0)


def test_tensor_encoder_is_deterministic() -> None:
    encoder = SplendorTensorEncoder()
    state = State()

    tensor_a = encoder.encode(state, player_id=0, turn_count=7)
    tensor_b = encoder.encode(state, player_id=0, turn_count=7)

    assert np.allclose(tensor_a, tensor_b)


def test_tensor_encoder_reserved_identity_changes_encoding() -> None:
    encoder = SplendorTensorEncoder()
    state_a = State()
    state_b = StateAsDict(state_a).to_state()

    board_cards = sorted(list(state_a.board.cards_on_board), key=lambda card: card.id)
    assert len(board_cards) >= 2

    state_a.list_of_players_hands[0].cards_reserved = {board_cards[0]}
    state_b.list_of_players_hands[0].cards_reserved = {board_cards[1]}

    tensor_a = encoder.encode(state_a, player_id=0, turn_count=0)
    tensor_b = encoder.encode(state_b, player_id=0, turn_count=0)

    assert not np.allclose(tensor_a, tensor_b)
