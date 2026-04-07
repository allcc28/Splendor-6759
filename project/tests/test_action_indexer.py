"""Tests for StableActionIndexer."""

import os
import sys

sys.path.insert(0, os.path.abspath("modules"))
sys.path.insert(0, os.path.abspath("project/src"))

from gym_splendor_code.envs.mechanics.action_space_generator import generate_all_legal_actions
from gym_splendor_code.envs.mechanics.state import State

from mcts.action_indexer import StableActionIndexer


def test_action_indexer_is_deterministic_for_same_action() -> None:
    state = State()
    actions = generate_all_legal_actions(state)
    assert len(actions) > 0

    indexer = StableActionIndexer(policy_size=256)
    first_action = actions[0]

    idx1 = indexer.action_index(first_action)
    idx2 = indexer.action_index(first_action)
    assert idx1 == idx2


def test_action_indexer_indices_within_policy_size() -> None:
    state = State()
    actions = generate_all_legal_actions(state)
    indexer = StableActionIndexer(policy_size=256)

    indices = indexer.legal_indices(actions)
    assert len(indices) == len(actions)
    assert int(indices.min()) >= 0
    assert int(indices.max()) < 256


def test_action_indexer_no_collisions_within_legal_set() -> None:
    state = State()
    actions = generate_all_legal_actions(state)
    indexer = StableActionIndexer(policy_size=2048)

    indices = indexer.legal_indices(actions)
    assert len(set(int(x) for x in indices)) == len(actions)


def test_action_indexer_trade_quantities_are_distinct() -> None:
    """Regression: gems_flow must encode quantities, not just colors.

    to_dict() returns a list of counts per color in fixed GemColor order,
    so two trades with the same colors but different amounts must get
    different policy-head indices.
    """
    from gym_splendor_code.envs.mechanics.action import ActionTradeGems
    from gym_splendor_code.envs.mechanics.gems_collection import GemsCollection
    from gym_splendor_code.envs.mechanics.enums import GemColor

    take_2_red = ActionTradeGems(GemsCollection({
        GemColor.GOLD: 0, GemColor.RED: 2, GemColor.GREEN: 0,
        GemColor.BLUE: 0, GemColor.WHITE: 0, GemColor.BLACK: 0,
    }))
    take_1_red = ActionTradeGems(GemsCollection({
        GemColor.GOLD: 0, GemColor.RED: 1, GemColor.GREEN: 0,
        GemColor.BLUE: 0, GemColor.WHITE: 0, GemColor.BLACK: 0,
    }))

    indexer = StableActionIndexer(policy_size=2048)
    idx_2 = indexer.action_index(take_2_red)
    idx_1 = indexer.action_index(take_1_red)

    assert idx_2 != idx_1, "Different gem quantities must map to different policy indices"
