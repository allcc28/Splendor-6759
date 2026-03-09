import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath("modules"))
sys.path.insert(0, os.path.abspath("project/src"))
sys.path.insert(0, os.path.abspath("project/src/reward"))
sys.path.insert(0, os.path.abspath("project/src/utils"))

from gym_splendor_code.envs.mechanics.action import ActionBuyCard
from event_based_reward import compute_event_reward
from event_detector import PlayerSnapshot, StateSnapshot, detect_events


def test_compute_event_reward_custom_weights():
    events = np.array([1, 0, 1, 0, 0, 0, 0, 0, 1], dtype=np.int32)
    weights = np.array([0.5, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0], dtype=np.float32)
    assert compute_event_reward(events, weights=weights) == 1.5


def test_detect_events_flags_buy_reserved():
    prev = StateSnapshot(
        player=PlayerSnapshot(
            score=10,
            gems=(0, 0, 0, 0, 0, 0),
            discounts=(1, 1, 1, 1, 1),
            reserved_card_ids=frozenset({42}),
            can_afford_board_card_ids=frozenset({42}),
            can_afford_board_count=1,
            purchased_card_count=5,
            discount_total=5,
        ),
        opponent=PlayerSnapshot(
            score=9,
            gems=(0, 0, 0, 0, 0, 0),
            discounts=(0, 0, 0, 0, 0),
            reserved_card_ids=frozenset(),
            can_afford_board_card_ids=frozenset(),
            can_afford_board_count=0,
            purchased_card_count=4,
            discount_total=0,
        ),
        board_gems=(0, 4, 4, 4, 4, 4),
    )
    next_state = StateSnapshot(
        player=PlayerSnapshot(
            score=12,
            gems=(0, 0, 0, 0, 0, 0),
            discounts=(1, 1, 2, 1, 1),
            reserved_card_ids=frozenset(),
            can_afford_board_card_ids=frozenset({10, 11}),
            can_afford_board_count=2,
            purchased_card_count=6,
            discount_total=6,
        ),
        opponent=prev.opponent,
        board_gems=prev.board_gems,
    )

    class DummyCard:
        id = 42
        victory_points = 2

    action = ActionBuyCard(DummyCard())
    event_vec = detect_events(prev, action, next_state)

    assert event_vec[1] == 1
    assert event_vec[3] == 1
    assert event_vec[7] == 1


def test_event_reward_wrapper_extends_observation():
    pytest.importorskip("gymnasium")
    from utils.event_reward_wrapper import EventRewardWrapper
    from utils.splendor_gym_wrapper import SplendorGymWrapper

    base_env = SplendorGymWrapper(reward_mode="score_progress")
    env = EventRewardWrapper(
        base_env,
        {
            "combine_with_score": True,
            "include_gem_gaps": True,
            "append_last_event": True,
        },
    )

    try:
        obs, info = env.reset()
        assert obs.shape == (204,)
        assert obs.dtype == np.float32

        next_obs, reward, terminated, truncated, info = env.step(0)
        assert next_obs.shape == (204,)
        assert isinstance(reward, float)
        assert "last_event" in info
        assert info["last_event"].shape == (9,)
    finally:
        env.close()
