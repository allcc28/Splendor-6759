import copy
import numpy as np
from gym_splendor_code.envs.mechanics.game_settings import POINTS_TO_WIN
from gym_splendor_code.envs.mechanics.state_as_dict import StateAsDict


# -----------------------------
# Helpers (robust feature access)
# -----------------------------
def _get_attr(obj, *names, default=None):
    """Try multiple attribute names and return the first existing one."""
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def _as_number(x, default=0.0):
    """Convert common numeric wrapper types to float."""
    if x is None:
        return float(default)
    if hasattr(x, "value"):
        try:
            return float(x.value)
        except Exception:
            pass
    try:
        return float(x)
    except Exception:
        return float(default)


def _sum_tokens(x, default=0.0):
    """Sum tokens stored as list/array or dict; return default if unknown."""
    if x is None:
        return float(default)
    if isinstance(x, dict):
        return float(sum(x.values()))
    try:
        return float(np.sum(x))
    except Exception:
        return float(default)


def normalize(w):
    """L2-normalize a weight vector (returns numpy array)."""
    w = np.array(w, dtype=np.float32)
    n = np.linalg.norm(w)
    return (w / n) if n > 0 else w


# -----------------------------
# State simulation
# -----------------------------
def simulate_next_state(current_state, action):
    """
    Create a copy of the state and execute the action on it.
    StateAsDict round-trip is usually safer than deepcopy for this env.
    """
    try:
        s_copy = StateAsDict(current_state).to_state()
    except Exception:
        s_copy = copy.deepcopy(current_state)
    action.execute(s_copy)
    return s_copy


# -----------------------------
# State feature extraction (for value-based)
# -----------------------------
def get_active_hand(state):
    """Return the active player's hand object."""
    if hasattr(state, "active_players_hand"):
        return state.active_players_hand()
    # Fallbacks (rare): try to access via players list
    pid = _get_attr(state, "active_player_id", default=0)
    players = _get_attr(state, "players", default=None)
    if players is not None and len(players) > int(pid):
        p = players[int(pid)]
        return _get_attr(p, "hand", "player_hand", "players_hand", default=p)
    return None


def get_points(hand):
    """Victory points of the given hand."""
    if hand is None:
        return 0.0
    if hasattr(hand, "number_of_my_points"):
        try:
            return float(hand.number_of_my_points())
        except Exception:
            pass
    return _as_number(_get_attr(hand, "victory_points", "points", "score", default=0), default=0.0)


def get_owned_nobles(hand):
    """Number of nobles owned by the player (if the env stores them)."""
    if hand is None:
        return 0.0
    nobles = _get_attr(hand, "nobles", "owned_nobles", "noble_tiles", default=None)
    if nobles is None:
        return 0.0
    try:
        return float(len(nobles))
    except Exception:
        return _as_number(nobles, default=0.0)


def get_reserved_count(hand):
    """Number of reserved cards (if stored)."""
    if hand is None:
        return 0.0
    reserved = _get_attr(hand, "reserved_cards", "reserved", "reserve", default=None)
    if reserved is None:
        return 0.0
    try:
        return float(len(reserved))
    except Exception:
        return _as_number(reserved, default=0.0)


def get_tokens_total(hand):
    """Total number of tokens/gems currently held."""
    if hand is None:
        return 0.0
    gems = _get_attr(hand, "gems", "tokens", "chips", default=None)
    return _sum_tokens(gems, default=0.0)


def get_engine_strength(hand):
    """
    Proxy for long-term power (permanent discounts / bonuses).
    Prefer explicit bonuses/discounts; fallback to number of bought cards.
    """
    if hand is None:
        return 0.0

    bonuses = _get_attr(hand, "bonuses", "discounts", "bonus", default=None)
    if bonuses is not None:
        return _sum_tokens(bonuses, default=0.0)

    cards = _get_attr(hand, "cards", "bought_cards", "purchased_cards", "development_cards", default=None)
    if cards is None:
        return 0.0
    if isinstance(cards, dict):
        try:
            return float(sum(len(v) for v in cards.values()))
        except Exception:
            return 0.0
    try:
        return float(len(cards))
    except Exception:
        return 0.0


# ============================================================
# 1) VALUE-BASED (STATE-BASED) evaluator: V(next_state)
# ============================================================
class ValueBasedEvaluator:
    """
    Value-based / score-based evaluator (true state-based):
    - Scores the NEXT STATE after applying an action.
    - Does NOT use action.evaluate(...) event dict.
    - Intended to approximate V(s): overall state quality.
    """

    def __init__(self, weights=None):
        # Feature weights: [win_bonus, points, nobles_owned, engine_strength, tokens_total, reserved_penalty]
        default = [500.0, 10.0, 30.0, 3.0, 0.5, -0.5]
        self.w = normalize(weights if weights is not None else default)

    def score_next_state(self, next_state) -> float:
        hand = get_active_hand(next_state)
        points = get_points(hand)
        nobles_owned = get_owned_nobles(hand)
        engine = get_engine_strength(hand)
        tokens = get_tokens_total(hand)
        reserved = get_reserved_count(hand)

        win = 1.0 if points >= POINTS_TO_WIN else 0.0

        return float(
            self.w[0] * win +
            self.w[1] * points +
            self.w[2] * nobles_owned +
            self.w[3] * engine +
            self.w[4] * tokens +
            self.w[5] * reserved
        )


# ============================================================
# 2) EVENT-BASED (TRANSITION-BASED) evaluator: R_event(s,a,s')
# ============================================================
class EventBasedEvaluator:
    """
    Event-based evaluator (true event-based):
    - Scores ONLY what happened in this action, using `ae = action.evaluate(state)`.
    - Does NOT read overall next_state strength (engine/tokens in hand, etc.).
    - This aligns with event-value style evaluation.
    """

    def __init__(self, event_weights=None):
        # You can tune these later.
        default = dict(
            win=500.0,
            buy_card=30.0,
            reserve_card=5.0,
            gain_points=80.0,     # points gained from the purchased card (not including noble bonus)
            gain_noble=200.0,
            token_gain=1.0,
            token_spend=1.5,
            overflow_penalty=20.0,  # penalize token overflow risk (approximate)
        )
        self.ew = default if event_weights is None else {**default, **event_weights}

    def score_event(self, current_points: int, ae: dict) -> float:
        """
        Score the action's event dict.
        Expected keys often include: "card", "nobles", "gems_flow", and for reserve: "card_booked".
        This function is robust if some keys are missing.
        """
        card = ae.get("card", [0, None, 0])
        nobles_gained = float(ae.get("nobles", 0))
        gems_flow = ae.get("gems_flow", 0.0)

        # Identify action types via ae fields
        is_buy = 1.0 if (len(card) > 0 and float(card[0]) == 1.0) else 0.0
        is_reserve = 1.0 if ("card_booked" in ae and isinstance(ae["card_booked"], (list, tuple)) and len(ae["card_booked"]) > 0 and float(ae["card_booked"][0]) == 1.0) else 0.0

        # Points gained:
        # Some implementations pack noble bonus into card[2]. To keep "points" and "noble" separated,
        # we subtract 3*nobles if this was a buy (common Splendor scoring: noble=3 points).
        raw_card_points = float(card[2]) if len(card) > 2 else 0.0
        points_from_card = raw_card_points - (3.0 * nobles_gained if is_buy > 0 else 0.0)
        points_from_card = max(0.0, points_from_card)

        # Token net change from gems_flow (positive = gain, negative = spend/return)
        try:
            token_net = float(np.sum(gems_flow))
        except Exception:
            token_net = 0.0

        # Win event (if this action reaches the winning threshold)
        win = 1.0 if (current_points + points_from_card + 3.0 * nobles_gained) >= POINTS_TO_WIN else 0.0

        # Overflow approximation: if token_net is big positive, it may push over the 10-token limit.
        # We cannot know exact hand size here without state, so this is a light proxy.
        overflow_risk = max(0.0, token_net - 3.0)  # heuristic: taking "too many" tokens is risky

        r = 0.0
        r += self.ew["win"] * win
        r += self.ew["buy_card"] * is_buy
        r += self.ew["reserve_card"] * is_reserve
        r += self.ew["gain_points"] * points_from_card
        r += self.ew["gain_noble"] * nobles_gained
        r += self.ew["token_gain"] * max(0.0, token_net)
        r -= self.ew["token_spend"] * max(0.0, -token_net)
        r -= self.ew["overflow_penalty"] * overflow_risk
        return float(r)
