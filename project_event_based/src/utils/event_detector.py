import numpy as np
from typing import Dict, Any


def detect_events(prev_vec: np.ndarray, action: Dict[str, Any], next_vec: np.ndarray) -> np.ndarray:
    """
    Detect 9 events from state transition.
    
    Vector structure (40-dim from state_vectorizer_event):
    [0:6]       board gems
    [6]         player score
    [7:13]      player gems
    [13:19]     player discounts
    [19]        player reserved count
    [20]        opponent score
    [21:27]     opponent gems
    [27:33]     opponent discounts
    [33]        opponent reserved count
    """
    ev = np.zeros(9, dtype=np.int32)
    
    # 1. Action type identification (adapt to class names like ActionTradeGems)
    a_name = type(action).__name__.lower()
    is_take = any(x in a_name for x in ['take', 'trade', 'tokens'])
    is_buy = 'buy' in a_name
    is_reserve = 'reserve' in a_name

    # 2. State feature extraction (ensure all variables are defined)
    # Previous state features
    player_score_prev = prev_vec[6]
    player_gems_prev = prev_vec[7:13]
    player_disco_prev = prev_vec[13:19]
    player_reserved_prev = prev_vec[19]
    opp_score_prev = prev_vec[20]

    # Next state features 
    board_prev, board_next = prev_vec[0:6], next_vec[0:6]
    player_score_next = next_vec[6]
    player_gems_next = next_vec[7:13]
    player_disco_next = next_vec[13:19] 
    player_reserved_next = next_vec[19]

    # Event detection logic

    # Event 0: Take/trade gems
    if is_take or (player_gems_next.sum() > player_gems_prev.sum()):
        ev[0] = 1

    # Event 1: Purchase development card
    if is_buy or (player_disco_next.sum() > player_disco_prev.sum()):
        ev[1] = 1

    # Event 2: reserve a card
    if is_reserve or (player_reserved_next > player_reserved_prev):
        ev[2] = 1

    # Event 3 & 4: Score gain and victory condition
    score_diff = player_score_next - player_score_prev
    if score_diff > 0: ev[3] = 1
    if player_score_next >= 15: ev[4] = 1

    # Event 5-8: Advanced strategic events (depend on basic events)
    if ev[0]: # Grab scarce resources
        if np.any(board_next[:5] < 4): ev[5] = 1

    # Event 6: Malicious blocking — reserve the card that opponent is closest to buying
    # Condition: player reserved a card AND opponent's gap for that exact card is small (≤2 gems away)
    if ev[2] and hasattr(action, 'card') and action.card is not None:
        from gym_splendor_code.envs.mechanics.enums import GemColor
        # prev_vec[21:27] = opp gems in enum order [GOLD,RED,GREEN,BLUE,WHITE,BLACK]
        # prev_vec[27:33] = opp discounts in the same order
        OPP_GEMS_BASE = 21
        OPP_DISC_BASE = 27
        opp_gold = prev_vec[OPP_GEMS_BASE + GemColor.GOLD.value]
        card_cost = action.card.price.gems_dict
        total_gap = 0
        for c in [GemColor.RED, GemColor.GREEN, GemColor.BLUE, GemColor.WHITE, GemColor.BLACK]:
            cost = card_cost.get(c, 0)
            opp_has = prev_vec[OPP_GEMS_BASE + c.value] + prev_vec[OPP_DISC_BASE + c.value]
            total_gap += max(0, cost - opp_has)
        total_gap = max(0, total_gap - opp_gold)  # opponent's gold covers part of gap
        if total_gap <= 2:
            ev[6] = 1
    elif ev[2]:
        # Fallback when action has no card info (shouldn't happen but keeps backward compat)
        if opp_score_prev >= 10:
            ev[6] = 1

    if ev[1] and player_reserved_next<player_reserved_prev: ev[7] = 1 # Clear reserved card inventory (buy using reserved card)
    if score_diff >= 3: ev[8] = 1 # Explosive scoring (gained ≥3 points in one step)

    return ev


if __name__ == '__main__':
    import numpy as _n
    pv = _n.zeros(40, dtype=_n.float32)
    nv = pv.copy()
    pv[0:6] = _n.array([4, 2, 2, 3, 1, 1])
    nv[0:6] = _n.array([3, 2, 2, 3, 1, 1])
    pv[6] = 5
    nv[6] = 8
    pv[7:13] = _n.array([1, 0, 0, 0, 0, 0])
    nv[7:13] = _n.array([1, 0, 0, 0, 0, 0])
    pv[19] = 1
    nv[19] = 0
    a = {'type': 'buy_card'}
    print(detect_events(pv, a, nv))
