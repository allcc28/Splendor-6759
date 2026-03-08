import numpy as np


def vectorize_state_event(state: dict, active_player_index: int = 0) -> np.ndarray:
    """
    Build a 40-dim state vector as specified in the request.
    """
    vec = np.zeros(40, dtype=np.float32)

    # Board gems (0-5)
    board_gems = list(state.get('board', {}).get('gems', [0] * 6))
    if len(board_gems) < 6:
        board_gems = (board_gems + [0] * 6)[:6]
    vec[0:6] = np.array(board_gems, dtype=np.float32)

    def player_features(p):
        score = float(p.get('score', 0))
        gems = list(p.get('gems', [0] * 6))
        if len(gems) < 6:
            gems = (gems + [0] * 6)[:6]
        discounts = list(p.get('discounts', [0] * 6))
        if len(discounts) < 6:
            discounts = (discounts + [0] * 6)[:6]
        reserved = float(p.get('reserved_count', 0))
        return score, gems, discounts, reserved

    players = state.get('players', [])
    if len(players) < 2:
        while len(players) < 2:
            players.append({'score': 0, 'gems': [0] * 6, 'discounts': [0] * 6, 'reserved_count': 0})

    ap = players[active_player_index]
    op = players[1 - active_player_index]

    score, gems, discounts, reserved = player_features(ap)
    vec[6] = score
    vec[7:13] = np.array(gems, dtype=np.float32)
    vec[13:19] = np.array(discounts, dtype=np.float32)
    vec[19] = reserved

    score_o, gems_o, discounts_o, reserved_o = player_features(op)
    vec[20] = score_o
    vec[21:27] = np.array(gems_o, dtype=np.float32)
    vec[27:33] = np.array(discounts_o, dtype=np.float32)
    vec[33] = reserved_o

    return vec


if __name__ == '__main__':
    example = {
        'board': {'gems': [4, 3, 2, 4, 1, 1]},
        'players': [
            {'score': 5, 'gems': [1, 0, 2, 0, 0, 0], 'discounts': [0, 1, 0, 0, 0, 0], 'reserved_count': 1},
            {'score': 8, 'gems': [0, 1, 0, 0, 0, 0], 'discounts': [0, 0, 0, 0, 0, 0], 'reserved_count': 0},
        ]
    }
    print(vectorize_state_event(example))
