# State Vector Representation Specification

**Author**: AI Agent  
**Date**: 2026-02-24  
**Version**: 1.0  
**Status**: Draft

## 1. Overview

This document specifies the fixed-size state vector representation for the Splendor Score-based RL agent. The vector will be consumed by the PPO policy network (MLP) and must capture all relevant game information while maintaining a consistent shape.

## 2. Design Principles

- **Fixed Size**: Vector size must be constant regardless of game state
- **Normalized**: All values scaled to reasonable ranges (typically [0, 1] or [-1, 1])
- **Float32**: Use float32 dtype for PyTorch compatibility
- **Complete**: Capture all information relevant to score-based decision making
- **Efficient**: ~150-200 dimensions for fast MLP processing

## 3. State Vector Structure

### 3.1 Active Player Hand (Self) - 35 dimensions

| Feature | Dimensions | Range | Description |
|---------|-----------|-------|-------------|
| Gems possessed | 6 | [0, 1] | (Gold, Red, Green, Blue, White, Black) / 10 |
| Card discounts | 5 | [0, 1] | Count of cards providing each discount (excl. Gold) / 10 |
| Reserved cards count | 1 | [0, 1] | Number of reserved cards / 3 |
| Victory points | 1 | [0, 1] | Current VP / 20 (normalized beyond win threshold) |
| Nobles possessed | 1 | [0, 1] | Count of nobles / 3 |
| Can afford count | 1 | [0, 1] | # of cards on board player can afford / 12 |
| Reserved cards features | 20 | [0, 1] | Top 3 reserved cards (row, discount, VP, 5 price gems) × 3 + padding |

**Rationale**: The active player's hand is the most important for decision-making. We track both raw resources and derived features (can_afford_count) to help the policy learn faster.

### 3.2 Opponent Hand - 14 dimensions

| Feature | Dimensions | Range | Description |
|---------|-----------|-------|-------------|
| Gems possessed | 6 | [0, 1] | Opponent's gems / 10 |
| Card discounts | 5 | [0, 1] | Count of discount cards / 10 |
| Victory points | 1 | [0, 1] | Opponent VP / 20 |
| Nobles possessed | 1 | [0, 1] | Count of nobles / 3 |
| Reserved cards count | 1 | [0, 1] | Number of reserved cards / 3 |

**Rationale**: Opponent state is important but less detailed than self. We omit specific reserved card features to keep dimensionality manageable.

### 3.3 Board State - 108 dimensions

#### Gems on Board - 6 dimensions
| Feature | Dimensions | Range | Description |
|---------|-----------|-------|-------------|
| Available gems | 6 | [0, 1] | (Gold, Red, Green, Blue, White, Black) / 7 |

**Rationale**: Normalized by max initial value (7 for colored gems, 5 for gold).

#### Cards on Board - 96 dimensions (12 cards × 8 features)

For each of the 12 cards on board:
| Feature | Dimensions | Range | Description |
|---------|-----------|-------|-------------|
| Row indicator | 3 | {0, 1} | One-hot: [CHEAP, MEDIUM, EXPENSIVE] |
| Discount profit | 5 | {0, 1} | One-hot: [Red, Green, Blue, White, Black] |
| Victory points | 1 | [0, 1] | VP / 5 (max VP is 5) |
| Price | 5 | [0, 1] | (Red, Green, Blue, White, Black) / 7 |

**Card Ordering**: Fixed positions [0-3: CHEAP, 4-7: MEDIUM, 8-11: EXPENSIVE]. Empty positions filled with zeros.

**Rationale**: One-hot encoding for categorical features (row, discount) ensures the network learns their discrete nature. Price and VP are normalized continuous values.

#### Nobles on Board - 6 dimensions (3 nobles × 2 features)

For each of the 3 nobles:
| Feature | Dimensions | Range | Description |
|---------|-----------|-------|-------------|
| Requirement gems count | 1 | [0, 1] | Total discount cards needed / 10 |
| Can obtain now | 1 | {0, 1} | Binary: active player meets requirements |

**Noble Ordering**: Arbitrary but fixed within an episode. Empty positions filled with zeros.

**Rationale**: Simplified noble representation - focus on obtainability rather than full requirement details.

### 3.4 Game Progress - 2 dimensions

| Feature | Dimensions | Range | Description |
|---------|-----------|-------|-------------|
| Turn progress | 1 | [0, 1] | Current turn / 120 (max turns) |
| Is active player | 1 | {0, 1} | Always 1 (state always from active player perspective) |

**Rationale**: Turn progress helps the agent learn urgency. `is_active_player` is constant (always 1) but included for potential future multi-agent extensions.

## 4. Total Dimensionality

| Component | Dimensions |
|-----------|-----------|
| Active player hand | 35 |
| Opponent hand | 14 |
| Board gems | 6 |
| Board cards | 96 |
| Board nobles | 6 |
| Game progress | 2 |
| **TOTAL** | **159** |

## 5. Implementation Notes

### 5.1 Vectorization Process

```python
def vectorize(state: State, player_id: int) -> np.ndarray:
    """
    Convert Splendor game state to fixed-size vector.
    
    Args:
        state: Splendor State object
        player_id: ID of the player for whom to generate observation
        
    Returns:
        np.ndarray: Shape (159,) dtype float32
    """
    # 1. Extract active and opponent hands
    # 2. Extract board state (gems, cards, nobles)
    # 3. Compute derived features (can_afford, can_obtain_noble)
    # 4. Concatenate all sub-vectors
    # 5. Clip to [0, 1] and return as float32
```

### 5.2 Normalization Strategy

- **Gem counts**: Divide by 10 (max on hand)
- **Card counts**: Divide by 10 (reasonable upper bound)
- **Victory points**: Divide by 20 (beyond win threshold of 15)
- **Card prices**: Divide by 7 (max single gem cost)
- **One-hot vectors**: No normalization needed (already {0, 1})
- **Turn counter**: Divide by 120 (max game length)

### 5.3 Empty Position Handling

- **Missing cards on board**: Fill with zero vector (8 zeros)
- **Missing nobles**: Fill with zero vector (2 zeros)
- **Missing reserved cards**: Fill with zero vector

This ensures fixed size and allows the network to learn that zeros = "no card/noble".

### 5.4 Card Ordering Strategy

**Board cards**: Sort by (row, then by internal ID if deterministic, or fixed positions)
- Positions 0-3: CHEAP row cards
- Positions 4-7: MEDIUM row cards
- Positions 8-11: EXPENSIVE row cards

**Reserved cards**: Sort by reservation time (oldest first) or ID for determinism.

## 6. Validation Requirements

The state vectorizer must pass these tests:

1. **Shape consistency**: Always returns shape (159,)
2. **No NaN/Inf**: Assert no invalid values
3. **Range bounds**: All values in [0, 1]
4. **Dtype correctness**: np.float32
5. **State change detection**: Different states → different vectors (excluding ties)
6. **Determinism**: Same state → same vector (if sorted properly)

## 7. Future Enhancements

Potential improvements for later iterations:

- **Reserved card details**: Expand from 20 to include full price vectors
- **Deck information**: Track remaining cards in each row (3 dims)
- **Opponent reserved cards**: If observable, add features
- **Action history**: Last N actions encoded (temporal context)
- **Positional embeddings**: For cards/nobles to break symmetry

These are deferred to avoid premature complexity during initial training.

## 8. Comparison with AlphaZero State (Phase 2)

This state representation is **simpler** than what AlphaZero will need:
- **No action history**: AlphaZero uses last 8 moves for temporal context
- **No board image**: AlphaZero uses spatial convolutions (we use flat MLP)
- **No hidden information modeling**: AlphaZero tracks probability distributions over unknown cards

For Phase 1 (Score-based agent), this simpler representation is sufficient because:
1. PPO learns from immediate rewards (score changes)
2. No need for deep forward simulation (MCTS does that in Phase 2)
3. MLP policy is faster to train than CNN policy

## 9. References

- Splendor game rules: https://en.wikipedia.org/wiki/Splendor_(game)
- PPO paper: https://arxiv.org/abs/1707.06347
- AlphaZero paper: https://arxiv.org/abs/1712.01815 (for Phase 2 comparison)

---

**Next Steps**: Implement `SplendorStateVectorizer` class in `project/src/utils/state_vectorizer.py`
