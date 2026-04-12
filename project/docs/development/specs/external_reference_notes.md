# External Reference Notes

## Referenced Repositories

### 1. cestpasphoto/alpha-zero-general

- **URL**: https://github.com/cestpasphoto/alpha-zero-general
- **License**: MIT (inherited from suragnair/alpha-zero-general)
- **Relevance**: Splendor-specific AlphaZero with pretrained models, chance handling, 2-4 player support

**What we borrow (design ideas only, no code copy):**

| Module | Idea | Notes |
|--------|------|-------|
| `splendor/SplendorGame.py` | Game interface pattern (`getGameEnded`, `getValidMoves`, `getCanonicalForm`) | Informs our planning adapter design |
| `MCTS.py` | First Play Urgency (FPU) for unvisited nodes | Better than Q=0 default for exploration |
| `Coach.py` | Arena gating: new model must beat old model before acceptance | Consider for AlphaZero V2 trainer |
| `splendor/SplendorLogicNumba.py` | Deterministic card reveal seeds for MCTS reproducibility | Noted for future chance-aware search |

**What we do NOT borrow:**
- Their 81-action fixed action space (our dynamic 2048-indexer is more correct)
- Their simplified gem rules (no simultaneous take+return)
- Numba JIT compilation approach (our game logic is in `modules/`)
- Multi-player support (we are 2-player only)
- Their observation encoding (2D array vs our CNN-friendly 80-channel tensor)

**Known limitations of their implementation:**
- Some gem take/return real rules not fully covered
- Not a "perfect" Splendor implementation

### 2. werner-duvaud/muzero-general

- **URL**: https://github.com/werner-duvaud/muzero-general
- **License**: MIT
- **Relevance**: Production-quality MuZero framework, not Splendor-specific

**What we borrow (design patterns, adapted locally):**

| Module | Pattern | Adaptation |
|--------|---------|------------|
| `models.py` | `AbstractNetwork` interface: `initial_inference()` + `recurrent_inference()` | Direct adoption as our MuZero network API |
| `models.py` | Three-network residual architecture (Representation, Dynamics, Prediction) | Simplified for our (80,3,4) input, no downsampling |
| `self_play.py` | Self-play loop structure, GameHistory container | Synchronous (no Ray), adapted for our adapter |
| `self_play.py` | MCTS with `MinMaxStats` Q-value normalization, `Node` class | Added hybrid real-state tracking for legal actions |
| `replay_buffer.py` | Unroll target building from sampled positions | Uniform sampling (no prioritized replay for V1) |
| `trainer.py` | Unroll training with gradient scaling (0.5x at dynamics boundary) | Loss weights: policy CE + value MSE x0.25 + reward MSE |

**What we do NOT borrow:**
- Ray distributed actors (V1 is synchronous)
- Prioritized experience replay (V1 uses uniform)
- Observation stacking (not needed for Splendor)
- `support_to_scalar` / categorical value transforms (V1 uses scalar MSE)
- SharedStorage actor pattern

## Attribution

Any closely mirrored patterns from the above repos are attributed in code comments with the source repo URL. All code is implemented locally in this repository.
