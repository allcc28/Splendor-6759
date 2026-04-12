# AlphaZero V2 + MuZero V1 Implementation Plan

## Overview

This document specifies the implementation plan for hardening AlphaZero (V2) and building a MuZero (V1) prototype for the Splendor game project. Both algorithms share a common planning adapter for game logic.

## Scope

**In scope:**
- Shared planning adapter with correct `let_all_move` terminal logic
- AlphaZero V2: refactored onto shared adapter, replay buffer mandatory, FPU
- MuZero V1: end-to-end prototype with hybrid latent/real-state MCTS
- Unified evaluation framework for both algorithms
- Public-information observation mode (no opponent reserved card identity leak)

**Out of scope (this cycle):**
- Full belief-state / information-set MCTS
- Full stochastic MuZero (chance-aware dynamics)
- Ray / distributed training
- External framework wrapping
- Categorical support transforms for MuZero value/reward
- Prioritized experience replay for MuZero
- Observation stacking for MuZero

## Architecture

### Shared Planning Adapter

`project/src/planning/adapter.py` — `SplendorPlanningAdapter`

Interface:
- `reset() -> State`
- `clone() -> SplendorPlanningAdapter`
- `current_player -> int`
- `legal_actions -> list[Action]`
- `step(action) -> None`
- `is_terminal() -> bool`
- `terminal_value(player_id) -> float` (+1/0/-1)
- `encode_observation(player_id, turn_count) -> Tensor`

Terminal logic: `let_all_move` — when any player reaches 15 points, the current round completes (all players get equal turns), then the winner is resolved by highest points, with fewest-cards tiebreak.

### AlphaZero V2

Refactored `alphazero_mcts.py` and `alphazero_trainer.py` to depend on the shared adapter. Replay buffer mandatory. Optional FPU for unvisited nodes.

Config naming: `alphazero_v2_*.yaml`

### MuZero V1

New package `project/src/muzero/` with:
- `network.py`: RepresentationNet, DynamicsNet, PredictionNet, combined MuZeroNetwork
- `mcts.py`: Latent-space MCTS with hybrid real-state tracking for legal action queries
- `replay.py`: Uniform replay buffer with K=5 unroll target building
- `history.py`: GameHistory data container
- `trainer.py`: MuZeroTrainer with self-play + unroll training

Config naming: `muzero_v1_*.yaml`

## Evaluation Convention

- Smoke runs: low sample count, `archive` bucket
- Canonical runs: `n=100`, `canonical` bucket
- Output: JSON + Markdown reports under `project/experiments/evaluation/robust/{mcts,muzero}/{archive,canonical}/`
- Matchups: RandomAgent, GreedyAgentBoost, optional PPO

## Acceptance Gates

See the approved plan for detailed acceptance criteria per phase.
