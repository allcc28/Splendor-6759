# Project Structure (IFT6759 Splendor)

This folder organizes the work described in the project plan into clear phases and modules.

## Top-Level Layout
- `project/configs/`: All experiment configs (env, reward shaping, MCTS, training).
- `project/src/`: Core implementation (agents, reward shaping, MCTS, NN, training, evaluation).
- `project/experiments/`: Phase-based experiment runs and artifacts.
- `project/data/`: Raw/processed/self-play/supervised datasets.
- `project/outputs/`: Figures, checkpoints, and reports.
- `project/logs/`: Training and evaluation logs.
- `project/scripts/`: Entry scripts for training, evaluation, and tournaments.
- `project/notebooks/`: Exploration and analysis notebooks.
- `project/docs/`: Design notes, results, and meeting notes.
- `project/tests/`: Unit/integration tests.

## Phase Mapping (from plan)
- **Phase 1 (Weeks 1–3)**
  - Baselines + reward shaping: `src/agents/score_based`, `src/agents/event_based`, `src/reward`
  - Experiment configs: `configs/reward`, `configs/training`
  - Runs: `experiments/phase1_reward_shaping`

- **Phase 2 (Weeks 4–7)**
  - AlphaZero components: `src/agents/alphazero`, `src/mcts`, `src/nn`, `src/training`
  - Configs: `configs/mcts`, `configs/training`
  - Runs: `experiments/phase2_alphazero`

- **Phase 3 (Weeks 8–10)**
  - Tournament/evaluation: `src/evaluation`, `scripts`, `experiments/phase3_tournament`
  - Outputs: `outputs/figures`, `outputs/reports`

## Conventions
- Training entrypoints should live in `project/scripts/`.
- Configs are versioned in `project/configs/` and referenced by scripts.
- All generated artifacts go to `project/outputs/` and `project/logs/`.
