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

## Environment Setup (RL Pipeline)

Recommended Python version: `3.10` (tested) or `3.11`.
(`stable-baselines3` + CUDA stack may fail on newer interpreters like 3.13.)

Install project dependencies from repo root:

```bash
python -m pip install -r project/requirements.txt
```

For development/testing:

```bash
python -m pip install -r project/requirements-dev.txt
```

Run tests:

```bash
pytest -q project/tests
```

## Canonical Entrypoints

- Train PPO score-based baseline: `project/scripts/train_score_based.py`
- Train MaskablePPO variant: `project/scripts/train_maskable_ppo.py`
- Evaluate PPO (current canonical evaluator): `project/scripts/evaluate_score_based_v3.py`

Deprecated evaluator:

- `project/scripts/evaluate_score_based.py` (kept for history, do not use)
