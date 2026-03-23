# Experiment Organization Guide

This folder stores experiment artifacts by purpose, while keeping raw run outputs under `project/logs/`.

## What Belongs Here

- `evaluation/`: generated evaluation outputs.
  - JSON metrics, Markdown reports, and benchmark summaries live here.
  - Start with `evaluation/LATEST.md` if you want the current headline results.
- `reports/`: curated summaries and generated indices.
  - `EXPERIMENT_INDEX.md` is the best single file for browsing all current runs.
- `phase1_reward_shaping/`, `phase2_alphazero/`, `phase3_tournament/`:
  historical phase buckets kept for course-structure alignment.

## Source Of Truth

- Raw checkpoints: `project/logs/*/final_model.zip` and `project/logs/*/eval/best_model.zip`
- Official robust PPO comparisons: `project/experiments/evaluation/robust/ppo_robust/**/*.json`
- Official MCTS comparisons: `project/experiments/evaluation/robust/mcts/**/*.json`

## Recommended Workflow

1. Train a model. This creates a new `project/logs/<run_name>/` directory.
2. Run evaluation scripts so outputs land in `project/experiments/evaluation/...`.
3. Refresh the experiment index:

```bash
python project/scripts/refresh_experiment_index.py
```

4. Read the current summary in:
   `project/experiments/reports/EXPERIMENT_INDEX.md`

## Naming Conventions

- Run folders: `<family>_<variant>_<YYYYMMDD_HHMMSS>`
- Robust eval files: `robust_eval_<tag>_<YYYYMMDD_HHMMSS>.json`
- MCTS benchmark files: `mcts_benchmark_<tag>_<YYYYMMDD_HHMMSS>.json`

## Cleanup Policy

- Do not delete historical run folders that are referenced by robust-eval or benchmark files.
- Keep publication-worthy outputs in stable folders, and move debug/smoke artifacts into archive buckets.
- Avoid mixing JSON and report files loosely in folder roots; prefer the organized subtrees documented in `evaluation/README.md`.
