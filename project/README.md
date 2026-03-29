# Project Structure

This `project/` folder is the active RL workspace. If you want to understand the repo quickly, start here instead of browsing the whole tree.

## Active Areas

- `project/src/`: active environment wrappers, reward shaping logic, callbacks, and PPO agent code.
- `project/configs/training/`: versioned training configs for score-based, event-based, and curriculum runs.
- `project/scripts/`: canonical entrypoints for training, evaluation, diagnostics, and experiment indexing.
- `project/scripts/helpers/`: convenience scripts for ad hoc monitoring and small utilities.
- `project/scripts/setup/`: local environment verification and setup helpers.
- `project/tests/`: focused tests for the current RL pipeline.
- `project/docs/development/`: specs, progress notes, ADRs, sprint notes, and dev logs.

## Experiment Artifacts

- `project/logs/`: raw run directories created by training scripts.
  - Each run folder is the source of truth for checkpoints and run-local `config.yaml`.
  - Typical contents: `final_model.zip`, `eval/best_model.zip`, TensorBoard files, console logs.
- `project/experiments/evaluation/`: machine-readable evaluation outputs and generated reports.
  - `robust/ppo_robust/score_based/`: official score-based robust evals.
  - `robust/ppo_robust/event_based/`: official event-based robust evals.
  - `robust/mcts/canonical/`: citation-worthy MCTS benchmark runs.
  - `robust/mcts/archive/`: smoke tests and intermediate MCTS diagnostics.
  - other sibling folders: one-off evaluation families such as `maskable_ppo_eval/` and `behavior_metrics/`.
- `project/experiments/reports/`: curated summaries and generated indices.
  - `EXPERIMENT_INDEX.md` is the main lookup table.
  - `EXPERIMENT_INDEX.json` is the machine-readable companion.
  - `raw_logs/` is the holding area for local console logs referenced by reports.

## Fast Navigation

- Current evaluation entry point: `project/experiments/evaluation/LATEST.md`
- Current experiment index: `project/experiments/reports/EXPERIMENT_INDEX.md`
- Robust evaluation layout notes: `project/experiments/evaluation/robust/README.md`

## Working Rules

- New training runs should write only to `project/logs/<run_name>/`.
- New robust PPO evaluations should be generated through `project/scripts/evaluate_robust.py`, which routes files into `robust/ppo_robust/<family>/`.
- New MCTS benchmarks should be generated through `project/scripts/benchmark_mcts.py`, which routes files into `robust/mcts/<bucket>/`.
- After adding or moving experiment artifacts, refresh the index with:

```bash
python project/scripts/refresh_experiment_index.py
```

## Canonical Entrypoints

- Train MaskablePPO: `project/scripts/train_maskable_ppo.py`
- Evaluate MaskablePPO: `project/scripts/evaluate_maskable_ppo.py`
- Run robust evaluation: `project/scripts/evaluate_robust.py`
- Benchmark MCTS: `project/scripts/benchmark_mcts.py`
- Refresh experiment index: `project/scripts/refresh_experiment_index.py`
