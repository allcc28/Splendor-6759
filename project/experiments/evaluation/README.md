# Evaluation Layout

This folder is for generated evaluation outputs only. Raw checkpoints still belong in `project/logs/`.

## Entry Points

- `LATEST.md`: quick pointer to the current best robust reports and canonical MCTS benchmark.
- `robust/README.md`: structure and rules for robust PPO and MCTS comparisons.
- `../reports/EXPERIMENT_INDEX.md`: broader experiment index across runs and evaluations.

## Current Subfolders

- `maskable_ppo_eval/`: current general PPO evaluation outputs.
- `maskable_ppo_v3_eval/`: older V3-era evaluation outputs kept for history.
- `behavior_metrics/`: behavior analysis JSON outputs.
- `ppo_score_based_*`: legacy score-based evaluation folders retained as-is.
- `robust/`: organized robust comparisons and MCTS benchmarks.

## Organization Rule

- New one-off evaluators should write into a dedicated subfolder under `evaluation/`.
- New robust PPO results should not be written directly into `robust/`; use `robust/ppo_robust/<family>/`.
- New MCTS benchmark results should not be written directly into `robust/`; use `robust/mcts/<bucket>/`.
