# Script Guide

This guide lists recommended entry points and where to find specialized utilities.

## Canonical Training

- `train_maskable_ppo.py`: primary training entry for MaskablePPO runs.
- `train_curriculum.py`: curriculum training variants.
- `train_score_based.py`: legacy PPO baseline training.

## Canonical Evaluation

- `evaluate_maskable_ppo.py`: primary model evaluation script.
- `evaluate_robust.py`: robust batched evaluation with confidence intervals.
- `benchmark_mcts.py`: PPO vs MCTS benchmark runner.

## Analysis & Indexing

- `run_eval_pipeline.py`: evaluation pipeline helper.
- `compare_checkpoints.py`: checkpoint comparisons.
- `refresh_experiment_index.py`: regenerate experiment index docs.

## Diagnostics

- `debug/`: ad hoc debug scripts (non-canonical, use on demand).
- `sanity_check_greedy.py`: opponent sanity check.

## Legacy / Specialized

- `evaluate_score_based.py`
- `evaluate_score_based_fixed.py`
- `evaluate_score_based_v2.py`
- `evaluate_score_based_v3.py`

These score-based evaluator variants are retained for historical context. Prefer robust and maskable evaluators for current conclusions.
