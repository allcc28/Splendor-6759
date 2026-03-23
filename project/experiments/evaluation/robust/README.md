# Robust Evaluation Layout

This folder is organized for fast lookup and for keeping future outputs tidy.

## Where Things Live

- `ppo_robust/score_based/`: official score-based robust PPO reports.
- `ppo_robust/event_based/`: official event-based robust PPO reports.
- `ppo_robust/other/`: fallback bucket for robust runs that do not fit the main families.
- `mcts/canonical/`: main MCTS benchmark results worth citing.
- `mcts/archive/`: smoke tests, iteration sweeps, and debugging runs.

## Script Rules

- Use `python project/scripts/evaluate_robust.py ...` for robust PPO evaluations.
  - Default output is routed into `ppo_robust/<family>/`.
  - Override family explicitly with `--family score_based` or `--family event_based` if needed.
- Use `python project/scripts/benchmark_mcts.py ...` for MCTS benchmarks.
  - Use `--bucket archive` for smoke/debug runs.
  - Use `--bucket canonical` only for final runs you want to cite.

## Quick Navigation

- Best score-based robust report:
  `ppo_robust/score_based/robust_eval_v4a_20260308_143224_report.md`
- Best event-based robust report:
  `ppo_robust/event_based/robust_eval_v5_event_20260309_211510_report.md`
- Strong later event-based candidate:
  `ppo_robust/event_based/robust_eval_e2_stage_b_20260317_130241_report.md`
- Current canonical MCTS run:
  `mcts/canonical/mcts_benchmark_overnight_g500_20260322_051530.md`

## Rules

- Do not keep mixed JSON and Markdown files loose in this folder root.
- Keep only publication-worthy MCTS files in `mcts/canonical/`.
- Move smoke and short iteration tests to `mcts/archive/`.
- Refresh the index after adding or moving files:

```bash
python project/scripts/refresh_experiment_index.py
```
