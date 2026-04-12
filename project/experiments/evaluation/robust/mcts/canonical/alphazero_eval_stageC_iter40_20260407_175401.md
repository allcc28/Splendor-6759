# AlphaZero Robust Evaluation Report

- Timestamp: 2026-04-07T17:54:01
- Games per matchup: 100
- Alpha checkpoint: project/logs/alphazero_stageC_wsl_20260331_231447/alphazero_iter_0040.pt
- Bucket: canonical

## Results

| Matchup | Path | Win rate | 95% CI | Alpha pts | Opp pts | Sec/game |
|---|---:|---:|---:|---:|---:|---:|
| AlphaZeroMCTS vs random_uniform_on_types | local-duel | 30.0% | [21.9, 39.6] | 5.82 | 11.67 | 43.051 |
| AlphaZeroMCTS vs greedy_value | local-duel | 5.0% | [2.2, 11.2] | 0.95 | 14.70 | 27.577 |
