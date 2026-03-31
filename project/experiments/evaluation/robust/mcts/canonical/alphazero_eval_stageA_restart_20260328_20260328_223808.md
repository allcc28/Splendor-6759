# AlphaZero Robust Evaluation Report

- Timestamp: 2026-03-28T22:38:08
- Games per matchup: 100
- Alpha checkpoint: project/logs/alphazero_stageA_wsl_20260328_211928/final_model.pt
- Bucket: canonical

## Results

| Matchup | Path | Win rate | 95% CI | Alpha pts | Opp pts | Sec/game |
|---|---:|---:|---:|---:|---:|---:|
| AlphaZeroMCTS vs random_uniform_on_types | local-duel | 59.0% | [49.2, 68.1] | 9.94 | 9.95 | 10.828 |
| AlphaZeroMCTS vs greedy_value | local-duel | 24.0% | [16.7, 33.2] | 2.72 | 11.15 | 10.976 |
| AlphaZeroMCTS vs PPO | gym-wrapper | 0.0% | [0.0, 3.7] | 1.02 | 15.04 | 9.783 |
