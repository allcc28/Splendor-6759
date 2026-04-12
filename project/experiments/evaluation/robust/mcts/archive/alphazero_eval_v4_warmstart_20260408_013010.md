# AlphaZero Robust Evaluation Report

- Timestamp: 2026-04-08T01:30:10
- Games per matchup: 50
- Alpha checkpoint: project/logs/alphazero_v4_warmstart_wsl_20260407_210410/final_model.pt
- Bucket: archive

## Results

| Matchup | Path | Win rate | 95% CI | Alpha pts | Opp pts | Sec/game |
|---|---:|---:|---:|---:|---:|---:|
| AlphaZeroMCTS vs random_uniform_on_types | local-duel | 6.0% | [2.1, 16.2] | 1.78 | 12.32 | 20.419 |
| AlphaZeroMCTS vs greedy_value | local-duel | 2.0% | [0.4, 10.5] | 0.10 | 16.00 | 15.256 |
