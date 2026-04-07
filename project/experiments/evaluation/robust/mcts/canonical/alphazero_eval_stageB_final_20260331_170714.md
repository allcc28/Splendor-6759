# AlphaZero Robust Evaluation Report

- Timestamp: 2026-03-31T17:07:14
- Games per matchup: 100
- Alpha checkpoint: project/logs/alphazero/alphazero_stageB_wsl_20260328_225252/final_model.pt
- Bucket: canonical

## Results

| Matchup | Path | Win rate | 95% CI | Alpha pts | Opp pts | Sec/game |
|---|---:|---:|---:|---:|---:|---:|
| AlphaZeroMCTS vs random_uniform_on_types | local-duel | 7.0% | [3.4, 13.7] | 0.94 | 15.39 | 26.271 |
| AlphaZeroMCTS vs greedy_value | local-duel | 1.0% | [0.2, 5.4] | 0.03 | 16.26 | 17.822 |
