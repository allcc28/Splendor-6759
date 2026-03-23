# MCTS Benchmark Report

- Timestamp: 2026-03-22T05:15:30
- Games per matchup: 500
- Iterations tested: [10]
- PPO model: project/logs/maskable_ppo_event_e5_v6_candidate_20260320_220734/final_model.zip

## Results

| Matchup | Path | Win rate | 95% CI | MCTS pts | Opp pts | Sec/game | Sanity |
|---|---:|---:|---:|---:|---:|---:|---:|
| MCTS[pure_dummy,iter=10] vs random_uniform_on_types | local-duel | 27.4% | [23.7, 31.5] | 4.84 | 10.85 | 24.229 | 154 |
| MCTS[pure_dummy,iter=10] vs greedy_value | local-duel | 16.4% | [13.4, 19.9] | 0.83 | 12.21 | 21.870 | 131 |
| MCTS[pure_dummy,iter=10] vs PPO | gym-wrapper | 25.2% | [21.6, 29.2] | 6.87 | 12.46 | 16.713 | 11 |

## Notes

- local-duel path is used for MCTS vs classic agents (Random/Greedy).
- Gym-wrapper path is used for MCTS vs PPO to preserve PPO action-index semantics.
