# MCTS Benchmark Report

- Timestamp: 2026-03-21T19:52:37
- Games per matchup: 5
- Iterations tested: [10]
- PPO model: project/logs/maskable_ppo_event_e5_v6_candidate_20260320_220734/final_model.zip

## Results

| Matchup | Path | Win rate | 95% CI | MCTS pts | Opp pts | Sec/game | Sanity |
|---|---:|---:|---:|---:|---:|---:|---:|
| MCTS[pure_dummy,iter=10] vs random_uniform_on_types | local-duel | 40.0% | [11.8, 76.9] | 3.80 | 8.60 | 21.217 | 2 |
| MCTS[pure_dummy,iter=10] vs greedy_value | local-duel | 20.0% | [3.6, 62.4] | 0.20 | 13.40 | 24.120 | 1 |
| MCTS[pure_dummy,iter=10] vs PPO | gym-wrapper | 60.0% | [23.1, 88.2] | 9.80 | 8.60 | 10.433 | 0 |

## Notes

- local-duel path is used for MCTS vs classic agents (Random/Greedy).
- Gym-wrapper path is used for MCTS vs PPO to preserve PPO action-index semantics.
