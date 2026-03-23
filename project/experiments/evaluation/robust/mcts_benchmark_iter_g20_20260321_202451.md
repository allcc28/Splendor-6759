# MCTS Benchmark Report

- Timestamp: 2026-03-21T20:24:51
- Games per matchup: 20
- Iterations tested: [10]
- PPO model: project/logs/maskable_ppo_event_e5_v6_candidate_20260320_220734/final_model.zip

## Results

| Matchup | Path | Win rate | 95% CI | MCTS pts | Opp pts | Sec/game | Sanity |
|---|---:|---:|---:|---:|---:|---:|---:|
| MCTS[pure_dummy,iter=10] vs random_uniform_on_types | local-duel | 15.0% | [5.2, 36.0] | 2.65 | 8.85 | 22.277 | 9 |
| MCTS[pure_dummy,iter=10] vs greedy_value | local-duel | 20.0% | [8.1, 41.6] | 0.45 | 11.85 | 20.037 | 6 |
| MCTS[pure_dummy,iter=10] vs PPO | gym-wrapper | 25.0% | [11.2, 46.9] | 7.00 | 13.30 | 19.718 | 1 |

## Notes

- local-duel path is used for MCTS vs classic agents (Random/Greedy).
- Gym-wrapper path is used for MCTS vs PPO to preserve PPO action-index semantics.
