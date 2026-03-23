# MCTS Benchmark Report

- Timestamp: 2026-03-21T17:01:09
- Games per matchup: 1
- Iterations tested: [10]
- PPO model: project/logs/maskable_ppo_event_e5_v6_candidate_20260320_220734/final_model.zip

## Results

| Matchup | Path | Win rate | MCTS pts | Opp pts | Sec/game |
|---|---:|---:|---:|---:|---:|
| MCTS[pure_dummy,iter=10] vs random_uniform_on_types | local-duel | 100.0% | 15.00 | 13.00 | 36.623 |
| MCTS[pure_dummy,iter=10] vs greedy_value | local-duel | 0.0% | 3.00 | 15.00 | 22.955 |
| MCTS[pure_dummy,iter=10] vs PPO | gym-wrapper | 0.0% | 8.00 | 15.00 | 14.196 |

## Notes

- local-duel path is used for MCTS vs classic agents (Random/Greedy).
- Gym-wrapper path is used for MCTS vs PPO to preserve PPO action-index semantics.
