# MCTS Benchmark Report

- Timestamp: 2026-03-21T19:46:00
- Games per matchup: 1
- Iterations tested: [10]
- PPO model: project/logs/maskable_ppo_event_e5_v6_candidate_20260320_220734/final_model.zip

## Results

| Matchup | Path | Win rate | MCTS pts | Opp pts | Sec/game |
|---|---:|---:|---:|---:|---:|
| MCTS[pure_dummy,iter=10] vs random_uniform_on_types | local-duel | 0.0% | 0.00 | 15.00 | 26.177 |
| MCTS[pure_dummy,iter=10] vs greedy_value | local-duel | 100.0% | 0.00 | 1.00 | 9.408 |
| MCTS[pure_dummy,iter=10] vs PPO | gym-wrapper | 0.0% | 1.00 | 16.00 | 27.915 |

## Notes

- local-duel path is used for MCTS vs classic agents (Random/Greedy).
- Gym-wrapper path is used for MCTS vs PPO to preserve PPO action-index semantics.
