# MCTS Benchmark Report

- Timestamp: 2026-03-21T16:59:20
- Games per matchup: 2
- Iterations tested: [50]
- PPO model: project/logs/maskable_ppo_event_e5_v6_candidate_20260320_220734/final_model.zip

## Results

| Matchup | Path | Win rate | MCTS pts | Opp pts | Sec/game |
|---|---:|---:|---:|---:|---:|
| MCTS[pure_dummy,iter=50] vs random_uniform_on_types | local-duel | 0.0% | 1.00 | 8.50 | 81.469 |
| MCTS[pure_dummy,iter=50] vs greedy_value | local-duel | 0.0% | 0.00 | 0.50 | 26.510 |
| MCTS[pure_dummy,iter=50] vs PPO | gym-wrapper | 50.0% | 10.00 | 12.00 | 41.498 |

## Notes

- Arena path is used for MCTS vs classic agents (Random/Greedy).
- Gym-wrapper path is used for MCTS vs PPO to preserve PPO action-index semantics.
