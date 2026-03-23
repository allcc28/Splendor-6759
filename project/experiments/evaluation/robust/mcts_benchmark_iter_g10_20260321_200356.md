# MCTS Benchmark Report

- Timestamp: 2026-03-21T20:03:56
- Games per matchup: 10
- Iterations tested: [10]
- PPO model: project/logs/maskable_ppo_event_e5_v6_candidate_20260320_220734/final_model.zip

## Results

| Matchup | Path | Win rate | 95% CI | MCTS pts | Opp pts | Sec/game | Sanity |
|---|---:|---:|---:|---:|---:|---:|---:|
| MCTS[pure_dummy,iter=10] vs random_uniform_on_types | local-duel | 30.0% | [10.8, 60.3] | 2.90 | 11.80 | 22.852 | 3 |
| MCTS[pure_dummy,iter=10] vs greedy_value | local-duel | 0.0% | [0.0, 27.8] | 0.80 | 17.40 | 25.899 | 0 |
| MCTS[pure_dummy,iter=10] vs PPO | gym-wrapper | 10.0% | [1.8, 40.4] | 4.30 | 15.20 | 18.507 | 0 |

## Notes

- local-duel path is used for MCTS vs classic agents (Random/Greedy).
- Gym-wrapper path is used for MCTS vs PPO to preserve PPO action-index semantics.
