# Experiment Index

- Generated: 2026-03-23T11:46:24
- Total log runs: 25
- Robust eval files: 3
- MCTS benchmark files: 8

## Quick Entry Points

- Latest training run: project/logs/maskable_ppo_event_e5_v6_candidate_20260320_220734
- Best score-based robust report: project/experiments/evaluation/robust/ppo_robust/score_based/robust_eval_v4a_20260308_143224_report.md
- Best event-based robust report: project/experiments/evaluation/robust/ppo_robust/event_based/robust_eval_v5_event_20260309_211510_report.md
- Canonical MCTS benchmark: project/experiments/evaluation/robust/mcts/canonical/mcts_benchmark_overnight_g500_20260322_051530.md

## Best Overall Robust Model

- Model path: project/logs/maskable_ppo_v4a_ent_lr_20260306_213530/eval/best_model
- Evidence file: project/experiments/evaluation/robust/ppo_robust/score_based/robust_eval_v4a_20260308_143224_report.md
- Composite score (mean win% vs Random/RandomAgent/Greedy): 87.27

## Robust Leaderboard

| Rank | Family | Eval Tag | Model Path | Rnd(wrapper) | RandomAgent | Greedy | Composite | Report |
|---:|---|---|---|---:|---:|---:|---:|---|
| 1 | score_based | v4a | project/logs/maskable_ppo_v4a_ent_lr_20260306_213530/eval/best_model | 94.8% | 91.2% | 75.8% | 87.27 | project/experiments/evaluation/robust/ppo_robust/score_based/robust_eval_v4a_20260308_143224_report.md |
| 2 | event_based | v5_event | project/logs/maskable_ppo_event_v1_20260309_110155/eval/best_model | 94.3% | 88.8% | 77.9% | 87.00 | project/experiments/evaluation/robust/ppo_robust/event_based/robust_eval_v5_event_20260309_211510_report.md |
| 3 | event_based | e2_stage_b | project/logs/maskable_ppo_event_e2_stage_b_20260316_202713/eval/best_model | 93.6% | 89.7% | 73.8% | 85.70 | project/experiments/evaluation/robust/ppo_robust/event_based/robust_eval_e2_stage_b_20260317_130241_report.md |

## Latest 10 Training Runs

| Run | Last Modified | best_model.zip | final_model.zip | config.yaml |
|---|---|---|---|---|
| maskable_ppo_event_e5_v6_candidate_20260320_220734 | 2026-03-21T02:00:17 | project/logs/maskable_ppo_event_e5_v6_candidate_20260320_220734/eval/best_model.zip | project/logs/maskable_ppo_event_e5_v6_candidate_20260320_220734/final_model.zip | project/logs/maskable_ppo_event_e5_v6_candidate_20260320_220734/config.yaml |
| maskable_ppo_event_e4_mixed_opp_20260319_212140 | 2026-03-20T01:18:46 | project/logs/maskable_ppo_event_e4_mixed_opp_20260319_212140/eval/best_model.zip | project/logs/maskable_ppo_event_e4_mixed_opp_20260319_212140/final_model.zip | project/logs/maskable_ppo_event_e4_mixed_opp_20260319_212140/config.yaml |
| maskable_ppo_event_e3_lite_reward_20260319_115615 | 2026-03-19T13:06:34 | project/logs/maskable_ppo_event_e3_lite_reward_20260319_115615/eval/best_model.zip | project/logs/maskable_ppo_event_e3_lite_reward_20260319_115615/final_model.zip | project/logs/maskable_ppo_event_e3_lite_reward_20260319_115615/config.yaml |
| maskable_ppo_event_e3_lite_reward_20260319_114834 | 2026-03-19T12:59:21 | project/logs/maskable_ppo_event_e3_lite_reward_20260319_114834/eval/best_model.zip | project/logs/maskable_ppo_event_e3_lite_reward_20260319_114834/final_model.zip | project/logs/maskable_ppo_event_e3_lite_reward_20260319_114834/config.yaml |
| maskable_ppo_event_e3_lite_reward_20260319_114534 | 2026-03-19T11:45:35 | - | - | project/logs/maskable_ppo_event_e3_lite_reward_20260319_114534/config.yaml |
| maskable_ppo_event_e2_stage_b_20260316_202713 | 2026-03-17T00:05:12 | project/logs/maskable_ppo_event_e2_stage_b_20260316_202713/eval/best_model.zip | project/logs/maskable_ppo_event_e2_stage_b_20260316_202713/final_model.zip | project/logs/maskable_ppo_event_e2_stage_b_20260316_202713/config.yaml |
| maskable_ppo_event_e1_no_gap_20260315_203357 | 2026-03-15T21:41:22 | project/logs/maskable_ppo_event_e1_no_gap_20260315_203357/eval/best_model.zip | project/logs/maskable_ppo_event_e1_no_gap_20260315_203357/final_model.zip | project/logs/maskable_ppo_event_e1_no_gap_20260315_203357/config.yaml |
| maskable_ppo_event_e2_no_last_event_20260310_222207 | 2026-03-10T23:27:30 | project/logs/maskable_ppo_event_e2_no_last_event_20260310_222207/eval/best_model.zip | project/logs/maskable_ppo_event_e2_no_last_event_20260310_222207/final_model.zip | project/logs/maskable_ppo_event_e2_no_last_event_20260310_222207/config.yaml |
| maskable_ppo_event_v1_20260309_110155 | 2026-03-09T14:39:53 | project/logs/maskable_ppo_event_v1_20260309_110155/eval/best_model.zip | project/logs/maskable_ppo_event_v1_20260309_110155/final_model.zip | project/logs/maskable_ppo_event_v1_20260309_110155/config.yaml |
| maskable_ppo_event_v1_20260309_110034 | 2026-03-09T11:00:34 | - | - | project/logs/maskable_ppo_event_v1_20260309_110034/config.yaml |

## Latest 10 MCTS Benchmarks

| Bucket | File | Timestamp | Games/Matchup | Iterations | PPO Model |
|---|---|---|---:|---|---|
| canonical | project/experiments/evaluation/robust/mcts/canonical/mcts_benchmark_overnight_g500_20260322_051530.md | 2026-03-22T05:15:30 | 500 | [10] | project/logs/maskable_ppo_event_e5_v6_candidate_20260320_220734/final_model.zip |
| archive | project/experiments/evaluation/robust/mcts/archive/mcts_benchmark_iter_g20_20260321_202451.md | 2026-03-21T20:24:51 | 20 | [10] | project/logs/maskable_ppo_event_e5_v6_candidate_20260320_220734/final_model.zip |
| archive | project/experiments/evaluation/robust/mcts/archive/mcts_benchmark_iter_g10_20260321_200356.md | 2026-03-21T20:03:56 | 10 | [10] | project/logs/maskable_ppo_event_e5_v6_candidate_20260320_220734/final_model.zip |
| archive | project/experiments/evaluation/robust/mcts/archive/mcts_benchmark_iter_g5_20260321_195237.md | 2026-03-21T19:52:37 | 5 | [10] | project/logs/maskable_ppo_event_e5_v6_candidate_20260320_220734/final_model.zip |
| archive | project/experiments/evaluation/robust/mcts/archive/mcts_benchmark_smoke_fix_20260321_194600.md | 2026-03-21T19:46:00 | 1 | [10] | project/logs/maskable_ppo_event_e5_v6_candidate_20260320_220734/final_model.zip |
| archive | project/experiments/evaluation/robust/mcts/archive/mcts_benchmark_smoke_20260321_194300.md | 2026-03-21T19:43:00 | 1 | [10] | project/logs/maskable_ppo_event_e5_v6_candidate_20260320_220734/final_model.zip |
| archive | project/experiments/evaluation/robust/mcts/archive/mcts_benchmark_smoke2_20260321_170109.md | 2026-03-21T17:01:09 | 1 | [10] | project/logs/maskable_ppo_event_e5_v6_candidate_20260320_220734/final_model.zip |
| archive | project/experiments/evaluation/robust/mcts/archive/mcts_benchmark_smoke_20260321_165920.md | 2026-03-21T16:59:20 | 2 | [50] | project/logs/maskable_ppo_event_e5_v6_candidate_20260320_220734/final_model.zip |

## How To Refresh

Run: python project/scripts/refresh_experiment_index.py

