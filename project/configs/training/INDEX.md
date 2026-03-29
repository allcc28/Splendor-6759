# Training Config Index

This index clarifies which configs are currently recommended versus historical variants.

## Current Canonical

- `maskable_ppo_score_based.yaml`: canonical score-based MaskablePPO baseline.
- `maskable_ppo_event_v6_candidate.yaml`: latest event-based candidate training.
- `maskable_ppo_event_v5_mixed_opp.yaml`: mixed-opponent event training baseline.

## Stable Historical Baselines

- `maskable_ppo_event_v1.yaml`: canonical event-based V5 baseline.
- `maskable_ppo_v4a_ent_lr.yaml`: canonical score-based robust baseline (V4a).
- `ppo_score_based.yaml`: original PPO score-based baseline.
- `ppo_score_based_v2_greedy_opp.yaml`: historical PPO+Greedy run family.

## Experimental / Ablation

- `maskable_ppo_event_v5_no_gap.yaml`
- `maskable_ppo_event_v5_no_last_event.yaml`
- `maskable_ppo_event_v5_lite_reward.yaml`
- `maskable_ppo_event_e2_stage_b.yaml`
- `maskable_ppo_v4b_rollout_curriculum.yaml`
- `maskable_ppo_v4c_curriculum.yaml`

## Quick Utility

- `ppo_quick_test.yaml`: short smoke test config.
