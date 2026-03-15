# Phase 11 Experiment Plan: Event-Based V5 to V6

**Date**: 2026-03-10  
**Owner**: RL branch / event-based track  
**Status**: Ready to execute  
**Goal**: Determine whether event-based shaping can produce a robust, general improvement over V4a score-based training.

---

## Baseline

Current authoritative baselines:

| Model | Training | vs Random | vs RandomAgent | vs Greedy | Protocol |
|-------|----------|-----------|----------------|-----------|----------|
| V4a score-based | RandomAgent | 94.8% | 91.2% | 75.8% | n=1000 robust |
| V5 event-based | RandomAgent | 94.3% | 88.8% | 77.9% | n=1000 robust |

Interpretation:
- V5 is competitive with V4a and slightly better vs Greedy.
- The V5 vs V4a Greedy improvement (+2.1 pp) is not statistically significant at n=1000.
- V5 regresses vs RandomAgent, so the current shaping is not yet a strict upgrade.

---

## Main Risks To Address

1. Event reward dominates base reward, so the policy may optimize the shaping proxy instead of winning.
2. `reserve_card`, `block_reserve`, and `buy_reserved` are nearly absent, meaning the shaping design still misses part of real Splendor strategy.
3. The extra observation features may be helping, hurting, or doing nothing; this is not yet isolated.
4. Training only against `RandomAgent` may bias the learned policy and explain the drop in generalization.

---

## Decision Rule

Promote a new event-based model only if all of the following hold:

1. `vs Greedy` is at least as good as V5 under robust n=1000 evaluation.
2. `vs RandomAgent` is no worse than 1 pp below V5.
3. The model shows healthier behavior metrics:
   - non-zero reservation frequency
   - non-zero `buy_reserved` frequency or a clear reason it remains low
   - stable score progression by turn
4. No observation-shape mismatch or eval-config mismatch is introduced.

If none of the planned runs beat V5, freeze V5 as the event-based endpoint and stop spending cycles on reward shaping.

---

## Experiment Matrix

Run all experiments in two stages:
- Stage A: short screen at `300k` steps
- Stage B: full train at `1M` steps only for shortlisted runs

Planned config names below are proposed names to create for the run.

| Run ID | Planned config | Change vs V5 | Hypothesis | Stage A gate | Promote to Stage B if... |
|--------|----------------|--------------|------------|--------------|--------------------------|
| E0 | existing `maskable_ppo_event_v1.yaml` | No change | Reference only; do not retrain unless reproduction check is needed | None | Baseline only |
| E1 | `maskable_ppo_event_v5_no_gap.yaml` | Remove 60-dim gem-gap features; keep 9 last-event flags | Tests whether the gain comes from shaping rather than hand-crafted affordance features | Quick eval n=200 | `vs Greedy >= 76%` and no training instability |
| E2 | `maskable_ppo_event_v5_no_last_event.yaml` | Remove 9 last-event flags; keep gem-gap features | Tests whether last-event memory is actually useful | Quick eval n=200 | `vs Greedy >= 76%` and no training instability |
| E3 | `maskable_ppo_event_v5_lite_reward.yaml` | Reduce shaping magnitude: lower `buy_card`, `engine_spike`, `reach_15`; keep `reserve_card >= 0` | Reduces proxy over-optimization and may recover generality | Quick eval n=200 + event stats | `vs RandomAgent >= 88%` and `reserve_card > 0` |
| E4 | `maskable_ppo_event_v5_mixed_opp.yaml` | Mixed-opponent training instead of RandomAgent-only | Improves generalization and reduces opponent-specific overfitting | Quick eval n=200 vs all 3 opponents | `vs RandomAgent >= 89%` and `vs Greedy >= 77%` |
| E5 | `maskable_ppo_event_v6_candidate.yaml` | Combine best result from E3 and E4 | Best chance of a true V6 upgrade | Full run only after E3/E4 | Promote directly to robust eval |

---

## Exact Parameter Proposal

Use the current V5 weights as the reference point:

| Event | V5 |
|------|----|
| `take_gems` | 0.01 |
| `buy_card` | 10.0 |
| `reserve_card` | 0.05 |
| `score_up` | 5.0 |
| `reach_15` | 25.0 |
| `scarcity_take` | 0.20 |
| `block_reserve` | 1.0 |
| `buy_reserved` | 2.0 |
| `engine_spike` | 5.0 |

For `E3`, start with:

| Event | Proposed E3 |
|------|-------------|
| `take_gems` | 0.01 |
| `buy_card` | 6.0 |
| `reserve_card` | 0.10 |
| `score_up` | 5.0 |
| `reach_15` | 10.0 |
| `scarcity_take` | 0.20 |
| `block_reserve` | 1.0 |
| `buy_reserved` | 3.0 |
| `engine_spike` | 3.0 |

Rationale:
- `buy_card`, `reach_15`, and `engine_spike` currently dominate.
- `reserve_card` should not be punished.
- `buy_reserved` needs a stronger incentive if reservation is expected to appear at all.

Do not tune more than one reward table at a time in this cycle.

---

## Execution Order

Run in this order:

1. `E1` and `E2`
2. `E3`
3. `E4`
4. `E5` only if `E3` or `E4` is clearly promising

Reason:
- First isolate observation contributions.
- Then fix reward dominance.
- Then test whether the opponent distribution is the real bottleneck.

---

## Evaluation Protocol

### Stage A: Short Screen

Per run:
- Train to `300k` steps
- Run `evaluate_maskable_ppo.py --games 200`
- Record TensorBoard metrics and event rates

Minimum logs to capture:
- eval reward
- explained variance
- `event_i` rates
- fps
- quick win rate vs all 3 opponents

Hard stop conditions:
- NaN or divergence
- explained variance stays poor after early learning
- obvious collapse in action diversity

### Stage B: Full Candidate Run

For shortlisted runs only:
- Train to `1M` steps
- Save best model and final model
- Run `evaluate_robust.py` with `n=1000`

Required comparison set:
- V4a robust baseline
- V5 robust baseline
- shortlisted run robust result

---

## Behavior Metrics To Add To Every Result Table

Win rate alone is not enough. Each promoted run should also report:

- reservation frequency
- `buy_reserved` frequency
- noble acquisition rate
- average score by turn 20 / 40 / 60
- average purchased cards by color
- average game length

If a run improves vs Greedy but still never reserves, call that out explicitly.

---

## Recommended Success Thresholds

Use these thresholds to avoid self-deception:

| Metric | Minimum target |
|--------|----------------|
| Robust vs Greedy | `>= 78.5%` |
| Robust vs RandomAgent | `>= 89.0%` |
| Robust vs Random | `>= 94.0%` |
| Reserve frequency | `> 0%` |
| CI claim | Preferably narrower than or clearly above V4a |

Interpretation:
- `78.5%` vs Greedy is not magical; it is simply the point where the gain over V4a starts to look practical instead of cosmetic.
- If Greedy rises but RandomAgent falls further, do not call it a net improvement.

---

## Command Templates

Short-screen train:

```bash
python project/scripts/train_maskable_ppo.py --config project/configs/training/<planned_config>.yaml
```

Quick eval:

```bash
python project/scripts/evaluate_maskable_ppo.py \
  --model project/logs/<run_dir>/eval/best_model \
  --config project/configs/training/<planned_config>.yaml \
  --games 200
```

Robust eval:

```bash
python project/scripts/evaluate_robust.py \
  --model project/logs/<run_dir>/eval/best_model \
  --config project/configs/training/<planned_config>.yaml \
  --games 1000 \
  --batches 10
```

---

## Expected Outcomes

Most likely outcomes:

1. Observation ablations show that one of `gem-gap` or `last-event flags` is unnecessary.
2. Lite reward improves generalization more than it improves peak Greedy win rate.
3. Mixed-opponent training is the most likely path to a true V6 improvement.

If `E4` fails badly, that is still useful: it means the current event representation is not stable enough for broader training and should be simplified before more experiments.

---

## Exit Criteria For Phase 11

Phase 11 can end when one of these is true:

1. A V6 candidate clearly beats both V4a and V5 under robust evaluation.
2. All planned ablations fail to improve V5 materially, in which case V5 becomes the final event-based result and the project should pivot to search / planning / hybrid methods.
