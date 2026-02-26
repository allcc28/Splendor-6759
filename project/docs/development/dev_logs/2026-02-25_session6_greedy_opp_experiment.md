# Dev Log — Session 6: Greedy Opponent Training Experiment

**Date**: 2026-02-25  
**Duration**: TBD (training in progress)  
**Goal**: Experiment 1 — Does training vs. a stronger opponent produce a better agent?

---

## Context

After completing Phase 1, we discovered the original fork's codebase contains a well-tuned
`StateEvaluatorHeuristic` (5-weight formula) that GreedyAgentBoost uses. This gives us a
well-defined "strong baseline" to train against.

**Key insight from analysis**: v1 was trained vs. RandomAgent — a trivially weak opponent.
The concern is that v1 may have learned shortcuts (e.g., just avoid losing quickly) rather
than genuine long-term strategies. By training v2 against GreedyAgentBoost, we force the
agent to develop more structured play.

---

## Hypothesis

> Training against a stronger opponent (GreedyAgentBoost) with the same PPO + score_progress
> reward will yield a policy that outperforms the v1 (random-opponent) agent in head-to-head.

**Prediction**:
- v2 will start slower (harder opponent = more losses early on)
- v2 will converge lower in absolute reward (greedy opponent is genuinely hard)
- v2 will **beat v1 > 55% of the time** in direct head-to-head evaluation
- v2 will score more VP per game on average

---

## Experiment Design

| Property         | v1 (Baseline)                | v2 (This run)                |
|------------------|------------------------------|------------------------------|
| Opponent         | RandomAgent (uniform random) | GreedyAgentBoost             |
| Reward mode      | score_progress               | score_progress (identical)   |
| PPO params       | [256, 256, 128], lr=0.0003   | identical                    |
| Timesteps        | 1,000,000                    | 1,000,000                    |
| Seed             | 42                           | 42                           |

Only one variable changed: the training opponent. This makes the experiment **controlled**.

---

## Files Created / Modified

| File | Action | Description |
|------|--------|-------------|
| `project/configs/training/ppo_score_based_v2_greedy_opp.yaml` | Created | V2 training config |
| `project/scripts/train_score_based_v2.py` | Created | V2 training script |
| `project/scripts/evaluate_v1_vs_v2.py` | Planned | Head-to-head eval script |
| `project/docs/development/PROGRESS.md` | Updated | Added Phase 8 |

---

## Training Run

- **Start time**: 2026-02-25 (evening)
- **Script**: `python project/scripts/train_score_based_v2.py`
- **tmux session**: `train_v2`
- **Log dir**: `project/logs/ppo_score_based_v2_greedy_opp_YYYYMMDD_HHMMSS/`
- **Expected duration**: ~1 hour (RTX 4090, identical to v1)

**What to watch in TensorBoard:**
1. `rollout/ep_rew_mean` — should start *lower* than v1 (harder opponent)
2. `rollout/ep_len_mean` — will be shorter initially (greedy wins faster)
3. `train/explained_variance` — should still converge toward 0.5+
4. **Compare v1 vs v2 tensorboard side-by-side**: `tensorboard --logdir project/logs`

---

## Expected Outcomes & Success Criteria

| Metric | Success Threshold | Notes |
|--------|------------------|-------|
| v2 wins vs. v1 (head-to-head) | > 55% | Primary success criterion |
| v2 avg VP vs. GreedyAgent (fallback) | > 9.5 (v1 baseline) | Should learn better strategy |
| v2 explained_variance at 1M | > 0.5 | Training health check |
| v2 training converges | ep_rew_mean > 0 at 500K | Confirms opponent not too hard |

---

## Risks

- **Opponent too hard**: If GreedyAgent dominates throughout, the reward signal might be too sparse.  
  Mitigation: The `0.01/step` progress reward in `score_progress` mode ensures dense signal.
- **No improvement**: If v2 ≈ v1, hypothesis is wrong — suggests opponent strength matters less
  than reward shaping for PPO. This is still a valid scientific finding.

---

## Next Steps After Training

1. Run `evaluate_v1_vs_v2.py` for head-to-head result (100 games)
2. Compare TensorBoard learning curves side-by-side
3. If v2 > v1: use **v2 as the new baseline for Phase 2** (Event-based reward)
4. If v2 ≈ v1: proceed with MaskablePPO + event-based reward as the next lever
5. Document results in `project/experiments/reports/experiment1_greedy_opp_training.md`

---

**Status**: Training in progress  
**Last updated**: 2026-02-25
