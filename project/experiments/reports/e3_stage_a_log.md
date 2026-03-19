# E3 Stage A Experiment Log - Lite Reward Shaping

**Date Started**: 2026-03-19 11:56 EDT  
**Status**: ⏳ In Progress  
**Target**: 300k steps, quick eval n=200, focus on event stats

---

## Experiment Configuration

**Config**: `maskable_ppo_event_v5_lite_reward.yaml`  
**Log Directory**: `project/logs/maskable_ppo_event_e3_lite_reward_20260319_115615`

### Reward Weights Comparison

| Event | V1 (Baseline) | E3 (Lite) | Change | Rationale |
|-------|---------------|-----------|--------|-----------|
| `take_gems` | 0.01 | 0.01 | — | unchanged |
| `buy_card` | 10.0 | 6.0 | ↓ -40% | Reduce shaping dominance |
| `reserve_card` | 0.05 | 0.10 | ↑ +100% | Encourage reservation |
| `score_up` | 5.0 | 5.0 | — | unchanged |
| `reach_15` | 25.0 | 10.0 | ↓ -60% | Major reduction in winning bonus |
| `scarcity_take` | 0.20 | 0.20 | — | unchanged |
| `block_reserve` | 1.0 | 1.0 | — | unchanged |
| `buy_reserved` | 2.0 | 3.0 | ↑ +50% | Encourage buying reserved cards |
| `engine_spike` | 5.0 | 3.0 | ↓ -40% | Reduce dominance |

### Hypothesis

The current V1 event shaping over-optimizes proxy rewards, especially `reach_15` (25.0), which dominates all other signals. This likely causes:
- Low reservation frequency (0.05 weight vs 10.0 vs 25.0)
- Weak generalization to GreedyAgent (policy learned GreedyAgent-specific patterns)

By reducing these dominant weights and boosting `reserve_card` + `buy_reserved`:
- Reserve-card and buy-reserved behaviors should emerge naturally
- Policy should generalize better to random opponent, keeping vs RandomAgent ≥ 88%
- vs Greedy may improve due to more robust, less opponent-specific learned behavior

---

## Key Metrics To Track

### Primary Gate Conditions (Stage A)
1. ✓ **vs RandomAgent ≥ 88%** (vs V1: 88.8%)
2. ✓ **reserve_card > 0** (appearance in event logs)
3. ✓ **buy_reserved > 0** (appearance in event logs)
4. ⚠️ Training stability (no divergence)

### Secondary Metrics (for decision-making on Stage B)
- vs Greedy win rate (target: ≥ 76% baseline from V1)
- vs Random win rate (target: ≥ 94%)
- Agent score progression (smooth learning curve)
- Explained variance (target: > 0.5)
- Invalid action rate (should be 0% with masking)

---

## Training Timeline

- **Training Start**: 2026-03-19 11:56 EDT
- **Expected Finish**: ~6-8 hours (300k steps ÷ 50k steps/hour)
- **Quick Eval**: Immediately after training (n=200, ~30 min)
- **Decision Point**: vs RandomAgent ≥ 88% + reserve_card > 0 → Promote to Stage B

---

## Execution Steps

### Step 1: Monitor Training
```bash
# In WSL terminal
tail -f _e3_stage_a_training.log

# Or check specific metrics
tensorboard --logdir project/logs --host 0.0.0.0 --port 6006 &
# then open http://localhost:6006 in Windows browser
```

### Step 2: Quick Evaluation (after training)
```bash
# Run quick eval script
python project/scripts/e3_stage_a_monitor.py --log-dir project/logs/maskable_ppo_event_e3_lite_reward_20260319_115615 --skip-wait --games 200
```

### Step 3: Event Statistics Analysis
Extract from TensorBoard or log parsing:
- `event_i/step` trends (reserve_card, buy_reserved, etc.)
- Frequency per game (games where each event fired)

### Step 4: Decision
- **If conditions met** → Create Stage B config, prepare 1M training
- **If not** → Analyze why, adjust hypothesis, or skip to E4 mixed-opponent

---

## Live Monitoring Resources

### TensorBoard (Windows)
1. Open WSL terminal, run: `tensorboard --logdir project/logs --load_fast=false --port 6006`
2. Open http://localhost:6006 in browser

Key scalars to watch:
- `train/explained_variance`
- `train/reward` (eval reward every 10k steps)
- `event_i/frequency` tabs (if logging is enabled)

### Log File
```bash
# Live tail in WSL
tail -f _e3_stage_a_training.log

# Count training progress (lines mentioning timestep)
grep -o "[0-9]*/300000" _e3_stage_a_training.log | tail -1
```

---

## Expected Outcomes & Contingencies

### Outcome A: ✅ All gates passed
- vs RandomAgent: 88%+
- vs Greedy: 76%+
- reserve_card and buy_reserved both > 0%
- **Action**: Proceed to Stage B (1M steps)

### Outcome B: ⚠️ Greedy regresses significantly
- vs RandomAgent: 88%+
- vs Greedy: < 75%
- **Action**: Decision needed — is lite reward the problem or opponent-overfitting? Run E4 mixed-opponent to test.

### Outcome C: ❌ RandomAgent generalization fails
- vs RandomAgent: < 88%
- **Action**: Reward was reduced too much. E3 fails; skip to E4 or E1/E2 ablations.

### Outcome D: 🔴 No reservation behavior appears
- reserve_card frequency still 0%
- **Action**: Reward boost insufficient; need qualitative investigation of state space / masks.

---

## Notes & Observations

(Will be filled during training)

- **11:56 EDT**: Training started successfully
- Status: Monitoring...

