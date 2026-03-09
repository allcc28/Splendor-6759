# Session 8: Phase 9 Complete — MaskablePPO V3 Training, Evaluation & Bug Fix

**Date**: 2026-03-03 to 2026-03-06  
**Phase**: 9  
**Status**: ✅ Complete

---

## What Was Done

### 1. MaskablePPO Training (2026-03-03)

Completed Phase 9 training from `train_maskable_ppo.py`:
- 1M timesteps, `GreedyAgentBoost` opponent replaced by `RandomAgent` (training only)
- `MaskablePPO` from `sb3-contrib 2.7.1` with `ActionMasker` wrapper
- **0 invalid actions** throughout all 1M steps
- Peak eval reward: **67.8** @ 820K steps (vs V1 peak ~38)
- 20 checkpoints saved (every 50K steps)

Model location: `project/logs/maskable_ppo_score_v3_20260303_183435/`

### 2. First Evaluation Run (2026-03-06 — CONTAINS BUG)

Created `evaluate_maskable_ppo.py` and ran 100 games per opponent.

**Initial results** (JSON: `eval_v3_maskable_20260306_190731.json`):
| Opponent | Win Rate | Agent Score | Opp Score |
|----------|----------|-------------|-----------|
| Random   | 95.0%    | 15.4 pts    | 0.9 pts   |
| RandomAgent | 93.0% | 15.5 pts   | 5.1 pts   |
| GreedyAgent | **94.0%** | 15.3 pts | **1.1 pts** |

**Red flag**: GreedyAgent only scored 1.1 pts vs RandomAgent's 5.1 pts — a properly functioning GreedyAgent should significantly outperform RandomAgent.

### 3. Bug Discovery and Fix (2026-03-06)

Created `sanity_check_greedy.py` to investigate. Tests revealed:

**Root cause**: `simulate_next_state()` in `evaluators.py` calls `action.execute(state_copy)`, which **switches `active_player_id`** in the Splendor state. The `ValueBasedEvaluator.score_next_state()` was calling `get_active_hand()` which returned `active_players_hand()` — now pointing to the **opponent's hand** (PPO agent), not the GreedyAgent's own hand.

Result: GreedyAgent was unknowingly optimizing moves that benefited the PPO agent, not itself.

**Fix**: Added `get_actor_hand(next_state)` to `modules/evaluators.py`:
```python
def get_actor_hand(next_state):
    pid_now = int(next_state.active_player_id)   # opponent is now active
    actor_pid = 1 - pid_now                       # the one who just moved
    return next_state.list_of_players_hands[actor_pid]
```

**Verification** (head-to-head before/after fix):
| Condition | Greedy score | Random score | Greedy win rate |
|-----------|-------------|--------------|-----------------|
| Before fix (broken) | 3.2 pts | 7.8 pts | 10% |
| After fix (correct) | 14.4 pts | 5.3 pts | **85%** |

### 4. Corrected Evaluation (2026-03-06)

Re-ran all 100 games per opponent with fixed evaluator.

**Corrected results** (JSON: `eval_v3_maskable_20260306_193442.json`):
| Opponent | Win Rate | Agent Score | Opp Score |
|----------|----------|-------------|-----------|
| Random   | 96.0%    | 15.7 pts    | 0.9 pts   |
| RandomAgent | 90.0% | 15.0 pts   | 5.9 pts   |
| GreedyAgent | **67.0%** | 13.0 pts | **7.9 pts** |

### 5. Comparison Plots & Reports

- Regenerated `project/experiments/reports/v3_figures/v1_vs_v3_win_rates.png` with corrected data
- Wrote `project/experiments/reports/v3_validation_report.md` documenting the full audit trail

---

## Files Created/Modified

| File | Status | Notes |
|------|--------|-------|
| `project/scripts/evaluate_maskable_ppo.py` | Created | MaskablePPO evaluator |
| `project/scripts/extract_tb_v3.py` | Created | V1 vs V3 comparison plots |
| `project/scripts/sanity_check_greedy.py` | Created | Diagnostic tool (3 tests) |
| `modules/evaluators.py` | Fixed | `get_actor_hand()` + `score_next_state` fix |
| `project/experiments/reports/v3_validation_report.md` | Created | Full audit report |
| `project/experiments/reports/v3_figures/*.png` | Updated | Corrected win rate bar chart |
| `project/docs/development/PROGRESS.md` | Updated | Phase 9 complete with corrected results |
| `.github/copilot-instructions.md` | Updated | Bug fix in pitfalls, new file refs |
| `docs/plan.md` | Updated | Phase summary status table |

---

## Key Decisions

1. **Do NOT report uncorrected 94% GreedyAgent win rate** — retracted in favor of 67%
2. **evaluators.py is now the authoritative place for state evaluation utilities** — any future agent using `simulate_next_state` must use `get_actor_hand()` 
3. **Phase 10 (event-based reward) is still deferred** — Phase 9 results (67% vs GreedyAgent) are good enough to proceed to Phase 11 (MCTS) if team agrees

---

## Open Questions for Next Session

- At 67% vs GreedyAgent, is the V3 MaskablePPO agent strong enough as a starting point for MCTS?
- Should Phase 10 (event-based rewards) be done before Phase 11 (planning) to improve baseline?
- Consider longer training (2M+ steps) or tuned hyperparameters before MCTS integration
