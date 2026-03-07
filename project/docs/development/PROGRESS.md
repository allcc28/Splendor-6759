# Implementation Progress Tracker

**Start Date**: 2026-02-24  
**Project**: PPO Score-Based RL Agent for Splendor  
**Current Phase**: Phase 9 Complete ✅ — MaskablePPO (V3) trained & evaluated

---

## ✅ Completed Tasks (2026-02-24 to 2026-02-25)

### Phase 8: Experiment 1 — Greedy Opponent (Negative Result) ✅
- [x] **Task 8.1**: Create training script vs. GreedyAgentBoost
  - File: `project/scripts/train_v2_greedy.py` (fixed `sys.path` for legacy agents)
- [x] **Task 8.1**: Run 1M timestep training
  - Result: Failed to learn scoring. Agent became "passive" (0 points, long survival) to avoid the -10 invalid action penalty.
  - Metrics: Negative Explained Variance (-0.167).
- [x] **Task 8.3**: Comparative Evaluation & Reporting
  - Report: `project/experiments/reports/ppo_v2_greedy_opponent_report.md`
  - Findings: Pure PPO cannot handle a 200-dim action space where >90% of actions are illegal against a strong opponent.

---

## ✅ Phase 9 — Architecture Pivot: MaskablePPO (COMPLETE)

**Date**: 2026-03-03 to 2026-03-06  
**Model**: `project/logs/maskable_ppo_score_v3_20260303_183435/`

- [x] **Task 9.1**: Install `sb3-contrib` → MaskablePPO 2.7.1 installed
- [x] **Task 9.2**: Added `action_masks()` method to `SplendorGymWrapper`
  - Also fixed Issue 4: `cached_legal_actions == 0` now triggers `truncated = True`
- [x] **Task 9.3**: Created `project/scripts/train_maskable_ppo.py`
- [x] **Task 9.4**: Training verified — 0 invalid actions during all 1M steps
- [x] **Task 9.5**: 1M timestep training completed
  - Final model: `final_model.zip` + `eval/best_model.zip` (at 820K steps)
  - Peak eval reward: **67.8** (vs V1 peak: ~38, V2 peak: ~6.6)
  - Mean reward stabilized at **60–67** range from step 10K onward
  - 20 checkpoints saved (every 50K steps)
- [x] **Task 9.6**: Evaluation script created: `project/scripts/evaluate_maskable_ppo.py`
- [x] **Task 9.7**: Comparison plotting script: `project/scripts/extract_tb_v3.py`

### V3 Training Metrics (eval rewards from `evaluations.npz`)
| Timestep | Mean Reward |
|----------|-------------|
| 10,000   | 66.2        |
| 310,000  | 66.2        |
| 610,000  | 66.8        |
| 820,000  | **67.8** (peak) |
| 1,000,000 | 59.7       |

### Three-way comparison
| Metric | V1 (PPO) | V2 (PPO+Greedy) | V3 (MaskablePPO) |
|--------|----------|-----------------|------------------|
| Peak eval reward | ~38 | ~6.6 | **67.8** |
| Final eval reward | ~28 | -3.9 | 59.7 |
| Explained variance | 0.54 | -0.167 | 0.54+ |
| Invalid actions/game | 40-60% | ~0 (but passive) | **0% by design** |

### V3 Evaluation Results (canonical — 2026-03-06, `eval_v3_maskable_20260306_204610.json`)
| Opponent | Win Rate | Agent Score | Opp Score | Invalid Actions |
|----------|----------|-------------|-----------|------------------|
| Random (wrapper) | **95.0%** (95/2/3) | 15.6 ± 3.8 | 1.4 ± 2.6 | 0 |
| RandomAgent | **91.0%** (91/4/5) | 15.4 ± 4.4 | 5.4 ± 4.0 | 0 |
| GreedyAgent | **78.0%** (78/19/3) | 14.5 ± 4.4 | 8.1 ± 5.2 | 0 |

> **Two sequential bugs were found and fixed (2026-03-06)**:
> 1. `ValueBasedEvaluator.score_next_state` evaluated the opponent’s hand instead of the agent’s after `action.execute()` switches `active_player_id`. Fixed via `get_actor_hand()` in `modules/evaluators.py` — this raised GreedyAgent result from ~~94%~~ → 67%.
> 2. Eval script always assigned agent as player 0 (first-mover advantage). Fixed via `player_id = game_idx % 2` — this raised result from 67% → **78%**.
> Superseded runs: `eval_v3_maskable_20260306_190731.json` (94%, retracted) and `eval_v3_maskable_20260306_193442.json` (67%, superseded). See `project/experiments/reports/v3_validation_report.md`.

### V1 vs V3 Win Rate Comparison (canonical)
| Opponent | V1 (PPO, with fallback) | V3 (MaskablePPO) | Improvement |
|----------|------------------------|------------------|-------------|
| Random   | 51%                    | **95%**          | +44 pp      |
| RandomAgent | 43%                 | **91%**          | +48 pp      |
| GreedyAgent | 53%                 | **78%**          | +25 pp      |

Comparison plots: `project/experiments/reports/v3_figures/`
- `v1_vs_v3_eval_reward.png` — training eval reward curves
- `v3_eval_reward.png` — V3-only detailed curve
- `v1_vs_v3_win_rates.png` — win rate bar chart (regenerated after bug fix)

- [x] **Task 9.8**: V3 evaluation vs all opponents — complete (2026-03-06)
- [x] **Task 9.9**: V1 vs V3 comparison report/plots generated
- [x] **Task 9.10**: Bug found & fixed — `ValueBasedEvaluator` perspective error; intermediate eval JSON: `eval_v3_maskable_20260306_193442.json` (67%, superseded)
- [x] **Task 9.11**: First-mover bias fixed (`game_idx % 2`); canonical eval JSON: `eval_v3_maskable_20260306_204610.json` (78% vs GreedyAgent)

---

## 📅 Phase 10: Score-Based Ceiling Hunt (2026-03-06)

**Goal**: Determine the maximum win rate achievable with score-based rewards before pivoting to event-based shaping.  
**Stop-loss**: If best V4 experiment improves vs-Greedy by < 5 pp over V3 (78%), cut score-based and start Phase 11 (event rewards).

### Experiment Matrix

| Run | Config | Key change vs V3 | Hypothesis | Status |
|-----|--------|-----------------|------------|--------|
| V4a | `maskable_ppo_v4a_ent_lr.yaml` | `ent_coef` 0.005→0.01, `lr` 3e-4→1e-4 | More exploration, slower convergence | ⏳ Queued |
| V4b | `maskable_ppo_v4b_rollout_curriculum.yaml` | `n_steps` 2048→4096, greedy opp last 300K | Longer credit assignment + hard opponent fine-tune | ⏳ Queued |
| V4c | `maskable_ppo_v4c_curriculum.yaml` | 3-stage curriculum: none→random→greedy | Smooth difficulty ramp | ⏳ Queued |

### Evaluation Protocol
- After each run: `python project/scripts/evaluate_maskable_ppo.py --games 100`
- Compare to V3 baseline (78% vs GreedyAgent) using `compare_checkpoints.py`
- Decision gate: if best V4 < 83% vs Greedy → pivot to Phase 11

### Run Order
1. Launch V4a first (fastest, 1M steps, ~1h on RTX 4090)
2. While V4a trains, review TensorBoard at ~500K steps
3. If V4a looks promising → run V4b; else skip to V4c
4. Final decision at end of day

```bash
# V4a (run first)
python project/scripts/train_maskable_ppo.py \
  --config project/configs/training/maskable_ppo_v4a_ent_lr.yaml

# V4b (2-stage curriculum via train_curriculum.py)
python project/scripts/train_curriculum.py \
  --config project/configs/training/maskable_ppo_v4b_rollout_curriculum.yaml

# V4c (3-stage full curriculum)
python project/scripts/train_curriculum.py \
  --config project/configs/training/maskable_ppo_v4c_curriculum.yaml
```

### Results (fill in after each eval)
| Run | vs Random | vs RandomAgent | vs Greedy | Δ vs V3 Greedy | Decision |
|-----|-----------|---------------|-----------|----------------|----------|
| V3 (baseline) | 95% | 91% | **78%** | — | canonical |
| V4a (best@860K) | 90% | 89% | **82%** | **+4 pp** | ✅ improvement, 1pp below stop-loss |
| V4b | | | | | |
| V4c | | | | | |

> **V4a notes** (`maskable_ppo_v4a_ent_lr_20260306_213530/eval/best_model`, eval `20260307_090003.json`):  
> Higher `ent_coef` (0.01) + lower `lr` (1e-4) improved vs-Greedy by **+4 pp** (78→82%) but traded off vs-Random (95→90%) and vs-RandomAgent (91→89%). This is expected: more exploration discovers better greedy-countering strategies at the cost of random-opponent polish.  
> Stop-loss threshold: 83% — V4a reaches 82%, 1 pp short. **Recommend running V4c (curriculum) before deciding.**

---

## 📅 Phase 11 (Event-based rewards): pending Phase 10 stop-loss decision

### Documentation & Planning
- [x] Created implementation plan with 15 detailed tasks
- [x] Created ADR-001 (Use PPO)
- [x] Created training monitoring guide (TensorBoard)
- [x] Created score-based agent design document
- [x] Created environment verification script
- [x] Created WSL2 setup guide
- [x] Created state representation specification (135-dim vector)

### Phase 1: Environment Setup ✅
- [x] **Task 1.1**: WSL2 + GPU environment configured
  - Python 3.10.19 installed via Miniconda
  - PyTorch 2.5.1 + CUDA 12.1 installed
  - GPU verified: NVIDIA GeForce RTX 4090 (25.76 GB)
  - Stable-Baselines3 2.7.1 installed
  - All dependencies installed (gymnasium, pytest, pyyaml, tensorboard, etc.)
  - Splendor game environment tested and working

### Phase 2: State Representation ✅
- [x] **Task 2.1**: Design state vector specification (20 min)
  - 135-dim fixed-size vector designed
  - Active player hand: 35 dims
  - Opponent hand: 14 dims
  - Board state: 84 dims (gems, cards, nobles)
  - Game progress: 2 dims
  - Documentation: `specs/state_representation_spec.md`

- [x] **Task 2.2**: Implement state vectorizer (40 min)
  - File: `project/src/utils/state_vectorizer.py`
  - Class: `SplendorStateVectorizer`
  - Fixed-size output: (135,) float32
  - All values normalized to [0, 1]

- [x] **Task 2.3**: Test state vectorizer (20 min)
  - File: `project/tests/test_state_vectorizer.py`
  - 13 tests implemented and passing
  - Validated: shape, dtype, determinism, state changes, normalization

### Phase 3: Gym Wrapper ✅
- [x] **Task 3.1**: Create SB3-compatible wrapper (1 hour)
  - File: `project/src/utils/splendor_gym_wrapper.py`
  - Class: `SplendorGymWrapper`
  - Observation space: Box(0, 1, (135,), float32)
  - Action space: Discrete(200) with masking
  - 3 reward modes: score_naive, score_win_bonus, score_progress

- [x] **Task 3.2**: Test with SB3 check_env (30 min)
  - File: `project/tests/test_gym_wrapper.py`
  - 11 tests implemented and passing
  - SB3 `check_env()` passed ✅

### Phase 4: PPO Integration ✅
- [x] **Task 4.1**: Create training config (30 min)
  - File: `project/configs/training/ppo_score_based.yaml`
  - Hyperparameters: lr=0.0003, n_steps=2048, batch_size=64
  - Network: 3-layer MLP [256, 256, 128]
  - 1M timesteps planned

- [x] **Task 4.2**: Create training script (30 min)
  - File: `project/scripts/train_score_based.py`
  - TensorBoard logging enabled
  - Checkpoint callback (save every 50k steps)
  - Evaluation callback (eval every 10k steps)

### Phase 5: Quick Test ✅
- [x] **Task 5.1**: Run 10k step validation (10 min)
  - Config: `ppo_quick_test.yaml`
  - Training completed successfully
  - Speed: ~520 FPS on CPU

- [x] **Task 5.2**: Check TensorBoard logs (5 min)
  - Logs saved to: `project/logs/ppo_quick_test_20260224_112355/`
  - Agent learned to play: 1→24 step survival
  - Reward improved: -10 → -6.37
  - Model saved: `final_model.zip`

### Phase 6: Full Training & Evaluation ⚠️ (PARTIALLY INVALID)
- [x] **Task 6.1**: Launch 1M timestep training (1 hour)
  - Training completed: 2026-02-24 11:35 - 12:36
  - Model: `project/logs/ppo_score_based_v1_20260224_113524/final_model.zip`
  - Final reward: +27.99 ± 37.87 (improvement from -9.91)
  - Episode length: 29.7 ± 16.68 steps
  - 20 checkpoints saved (every 50K steps)
  - TensorBoard logs: `project/logs/ppo_score_based_v1_20260224_113524/logs/tensorboard`
  - ⚠️ **Training used `opponent_agent=None` (random branch only)**
  - ⚠️ **Wrapper `_opponent_move()` had wrong API call (`choose_action` → should be `choose_act`)**
  - ✅ Training itself was valid: random opponent path (`opponent_agent is None`) works correctly

- [x] **Task 6.2**: Create evaluation framework (1 hour)
  - File: `project/src/agents/ppo_agent.py` (Agent wrapper for arena)
  - File: `project/scripts/evaluate_score_based.py` (Evaluation script - **BUGGY**)
  - ❌ **CRITICAL BUG**: Evaluation script bypassed the Gym wrapper entirely
  - ❌ **BUG**: Direct SplendorEnv interaction had wrong game loop logic
  - ❌ **BUG**: `env.action_space.list_of_actions` returned actions for wrong player
  - ❌ **RESULT**: PPO always got invalid action_idx → fell back to action 0 → repeated `trade_gems` forever

- [❌] **Task 6.3**: Evaluate vs RandomAgent (100 games) — **RESULTS INVALID**
  - ~~Win Rate: 62.0%~~ → **INVALID** — games never completed properly
  - Debug revealed: Both players scored 0 points, games hit 200-turn limit
  - PPO always chose invalid actions → fallback to random → scores meaningless
  - "62% win rate" was artifact of broken scoring: `ppo_points >= opponent_points` when both = 0

- [❌] **Task 6.4**: Evaluate vs GreedyAgent-value (100 games) — **RESULTS INVALID**
  - ~~Win Rate: 60.0%~~ → **INVALID** — same bugs as above
  - Same broken evaluation logic

- [x] **Task 6.5**: Generate training report
  - File: `project/experiments/reports/ppo_score_based_training_report.md`
  - ⚠️ **Report contains invalid evaluation numbers** — needs update after re-evaluation

### Phase 7: Bug Discovery & Fix ✅ COMPLETE
- [x] **Task 7.1**: Debug evaluation quality (2026-02-25)
  - Created `debug_evaluation.py` to trace game execution
  - Discovered: PPO outputs action_idx=14, but only 1-2 legal actions exist
  - Discovered: games stuck in infinite `trade_gems` loop, 0 points
  - Root cause: evaluation script did NOT use Gym wrapper's `_update_legal_actions()`

- [x] **Task 7.2**: Identify wrapper bugs
  - Bug 1: `_opponent_move()` called `choose_action()` instead of `choose_act()` → FIXED Session 4
  - Bug 2: `_opponent_move()` called `choose_act()` directly, bypassing `load_observation()` on agent's private env → FIXED Session 5
  - Bug 3: Agent's private `self.env` never receives game state → root cause of Bug 2
  - ✅ Final fix: Call `choose_action(observation, [])` which goes through proper API path:
    `show_observation()` → `load_observation()` → `update_actions_light()` → `choose_act()`

- [x] **Task 7.3**: Fix evaluation script to use Gym wrapper
  - v1 (`evaluate_score_based_fixed.py`): Crashed on wrong API, then 0% win rate
  - v2 (`evaluate_score_based_v2.py`): Works but RandomAgent stuck (observation not loaded)
  - v3 (`evaluate_score_based_v3.py`): Fully working with fallback mode ✅

- [x] **Task 7.4**: Deep investigation of training quality
  - Created `debug_deep_diagnostic.py`: model DOES work inside wrapper
  - Model wins some games with 15-22 points (legitimate Splendor scores)
  - Main issue: ~40-60% of games have invalid actions (model picks idx >= n_legal)
  - This is inherent to Discrete(200) without action masking

- [x] **Task 7.5**: Re-evaluate with corrected code ✅
  - **v3 evaluation results (100 games per opponent, with fallback):**

  | Opponent | Win Rate | W/L/D | Agent Avg | Opp Avg | Max Score |
  |----------|----------|-------|-----------|---------|-----------|
  | Random (wrapper) | **51%** | 51/26/23 | 9.5 ± 7.6 | 6.5 ± 7.1 | 21 |
  | RandomAgent | **43%** | 43/37/20 | 9.0 ± 7.1 | 10.0 ± 6.6 | 20 |
  | GreedyAgent | **53%** | 53/30/17 | 10.1 ± 7.6 | 7.0 ± 7.2 | 22 |
  | Random (strict) | 31% | 31/9/60 | 5.2 ± 7.6 | 2.2 ± 4.9 | 24 |

  - **Fallback mode**: When model picks invalid action (idx >= n_legal), fall back to random legal action
  - **Strict mode**: Terminate episode on invalid action (same as training)
  - 40-60% of games have at least one invalid action → shows need for action masking
  - ~17-23% of games encounter 0 legal actions (Splendor engine edge case)

- [x] **Task 7.6**: Assessment — retraining NOT needed for Phase 1
  - Model has learned genuine Splendor strategy (53% vs Greedy is above random)
  - Action masking should be added in next training iteration
  - Current results establish a valid Phase 1 baseline

---

## 🚨 Critical Issues Found & Resolved (2026-02-25)

### Issue 1: Evaluation Script Bypassed Gym Wrapper ✅ RESOLVED
**Severity**: CRITICAL  
**Impact**: All Phase 6 evaluation results (62%, 60%) were INVALID  
**Root Cause**: `evaluate_score_based.py` used `SplendorEnv` directly instead of `SplendorGymWrapper`  
**Fix**: Created v2/v3 evaluation scripts that use the wrapper  
**Status**: ✅ FIXED — v3 evaluation produces real results (51%/43%/53% win rates)

### Issue 2: Wrapper `_opponent_move()` Never Loaded Game State ✅ RESOLVED
**Severity**: HIGH  
**Impact**: Opponent agents (RandomAgent, GreedyAgent) picked actions from stale/blank game state  
**Root Cause Two-Part**:
  - Session 4 fix: Changed `choose_action(obs)` → `choose_act('deterministic')` — but this bypassed `load_observation()`
  - Session 5 fix: Changed to `choose_action(observation, [])` which goes through the proper API path
**Fix Flow**: `show_observation('deterministic')` → `choose_action()` → `deterministic_choose_action()` → `load_observation()` → `update_actions_light()` → `choose_act()`  
**Status**: ✅ FIXED

### Issue 3: Model Outputs Invalid Action Indices ⚠️ KNOWN LIMITATION
**Severity**: MEDIUM  
**Impact**: 40-60% of games have at least one action_idx >= n_legal_actions  
**Root Cause**: `Discrete(200)` action space without action masking — model doesn't know how many actions are legal  
**Mitigation**: Fallback-to-random mode in evaluation (v3)  
**Permanent Fix**: Add action masking to training (planned for Phase 2)

### Issue 4: 0 Legal Actions Engine Edge Case ⚠️ KNOWN  
**Severity**: LOW  
**Impact**: ~17-23% of games encounter states with 0 legal actions  
**Root Cause**: Splendor engine returns empty action list in certain board states  
**Status**: Needs investigation (may be valid game-over state not caught by `is_done`)

---

## 🚀 Phase 1 Status — COMPLETE ✅

**Completion Date**: 2026-02-25  
**Total Duration**: 2 days (2026-02-24 to 2026-02-25, 3 sessions)  
**Overall Status**: ✅ TRAINING COMPLETE, EVALUATION VALIDATED

### Final Results (v3 evaluation, 100 games per opponent)
| Opponent | Win Rate (fallback) | Win Rate (strict) |
|----------|--------------------|--------------------|
| Random (built-in) | **51%** | 31% |
| RandomAgent | **43%** | — |
| GreedyAgent | **53%** | — |

### What Works
1. **Training Pipeline** ✅
   - PPO agent trained to 1M timesteps with random opponent
   - Stable learning curve: -9.91 → +27.99 reward
   - Episode length: 2 → 30 steps

2. **Evaluation Pipeline** ✅
   - v3 evaluation uses Gym wrapper (same interface as training)
   - Fallback mode for invalid actions allows measuring strategic capability
   - Strict mode for measuring raw policy quality
   - Proper opponent integration via `choose_action()` API

3. **Model Quality** ✅
   - Beats GreedyAgent 53% (with fallback) — above random baseline
   - Scores 9-10 avg points (legitimate Splendor gameplay, near 15-point threshold)
   - Reaches 20-24 points in best games

4. **Code Quality** ✅
   - 24/24 unit tests passing
   - State vectorizer + Gym wrapper fully functional
   - Comprehensive debug tooling

### Known Limitations
1. **No action masking**: 40-60% of games have invalid actions → need `MaskablePPO`
2. **0 legal actions edge case**: ~20% of games affected by Splendor engine quirk
3. **Only trained vs random**: Performance vs structured opponents not optimized

---

## 📊 Statistics (Updated)

**Date Range**: 2026-02-24 to 2026-02-25  
**Time Invested**: ~9 hours total
  - Day 1 Session 1 (Setup + Quick Test): 4 hours
  - Day 1 Session 2-3 (Training + Broken Eval + Report): 2 hours
  - Day 2 Session 4 (Bug Discovery): 1 hour
  - Day 2 Session 5 (Bug Fix + Re-evaluation): 2 hours
**Tasks Completed**: 14/14 (Phases 1-7) ✅  
**Code Written**: ~3500 lines  
**Tests Passing**: 24/24 (100%)  
**Training Completed**: 1,000,000 timesteps  
**Valid Evaluation Games**: 400 (v3: 100 per opponent × 4 configs)

**Bugs Found**: 4 (2 critical fixed, 2 known limitations)
**Evaluation Results JSON**: `project/experiments/evaluation/ppo_score_based_eval_v3/`

---

## ❌ Phase 8: Experiment 1 — Greedy Opponent Training (COMPLETE — NEGATIVE RESULT)

**Date**: 2026-02-25 to 2026-02-26  
**Status**: ❌ Hypothesis REJECTED  
**Session**: 6-7  
**Report**: `project/experiments/reports/ppo_v2_greedy_opponent_report.md`

### Hypothesis
Training PPO with a stronger (Greedy) opponent — identical hyperparams and reward to v1 — yields
a policy that beats v1 in head-to-head play, proving that opponent strength matters for PPO training.

### Result: **HYPOTHESIS REJECTED**
V2 performed dramatically worse than V1 in all matchups. Training against a Greedy opponent without
Action Masking causes the agent to learn passive, avoidance behavior instead of winning strategies.

### Tasks
- [x] Analyzed original fork codebase: found `StateEvaluatorHeuristic` (5-weight formula), `alpaca` MCTS framework
- [x] Created v2 config: `project/configs/training/ppo_score_based_v2_greedy_opp.yaml`
- [x] Created v2 training script: `project/scripts/train_score_based_v2.py`
- [x] Created dev log: `dev_logs/2026-02-25_session6_greedy_opp_experiment.md`
- [x] Launched training: tmux session `train_v2`
- [x] Training complete: 1M steps, final reward -3.90 (never sustained positive)
- [x] Evaluated v2 vs all opponents (100 games each)
- [x] Generated TensorBoard plots and comparison charts
- [x] Documented results in report

### V2 Evaluation Results (100 games each, fallback mode)

| Opponent | V2 Win Rate | V1 Win Rate | V2 Agent Score | V1 Agent Score |
|----------|-------------|-------------|----------------|----------------|
| Random (wrapper) | **3.0%** | 51.0% | 1.9 ± 3.4 | 9.5 ± 7.6 |
| RandomAgent | **3.0%** | 43.0% | 3.1 ± 4.2 | 9.0 ± 7.1 |
| GreedyAgent | **7.0%** | 53.0% | 2.8 ± 4.7 | 10.1 ± 7.6 |
| Random (strict) | **2.0%** | 31.0% | 1.5 ± 3.2 | 5.2 ± 7.6 |

### Key Training Metrics Comparison

| Metric | V1 | V2 |
|--------|----|----|
| Final Reward | +27.99 | -3.90 |
| Peak Reward | +38.43 | +6.62 |
| Explained Variance | 0.541 | **-0.167** |
| Clip Fraction | 0.111 | **0.346** |
| Invalid Actions/Game | ~10+ | **0.03-0.11** |

### Key Insight
V2's one "success" — near-zero invalid actions — came at the cost of strategic capability.
The agent learned to pick safe (always-legal) actions like gem collection, but never learned
to buy cards or pursue victory points. **Action Masking is prerequisite before training
against harder opponents.**

### Success Criteria (All FAILED)
- ❌ v2 wins > 55% vs v1 → v2 wins 3-7% against all opponents
- ❌ v2 avg VP ≥ v1 against GreedyAgent → v2: 2.8 vs v1: 10.1
- ❌ Training converges: `ep_rew_mean > 0` by 500K → Never sustained positive

---

## 📁 Artifacts

### Source Code
```
project/src/utils/
├── state_vectorizer.py          (440 lines, 13 tests)
├── splendor_gym_wrapper.py      (330 lines, 11 tests)
└── __init__.py

project/scripts/
├── train_score_based.py         (220 lines — v1, random opponent)
├── train_score_based_v2.py      (NEW — v2, greedy opponent)
├── evaluate_score_based.py      (DEPRECATED — buggy)
├── evaluate_score_based_v2.py   (266 lines — partial fix)
├── evaluate_score_based_v3.py   (276 lines — FINAL, with fallback mode)
├── evaluate_v1_vs_v2.py         (PLANNED — head-to-head tournament)
├── export_training_plots.py     (TensorBoard → PNG export)
├── debug_evaluation.py          (diagnostic tool)
├── debug_deep_diagnostic.py     (wrapper diagnostic)
└── debug_zero_actions.py        (engine edge case diagnostic)

project/configs/training/
├── ppo_score_based.yaml                   (v1 — Full 1M training, random opp)
├── ppo_score_based_v2_greedy_opp.yaml     (v2 — Full 1M training, greedy opp)
└── ppo_quick_test.yaml                    (10k validation)
```

### Documentation
```
project/docs/development/
├── PROGRESS.md                          (this file)
├── specs/state_representation_spec.md
└── dev_logs/
    ├── 2026-02-24_session1_project_setup.md
    ├── 2026-02-25_session4_bug_discovery.md
    ├── 2026-02-25_session5_bug_fix_evaluation.md
    └── 2026-02-25_session6_greedy_opp_experiment.md  (NEW)
```

### Evaluation Results
```
project/experiments/evaluation/
├── ppo_score_based_eval_v2/     (partial — before opponent fix)
├── ppo_score_based_eval_v3/     (FINAL v1 results)
│   └── eval_v3_20260225_*.json
└── ppo_score_based_v2_eval/     (v2 Greedy Opponent results)
    └── eval_v3_20260226_083927.json
```

### Experiment Reports
```
project/experiments/reports/
├── ppo_score_based_training_report.md        (v1 Phase 1 report)
├── ppo_v2_greedy_opponent_report.md          (v2 Experiment 1 report — NEGATIVE RESULT)
├── figures/                                   (v1 training plots)
│   ├── episode_reward_mean.png
│   └── episode_length_mean.png
└── v2_figures/                                (v2 training plots)
    ├── v2_episode_reward_mean.png
    ├── v2_episode_length_mean.png
    └── v1_vs_v2_training_comparison.png
```

---

## 🔧 Technical Notes

### Environment
- **Platform**: WSL2 (Ubuntu 22.04)
- **Python**: 3.10.19
- **PyTorch**: 2.5.1+cu121
- **SB3**: 2.7.1
- **GPU**: RTX 4090 (25.76 GB, CUDA 12.1)

### Performance
- **State vectorization**: <1ms per call
- **Training speed**: ~520 FPS (CPU), ~1000+ FPS (GPU expected)
- **Validation results** (10k steps):
  - Episode survival: 1 → 24 steps (24x)
  - Mean reward: -10 → -6.37 (36% improvement)

### Known Issues
- SB3 warning about GPU utilization with MLP (expected, not critical)
- Legacy gym deprecation warnings (harmless, from old Splendor code)
- **RESOLVED**: `_opponent_move()` now uses `choose_action(observation, [])` API path
- **RESOLVED**: Evaluation scripts now use Gym wrapper
- **KNOWN**: No action masking → 40-60% of games have invalid actions
- **KNOWN**: ~20% of games hit 0 legal actions (engine edge case)

---

## 📝 Lessons Learned

1. **WSL2 > Windows** for ML workflows (10-20x I/O, native packages)
2. **Pip > Conda** for PyTorch on WSL2 (fewer conflicts)
3. **Quick validation tests** save time before long runs
4. **135-dim state** sufficient for score-based agent
5. **Test coverage** caught 4 bugs before training
6. **Always evaluate through the same interface as training** — evaluating outside the wrapper causes action space mismatch
7. **Debug with game traces** — running a single verbose game reveals bugs instantly
8. **Low scores (0-2 pts) in Splendor = agent not playing real game** — legitimate games reach 15+ points
9. **Legacy agent API: `choose_action()` not `choose_act()`** — `choose_action()` loads the observation into the agent's private env, `choose_act()` assumes it's already loaded
10. **Action masking is critical** — without it, Discrete(200) lets the model pick indices that exceed the number of legal actions
11. **Don't train against too-strong opponents without Action Masking** — the agent learns avoidance behavior (picking safe but useless actions) instead of winning strategies
12. **Negative explained variance = failed training** — V2's -0.167 means the value network is worse than a constant predictor
13. **Clip fraction > 0.3 is a warning sign** — V2's 0.346 suggests the learning rate is too aggressive for the problem difficulty

---

**Last Updated**: 2026-02-26 (Session 7 — Experiment 1 Evaluation Complete)  
**Next Steps**: Implement MaskablePPO with Action Masking (Phase 2 priority). V1 remains the baseline.
