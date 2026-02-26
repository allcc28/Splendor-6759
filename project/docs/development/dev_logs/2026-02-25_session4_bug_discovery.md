# Session 4: Evaluation Bug Discovery & Investigation

**Date**: 2026-02-25  
**Duration**: ~1.5 hours  
**Focus**: Evaluation code audit, critical bug discovery, documentation correction  

---

## Session Goals

1. Re-run evaluation to verify results for user
2. Audit evaluation code quality and correctness
3. Document and fix any discovered issues
4. Update all project documentation

---

## Timeline

### 1. Re-run Evaluation Demo (15 min)

Ran `evaluate_score_based.py --games 20` to demonstrate evaluation process:
- vs RandomAgent: 65% win rate (2 seconds, 9.42 it/s)
- vs GreedyAgent: 65% win rate (29 seconds, 1.47 s/it)

**Initial suspicion**: Games completed suspiciously fast for Splendor.

### 2. User Questions Evaluation Logic (10 min)

User asked: "How do you prove our evaluation? How does it fight against random, what are the win rules, how is win rate calculated?"

Provided detailed code walkthrough of:
- `play_game_ppo_vs_agent()` function (lines 33-94)
- Win condition logic (line 92)
- Win rate calculation (line 152)

### 3. Critical Bug Discovery (30 min)

**User said**: "How can it be this fast? Review and check all our code"

Created `debug_evaluation.py` to trace a single game turn-by-turn.

#### Debug Output (DEVASTATING)
```
Turn 0: Active Player = 0, PPO Score: 0, Random Score: 0
  PPO chooses: trade_gems
  WARNING: Invalid action_idx=13, legal=2
Turn 20: PPO Score: 0, Random Score: 0
  WARNING: Invalid action_idx=14, legal=1
Turn 40: PPO Score: 0, Random Score: 0
  WARNING: Invalid action_idx=14, legal=1
...
Turn 200: Game hit max turns
FINAL: PPO=0, Random=0, Winner=Random (by tiebreak)
```

#### Root Cause Analysis

**Bug 1: Evaluation bypassed Gym wrapper**

The evaluation script (`evaluate_score_based.py`) interacted directly with `SplendorEnv` instead of going through `SplendorGymWrapper`. This caused a fundamental mismatch:

| Aspect | Training (Wrapper) | Evaluation (Direct) |
|--------|-------------------|-------------------|
| Legal actions | `_update_legal_actions()` → `cached_legal_actions` | `env.action_space.list_of_actions` (unsynced) |
| Action mapping | `action_idx` → `cached_legal_actions[idx]` | `action_idx` → same list but different timing |
| Opponent turn | `_opponent_move()` auto-called | Manual alternation (but env auto-switches!) |
| Player tracking | `self.player_id` fixed | `env.active_player_id()` changes |

The PPO model outputs indices into `cached_legal_actions` (updated by `_update_legal_actions()`), but the evaluation code called `env.action_space.list_of_actions` directly — producing a different, unsynced list.

**Bug 2: Action index always out of range**

PPO consistently output action_idx=14, but there were typically only 1-2 legal actions. The fallback code `action_idx = 0` always picked the first action (trade_gems), leading to an infinite gem-trading loop with 0 score.

**Bug 3: Double player switching**

`SplendorEnv.step()` internally switches the active player via `action.execute(state)`. The evaluation code also manually checked `env.active_player_id()` to decide whose turn it was. This caused desync where multiple moves went to the same player or turns were skipped.

**Bug 4: _opponent_move() wrong API (in wrapper)**

```python
# WRONG (wrapper line 243):
action = self.opponent_agent.choose_action(observation_obj)

# CORRECT:
action = self.opponent_agent.choose_act('deterministic')
```

`choose_action()` requires `(observation, previous_actions)` — a different parent class method. All legacy agents use `choose_act(mode)` which internally accesses `self.env`.

### 4. Impact Assessment (15 min)

**What's valid:**
- ✅ Training pipeline (used `opponent_agent=None` → random branch, no API bug)
- ✅ 1M timesteps completed, reward improved -9.91 → +27.99
- ✅ Unit tests (24/24 passing — but they don't test evaluation!)
- ✅ State vectorizer (independent of evaluation)

**What's invalid:**
- ❌ 62% win rate vs RandomAgent — games never properly completed
- ❌ 60% win rate vs GreedyAgent — same broken evaluation
- ❌ Training report evaluation sections — contain fabricated numbers
- ❌ `ppo_agent.py` — likely has similar wrapper bypass issues

**Why "62% win rate" appeared:**
```python
# Line 92 of evaluate_score_based.py:
ppo_won = info.get('winner_id') == ppo_id if done else (ppo_points >= opponent_points)
```
When both players score 0 (game never really played), and `done=False` (hit max turns), the tiebreak `ppo_points >= opponent_points` → `0 >= 0` → `True` → PPO "wins". Approximately half the games showed PPO "winning" this way, giving ~60% appearance.

### 5. First Fix Attempt (20 min)

Created `evaluate_score_based_fixed.py` using `SplendorGymWrapper` directly (same as training).

**First attempt**: Crashed with `choose_action() missing argument` — the wrapper bug.  
**Fixed**: Changed `_opponent_move()` to use `choose_act('deterministic')`.  
**Second attempt result**:
```
vs RandomAgent (10 games): 0% win rate, 0 avg score, 401 avg turns
vs GreedyAgent: 0% (3 games completed before timeout, ~50s each)
```

The wrapper-based evaluation also shows 0 scores. This means the PPO model itself may have learned a non-functional policy, or there's still a remaining issue with how the model interacts with the wrapper during evaluation.

---

## Bugs Found Summary

| # | Bug | Severity | Location | Status |
|---|-----|----------|----------|--------|
| 1 | Eval bypasses wrapper | CRITICAL | `evaluate_score_based.py` | Identified, fix in progress |
| 2 | `_opponent_move()` wrong API | HIGH | `splendor_gym_wrapper.py:243` | ✅ FIXED |
| 3 | `_opponent_move()` missing `update_actions_light()` | HIGH | `splendor_gym_wrapper.py` | ✅ FIXED |
| 4 | PPO always outputs invalid action_idx | MEDIUM | Model behavior | Under investigation |
| 5 | "62% win rate" is artifact of 0=0 tiebreak | CRITICAL | `evaluate_score_based.py:92` | All results invalidated |

---

## File Changes

### Modified
- `project/src/utils/splendor_gym_wrapper.py` — Fixed `_opponent_move()` API call
- `project/docs/development/PROGRESS.md` — Rewrote Phase 6 status to reflect invalid evaluation

### Created
- `project/scripts/debug_evaluation.py` — Game trace debugging tool
- `project/scripts/evaluate_score_based_fixed.py` — Wrapper-based evaluation (WIP)
- `project/docs/development/dev_logs/2026-02-25_session4_bug_discovery.md` — This file

---

## Open Questions

1. **Did the PPO model learn anything useful?**
   - Training reward increased -9.91 → +27.99
   - But this was against random opponent inside wrapper
   - The model outputs action_idx=14 consistently — is this a valid strategy inside the wrapper?

2. **Why does wrapper evaluation also give 0 scores?**
   - The model was trained through the wrapper, so it should work
   - Possible: max_turns difference (training=120, eval=500)
   - Possible: opponent type difference (training=None/random, eval=RandomAgent object)
   - Need to test with exact same config as training

3. **Is the training reward metric meaningful?**
   - +27.99 reward comes from `score_progress` mode: `score_diff + 0.01 + 50*win`
   - If PPO gets -10 for invalid actions, and some episodes it gets 0.01*30 + 50 = 50.3 for winning
   - Need to check if training episodes actually had wins

---

## Next Steps (Session 5)

1. Debug with exact training configuration (opponent=None, max_turns=120)
2. Check if model works at all inside the wrapper
3. If model is broken: analyze why and retrain
4. If model works: fix evaluation to match training config
5. Re-evaluate and update all reports with real numbers
6. Commit fixes and push

---

## Reflection

This session revealed a critical oversight: **we never validated the evaluation pipeline end-to-end**. The unit tests checked individual components (vectorizer, wrapper) but not the full game loop. The "62% win rate" passed casual inspection because it was a plausible number, but the underlying data (avg 2 points in a game that requires 15 to win) should have raised red flags immediately.

**Key lesson**: Always sanity-check evaluation metrics against domain knowledge. In Splendor, any evaluation showing average scores below 10 (when 15 is needed to win) is suspicious and likely indicates broken game logic.
