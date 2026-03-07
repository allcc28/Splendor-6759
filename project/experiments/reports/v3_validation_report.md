# V3 MaskablePPO Evaluation Validation Report

**Date**: 2026-03-06  
**Author**: Project session (Yan Hao)  
**Purpose**: Verify the authenticity of V3 agent evaluation results before publishing

---

## 1. Summary

Initial evaluation results showed V3 MaskablePPO winning **94% of games against GreedyAgent** — a suspiciously high figure even higher than against random opponents (95%). An audit revealed a critical bug in the opponent evaluator. After fixing the bug, the **corrected win rate vs GreedyAgent is 67%**, which is both credible and still a substantial improvement over V1 PPO (53%).

---

## 2. Initial (Buggy) Results

**Eval run**: `eval_v3_maskable_20260306_190731.json`  
**Model**: `project/logs/maskable_ppo_score_v3_20260303_183435/final_model.zip`

| Opponent | Win Rate | Agent Score | Opp Score |
|----------|----------|-------------|-----------|
| Random (wrapper) | 95.0% (95/0/5) | 15.4 ± 4.1 | 0.9 ± 1.7 |
| RandomAgent | 93.0% (93/4/3) | 15.5 ± 3.5 | 5.1 ± 4.0 |
| GreedyAgent | 94.0% (94/1/5) | 15.3 ± 4.2 | **1.1 ± 2.0** |

**Red flags that triggered the audit:**
- GreedyAgent (a state-evaluating heuristic) only scored **1.1 pts** — lower than RandomAgent's **5.1 pts**
- A properly functioning GreedyAgent should outperform RandomAgent by a wide margin
- Win rates vs GreedyAgent (94%) and Random (95%) were nearly identical
- Games vs GreedyAgent took 6.4s each while Random took 0.12s, confirming computation was happening — but it was computing wrongly

---

## 3. Bug Discovery

### 3.1 Root Cause

`GreedyAgentBoost` (mode="value") evaluates actions by:

```python
# modules/agents/greedy_agent_boost.py
def choose_act(self, mode):
    for action in actions:
        next_state = simulate_next_state(state, action)
        s = self.value_eval.score_next_state(next_state)
```

`simulate_next_state` creates a state copy and calls `action.execute(s_copy)`. The critical issue: **`action.execute()` switches the active player in Splendor's state object**. The `active_player_id` changes from 0 → 1 after execution.

`ValueBasedEvaluator.score_next_state` then called `get_active_hand(next_state)`, which called `next_state.active_players_hand()` — returning the **opponent's (PPO agent's) hand**, not the GreedyAgent's own hand.

### 3.2 Evidence from Sanity Check

Running `project/scripts/sanity_check_greedy.py` (Test 1):

```
Active player BEFORE execute: 0
Active player AFTER  execute: 1

❌ BUG CONFIRMED: action.execute() switches active player!
   GreedyAgent evaluates opponent's (PPO agent's) hand — WRONG!
```

Head-to-head before the fix (Greedy vs RandomAgent, 20 games):
```
Greedy avg score:  3.2 ± 5.0
Random avg score:  7.8 ± 8.1
Greedy win rate:   2/20 = 10%
❌ WARNING: Greedy scores LESS than Random
```

This confirms the evaluator was completely inverted — GreedyAgent was optimizing moves that *benefited the opponent*, not itself.

---

## 4. Fix Applied

**File**: `modules/evaluators.py`

Added `get_actor_hand()` function:

```python
def get_actor_hand(next_state):
    """
    Return the hand of the player who JUST ACTED (not the now-active player).
    
    IMPORTANT: action.execute(state) SWITCHES the active player in Splendor.
    So after simulate_next_state(), active_player_id points to the OPPONENT,
    not the agent who made the move.
    """
    try:
        pid_now = int(next_state.active_player_id)  # opponent is now active
        actor_pid = 1 - pid_now                      # the one who just moved
        return next_state.list_of_players_hands[actor_pid]
    except Exception:
        return get_active_hand(next_state)  # fallback
```

Changed `ValueBasedEvaluator.score_next_state` from `get_active_hand(next_state)` → `get_actor_hand(next_state)`.

**Verification** (Greedy vs RandomAgent, 20 games, post-fix):
```
Greedy avg score:  14.4 ± 5.0
Random avg score:  5.3 ± 3.8
Greedy win rate:   17/20 = 85%
✅ Greedy outperforms Random as expected
```

---

## 5. Corrected Results

**Eval run**: `eval_v3_maskable_20260306_193442.json`  
**Model**: same (`final_model.zip`)

| Opponent | Win Rate | Agent Score | Opp Score |
|----------|----------|-------------|-----------|
| Random (wrapper) | **96.0%** (96/0/4) | 15.7 ± 3.8 | 0.9 ± 1.4 |
| RandomAgent | **90.0%** (90/7/3) | 15.0 ± 3.9 | 5.9 ± 4.5 |
| GreedyAgent | **67.0%** (67/24/9) | 13.0 ± 5.8 | 7.9 ± 5.6 |

The vs-Random and vs-RandomAgent results are consistent with the previous run (slight variation due to stochasticity), confirming that evaluator was not involved in those cases (both use the `opponent_agent=None` path or `RandomAgent.choose_action` which doesn't call `simulate_next_state`).

---

## 6. Interpretation of Corrected Results

### 6.1 V1 (PPO) vs V3 (MaskablePPO) Win Rate Comparison

| Opponent | V1 (PPO) | V3 (MaskablePPO) | Improvement |
|----------|----------|------------------|-------------|
| Random   | 51%      | **96%**          | +45 pp      |
| RandomAgent | 43%   | **90%**          | +47 pp      |
| GreedyAgent | 53%   | **67%**          | +14 pp      |

### 6.2 What the Results Say

**vs Random (wrapper)**: 96% is expected — same distribution as training. Agent has learned excellent coverage of all game situations arising when the opponent plays uniformly randomly.

**vs RandomAgent**: 90% shows generalization beyond training conditions. RandomAgent uses Splendor's proper action-type distribution (not uniform across all 200 indices). Small drop from 96% is expected.

**vs GreedyAgent: 67%** is the most informative result:
- GreedyAgent wins 24% of games (24/100 strict losses + 9 draws) — it is a formidable opponent
- Agent average score drops to 13.0 pts vs 15.x vs random opponents — GreedyAgent forces harder games
- Game length drops slightly (29.9 vs 32 turns) — GreedyAgent converges faster
- **67% vs a hand-crafted greedy heuristic is solid for a model trained with only score-based rewards** (no game-specific knowledge)

### 6.3 Remaining Limitations

1. **Score-based reward only**: V3 was trained purely on score difference + win bonus. Higher win rates vs GreedyAgent would likely require event-based reward shaping (Phase 10) that incentivizes strategic objectives (gem engine, card paths).

2. **GreedyAgent as benchmark ceiling**: GreedyAgent uses lookahead simulation + value function. V3's 67% win rate is a realistic competitive baseline before adding planning (Phase 3 MCTS).

3. **Evaluation sample size**: 100 games per opponent. Confidence interval at 95%: ±4.4 pp for a 67% estimate. The vs-GreedyAgent 67% result should be read as approximately 63–71%.

---

## 7. Impact on Previous Claims

The following results from earlier documentation must be treated as **INVALID** due to the evaluator bug:

- `eval_v3_maskable_20260306_190731.json`: V3 vs GreedyAgent 94.0% win — **retracted**  
- Any comparison table showing "V3 vs GreedyAgent ≥ 90%" — **use 67% from corrected run**
- PROGRESS.md has been updated to reflect corrected figures

All other V3 results (vs Random, vs RandomAgent) are valid and consistent across both runs.

---

## 8. Lessons Learned

1. **Cross-validate opponent behavior**: When an opponent scores significantly less than random, the opponent is likely broken — audit before reporting.
2. **State simulation perspective**: In turn-based games, `action.execute()` typically switches the active player. Any evaluator using `simulate_next_state()` must explicitly get the hand of the player who acted, not the current `active_player()`.
3. **Sanity checks should be mandatory**: `project/scripts/sanity_check_greedy.py` was created specifically for this audit. Should be added to the test suite.

---

*Generated during validation session, 2026-03-06*
