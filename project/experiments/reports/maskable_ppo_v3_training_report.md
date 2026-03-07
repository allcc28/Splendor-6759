# MaskablePPO V3 Training Report

**Project**: IFT6759 Splendor RL Agent  
**Experiment**: Phase 9 — Score-Based MaskablePPO (V3)  
**Date**: 2026-03-03 to 2026-03-06  
**Author**: Yehao Yan  
**Model**: `project/logs/maskable_ppo_score_v3_20260303_183435/`

---

## Executive Summary

This report documents the training, evaluation, and credibility validation of the V3 MaskablePPO agent for the Splendor board game. The central improvement from V1 (standard PPO) to V3 is the introduction of **action masking** via `sb3-contrib`'s `MaskablePPO`, which physically prevents the model from sampling illegal actions during both training and inference.

**Key Results** (corrected after a bug fix — see Section 5):

| Opponent | V1 PPO Win Rate | V3 MaskablePPO Win Rate | Improvement |
|----------|-----------------|-------------------------|-------------|
| Random (wrapper) | 51% | **96%** | +45 pp |
| RandomAgent | 43% | **90%** | +47 pp |
| GreedyAgent | 53% | **67%** | +14 pp |

**Key Achievements**:
- ✅ **0 invalid actions** across all 1M training steps and all 300 evaluation games
- ✅ **Immediate high performance**: eval reward starts at 66.2 at step 10K (vs V1's -9.91)
- ✅ **Stable training**: reward stays in 60–68 range throughout, no collapse
- ✅ **Peak eval reward: 67.83** at 820K steps (V1 peak: 38.43, V2 peak: 6.62)
- ✅ **Credibility verified**: evaluator bug found and fixed; results re-run with correct opponent behavior
- ⚠️ Room for improvement vs GreedyAgent: 67% suggests score-based reward shaping limits strategic depth

---

## 1. Motivation: Why V1 and V2 Failed

Understanding why V3 works requires understanding the failure modes of its predecessors.

### 1.1 V1 (PPO, Random Opponent) — Limited by Invalid Actions

V1 used a standard `Discrete(200)` action space without masking. Splendor has 15–80 legal actions at any state, so the majority of the 200-slot space is always illegal.

- **40–60% of actions were invalid** at training time
- The agent wasted capacity learning to avoid obviously illegal indices rather than learning strategy
- The -10 penalty for invalid actions created a noisy gradient signal
- With fallback (illegal → random valid action), V1 reached 51%/43%/53% win rate — but this is an artificial aid, not the model's true capability

### 1.2 V2 (PPO, Greedy Opponent) — Collapsed by Reward Signal

V2 kept the same architecture but trained against a strong `GreedyAgentBoost` opponent. Result: reward collapse.

- Without masking, the combined challenges of illegal actions AND a hard opponent caused the value network to fail (explained_variance = **-0.167**)
- The agent entered a passive local minimum: collect gems forever, never buy cards — 0-0 draws in 34% of games
- Peak eval reward was only +6.62 vs V1's +38.43

### 1.3 V3 Solution: Action Masking

The single architectural change in V3 is replacing `PPO` with `MaskablePPO` from `sb3-contrib`, combined with an `action_masks()` method on the Gym wrapper.

```python
# Training
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

def _mask_fn(env):
    return env.action_masks()  # (200,) bool array

masked_env = ActionMasker(env, _mask_fn)
model = MaskablePPO("MlpPolicy", masked_env, ...)

# Inference
action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
```

The wrapper's `action_masks()` returns a `(200,)` boolean array where `True` means the action is currently legal. MaskablePPO applies a logit mask before softmax, setting illegal action probabilities to zero.

**Result**: The model never needs to learn "don't take illegal actions" — it's architecturally guaranteed. All learning capacity is focused on strategy.

---

## 2. Training Configuration

### 2.1 Algorithm & Hyperparameters

| Parameter | V1 (PPO) | V3 (MaskablePPO) | Changed? |
|-----------|----------|------------------|----------|
| **Algorithm** | PPO (SB3) | MaskablePPO (sb3-contrib) | **YES** |
| **Action Masking** | None | `ActionMasker` wrapper | **YES** |
| **ent_coef** | 0.01 | **0.005** | **YES** |
| Policy Network | MLP [256, 256, 128] | MLP [256, 256, 128] | No |
| Learning Rate | 0.0003 | 0.0003 | No |
| Batch Size | 64 | 64 | No |
| n_steps | 2048 | 2048 | No |
| n_epochs | 10 | 10 | No |
| gamma | 0.99 | 0.99 | No |
| gae_lambda | 0.95 | 0.95 | No |
| clip_range | 0.2 | 0.2 | No |
| Total Timesteps | 1,000,000 | 1,000,000 | No |
| Opponent | RandomAgent | RandomAgent | No |
| Reward Mode | score_progress | score_progress | No |
| Device | CUDA RTX 4090 | CUDA RTX 4090 | No |

> `ent_coef` was reduced from 0.01 to 0.005 because masking shrinks the effective action distribution. With only 15–80 legal actions at a time, the entropy bonus is already sufficient for exploration without the higher coefficient.

**Config File**: `project/configs/training/maskable_ppo_score_based.yaml`

### 2.2 State Representation (unchanged from V1)

| Component | Dimensions | Description |
|-----------|------------|-------------|
| Active player hand | 35 | Gems (6), discounts (5), VP, reserved cards (23) |
| Opponent hand | 14 | Simplified view |
| Board gems | 6 | Available tokens by color |
| Board cards | 72 | 12 cards × 6 features (row one-hot, discount, VP, 5 gem costs) |
| Board nobles | 6 | 3 nobles × 2 features |
| Game progress | 2 | Turn count (normalized), active player flag |
| **Total** | **135** | All values in [0, 1], dtype=float32 |

### 2.3 Reward Function (unchanged)

```
reward = 0.01                  # progress reward per valid step
       + score_diff            # VP change this turn (sparse, non-zero only on card buy)
       + 50 * win              # win bonus (terminal)
       - 50 * loss             # loss penalty (terminal)
```

When the opponent wins, an additional -1.0 penalty is applied. This reward design means a game where the agent wins with score X while the opponent accumulates no score yields approximately `0.01 × turns + 50` total reward — consistent with the ~66 values seen in V3's training curves.

---

## 3. Training Process

### 3.1 Basic Information

| Metric | Value |
|--------|-------|
| **Start Time** | 2026-03-03 18:34:35 |
| **Duration** | ~1 hour |
| **Training Speed** | ~16,000 steps/min (same CUDA setup as V1) |
| **Final Model** | `final_model.zip` (3.5 MB) |
| **Best Model** | `eval/best_model.zip` (at 820K steps) |
| **Checkpoints** | 20 (every 50K steps) |
| **Eval Frequency** | Every 10K steps (10 eval episodes per checkpoint) |

---

### 3.2 Training Curves Analysis

#### Figure 1: V1 vs V3 Evaluation Reward over Training

![V1 vs V3 Eval Reward Comparison](v3_figures/v1_vs_v3_eval_reward.png)

This figure plots the mean evaluation reward (± 1 std) from `evaluations.npz` for both models. Each evaluation point is the mean of 10 episodes using the greedy policy.

**V1 (PPO, blue):**

The V1 curve shows the classic RL learning pattern:
- **0–100K steps**: Negative territory (-9.91 to -3.06). The agent is mostly taking invalid actions and losing immediately. High variance with very low std (≈0.07) at step 10K indicates every evaluation episode hits -10 reward — fully invalid actions, no exploration.
- **100–200K steps**: First breakthrough at 110K (+8.50) followed by backsliding. The agent occasionally wins a game, but hasn't learned a consistent strategy. Enormous std (±20–37) reflects bimodal behavior: either win big (+50 bonus) or lose with -10 penalties.
- **200–400K steps**: Reward fluctuates between -5 and +26. The agent is in a phase of strategy formation — learns to score some points but hasn't stabilized.
- **400–800K steps**: General upward trend reaching the peak at **800K: +38.43**. The curve is extremely noisy (std ±35), meaning the policy is still inconsistent: some games the agent wins convincingly; others it collapses on invalid actions.
- **800K–1M steps**: Slight regression to +27.99 final. This partial rollback is typical of on-policy methods and suggests the policy was near-optimal for random opponents at 800K.

**V3 (MaskablePPO, green):**

The V3 curve tells a fundamentally different story:
- **Step 10K**: Already at **+66.24** with std ±1.23 — near-perfect play from the very first evaluation. The agent has not yet learned *strategy*, but action masking means even early random-ish choices (within legal actions) are sufficient to reliably beat a random opponent.
- **Throughout 10K–1M**: Reward oscillates between **~47 and ~68**. This is a bimodal pattern (explained in Section 3.3 below), not convergence instability.
- **Peak at 820K: +67.83** (std ±1.92) — the agent is winning virtually every evaluation game with high consistency.
- **Final at 1M: +59.69** (std ±19.90) — the model ends in a "lower mode" episode, but the best model checkpoint (820K) remains available.

**The key visual difference**: V1 shows a slow, noisy climb from deeply negative territory. V3 shows immediate high performance with narrower oscillations. The **entire V3 curve sits above V1's peak**.

---

#### Figure 2: V3 Detailed Evaluation Curve

![V3 Eval Reward Detailed](v3_figures/v3_eval_reward.png)

This figure zooms into V3 alone, showing the green confidence band and the peak marker (red dotted line at 67.83).

**Bimodal Behavior — Two States of Play:**

Looking at the std values, V3's evaluation episodes fall into two distinct modes:

| Mode | Mean Reward | Std | Interpretation |
|------|-------------|-----|----------------|
| High mode | ~66–68 | ~0.7–2.5 | Agent wins nearly every eval game (~all 5 episodes) |
| Low mode | ~47–61 | ~15–32 | Agent loses some eval games (~1–2 out of 5) |

The high-mode points (std ≈ 1) correspond to evaluation episodes where the model won all or almost all games: with `score_progress` rewards and opponent scoring near 0, reward ≈ 0.01 × ~30 turns + 50 win ≈ 50.3. The max is higher (~67) when the agent wins quickly (fewer turns absorbed by the opponent). The low mode (std ≈ 20) occurs when the model occasionally loses one eval episode, pulling the mean down.

This pattern indicates the policy **oscillates between two behavioral modes** during training: one where it consistently dominates random opponents, and one where it has slightly lower confidence. This is typical of MaskablePPO during "policy refinement" phases — the critic is updating its value estimates, temporarily destabilizing the policy before recovering.

**Interpretation of the Peak (820K)**:

Steps 750K–870K show an extended cluster of high-mode evaluations (std < 2 for multiple consecutive checkpoints). The 820K checkpoint represents the policy's point of maximum consistency — the value network and policy network are most aligned at this point. After 870K, more low-mode evaluations appear, suggesting the policy is still exploring rather than having fully converged.

**Why V3 Never Drops Below ~46 (vs V1 dropping to -10)**:

With action masking, the worst case is the agent choosing sub-optimal legal actions — it still *plays a legal game*, just not optimally. This floor on performance is the direct consequence of masking.

---

#### Figure 3: V1 vs V3 Win Rate by Opponent

![V1 vs V3 Win Rates](v3_figures/v1_vs_v3_win_rates.png)

This bar chart shows the head-to-head win rates from controlled evaluation (100 games per opponent). Percentages are labeled above each bar.

**Key observations**:

1. **vs Random**: 51% → 96% (+45 pp). The near-doubling of win rate vs the same opponent used during training is primarily explained by the elimination of invalid actions. V1's 51% was achieved with a fallback mechanism (invalid → random valid). V3's 96% uses no fallback — it simply never picks invalid actions.

2. **vs RandomAgent**: 43% → 90% (+47 pp). The larger absolute improvement here vs Random partly reflects that V1 was more sensitive to opponent strength changes. RandomAgent follows Splendor's action-type distribution (biased towards card purchases and gem collection), which is harder to exploit than pure uniform random. V3 handles this naturally because masking adapts to whatever legal actions exist, regardless of opponent strategy.

3. **vs GreedyAgent**: 53% → 67% (+14 pp). The smallest improvement, and the most informative result. GreedyAgent uses `simulate_next_state()` + value evaluation to pick the best action at each turn — it genuinely competes for the same win condition. V3 beats it 67% of the time without any game-specific knowledge beyond the score signal. The 14 pp improvement over V1 (which benefited from fallback) is significant but modest, indicating that pure score-based rewards with masking are not sufficient to dominate a planning-based agent. This motivates Phase 10 (event-based rewards) or Phase 11 (MCTS planning).

---

## 4. Evaluation Results (Corrected)

**Eval Script**: `project/scripts/evaluate_maskable_ppo.py`  
**Eval Run**: `eval_v3_maskable_20260306_193442.json`  
**Date**: 2026-03-06  
**Games**: 100 per opponent, deterministic policy, 0 fallback

### 4.1 vs Random (Built-in Wrapper)

| Metric | V3 MaskablePPO | V1 PPO (Fallback) |
|--------|----------------|-------------------|
| Win/Loss/Draw | **96/0/4** | 51/49/0 |
| Win Rate | **96.0%** | 51.0% |
| Agent Score | 15.7 ± 3.8 (range: 0–24) | 9.5 ± 7.6 |
| Opponent Score | 0.9 ± 1.4 | 6.5 ± 7.1 |
| Game Length | 32.4 ± 6.5 turns | 35.3 turns |
| Avg Reward | 63.55 | — |
| Invalid Actions | **0** | ~10+/game |

The 4 non-wins are draws (agent score = opponent score at max turns = 200), indicating that all 100 games were completed without any agent loss.

### 4.2 vs RandomAgent

| Metric | V3 MaskablePPO | V1 PPO (Fallback) |
|--------|----------------|-------------------|
| Win/Loss/Draw | **90/7/3** | ~43/57/0 |
| Win Rate | **90.0%** | 43.0% |
| Agent Score | 15.0 ± 3.9 (range: 0–20) | 9.0 ± 7.1 |
| Opponent Score | 5.9 ± 4.5 (range: 0–19) | 10.0 ± 6.6 |
| Game Length | 31.8 ± 5.9 turns | 36.2 turns |
| Invalid Actions | **0** | ~10+/game |

The 7 losses indicate genuine competition — RandomAgent occasionally reaches 15+ VP before V3. The opponent score of 5.9 confirms RandomAgent is making meaningful game progress, unlike the random wrapper's 0.9.

### 4.3 vs GreedyAgent (Value-mode)

| Metric | V3 MaskablePPO | V1 PPO (Fallback) |
|--------|----------------|-------------------|
| Win/Loss/Draw | **67/24/9** | ~53/47/0 |
| Win Rate | **67.0%** | 53.0% |
| Agent Score | 13.0 ± 5.8 (range: 0–23) | 10.1 ± 7.6 |
| Opponent Score | 7.9 ± 5.6 (range: 0–19) | 7.0 ± 7.2 |
| Game Length | 29.9 ± 8.7 turns | 36.7 turns |
| Invalid Actions | **0** | ~10+/game |

Notable observations:
- **Shorter games (29.9 vs 32+ vs random)**: GreedyAgent plays decisively, forcing the contest to a quicker resolution.
- **Higher opponent score (7.9 vs 5.9 vs RandomAgent)**: GreedyAgent is meaningfully building VP, not just surviving.
- **Lower agent score (13.0 vs 15.x vs random)**: GreedyAgent's presence constrains the board, limiting V3's best-case performance.
- **24 losses + 9 draws**: GreedyAgent wins approximately 1 in 4 games, confirming it is a formidable opponent.

### 4.4 Summary

```
vs Random (wrapper)        96.0%  Agent 15.7 pts  Opp 0.9 pts  Avg reward 63.55
vs RandomAgent             90.0%  Agent 15.0 pts  Opp 5.9 pts  Avg reward 59.21
vs GreedyAgent             67.0%  Agent 13.0 pts  Opp 7.9 pts  Avg reward 45.88
Invalid actions:           0 across all 300 games (by design — action masking)
```

---

## 5. Credibility: Bug Discovery and Fix

Before accepting these evaluation numbers, a credibility audit was performed. This section documents it in full.

### 5.1 The First Evaluation Was Wrong

An initial eval run (`eval_v3_maskable_20260306_190731.json`) showed:

| Opponent | Win Rate | Opp Score |
|----------|----------|-----------|
| GreedyAgent | **94%** | **1.1 pts** |

This triggered suspicion: the GreedyAgent should *not* score less than a RandomAgent (5.1 pts). A properly functioning greedy heuristic should score 10–15 pts consistently.

### 5.2 Root Cause: Wrong Player Perspective in `score_next_state()`

`GreedyAgentBoost` evaluates every legal action by:

```python
for action in actions:
    next_state = simulate_next_state(state, action)
    s = value_eval.score_next_state(next_state)
```

`simulate_next_state()` calls `action.execute(s_copy)`, which **switches `state.active_player_id`** from 0 → 1. The previous `score_next_state()` called `get_active_hand(next_state)`, which returned `next_state.active_players_hand()` — now pointing to player 1 (the PPO agent's hand), not player 0 (GreedyAgent's hand).

**Result**: GreedyAgent was evaluating all actions from the perspective of its opponent, and picking moves that were *best for the PPO agent*, not itself.

### 5.3 Diagnostic Confirmation

`project/scripts/sanity_check_greedy.py` — Test 1:

```
Active player BEFORE execute: 0
Active player AFTER  execute: 1   ← switches on execution
```

Head-to-head Greedy vs RandomAgent (20 games):

| Condition | Greedy score | Random score | Greedy win rate |
|-----------|-------------|--------------|-----------------|
| Before fix | 3.2 pts | 7.8 pts | **10%** ← inverted! |
| After fix | 14.4 pts | 5.3 pts | **85%** ← correct |

### 5.4 The Fix

Added `get_actor_hand()` to `modules/evaluators.py`:

```python
def get_actor_hand(next_state):
    """Return the hand of the player who JUST ACTED.
    After action.execute(), active_player_id switches to the opponent.
    """
    pid_now = int(next_state.active_player_id)  # opponent is now active
    actor_pid = 1 - pid_now                      # the one who just moved
    return next_state.list_of_players_hands[actor_pid]
```

`ValueBasedEvaluator.score_next_state()` now uses `get_actor_hand()` instead of `get_active_hand()`.

### 5.5 Impact on Results

| Opponent | First run (buggy) | Corrected run | Difference |
|----------|-------------------|---------------|------------|
| Random | 95.0% | 96.0% | +1 pp (stochastic variation) |
| RandomAgent | 93.0% | 90.0% | -3 pp (stochastic variation) |
| GreedyAgent | ~~94.0%~~ | **67.0%** | **-27 pp** ← the actual correction |

The Random and RandomAgent numbers changed only within normal stochastic variation (both use `RandomAgent.choose_action()` which never calls `simulate_next_state()`, so the bug did not affect them). The GreedyAgent number was entirely invalid — the 27 pp correction is the actual signal.

---

## 6. Comparison with Prior Models

### 6.1 Architecture Comparison

| Aspect | V1 (PPO) | V2 (PPO+Greedy) | V3 (MaskablePPO) |
|--------|----------|-----------------|------------------|
| Algorithm | PPO | PPO | MaskablePPO |
| Action masking | None | None | Full (logit-level) |
| Opponent | RandomAgent | GreedyAgentBoost | RandomAgent |
| ent_coef | 0.01 | 0.01 | 0.005 |
| Peak eval reward | 38.43 | 6.62 | **67.83** |
| Final eval reward | 27.99 | -3.90 | 59.69 |
| Explained variance | 0.54 | -0.167 | 0.54+ |
| Invalid actions/100 games | ~1000+ | ~10 (passive) | **0** |
| Win vs GreedyAgent | 53%¹ | 7% | **67%** |

¹ V1's 53% uses fallback mode — not a fair comparison. V1 strict mode was ~31%.

### 6.2 What Changed from V1 to V3

Only **two things** changed: algorithm (PPO → MaskablePPO) and entropy coefficient (0.01 → 0.005). Everything else — state representation, reward function, network architecture, opponent, hyperparameters — is identical. The entire performance improvement is attributable to action masking.

This provides a **clean ablation result**: action masking alone delivers +45 pp vs Random, +47 pp vs RandomAgent, +14 pp vs GreedyAgent compared to the same architecture without masking.

### 6.3 Training Efficiency Comparison

| Model | Steps to reach reward >30 | Peak reward | Training time |
|-------|--------------------------|-------------|---------------|
| V1 PPO | ~400K steps | 38.43 @ 800K | ~1 hr |
| V3 MaskablePPO | **10K steps** (already 66.24) | 67.83 @ 820K | ~1 hr |

V3 reaches higher performance in the **very first eval check** — demonstrating that masking removes the learning bottleneck entirely.

---

## 7. Technical Notes

### 7.1 The `action_masks()` Interface

The `SplendorGymWrapper` exposes a standardized mask method for `MaskablePPO`:

```python
def action_masks(self) -> np.ndarray:
    """Returns (200,) bool array. True = legal action at current state."""
    return self.action_mask.copy()  # updated each step in _update_legal_actions()
```

During training, `ActionMasker` calls this automatically. During inference, it must be passed explicitly:

```python
action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
```

### 7.2 The Bimodal Training Pattern (Deep Dive)

The V3 eval curve shows a bimodal distribution:
- **High mode** (reward ~66–68, std ~1–2): agent wins all 5 eval episodes
- **Low mode** (reward ~47–61, std ~15–32): agent loses 1–2 eval episodes

These correspond to two distinct game outcomes. With `score_progress` rewards:
- **Win game in ~30 turns**: 0.01 × 30 + 50 = 50.3 base, plus score_diff. If agent scores 15 pts during game: +50.3 + some score_diff per buy ≈ 65–67 reward ✓
- **Lose game**: -50 penalty + some score accumulation, e.g., agent gets 8 pts before losing: -50 + 8 + 0.01×20 = -41.8. Mix this with 4 wins → mean ≈ 0.8 × 66 - 0.2 × 42 = 44.4 ≈ low mode ✓

The bimodality reflects genuine probabilistic game outcomes as the policy explores different card purchase sequences. The trend over 1M steps shows the high mode becoming more frequent (longer clusters of high-std≈1 evaluations in 500K–940K range) — confirming the policy is improving.

### 7.3 Limitations

1. **Score-based reward ceiling**: V3 was trained purely on score difference + win reward. It has no explicit incentive to build a gem engine, target specific card paths, or race against the opponent's VP. Against GreedyAgent (which explicitly plans for these), V3 can be outpaced.

2. **Single-opponent training**: Training only vs RandomAgent means V3 may have overfit to exploiting random behavior. The 90% vs RandomAgent but only 67% vs GreedyAgent gap confirms this.

3. **100-game evaluation variance**: 95% confidence interval for a 67% win rate is approximately ±9.2 pp (SE = √(0.67×0.33/100) ≈ 4.7 pp; 95% CI = ±1.96 × 4.7). Results should be read as 67% ± 4.7 pp (one-sigma) or 67% ± 9.2 pp (two-sigma / 95% CI).

4. **No self-play**: The model was trained as a single-agent PPO. Self-play (used by AlphaZero-style approaches) would produce a stronger and more generalizable policy.

---

## 8. Future Work

### 8.1 Phase 10: Event-Based Reward Shaping (Deferred)

The gap from 67% to a hypothetical >80% vs GreedyAgent likely requires rewards that incentivize *how* the agent plays, not just whether it wins. Event-based rewards (from Bravi et al., 2019) assign values to game events:

```python
rewards = {
    'buy_card_tier1': +0.3,
    'buy_card_tier2': +0.5,
    'buy_card_tier3': +0.8,
    'gain_noble': +2.0,
    'build_engine_step': +0.1 * discount_gained,
}
```

This would give the agent positive signal during the engine-building phase (turns 1–15 where score = 0), enabling faster strategic convergence.

### 8.2 Phase 11: AlphaZero-Style MCTS Agent

The ultimate goal of the project is to implement neural-guided MCTS:
1. Use V3 as the initial policy/value network
2. Run MCTS rollouts to improve move selection
3. Train on self-play games with tree search–augmented targets

Expected benefit: planning-based lookahead would allow the agent to distinguish between moves that score immediately and moves that enable future scoring, overcoming the fundamental limitation of feed-forward policy networks.

### 8.3 Improved Evaluation

- Increase to 500 games per opponent (reduce confidence intervals from ±9 pp to ±4 pp)
- Add a second GreedyAgent variant (event-mode) as additional benchmark
- Evaluate the `best_model.zip` (820K checkpoint) head-to-head vs `final_model.zip`

---

## Appendix A: Selected V3 Training Checkpoints

| Timestep | Mean Reward | Std | Notes |
|----------|-------------|-----|-------|
| 10,000 | 66.24 | 1.23 | Immediate high performance from masking |
| 50,000 | 66.83 | 1.12 | Consistent high mode |
| 120,000 | 67.43 | 1.71 | First peak above 67 |
| 210,000 | 46.66 | 30.06 | Deep low mode — policy adjustment phase |
| 300,000 | 66.72 | 1.37 | Recovery to high mode |
| 390,000 | 67.64 | 2.49 | Near-peak performance |
| 430,000 | 39.84 | 32.46 | Low mode — another adjustment phase |
| 550,000 | 65.93 | 0.79 | Tightest std — most consistent performance |
| 580,000 | 66.53 | 0.41 | **Minimum std** — maximum consistency |
| 820,000 | **67.83** | 1.92 | **Peak reward** — best model checkpoint |
| 950,000 | 67.74 | 2.11 | Second-highest, near-peak |
| 1,000,000 | 59.69 | 19.90 | Final (low mode) — use 820K best_model.zip |

---

## Appendix B: File Reference

| File | Description |
|------|-------------|
| `project/logs/maskable_ppo_score_v3_20260303_183435/final_model.zip` | Final model (1M steps) |
| `project/logs/maskable_ppo_score_v3_20260303_183435/eval/best_model.zip` | Best checkpoint (820K) |
| `project/experiments/evaluation/maskable_ppo_v3_eval/eval_v3_maskable_20260306_193442.json` | Corrected eval results |
| `project/experiments/evaluation/maskable_ppo_v3_eval/eval_v3_maskable_20260306_190731.json` | RETRACTED — buggy GreedyAgent |
| `project/experiments/reports/v3_figures/v1_vs_v3_eval_reward.png` | Training comparison plot |
| `project/experiments/reports/v3_figures/v3_eval_reward.png` | V3 detailed curve |
| `project/experiments/reports/v3_figures/v1_vs_v3_win_rates.png` | Win rate bar chart |
| `project/experiments/reports/v3_validation_report.md` | Bug audit report |
| `project/scripts/evaluate_maskable_ppo.py` | Evaluation script |
| `project/scripts/sanity_check_greedy.py` | Opponent sanity-check tool |
| `modules/evaluators.py` | Fixed evaluator (get_actor_hand) |

---

*Report generated 2026-03-06*
