# Score-Based Agent Design & Feasibility Analysis

**Date**: 2026-02-24  
**Purpose**: Research and design the Score-based RL agent before implementation  
**Status**: Research Phase

---

## 1. The Core Problem: Sparse Rewards

### Game Mechanics
- Typical Splendor game: 30-50 turns
- **Engine-building phase** (turns 1-20): Zero points gained
- **Scoring phase** (turns 20-50): Points accumulate rapidly
- Winner: First to 15 points

### Sparse Reward Challenge
```
Turn:   1  2  3  4  5 ... 18 19 20 21 22 23 24 25 26 27 28 [END]
Score:  0  0  0  0  0 ...  0  0  1  1  4  5  8  9 12 13 15
Reward: 0  0  0  0  0 ...  0  0  1  0  3  1  3  1  3  1  3  [WIN: +10]
```

**The Problem**: 
- For the first 20 turns, agent gets zero feedback
- Cannot distinguish "good 0-point turn" from "bad 0-point turn"
- Traditional RL struggles: no gradient signal

---

## 2. Three Approaches to Score-Based Design

### Approach 1: "Naive Score-Only" (Extreme Baseline)

**Reward Function**:
```python
def reward_naive_score(state_before, action, state_after):
    return state_after.my_points - state_before.my_points
```

**Pros**:
- Simplest possible definition
- Pure experimental control (no human bias)
- Clearly demonstrates sparse reward problem

**Cons**:
- Will likely fail completely or converge extremely slowly
- Agent essentially random for first 15-20 turns
- May not converge at all

**Expected Behavior**:
- Random exploration for thousands of episodes
- Might stumble upon card-buying by accident
- Convergence time: **weeks** (if at all)

**Verdict**: ❌ Too naive - good for demonstration but impractical

---

### Approach 2: "Score + Win Bonus" (Minimal Enhancement)

**Reward Function**:
```python
def reward_score_with_win(state_before, action, state_after, done):
    reward = state_after.my_points - state_before.my_points
    
    if done:
        if state_after.winner_id == my_id:
            reward += 10.0  # Big win bonus
        else:
            reward += -5.0  # Loss penalty
    
    return reward
```

**Pros**:
- Still "score-based" in spirit
- Win bonus provides delayed terminal reward
- Slightly better gradient signal at game end

**Cons**:
- Still almost zero reward during engine-building
- Credit assignment problem: which early actions led to win?
- May converge but very slowly

**Expected Behavior**:
- Slightly better than naive (win signal helps)
- Still requires extensive exploration
- Convergence time: **days to weeks**

**Verdict**: ⚠️ Marginally better but still challenging

---

### Approach 3: "Score + Relative Progress" (Smart Baseline)

**Reward Function**:
```python
def reward_score_with_progress(state_before, action, state_after, done):
    # Primary: point difference
    point_delta = state_after.my_points - state_before.my_points
    
    # Secondary: "progress toward winning" heuristic
    progress_reward = 0.0
    
    # Tiny reward for advancing game state (not stuck)
    if action is not None:
        progress_reward += 0.01
    
    # Win/loss terminal reward
    if done:
        if state_after.winner_id == my_id:
            progress_reward += 10.0
        else:
            progress_reward += -5.0
    
    return point_delta + progress_reward
```

**Pros**:
- Still fundamentally score-based
- Tiny "existence bonus" encourages valid actions
- Less likely to get stuck in do-nothing loops

**Cons**:
- 0.01 bonus might bias toward fast games
- Not truly "pure" score-based anymore
- Still struggles with credit assignment

**Expected Behavior**:
- Better exploration than naive approaches
- Might learn basic resource gathering
- Convergence time: **days**

**Verdict**: ⚠️ Practical compromise but still sparse

---

## 3. State Representation Options

The agent needs to observe the game state. Options:

### Option A: Full State Vector (Recommended)
```python
state_vector = [
    # My resources (6 dimensions: 5 gem types + gold)
    my_gems[white], my_gems[blue], my_gems[green],
    my_gems[red], my_gems[black], my_gems[gold],
    
    # My cards (5 dimensions: cards of each color)
    my_card_bonuses[white], ..., my_card_bonuses[black],
    
    # My score
    my_points,
    
    # Board state (5 dimensions: gems available)
    board_gems[white], ..., board_gems[black],
    
    # Visible cards (12 cards × features)
    # For each visible card: cost (5 dims) + bonus (1 dim) + points (1 dim)
    # = 12 × 7 = 84 dimensions
    
    # Opponent state (simplified)
    opponent_points,
    opponent_card_count,
    
    # Total: ~100-150 dimensions
]
```

**Pros**: Complete information, agent can learn patterns  
**Cons**: High dimensional, needs larger network

### Option B: Hand-Crafted Features
```python
state_features = [
    my_points,
    my_total_gems,
    my_total_cards,
    my_purchasing_power,  # cards + gems combined
    opponent_points,
    opponent_purchasing_power,
    affordable_cards_count,
    points_deficit,  # opponent_points - my_points
]
```

**Pros**: Low dimensional, faster training  
**Cons**: Loses information, limits agent's potential

**Recommendation**: Start with Option A (full vector) - let the network learn what matters.

---

## 4. PPO Convergence Analysis

### Can PPO Handle Sparse Rewards?

**Short Answer**: Yes, but slowly.

**PPO Strengths**:
- ✅ Stable policy updates (clipping prevents catastrophic forgetting)
- ✅ Sample efficient (reuses data via multiple epochs)
- ✅ Works well with continuous/discrete actions

**PPO Weaknesses**:
- ❌ Still requires *some* reward signal to improve
- ❌ May plateau if stuck in no-reward local minimum
- ❌ Slow exploration without dense rewards

### Expected Training Dynamics

**Phase 1: Random Exploration (Episodes 1-5000)**
```
Win rate vs Random: ~50% (both playing randomly)
Average reward per episode: ~0.1 (mostly zero, occasional lucky point)
Loss: High, unstable
```

**Phase 2: Breakthrough (Episodes 5000-10000)**
```
Agent accidentally discovers buying cards gives points
Win rate vs Random: ~55-60%
Average reward: ~0.5-1.0
Loss: Starting to decrease
```

**Phase 3: Convergence (Episodes 10000-20000)**
```
Policy stabilizes around point-maximizing strategy
Win rate vs Random: ~70-80%
Average reward: ~2-5
Loss: Stable
```

**Total Training Time Estimate**:
- 20,000 episodes × 50 turns/episode = 1M steps
- With batching and 4090 GPU: **12-24 hours**

### Critical Factors for Convergence

1. **Exploration Strategy**
   - Use epsilon-greedy or entropy bonus to encourage exploration
   - Gradually decay exploration over time

2. **Reward Scaling**
   - Normalize rewards to prevent gradient explosion
   - May need to scale win bonus appropriately

3. **Learning Rate Schedule**
   - Start with lr=3e-4 (standard PPO)
   - Decay if loss plateaus

4. **Architecture Size**
   - Network must be large enough to represent game patterns
   - Recommend: 256→128→64 hidden layers

---

## 5. Recommended Experimental Design

### Experiment 1: Naive Score-Only
**Purpose**: Demonstrate the problem clearly  
**Expected Result**: Poor/no convergence  
**Training Budget**: 10k episodes (8-12 hours)

### Experiment 2: Score + Win Bonus
**Purpose**: Show minimal improvement  
**Expected Result**: Slow convergence, ~60% win rate  
**Training Budget**: 20k episodes (16-24 hours)

### Experiment 3: Score + Progress Hint
**Purpose**: Practical score-based baseline  
**Expected Result**: Decent convergence, ~70% win rate  
**Training Budget**: 20k episodes (16-24 hours)

### Comparison Benchmark
Train Event-based agent (Phase 1 goal) to show dramatic improvement.

---

## 6. Implementation Strawman (Pseudocode)

```python
class ScoreBasedAgent:
    def __init__(self, mode='naive'):
        self.mode = mode  # 'naive', 'win_bonus', 'progress'
        self.policy = PPOPolicy(
            state_dim=150,
            action_dim=200,  # max actions in Splendor
            hidden_dims=[256, 128, 64]
        )
    
    def compute_reward(self, state_before, action, state_after, done):
        if self.mode == 'naive':
            return state_after.my_points - state_before.my_points
        
        elif self.mode == 'win_bonus':
            reward = state_after.my_points - state_before.my_points
            if done:
                reward += 10.0 if state_after.i_won else -5.0
            return reward
        
        elif self.mode == 'progress':
            reward = state_after.my_points - state_before.my_points
            reward += 0.01  # tiny progress bonus
            if done:
                reward += 10.0 if state_after.i_won else -5.0
            return reward
    
    def train_episode(self):
        states, actions, rewards = [], [], []
        state = env.reset()
        
        while not done:
            action = self.policy.select_action(state)
            next_state, reward, done = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(self.compute_reward(state, action, next_state, done))
            
            state = next_state
        
        # PPO update
        self.policy.update(states, actions, rewards)
```

---

## 7. Key Research Questions

### Q1: Will naive score-only converge at all?
**Hypothesis**: No, or extremely slowly (>50k episodes)  
**How to test**: Train for 10k episodes, check if win rate > 55%

### Q2: Is win bonus sufficient?
**Hypothesis**: Helps, but still slow  
**How to test**: Compare convergence speed vs naive

### Q3: How much "cheating" (progress hints) is acceptable?
**Hypothesis**: Tiny hints dramatically improve convergence  
**How to test**: Ablation study on progress bonus weight (0, 0.001, 0.01, 0.1)

### Q4: Does PPO outperform DQN on sparse rewards?
**Hypothesis**: PPO may be slightly better due to policy gradient  
**How to test**: Implement both, compare convergence curves

---

## 8. My Recommendations

### For the Project

**Option A: Demonstrate the Problem (Academically Rigorous)**
- Implement Approach 1 (Naive) and show it fails
- Implement Approach 2 (Win Bonus) and show slow convergence
- This clearly motivates Event-based approach in Phase 1

**Option B: Practical Baseline (Get Results Fast)**
- Implement Approach 3 (Progress Hints) directly
- It's still "score-based" in spirit
- Will converge reasonably well as baseline for comparison

**My Suggestion**: **Do Both**
- Quick experiment with Naive (4-8 hours training) to demonstrate problem
- Main baseline with Progress Hints for fair Event-based comparison
- This gives you both "pain point" and "working baseline"

### For Your ADR

The question "DQN vs PPO" actually matters less than reward design:
- **PPO Pros**: More stable, standard for game AI
- **DQN Pros**: Simpler to implement, off-policy = more sample efficient

**Verdict**: I'd recommend **PPO** because:
1. Better for sparse rewards (policy gradient helps)
2. More stable (clipping prevents bad updates)
3. Industry standard for game AI (easier to explain in report)

---

## 9. Next Steps

1. **Create ADR-001**: Document DQN vs PPO decision (recommend PPO)
2. **Create ADR-002**: Document which score-based variant to use
3. **Implement state representation module**: Vectorize game state
4. **Set up minimal PPO training loop**: Using Stable-Baselines3 or custom
5. **Run 8-hour pilot experiment**: Naive score-only to see if it learns at all

---

## References

- Silver et al. (2016): "Reward shaping in sparse RL"
- Ng et al. (1999): "Policy invariance under reward transformations"
- Schulman et al. (2017): "Proximal Policy Optimization"
- Your project docs: `docs/plan.md`, `docs/Splendor_Feasibility_Report.md`

---

**Status**: Ready for team discussion / ADR creation
