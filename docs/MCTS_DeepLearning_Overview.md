# MCTS + Deep Learning (AlphaZero) Overview for Splendor RL Project

## Executive Summary

This document explains why combining **Monte Carlo Tree Search (MCTS)** with **Deep Learning** (AlphaZero paradigm) represents an "Advanced" approach for the IFT6759 course Splendor RL project, providing both theoretical depth and state-of-the-art performance.

---

## 1. Why MCTS is "Advanced"

### Game AI Hierarchy
- **Level 1 (Basic):** Rule-based heuristics (e.g., "buy points whenever possible")
- **Level 2 (Intermediate):** Minimax search / Alpha-Beta pruning (undergraduate algorithms)
- **Level 3 (Advanced - RL):** DQN / PPO (standard deep RL, "intuition" only)
- **Level 4 (SOTA - MCTS + RL):** AlphaZero / MuZero (**industry state-of-the-art**)

### Three Dimensions of Complexity

#### 1. Theoretical Depth
- **Exploration vs. Exploitation:** Mathematical framework for balancing between trying new actions and exploiting known good ones
- **UCB (Upper Confidence Bound):** Theoretically grounded formula for node selection:
  ```
  UCB = X̄_j + C√(ln N / n_j)
  ```
  where:
  - `X̄_j` = average reward of node j (exploitation term)
  - `C√(ln N / n_j)` = confidence interval (exploration term)
  - `N` = parent visits, `n_j` = node visits
- **Tree Search Components:** Selection, Expansion, Simulation, Backpropagation
- **IS-MCTS (Information Set MCTS):** Required for imperfect information games like Splendor (hidden opponent cards)

#### 2. Implementation Complexity
- **DQN:** Single neural network training loop
- **MCTS:** Complete tree structure management system:
  - Dynamic tree construction and traversal
  - Recursive logic for all four phases
  - State management and memory efficiency
  - Parallelization for performance
- **Integration Challenge:** Synchronizing neural network inference with tree search

#### 3. Computational Challenge
- MCTS is compute-intensive; efficient implementation critical for deep search
- Poor performance = shallow trees = weak play
- Requires engineering optimization (vectorization, parallelization, caching)

---

## 2. Prerequisites

### Essential Knowledge
1. **Data Structures:**
   - Tree operations and traversal
   - Recursive algorithms
   
2. **UCB Formula Understanding:**
   - Intuition: balances "high win rate" vs. "under-explored" nodes
   - No need to memorize, but must understand the trade-off

3. **MDP (Markov Decision Process) Basics:**
   - State, Action, Reward, Transition
   - Standard RL formulation

4. **Neural Network Fundamentals:**
   - Fully connected layers or ResNet for state encoding
   - Basic backpropagation and training loops
   - Policy and value heads architecture

---

## 3. MCTS + Deep Learning Integration (The "AlphaZero Magic")

### Traditional MCTS Limitations
**Pure MCTS** uses random rollouts to evaluate positions:
- **Problem:** Too slow and naive for complex games like Go or Splendor
- Random playouts provide weak evaluation signals

### Deep Learning's Role: "Human Intuition"

DL accelerates search by replacing expensive simulations with learned heuristics.

#### A. Policy Network (π): "Intuition Guides Direction"

**Problem:**
- A position may have 50 legal moves
- Exploring all equally wastes computation

**DL Solution:**
- Network evaluates state and outputs probability distribution over actions
- MCTS focuses on moves suggested by policy network

**Effect:**
- Neural network **prunes search space**
- MCTS searches narrower but deeper

**Example in Splendor:**
```python
policy_probs = policy_network(current_state)  # [0.4, 0.35, 0.15, 0.05, 0.05]
# Focus search on top 2-3 actions, ignore low-probability moves
```

#### B. Value Network (v): "Intuition Judges Position"

**Problem:**
- Traditional MCTS simulates to terminal state (hundreds of moves)
- Extremely slow for deep games

**DL Solution:**
- Network evaluates current state directly: "I estimate 70% win probability"
- No need to play out to the end

**Effect:**
- MCTS searches **faster**
- Evaluations available at any depth

**Example in Splendor:**
```python
value_estimate = value_network(current_state)  # 0.67 (67% win probability)
# Use this instead of 100-move random rollout
```

#### C. Self-Improvement Loop (Closed-Loop Training)

This is the **key innovation** that makes AlphaZero work:

```
1. MCTS thinks deeply (seconds of search) → produces strong move
2. Neural network learns to imitate MCTS results
3. Improved network → better search guidance
4. Better search → stronger training signal
5. Repeat → exponential improvement
```

**Training Pipeline:**
```
Generate Self-Play Games:
  For each position:
    - Run MCTS (1000+ simulations) → get improved policy π_MCTS
    - Store (state, π_MCTS, game_outcome)
  
Train Neural Network:
  - Policy loss: cross-entropy(π_network, π_MCTS)
  - Value loss: MSE(v_network, game_outcome)
  
Iterate:
  - Use new network for next round of self-play
  - Network gets stronger → MCTS gets stronger → repeat
```

---

## 4. Why Not Just DQN? (Professor Pitch)

### DQN Limitations
- **Purely Reactive:** Maps state → action instantly (no planning)
- **Struggles with:**
  - Long-term strategic planning (e.g., reserving a card now to win 10 turns later)
  - Sparse rewards (in Splendor, reward only comes at game end)
  - Credit assignment over long horizons

### MCTS + DL Advantages

| Aspect | DQN | MCTS + DL |
|--------|-----|-----------|
| **Thinking Type** | System 1 (intuitive, fast) | System 1 + System 2 (deliberate planning) |
| **Planning Horizon** | 1-step lookahead (implicit) | N-step lookahead (explicit tree search) |
| **Sparse Rewards** | Struggles without reward shaping | Handles naturally via self-play outcomes |
| **Strategic Depth** | Limited by network capacity | Scales with compute (deeper search = better play) |
| **Sample Efficiency** | Needs many games | MCTS generates high-quality training data |

### Formal Pitch to Professor

> "DQN is purely reactive and suffers in games like Splendor which require **long-term strategic planning** (e.g., reserving a card now to win 10 turns later, or blocking an opponent's gem collection strategy).
>
> By integrating MCTS, we add a **deliberate planning component** (System 2 thinking) that reasons about future game states explicitly. The Deep Learning component (System 1 thinking) acts as a learned heuristic to guide this search, making it computationally tractable.
>
> This architecture solves the sparse reward problem much more effectively than vanilla Q-learning, and represents the current **state-of-the-art** for reasoning in strategic games (AlphaGo, AlphaZero, MuZero).
>
> This approach meets the 'Advanced' criteria for IFT6759 through its theoretical depth (UCB, IS-MCTS for imperfect information), implementation complexity (tree search + NN training pipeline), and computational challenges (efficient search algorithms)."

---

## 5. Project Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [ ] Implement basic Splendor game engine
- [ ] Create state representation for neural network
- [ ] Implement vanilla MCTS (without NN)

### Phase 2: Neural Network Integration (Weeks 3-4)
- [ ] Design policy/value network architecture
- [ ] Implement IS-MCTS for imperfect information
- [ ] Create self-play data generation pipeline

### Phase 3: Training Loop (Weeks 5-6)
- [ ] Implement training pipeline
- [ ] Run initial self-play experiments
- [ ] Tune hyperparameters (exploration constant, network architecture)

### Phase 4: Evaluation & Refinement (Weeks 7-8)
- [ ] Benchmark against baseline agents (random, heuristic, DQN)
- [ ] Analyze game understanding (visualize policy, value estimates)
- [ ] Write final report with ablation studies

---

## 6. Key References

1. **AlphaGo Zero Paper:** Silver et al. (2017) - Original AlphaZero algorithm
2. **MuZero Paper:** Schrittwieser et al. (2020) - Handling imperfect information
3. **IS-MCTS:** Cowling et al. (2012) - Information Set MCTS for hidden information games
4. **Splendor AI:** Existing work on Splendor strategy (search GitHub/ArXiv)

---

## 7. 中文总结 (Chinese Summary)

### 为什么选择 MCTS + 深度学习？

这个方法论代表了游戏 AI 的最高水平（AlphaZero 范式），具备以下优势：

1. **理论深度：**
   - 需要理解探索与利用的平衡（UCB 公式）
   - 需要实现复杂的树搜索算法（选择、扩展、模拟、回溯）
   - 对于 Splendor 这种非完全信息博弈，需要 IS-MCTS

2. **实现难度：**
   - 比简单的 DQN 训练循环复杂得多
   - 需要管理动态树结构和递归逻辑
   - 需要优化性能以支持深度搜索

3. **效果显著：**
   - DQN 只有"直觉"，MCTS + DL 同时具备"直觉"（神经网络）和"思考"（搜索）
   - 能够进行长期战略规划，解决稀疏奖励问题
   - 通过自我对弈不断提升，无需人类标注数据

### 核心创新点

**闭环提升机制：**
```
MCTS 深度思考 → 产生强力走法 → 神经网络学习模仿 → 网络变强 
→ 指导 MCTS 更高效搜索 → 产生更强走法 → 循环往复
```

这种"互相提升"的机制是 AlphaZero 能够超越人类水平的关键。

---

## Conclusion

MCTS + Deep Learning for Splendor RL represents a sophisticated approach that:
- ✅ Meets "Advanced" criteria through theoretical depth and implementation complexity
- ✅ Leverages state-of-the-art techniques (AlphaZero paradigm)
- ✅ Provides superior performance over reactive methods (DQN/PPO)
- ✅ Offers rich learning opportunities in tree search, neural architecture, and self-play training

This approach positions the project at the cutting edge of game AI research while remaining implementable within the course timeframe.
