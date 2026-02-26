# IFT6759 Course Project Plan: Optimizing Splendor AI Agents
# 课程项目规划：基于璀璨宝石 (Splendor) 的奖励设计与规划算法研究

> **Status:** Finalized Proposal | Phase 1 (Baselines) Complete | **Pivoting to Action Masking (Phase 2)**
> **Team:** Yehao Yan, Qianyun Shen, Xinyan Guo
> **Hardware:** AMD Ryzen Threadripper PRO 5955WX (16 Cores/32 Threads) + NVIDIA RTX 4090 (24GB VRAM)
> **Core Theme:** Reward Shaping (Score vs. Event) -> Advanced Planning (AlphaZero)
> **Key Pivot (2026-02-26):** Experiment 1 showed that standard PPO fails against strong opponents due to the large invalid action space. We are upgrading to **MaskablePPO** (from `sb3-contrib`) as the foundation for Phase 2.

---

## 📖 Project Narrative & Motivation (核心叙事)

### 1. The Problem: The "Silent" Engine Building Phase
Splendor is a game of two halves. In the first 10-20 turns, players build their "engine" (collecting gems, buying low-tier cards) without gaining any Prestige Points.
*   **Score-Based Limitation (Sparse Reward):** A traditional RL agent only rewarded by score sees a flat zero reward for half the game. It cannot distinguish between a "good 0-point turn" (rich resources) and a "bad 0-point turn". This leads to slow convergence or failure to learn.
*   **Stochasticity:** Random card replenishment makes pure search methods (standard Minimax/MCTS) unreliable due to the high branching factor.

### 2. Our Approach: A Two-Stage Advanced Study
我们采取层层递进的策略，既保证项目的基本分（通过对比实验），又冲击高分（通过引入高级规划算法）。

#### **Stage 1: The Reward Shaping Study (Score-based vs. Event-based)**
*   **Hypothesis:** An **Event-Based** reward system (assigning value to actions like 'reserving a card' or 'buying a tier-1 card') will provide a denser signal, leading to faster convergence and better engine building than a naive **Score-Based** system.
*   **Action:** We will train two distinct RL agents (likely PPO/DQN) using these different reward signals and compare their win rates.
*   **Context:** This builds directly on the *Rinascimento* (Bravi et al., 2019) research line.

#### **Stage 2: The "Advanced" Extension (AlphaZero Style)**
*   **The Paradox:** While Event-based rewards help learning, they introduce **Bias**. The agent might become "greedy" for events (e.g., obsessively reserving cards to get the +1 event reward) rather than winning the game.
*   **The Solution:** We propose to implement an **AlphaZero-style agent** (Neural Network + MCTS).
    *   **Why:** MCTS provides **LONG-TERM PLANNING**. It uses the Neural Network as intuition but simulates actual future outcomes to correct the "short-sightedness" of the Event-based agent.
    *   **Goal:** Prove that Planning (MCTS) > Heuristic Reward Shaping (Event-based).

---

## 🔬 Evaluation Plan (实验评估)

### 2.1 Feasibility & Computational Resources
We address the computational concerns with our high-performance hardware:
*   **RTX 4090 (24GB):** Allows for massive batch inference and rapid training of Value/Policy networks.
*   **Threadripper 5955WX (32 Threads):** Enables **massively parallel self-play**. While previous baselines (GTX 1080 Ti) took ~2 days, our 32-worker setup allows us to complete training iterations overnight (~8-10 hours).

### 2.2 Core Metrics
1.  **Win Rate Analysis:**
    *   Target: Event-Based Agent > Score-Based Agent (60% win rate).
    *   Target: AlphaZero Agent > Event-Based Agent (Show that Planning beats Bias).
    *   Setup: 500 matches per pair, swapping First/Second player to remove advantage bias.
2.  **Convergence Speed:** Plot "Win Rate vs Training Steps" to prove Event-based learns faster.
3.  **Ablation Study:** Train with *only* State-Vector vs. State + Event-Vector to isolate identifying feature importance.

---

## 📅 10-Week Execution Plan (执行计划)

### **Phase 1: Environment & Baselines (Weeks 1-3)**
*   **Week 1:** Setup `gym-splendor` (roeey777 or customization). Verify random agents run correctly on the local machine.
*   **Week 2:** Implement the **Score-Based Agent** (The weak sparse-reward baseline).
*   **Week 3:** Implement the **Event-Based Agent** (The dense-reward challenger). Define the event values carefully based on the *Rinascimento* paper.
*   **Milestone:** Generate the first "Score vs Event" comparison graph. (Safe Zone reached).

### **Phase 2: The Advanced Leap (Weeks 4-7)**
*   **Week 4:** Design the AlphaZero architecture (ResNet backbone).
*   **Week 5:** Implement MCTS logic and integrate with the Neural Network.
*   **Week 6:** Optimize the Self-Play Pipeline using `multiprocessing` on the Threadripper to saturate the 4090.
*   **Week 7:** Large-scale training of the AlphaZero agent.

### **Phase 3: Analysis & Reporting (Weeks 8-10)**
*   **Week 8:** **Tournament!** Run Round-Robin tournament between Score-Agent, Event-Agent, and AlphaZero-Agent.
*   **Week 9:** Analyze results. Did AlphaZero overcome the greediness of the Event-based agent?
*   **Week 10:** Final Report & Presentation.

---

## 📚 Key References
1.  *Mnih et al. "Human-level control through deep reinforcement learning" (Nature, 2015)*
2.  *Bravi et al. "Rinascimento: Optimising Statistical Forward Planning Agents for Playing Splendor" (2019)*
3.  *Silver et al. "Mastering the game of Go without human knowledge" (Nature, 2017)*
