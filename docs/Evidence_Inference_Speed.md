# AlphaZero/MCTS Evidence & Feasibility Analysis

## 1. Inference Speed vs. Training Time
**The Professor's Concern:** "Deep Learning is too slow for a real-time game/tournament."
**The Technical Rebuttal:** Confusion between *Training* time and *Inference* time.

| Phase | Activity | Compute Load | Time Scale (RTX 4090) |
| :--- | :--- | :--- | :--- |
| **Training** | Self-play, Gradient Descent, Backpropagation | massive (100% GPU Load) | Hours/Days |
| **Inference (Battle)** | Matrix Multiplication (Forward Pass) | Minimal (Tensor Cores) | **Milliseconds** |

**Why Inference is Instant:**
Once the file `best_model.pt` is saved, the Neural Network is a fixed function $f(s) \rightarrow (p, v)$.
- A single forward pass of a standard "Splendor-sized" ResNet (5-10 blocks) on an RTX 4090 takes **< 0.5 milliseconds**.
- Even with MCTS performing 50-100 simulations per move, the total decision time is **0.05 - 0.2 seconds**.
- This is significantly faster than the standard 1.0 second limit usually imposed in academic tournaments.

---

## 2. Key Evidence & Papers

### The "Iconic" Implementation
**Repository:** `suragnair/alpha-zero-general`
**URL:** [https://github.com/suragnair/alpha-zero-general](https://github.com/suragnair/alpha-zero-general)
**Relevance:** 
- This is the "reference" implementation for student implementations of AlphaZero.
- **Benchmark (Othello):** On a specialized NVIDIA K80 (very old), it took ~3 days. 
- **Modern Hardware translation:** Your RTX 4090 is approximately **20-30x faster** than a K80. 
- **Community Consensus:** Users routinely train Connect4/Othello agents on consumer laptops overnight.

### The "Efficiency" Paper
**Title:** *"Mastering Atari Games with Limited Data"* (EfficientZero)
**Authors:** Ye et al. (2021)
**Key Finding:** 
- Demonstrated that sample-efficient MuZero agents can be trained with drastically fewer interactions than original AlphaZero.
- While the paper focuses on Atari, the core finding—that improvements in MCTS value estimation reduce training time—applies directly to board games.

### The "Speed" Paper
**Title:** *"Parallel Monte-Carlo Tree Search"*
**Key Concept:** Tree Parallelization / Root Parallelization.
**Relevance:** Explains how 4090's CUDA cores allow running multiple MCTS simulations simultaneously (batch inference), reducing decision time from linear $O(N)$ to near-constant time for reasonable $N$.

---

## 3. Summary for Professor
"We are not training *during* the game. We are training *before* the game. During the tournament, our agent simply queries a pre-optimized 50MB weight file. The inference latency is effectively zero (sub-millisecond) on modern hardware, and MCTS simply structures these queries. We expect move times under 200ms."
