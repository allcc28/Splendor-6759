# Splendor RL: Reward Shaping & Advanced Planning

This repository is dedicated to the development and training of Reinforcement Learning agents for the board game **Splendor**. This project is a part of the IFT6759 course.

## 🚀 Project Overview
The objective is to study the impact of reward shaping (Score-based vs. Event-based) and advanced planning algorithms (AlphaZero-style MCTS) on agent performance in a complex, multi-modal strategy game.

### Latest Milestone: Phase 1 & 8 Complete
- ✅ **Phase 1 Baseline**: Successfully trained a PPO agent with simple score-based rewards against Random opponents.
- ❌ **Experiment 1 (v2) Insight**: Training against Greedy opponents without Action Masking led to "passive avoidance" behavior. 
- 🛠️ **Current Status**: Pivoting to **MaskablePPO** (using `sb3-contrib`) to handle the large discrete action space (200 actions) by masking illegal moves.

## 📂 Repository Structure

### 🛠️ Core Development (`project/` `project_event_based/`)
- `project/src/utils/`: SB3 Gym Wrapper, State Vectorizer (135-dim), and utilities.
- `project/configs/`: YAML configurations for training (PPO, Masking, etc.).
- `project/scripts/`: Pipelines for training, evaluation, and plotting.
- `project/experiments/`: Reports, evaluation results, and training figures.
- `project/logs/`: TensorBoard logs and model checkpoints.
- `project/docs/development/`: Progress trackers, ADRs, and session logs.

- `project_event_based/src/utils/`: SB3 Gym Wrapper, State Vectorizer (135-dim), and utilities.
- `project_event_based/configs/`: YAML configurations for training (PPO, Masking, etc.).
- `project_event_based/scripts/`: Pipelines for training, evaluation, and plotting.

### 📚 Documentation (`docs/`)
- `plan.md`: The 10-week execution roadmap.
- `Splendor_Feasibility_Report.md`: Technical feasibility study.
- `Evidence_Inference_Speed.md`: Performance profiling and inference speed analysis.

### 🏗️ Game Modules (`modules/`)
- `gym_splendor_code/`: The core Splendor environment logic.
- `agents/`: Legacy agents (Greedy, Random, MCTS) used as training opponents.
- `arena/`: Multi-agent match execution framework.

### 📦 Legacy Resources (`legacy/`)
- Archived scripts, data, and historical experiments.

## ⚙️ Environment Setup

### Prerequisites
- **OS**: WSL2 (Ubuntu 22.04 recommended)
- **GPU**: NVIDIA RTX 4090 (or similar) with CUDA 12.1+
- **Python**: 3.10.x (Miniconda recommended)

### Installation
1. Clone the repository.
2. Initialize the environment:
```bash
conda activate splendor
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install stable-baselines3[extra] gymnasium pyyaml tensorboard pytest
pip install sb3-contrib  # Required for MaskablePPO
```
or
```bash
conda activate splendor
pip install -r requirements.txt
```
## 🎮 How to Use

### Training
To train the baseline Score-based agent:
```bash
python project/scripts/train_score_based.py
```

To train the baseline Event-based agent:
notebook: project_event_based/notebook


### Evaluation
To evaluate a trained model against different opponents:
```bash
python project/scripts/evaluate_score_based_v3.py --model path/to/model.zip --games 100
```
To evaluate a trained model against different opponents:
notebook: project_event_based/notebook

### Monitoring
Launch TensorBoard from within WSL:
```bash
tensorboard --logdir project/logs --port 6006
```

## 📜 Key Research Themes
1. **Reward Shaping**: Comparing sparse score rewards with dense event-based signals.
2. **Action Masking**: Utilizing `MaskablePPO` to optimize learning in a high-branching-factor environment.
3. **Hybrid Planning**: Integrating Neural Network value/policy priors with Monte Carlo Tree Search (AlphaZero style).

---
*Developed for IFT6759 - Winter 2026*
