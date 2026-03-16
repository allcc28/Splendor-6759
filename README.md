# Splendor RL: Reward Shaping and Planning

This repository contains our IFT6759 project on reinforcement learning for **Splendor**. The current focus is to measure how far we can push a masked PPO agent with better reward design before moving on to search and hybrid planning methods.

## Project Status

Current phase: **Phase 11 - event-based reward shaping ablations**

Authoritative baselines so far:

| Model | Training setup | vs Random | vs RandomAgent | vs Greedy | Protocol |
|-------|----------------|-----------|----------------|-----------|----------|
| V4a MaskablePPO | score-based | 94.8% | 91.2% | 75.8% | robust eval, n=1000 |
| V5 MaskablePPO | event-based | 94.3% | 88.8% | 77.9% | robust eval, n=1000 |

Current interpretation:

- Action masking is already the default approach in this project.
- Event shaping is slightly better than the score-based baseline against `GreedyAgent`, but the gain is not yet statistically convincing at `n=1000`.
- The active work is **E1/E2 ablation screening** to test whether the V5 gains come from the shaping signal itself or from extra observation features such as gem-gap features and last-event flags.

Useful progress documents:

- `project/docs/development/PROGRESS.md`
- `project/docs/development/specs/phase11_event_based_experiment_plan.md`
- `project/experiments/evaluation/robust/robust_eval_v4a_20260308_143224_report.md`
- `project/experiments/evaluation/robust/robust_eval_v5_event_20260309_211510_report.md`

## Repository Structure

### Core development (`project/`)

- `project/src/`: wrappers, reward shaping, callbacks, and training utilities
- `project/configs/`: YAML experiment configs for score-based and event-based runs
- `project/scripts/`: training, evaluation, plotting, and diagnostics
- `project/experiments/`: reports, JSON evaluations, and generated analysis
- `project/logs/`: training runs, checkpoints, TensorBoard logs, and console logs
- `project/docs/development/`: progress trackers, specs, ADRs, and dev logs

### Documentation (`docs/`)

- `docs/plan.md`: high-level project roadmap
- `docs/Splendor_Feasibility_Report.md`: feasibility discussion
- `docs/Evidence_Inference_Speed.md`: inference-speed argumentation for tournament play

### Game modules (`modules/`)

- `modules/`: legacy environment, agents, and arena code used by the RL pipeline

### Legacy resources (`legacy/`)

- historical experiments and archived material kept for reference

## Environment Setup

Recommended environment:

- OS: `WSL2` on Windows
- Python: `3.10`
- GPU: CUDA-capable NVIDIA GPU

Install the core dependencies:

```bash
conda activate splendor
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install stable-baselines3[extra] gymnasium pyyaml tensorboard pytest
pip install sb3-contrib
```

## Common Commands

Train the current score-based MaskablePPO baseline:

```bash
python project/scripts/train_maskable_ppo.py \
  --config project/configs/training/maskable_ppo_v4a_ent_lr.yaml
```

Train the current event-based baseline:

```bash
python project/scripts/train_maskable_ppo.py \
  --config project/configs/training/maskable_ppo_event_v1.yaml
```

Run a quick evaluation against all three opponents:

```bash
python project/scripts/evaluate_maskable_ppo.py \
  --model project/logs/<run_dir>/eval/best_model \
  --config project/configs/training/<config>.yaml \
  --games 200
```

Run the robust evaluation protocol:

```bash
python project/scripts/evaluate_robust.py \
  --model project/logs/<run_dir>/eval/best_model \
  --config project/configs/training/<config>.yaml \
  --games 1000 \
  --batches 10
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir project/logs --port 6006
```

## Research Themes

1. Reward shaping: score-based rewards versus event-based shaping.
2. Action masking: making large discrete action spaces trainable with `MaskablePPO`.
3. Evaluation rigor: alternating first player, robust multi-batch evaluation, and Wilson confidence intervals.
4. Future direction: move to search or hybrid planning only if reward shaping stops producing meaningful gains.

---
Developed for IFT6759 - Winter 2026
