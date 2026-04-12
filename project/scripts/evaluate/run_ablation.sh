#!/bin/bash
# PPO+Lookahead Ablation Study
# Tests contribution of each component by removing/modifying one at a time.
# Total: 7 configs × ~90 min each ≈ ~10-11 hours
#
# Baseline (already done): d=1, K=15, α=0.3, β=0.5, γ=0.2 → 91.9% vs Greedy

set -e
cd /mnt/c/Users/yehao/Documents/03Study/IFT6759/Splendor-6759

PY=~/miniconda3/envs/splendor/bin/python
MODEL=project/logs/ppo_event_based/maskable_ppo_event_v1_20260309_110155/eval/best_model.zip
SCRIPT=project/scripts/evaluate/evaluate_ppo_lookahead.py
N=1000

echo "============================================"
echo "PPO+Lookahead Ablation Study — $(date)"
echo "============================================"
echo ""

echo "=== [1/7] No event reward (α=1.0, β=0.0, γ=0.0) ==="
echo "Tests: what happens without event-based evaluation"
$PY $SCRIPT --ppo-model $MODEL --games $N --depth 1 --top-k 15 \
  --alpha 1.0 --beta 0.0 --gamma 0.0 \
  --tag ablation_no_event --bucket ablation
echo ""

echo "=== [2/7] No PPO value estimate (α=0.3, β=0.7, γ=0.0) ==="
echo "Tests: what happens without future state value"
$PY $SCRIPT --ppo-model $MODEL --games $N --depth 1 --top-k 15 \
  --alpha 0.3 --beta 0.7 --gamma 0.0 \
  --tag ablation_no_value --bucket ablation
echo ""

echo "=== [3/7] No PPO probability (α=0.0, β=0.5, γ=0.5) ==="
echo "Tests: what happens without PPO policy guidance"
$PY $SCRIPT --ppo-model $MODEL --games $N --depth 1 --top-k 15 \
  --alpha 0.0 --beta 0.5 --gamma 0.5 \
  --tag ablation_no_prob --bucket ablation
echo ""

echo "=== [4/7] Event only (α=0.0, β=1.0, γ=0.0) ==="
echo "Tests: pure event-based evaluation without any PPO signal"
$PY $SCRIPT --ppo-model $MODEL --games $N --depth 1 --top-k 15 \
  --alpha 0.0 --beta 1.0 --gamma 0.0 \
  --tag ablation_event_only --bucket ablation
echo ""

echo "=== [5/7] Top-K=5 (fewer candidates) ==="
echo "Tests: effect of candidate pool size"
$PY $SCRIPT --ppo-model $MODEL --games $N --depth 1 --top-k 5 \
  --alpha 0.3 --beta 0.5 --gamma 0.2 \
  --tag ablation_k5 --bucket ablation
echo ""

echo "=== [6/7] Top-K=30 (more candidates) ==="
echo "Tests: more candidates vs default K=15"
$PY $SCRIPT --ppo-model $MODEL --games $N --depth 1 --top-k 30 \
  --alpha 0.3 --beta 0.5 --gamma 0.2 \
  --tag ablation_k30 --bucket ablation
echo ""

echo "=== [7/7] Heavy event weight (α=0.1, β=0.7, γ=0.2) ==="
echo "Tests: prioritizing event reward over PPO probability"
$PY $SCRIPT --ppo-model $MODEL --games $N --depth 1 --top-k 15 \
  --alpha 0.1 --beta 0.7 --gamma 0.2 \
  --tag ablation_heavy_event --bucket ablation
echo ""

echo "============================================"
echo "All ablations done! — $(date)"
echo "Results in: project/experiments/evaluation/robust/ppo_lookahead/ablation/"
echo "============================================"
