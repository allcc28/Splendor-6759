#!/bin/bash
# Event Weight Ablation Study
#
# 9 events: take_gems(0), buy_card(1), reserve_card(2), score_up(3),
#           reach_15(4), scarcity_take(5), block_reserve(6), buy_reserved(7), engine_spike(8)
#
# Default weights: 0.01, 1.0, 0.05, 0.8, 25.0, 0.25, 0.5, 0.4, 2.0
#
# Part A: Leave-one-out (9 runs) — zero out one event at a time
# Part B: Solo event (4 runs) — keep only one event
# Total: 13 runs × n=500 × ~45 min ≈ ~10 hours

set -e
cd /mnt/c/Users/yehao/Documents/03Study/IFT6759/Splendor-6759

PY=~/miniconda3/envs/splendor/bin/python
MODEL=project/logs/ppo_event_based/maskable_ppo_event_v1_20260309_110155/eval/best_model.zip
SCRIPT=project/scripts/evaluate/evaluate_ppo_lookahead.py
N=500

# Default weights for reference
# take_gems=0.01, buy_card=1.0, reserve_card=0.05, score_up=0.8,
# reach_15=25.0, scarcity_take=0.25, block_reserve=0.5, buy_reserved=0.4, engine_spike=2.0

echo "============================================"
echo "Event Weight Ablation Study — $(date)"
echo "============================================"
echo ""

# =============================================
# Part A: Leave-One-Out (remove one event at a time)
# =============================================

echo "===== PART A: Leave-One-Out ====="
echo ""

echo "[A1/9] Without take_gems"
$PY $SCRIPT --ppo-model $MODEL --games $N --depth 1 \
  --event-weights "0,1.0,0.05,0.8,25.0,0.25,0.5,0.4,2.0" \
  --tag loo_no_take_gems --bucket event_ablation
echo ""

echo "[A2/9] Without buy_card"
$PY $SCRIPT --ppo-model $MODEL --games $N --depth 1 \
  --event-weights "0.01,0,0.05,0.8,25.0,0.25,0.5,0.4,2.0" \
  --tag loo_no_buy_card --bucket event_ablation
echo ""

echo "[A3/9] Without reserve_card"
$PY $SCRIPT --ppo-model $MODEL --games $N --depth 1 \
  --event-weights "0.01,1.0,0,0.8,25.0,0.25,0.5,0.4,2.0" \
  --tag loo_no_reserve_card --bucket event_ablation
echo ""

echo "[A4/9] Without score_up"
$PY $SCRIPT --ppo-model $MODEL --games $N --depth 1 \
  --event-weights "0.01,1.0,0.05,0,25.0,0.25,0.5,0.4,2.0" \
  --tag loo_no_score_up --bucket event_ablation
echo ""

echo "[A5/9] Without reach_15"
$PY $SCRIPT --ppo-model $MODEL --games $N --depth 1 \
  --event-weights "0.01,1.0,0.05,0.8,0,0.25,0.5,0.4,2.0" \
  --tag loo_no_reach_15 --bucket event_ablation
echo ""

echo "[A6/9] Without scarcity_take"
$PY $SCRIPT --ppo-model $MODEL --games $N --depth 1 \
  --event-weights "0.01,1.0,0.05,0.8,25.0,0,0.5,0.4,2.0" \
  --tag loo_no_scarcity_take --bucket event_ablation
echo ""

echo "[A7/9] Without block_reserve"
$PY $SCRIPT --ppo-model $MODEL --games $N --depth 1 \
  --event-weights "0.01,1.0,0.05,0.8,25.0,0.25,0,0.4,2.0" \
  --tag loo_no_block_reserve --bucket event_ablation
echo ""

echo "[A8/9] Without buy_reserved"
$PY $SCRIPT --ppo-model $MODEL --games $N --depth 1 \
  --event-weights "0.01,1.0,0.05,0.8,25.0,0.25,0.5,0,2.0" \
  --tag loo_no_buy_reserved --bucket event_ablation
echo ""

echo "[A9/9] Without engine_spike"
$PY $SCRIPT --ppo-model $MODEL --games $N --depth 1 \
  --event-weights "0.01,1.0,0.05,0.8,25.0,0.25,0.5,0.4,0" \
  --tag loo_no_engine_spike --bucket event_ablation
echo ""

# =============================================
# Part B: Solo Event (keep only one event)
# =============================================

echo "===== PART B: Solo Event ====="
echo ""

echo "[B1/4] Only buy_card"
$PY $SCRIPT --ppo-model $MODEL --games $N --depth 1 \
  --event-weights "0,1.0,0,0,0,0,0,0,0" \
  --tag solo_buy_card --bucket event_ablation
echo ""

echo "[B2/4] Only score_up"
$PY $SCRIPT --ppo-model $MODEL --games $N --depth 1 \
  --event-weights "0,0,0,0.8,0,0,0,0,0" \
  --tag solo_score_up --bucket event_ablation
echo ""

echo "[B3/4] Only reach_15"
$PY $SCRIPT --ppo-model $MODEL --games $N --depth 1 \
  --event-weights "0,0,0,0,25.0,0,0,0,0" \
  --tag solo_reach_15 --bucket event_ablation
echo ""

echo "[B4/4] Only engine_spike"
$PY $SCRIPT --ppo-model $MODEL --games $N --depth 1 \
  --event-weights "0,0,0,0,0,0,0,0,2.0" \
  --tag solo_engine_spike --bucket event_ablation
echo ""

echo "============================================"
echo "All event ablations done! — $(date)"
echo "Results: project/experiments/evaluation/robust/ppo_lookahead/event_ablation/"
echo "============================================"
