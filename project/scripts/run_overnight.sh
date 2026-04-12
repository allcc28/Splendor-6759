#!/bin/bash
# Overnight experiment pipeline for AlphaZero hardening.
# Run from repo root: bash project/scripts/run_overnight.sh
#
# Plan:
#   1. Wait for V3 to finish (if still running)
#   2. Evaluate V3 (quick 20-game eval)
#   3. Distill PPO → AlphaZero pre-trained checkpoint
#   4. Train V4 (PPO warm-start + reward shaping, 60 iterations)
#   5. Evaluate V4
#   6. Also train V3-long (V3 config but 60 iterations, no warm-start) as control
#   7. Evaluate V3-long
#   8. Write summary

set -e
cd /mnt/c/Users/yehao/Documents/03Study/IFT6759/Splendor-6759
export PYTHONPATH=project/src:modules
PYTHON=~/miniconda3/envs/splendor/bin/python
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="project/logs/overnight_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=== Overnight Experiment Pipeline ===" | tee "$LOG_DIR/pipeline.log"
echo "Started: $(date)" | tee -a "$LOG_DIR/pipeline.log"

# ---------------------------------------------------------------
# Step 1: Wait for V3 if still running
# ---------------------------------------------------------------
echo "" | tee -a "$LOG_DIR/pipeline.log"
echo "--- Step 1: Checking V3 training ---" | tee -a "$LOG_DIR/pipeline.log"
while pgrep -f "alphazero_v3_shaped" > /dev/null 2>&1; do
    echo "  V3 still running... waiting 60s" | tee -a "$LOG_DIR/pipeline.log"
    sleep 60
done
echo "  V3 done or not running." | tee -a "$LOG_DIR/pipeline.log"

# Find V3 checkpoint
V3_CKPT=$(ls -t project/logs/alphazero_v3_shaped_wsl_*/final_model.pt 2>/dev/null | head -1)
echo "  V3 checkpoint: $V3_CKPT" | tee -a "$LOG_DIR/pipeline.log"

# ---------------------------------------------------------------
# Step 2: Quick eval V3
# ---------------------------------------------------------------
echo "" | tee -a "$LOG_DIR/pipeline.log"
echo "--- Step 2: Evaluating V3 ---" | tee -a "$LOG_DIR/pipeline.log"
if [ -n "$V3_CKPT" ]; then
    $PYTHON project/scripts/evaluate/evaluate_alphazero.py \
        --checkpoint "$V3_CKPT" \
        --games 50 \
        --bucket archive \
        --tag v3_shaped \
        --skip-ppo 2>&1 | tee -a "$LOG_DIR/pipeline.log"
fi

# ---------------------------------------------------------------
# Step 3: PPO distillation
# ---------------------------------------------------------------
echo "" | tee -a "$LOG_DIR/pipeline.log"
echo "--- Step 3: PPO → AlphaZero distillation ---" | tee -a "$LOG_DIR/pipeline.log"
PPO_MODEL="project/logs/ppo_event_based/maskable_ppo_event_e5_v6_candidate_20260320_220734/eval/best_model.zip"
DISTILL_CKPT="$LOG_DIR/distilled_warmstart.pt"

$PYTHON project/scripts/train/distill_ppo_to_alphazero.py \
    --ppo-model "$PPO_MODEL" \
    --config project/configs/mcts/alphazero_v4_warmstart_wsl.yaml \
    --output "$DISTILL_CKPT" \
    --games 200 \
    --epochs 15 \
    --batch-size 128 2>&1 | tee -a "$LOG_DIR/pipeline.log"

# ---------------------------------------------------------------
# Step 4: Train V4 (warm-start + shaped, 60 iter)
# ---------------------------------------------------------------
echo "" | tee -a "$LOG_DIR/pipeline.log"
echo "--- Step 4: Training AlphaZero V4 (warm-start + shaped, 60 iter) ---" | tee -a "$LOG_DIR/pipeline.log"
$PYTHON project/scripts/train/train_alphazero.py \
    --config project/configs/mcts/alphazero_v4_warmstart_wsl.yaml \
    --resume "$DISTILL_CKPT" 2>&1 | tee -a "$LOG_DIR/pipeline.log"

# Find V4 checkpoint
V4_CKPT=$(ls -t project/logs/alphazero_v4_warmstart_wsl_*/final_model.pt 2>/dev/null | head -1)
echo "  V4 checkpoint: $V4_CKPT" | tee -a "$LOG_DIR/pipeline.log"

# ---------------------------------------------------------------
# Step 5: Evaluate V4
# ---------------------------------------------------------------
echo "" | tee -a "$LOG_DIR/pipeline.log"
echo "--- Step 5: Evaluating V4 ---" | tee -a "$LOG_DIR/pipeline.log"
if [ -n "$V4_CKPT" ]; then
    $PYTHON project/scripts/evaluate/evaluate_alphazero.py \
        --checkpoint "$V4_CKPT" \
        --games 50 \
        --bucket archive \
        --tag v4_warmstart \
        --skip-ppo 2>&1 | tee -a "$LOG_DIR/pipeline.log"
fi

# ---------------------------------------------------------------
# Step 6: Train V3-long (shaped only, 60 iter, no warm-start) as control
# ---------------------------------------------------------------
echo "" | tee -a "$LOG_DIR/pipeline.log"
echo "--- Step 6: Training AlphaZero V3-long (shaped, 60 iter, no warm-start) ---" | tee -a "$LOG_DIR/pipeline.log"
$PYTHON project/scripts/train/train_alphazero.py \
    --config project/configs/mcts/alphazero_v3_shaped_wsl.yaml \
    --iterations 60 2>&1 | tee -a "$LOG_DIR/pipeline.log"

V3L_CKPT=$(ls -t project/logs/alphazero_v3_shaped_wsl_*/final_model.pt 2>/dev/null | head -1)

# ---------------------------------------------------------------
# Step 7: Evaluate V3-long
# ---------------------------------------------------------------
echo "" | tee -a "$LOG_DIR/pipeline.log"
echo "--- Step 7: Evaluating V3-long ---" | tee -a "$LOG_DIR/pipeline.log"
if [ -n "$V3L_CKPT" ]; then
    $PYTHON project/scripts/evaluate/evaluate_alphazero.py \
        --checkpoint "$V3L_CKPT" \
        --games 50 \
        --bucket archive \
        --tag v3_long \
        --skip-ppo 2>&1 | tee -a "$LOG_DIR/pipeline.log"
fi

# ---------------------------------------------------------------
# Step 8: Summary
# ---------------------------------------------------------------
echo "" | tee -a "$LOG_DIR/pipeline.log"
echo "=========================================" | tee -a "$LOG_DIR/pipeline.log"
echo "=== OVERNIGHT EXPERIMENTS COMPLETE ===" | tee -a "$LOG_DIR/pipeline.log"
echo "Finished: $(date)" | tee -a "$LOG_DIR/pipeline.log"
echo "" | tee -a "$LOG_DIR/pipeline.log"
echo "Results in: $LOG_DIR/pipeline.log" | tee -a "$LOG_DIR/pipeline.log"
echo "Eval reports in: project/experiments/evaluation/robust/mcts/archive/" | tee -a "$LOG_DIR/pipeline.log"
echo "Checkpoints:" | tee -a "$LOG_DIR/pipeline.log"
echo "  V3: $V3_CKPT" | tee -a "$LOG_DIR/pipeline.log"
echo "  V4: $V4_CKPT" | tee -a "$LOG_DIR/pipeline.log"
echo "  V3-long: $V3L_CKPT" | tee -a "$LOG_DIR/pipeline.log"
