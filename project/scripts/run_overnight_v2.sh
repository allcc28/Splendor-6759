#!/bin/bash
# Overnight Experiment Pipeline v2
# Runs curriculum training (GPU) + ablation experiments (CPU) in parallel
#
# Usage: wsl -e bash -c "cd /mnt/c/.../Splendor-6759 && bash project/scripts/run_overnight_v2.sh"

set -e
cd /mnt/c/Users/yehao/Documents/03Study/IFT6759/Splendor-6759

PY=~/miniconda3/envs/splendor/bin/python
EVAL_SCRIPT=project/scripts/evaluate/evaluate_ppo_lookahead.py
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================"
echo "Overnight Experiment Pipeline v2"
echo "Started: $(date)"
echo "============================================"
echo ""

# ─────────────────────────────────────────────
# Part A: Curriculum Training (GPU-bound)
# ─────────────────────────────────────────────
echo "[Part A] Starting Curriculum Training..."
$PY project/scripts/archive/train_curriculum.py \
  --config project/configs/training/maskable_ppo_event_curriculum_v1.yaml \
  2>&1 | tee project/logs/curriculum_training_${TIMESTAMP}.log &
TRAIN_PID=$!
echo "  Training PID: $TRAIN_PID"
echo ""

# ─────────────────────────────────────────────
# Part B: Ablation Experiments (CPU-bound)
# ─────────────────────────────────────────────
echo "[Part B] Starting Ablation Experiments..."
bash project/scripts/evaluate/run_ablation.sh \
  2>&1 | tee project/logs/ablation_${TIMESTAMP}.log &
ABLATION_PID=$!
echo "  Ablation PID: $ABLATION_PID"
echo ""

# ─────────────────────────────────────────────
# Wait for both to complete
# ─────────────────────────────────────────────
echo "Waiting for both jobs to complete..."
echo ""

# Wait for training first (usually longer)
wait $TRAIN_PID
TRAIN_EXIT=$?
echo "[Part A] Training finished (exit code: $TRAIN_EXIT) — $(date)"

if [ $TRAIN_EXIT -eq 0 ]; then
  # Find the latest curriculum model
  LATEST_DIR=$(ls -td project/logs/maskable_ppo_event_curriculum_* 2>/dev/null | head -1)
  if [ -z "$LATEST_DIR" ]; then
    LATEST_DIR=$(ls -td project/logs/ppo_event_based/maskable_ppo_event_curriculum_* 2>/dev/null | head -1)
  fi

  if [ -n "$LATEST_DIR" ]; then
    MODEL_PATH="${LATEST_DIR}/eval/best_model.zip"
    if [ ! -f "$MODEL_PATH" ]; then
      MODEL_PATH="${LATEST_DIR}/final_model.zip"
    fi

    if [ -f "$MODEL_PATH" ]; then
      echo ""
      echo "[Part C] Evaluating curriculum model: $MODEL_PATH"

      # Eval d=0
      echo "  Running d=0 eval (n=1000)..."
      $PY $EVAL_SCRIPT --ppo-model "$MODEL_PATH" \
        --games 1000 --depth 0 \
        --tag curriculum_d0 --bucket canonical

      # Eval d=1
      echo "  Running d=1 eval (n=1000)..."
      $PY $EVAL_SCRIPT --ppo-model "$MODEL_PATH" \
        --games 1000 --depth 1 --top-k 15 \
        --tag curriculum_d1 --bucket canonical

      echo "[Part C] Curriculum eval done!"
    else
      echo "  WARNING: No model found at $MODEL_PATH"
    fi
  else
    echo "  WARNING: No curriculum training directory found"
  fi
else
  echo "  WARNING: Training failed, skipping eval"
fi

# Wait for ablation
wait $ABLATION_PID
ABLATION_EXIT=$?
echo "[Part B] Ablation finished (exit code: $ABLATION_EXIT) — $(date)"

echo ""
echo "============================================"
echo "All overnight experiments complete!"
echo "Finished: $(date)"
echo ""
echo "Check results:"
echo "  Ablation:   project/experiments/evaluation/robust/ppo_lookahead/ablation/"
echo "  Curriculum:  project/experiments/evaluation/robust/ppo_lookahead/canonical/"
echo "  Logs:       project/logs/curriculum_training_${TIMESTAMP}.log"
echo "              project/logs/ablation_${TIMESTAMP}.log"
echo "============================================"
