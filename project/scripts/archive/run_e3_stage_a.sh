#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/c/Users/yehao/Documents/03Study/IFT6759/Splendor-6759"
PYTHON="$HOME/miniconda3/envs/splendor/bin/python"
CONFIG="project/configs/training/maskable_ppo_event_v5_lite_reward.yaml"

cd "$ROOT"
mkdir -p project/logs

ts="$(date +%Y%m%d_%H%M%S)"
log_file="project/logs/training_e3_stage_a_${ts}.log"
log_ptr_file="project/logs/training_e3_stage_a.latest"

echo "$log_file" > "$log_ptr_file"
echo "LOG=$log_file"
"$PYTHON" project/scripts/train_maskable_ppo.py --config "$CONFIG" 2>&1 | tee "$log_file"
