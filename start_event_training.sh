#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate splendor
cd /mnt/c/Users/yehao/Documents/03Study/IFT6759/Splendor-6759
tmux kill-session -t event_v1 2>/dev/null || true
tmux new-session -d -s event_v1 \
  "source /root/miniconda3/etc/profile.d/conda.sh && conda activate splendor && cd /mnt/c/Users/yehao/Documents/03Study/IFT6759/Splendor-6759 && python project/scripts/train_maskable_ppo.py --config project/configs/training/maskable_ppo_event_v1.yaml 2>&1 | tee project/logs/training_event_v1.log"
echo "tmux session event_v1 started"
tmux list-sessions
