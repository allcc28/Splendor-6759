#!/bin/bash
# E3 Stage A Quick Monitoring Script
# Usage: bash project/scripts/e3_monitor.sh

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

LOG_DIR="project/logs/maskable_ppo_event_e3_lite_reward_20260319_115615"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}E3 Stage A Monitoring Dashboard${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check if training is still running
if ps aux | grep -i "train_maskable_ppo" | grep -v grep > /dev/null; then
    echo -e "${GREEN}✓ Training is RUNNING${NC}"
else
    echo -e "${YELLOW}✗ Training appears to be COMPLETED or STOPPED${NC}"
fi

echo ""
echo -e "${BLUE}Log Directory:${NC} $LOG_DIR"
echo ""

# Try to show last few lines of training log
RAW_LOG="project/experiments/reports/raw_logs/_e3_stage_a_training.log"

if [ -f "$RAW_LOG" ]; then
    echo -e "${BLUE}Latest Training Output (last 10 lines):${NC}"
    tail -10 "$RAW_LOG"
    echo ""
fi

# Show available checkpoints
echo -e "${BLUE}Checkpoints/Models Available:${NC}"
if [ -d "$LOG_DIR/eval" ]; then
    ls -lh "$LOG_DIR/eval/"*.zip 2>/dev/null | awk '{print "  " $9}'
else
    echo "  (none yet - training in progress)"
fi

echo ""
echo -e "${BLUE}Quick Eval Command (when training is done):${NC}"
echo -e "${YELLOW}python project/scripts/e3_stage_a_monitor.py --skip-wait --games 200${NC}"
echo ""
echo -e "${BLUE}TensorBoard (in separate WSL terminal):${NC}"
echo -e "${YELLOW}tensorboard --logdir project/logs --load_fast=false --port 6006${NC}"
echo -e "Then open: ${YELLOW}http://localhost:6006${NC}"
echo ""
