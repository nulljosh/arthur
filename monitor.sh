#!/usr/bin/env bash
# Monitor active arthur training run, polling every 3 minutes.
set -e

LOG_FILE="logs/arthur_final_run.log"

while true; do
    if pgrep -f train_v2_no_early_stop > /dev/null; then
        epoch=$(tail -1 "$LOG_FILE" | grep -oE "Epoch [0-9]+/50" | grep -oE "[0-9]+" | head -1)
        loss=$(tail -20 "$LOG_FILE" | grep "Step" | tail -1 | grep -oE "loss = [0-9.]+" | grep -oE "[0-9.]+")
        echo "$(date '+%I:%M %p'): Epoch $epoch/50, Loss $loss"
    else
        echo "Training stopped!"
        break
    fi
    sleep 180
done
