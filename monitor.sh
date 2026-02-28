#!/bin/bash
while true; do
    if ps aux | grep -v grep | grep -q train_v2_no_early_stop; then
        epoch=$(tail -1 logs/arthur_final_run.log | grep -oE "Epoch [0-9]+/50" | grep -oE "[0-9]+" | head -1)
        loss=$(tail -20 logs/arthur_final_run.log | grep "Step" | tail -1 | grep -oE "loss = [0-9.]+" | grep -oE "[0-9.]+")
        echo "$(date '+%I:%M %p'): Epoch $epoch/50, Loss $loss"
    else
        echo "Training stopped!"
        break
    fi
    sleep 180
done
