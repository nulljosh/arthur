#!/bin/bash
# Qwen decides when to stop/continue training

check_training() {
    LOSS=$(tail -1 logs/training.log | grep -oE 'loss: [0-9.]+' | cut -d' ' -f2)
    
    DECISION=$(echo "Current loss: $LOSS. Target: 0.05. Should we continue training? Reply YES or NO only." | \
               ollama run qwen2.5:3b 2>/dev/null)
    
    if [[ "$DECISION" == *"NO"* ]]; then
        echo "🛑 Qwen says stop training"
        pkill -f train_v2.py
    else
        echo "✅ Qwen says continue"
    fi
}

while true; do
    check_training
    sleep 300  # Check every 5 minutes
done
