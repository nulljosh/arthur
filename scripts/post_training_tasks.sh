#!/bin/bash
# Post-training tasks - runs after training completes

echo "=================================================="
echo "POST-TRAINING TASKS"
echo "=================================================="
echo ""

# Wait for training process to finish
echo "Waiting for training to complete..."
while pgrep -f "train_comprehensive.py" > /dev/null; do
    sleep 30
done

echo "✓ Training finished!"
echo ""

# Push journal commit
echo "Pushing journal commit..."
cd ~/Documents/Code/journal
git push origin main 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Journal commit pushed successfully"
else
    echo "⚠ Git push failed - check manually"
fi

echo ""
echo "=================================================="
echo "ALL POST-TRAINING TASKS COMPLETE"
echo "=================================================="
