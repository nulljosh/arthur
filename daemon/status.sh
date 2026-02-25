#!/bin/bash
# Quick status check

ARTHUR_ROOT="/Users/joshua/Documents/Code/arthur"

echo ""
echo "=== ARTHUR TRAINING STATUS ==="
echo ""

# Daemon status
if [ -f "$ARTHUR_ROOT/daemon_state.json" ]; then
  EPOCH=$(grep epoch "$ARTHUR_ROOT/daemon_state.json" | head -1 | grep -o '[0-9]*' | head -1)
  TOTAL=$(grep total "$ARTHUR_ROOT/daemon_state.json" | head -1 | grep -o '[0-9]*' | head -1)
  echo "Progress: Epoch $EPOCH/$TOTAL"
else
  echo "Status: Not initialized"
fi

# Process status
if pgrep -f "train_v2.py" > /dev/null; then
  echo "Training: ACTIVE"
else
  echo "Training: IDLE"
fi

if pgrep -f "arthur_watchdog.py" > /dev/null; then
  echo "Daemon: RUNNING"
else
  echo "Daemon: STOPPED"
fi

# Storage
DISK_FREE=$(df / | tail -1 | awk '{print $4}')
DISK_FREE_GB=$((DISK_FREE / 1024 / 1024))
echo "Storage: ${DISK_FREE_GB}GB free"

# Recent checkpoint
LATEST=$(ls -t "$ARTHUR_ROOT/models"/arthur_v2_epoch*.pt 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
  SIZE=$(ls -lh "$LATEST" | awk '{print $5}')
  echo "Latest checkpoint: $(basename $LATEST) ($SIZE)"
fi

echo ""
