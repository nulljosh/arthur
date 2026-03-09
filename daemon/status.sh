#!/bin/bash
# Quick status check

ARTHUR_ROOT="/Users/joshua/Documents/Code/arthur"

echo ""
echo "=== ARTHUR TRAINING STATUS ==="
echo ""

# Daemon state
if [ -f "$ARTHUR_ROOT/daemon_state.json" ]; then
  read -r EPOCH TOTAL SIZE < <(python3 - <<'PY2'
import json
from pathlib import Path
state=json.loads(Path("/Users/joshua/Documents/Code/arthur/daemon_state.json").read_text())
print(state.get("epoch", "?"), state.get("total", "?"), state.get("size", "?"))
PY2
)
  echo "Progress: Epoch $EPOCH/$TOTAL ($SIZE)"
else
  echo "Status: Not initialized"
fi

# Process status
if pgrep -f "scripts/train.py" > /dev/null; then
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
LATEST=$(ls -t "$ARTHUR_ROOT/models"/arthur_v3_*_latest.pt "$ARTHUR_ROOT/models"/arthur_v3_*_best.pt 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
  SIZE=$(ls -lh "$LATEST" | awk '{print $5}')
  echo "Latest checkpoint: $(basename "$LATEST") ($SIZE)"
fi

echo ""
