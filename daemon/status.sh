#!/bin/bash
# Quick status check

ARTHUR_ROOT="/Users/joshua/Documents/Code/arthur"

echo ""
echo "=== ARTHUR DEMO STATUS ==="
echo ""

# Daemon state
echo "Mode: Demo / eval only"

if pgrep -f "scripts/train.py" > /dev/null; then
  echo "Training: ACTIVE (unexpected)"
else
  echo "Training: OFF"
fi

if pgrep -f "arthur_watchdog.py" > /dev/null; then
  echo "Watchdog: RUNNING (unexpected)"
else
  echo "Watchdog: OFF"
fi

if pgrep -f "ollama serve" > /dev/null || pgrep -f "ollama runner" > /dev/null; then
  echo "Local LLM: ACTIVE"
else
  echo "Local LLM: OFF"
fi

# Storage
DISK_FREE=$(df / | tail -1 | awk '{print $4}')
DISK_FREE_GB=$((DISK_FREE / 1024 / 1024))
echo "Storage: ${DISK_FREE_GB}GB free"

LATEST=$(ls -t "$ARTHUR_ROOT/models"/arthur_v3_*_latest.pt "$ARTHUR_ROOT/models"/arthur_v3_*_best.pt 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
  SIZE=$(ls -lh "$LATEST" | awk '{print $5}')
  echo "Latest checkpoint: $(basename "$LATEST") ($SIZE)"
fi

if [ -f "$ARTHUR_ROOT/logs/demo_smoke_latest.json" ]; then
  echo "Latest smoke: $(python3 - <<'PY2'
import json
from pathlib import Path
report = json.loads(Path("/Users/joshua/Documents/Code/arthur/logs/demo_smoke_latest.json").read_text())
status = "OK" if report.get("all_ok") else "FAIL"
print(f"{status} @ {report.get('timestamp', '?')}")
PY2
)"
fi

echo ""
