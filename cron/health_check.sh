#!/bin/bash
# Check demo-mode health every 6 hours

ARTHUR_ROOT="/Users/joshua/Documents/Code/arthur"
STATE_FILE="$ARTHUR_ROOT/daemon_state.json"
LOG_FILE="$ARTHUR_ROOT/logs/health.log"

{
  echo "=== Demo Health Check: $(date) ==="

  if pgrep -f arthur_watchdog.py > /dev/null; then
    echo "✗ Watchdog daemon was running; stopping it"
    pkill -TERM -f arthur_watchdog.py || true
  else
    echo "✓ Watchdog daemon: OFF"
  fi

  if pgrep -f "scripts/train.py" > /dev/null; then
    echo "✗ Training process was running; stopping it"
    pkill -TERM -f "scripts/train.py" || true
  else
    echo "✓ Training process: OFF"
  fi

  if [ -f "$ARTHUR_ROOT/logs/demo_smoke_latest.json" ]; then
    echo "✓ Demo smoke report: present"
  else
    echo "✗ Demo smoke report: missing"
  fi

  DISK_FREE=$(df / | tail -1 | awk '{print $4}')
  DISK_FREE_GB=$((DISK_FREE / 1024 / 1024))
  if [ $DISK_FREE_GB -lt 5 ]; then
    echo "✗ Low disk space: ${DISK_FREE_GB}GB"
  else
    echo "✓ Disk space: ${DISK_FREE_GB}GB"
  fi

  echo ""
} >> "$LOG_FILE"
