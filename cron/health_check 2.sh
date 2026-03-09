#!/bin/bash
# Check training health every 6 hours

ARTHUR_ROOT="/Users/joshua/Documents/Code/arthur"
STATE_FILE="$ARTHUR_ROOT/daemon_state.json"
LOG_FILE="$ARTHUR_ROOT/logs/health.log"

{
  echo "=== Health Check: $(date) ==="
  
  # Check if daemon is running
  if pgrep -f arthur_watchdog.py > /dev/null; then
    echo "✓ Watchdog daemon: RUNNING"
  else
    echo "✗ Watchdog daemon: STOPPED (restarting...)"
    launchctl load ~/Library/LaunchAgents/com.joshua.arthur.plist 2>/dev/null
  fi
  
  # Check if training is active
  if [ -f "$STATE_FILE" ]; then
    EPOCH=$(grep epoch "$STATE_FILE" | head -1 | grep -o '[0-9]*' | head -1)
    echo "✓ Training progress: Epoch $EPOCH"
  fi
  
  # Check disk space
  DISK_FREE=$(df / | tail -1 | awk '{print $4}')
  DISK_FREE_GB=$((DISK_FREE / 1024 / 1024))
  if [ $DISK_FREE_GB -lt 5 ]; then
    echo "✗ Low disk space: ${DISK_FREE_GB}GB"
  else
    echo "✓ Disk space: ${DISK_FREE_GB}GB"
  fi
  
  echo ""
} >> "$LOG_FILE"
