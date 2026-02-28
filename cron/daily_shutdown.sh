#!/bin/bash
# Graceful shutdown at end of day

ARTHUR_ROOT="/Users/joshua/Documents/Code/arthur"

# Don't kill if we're in middle of important work
if ps aux | grep -q "ssh"; then
  echo "SSH session active, skipping shutdown" >> "$ARTHUR_ROOT/logs/shutdown.log"
  exit 0
fi

# Save state and pause training
if pgrep -f "train_v2.py" > /dev/null; then
  echo "Pausing training for end-of-day..." >> "$ARTHUR_ROOT/logs/shutdown.log"
  pkill -TERM -f "train_v2.py"
  sleep 5
fi
