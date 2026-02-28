#!/bin/bash
# Auto-push to GitHub after refreshing docs

ARTHUR_ROOT="/Users/joshua/Documents/Code/arthur"
LOG_FILE="$ARTHUR_ROOT/logs/push.log"

cd "$ARTHUR_ROOT"

# Refresh docs before pushing
echo "[$(date)] Refreshing docs..." >> "$LOG_FILE"
python3 "$ARTHUR_ROOT/daemon/refresh_docs.py" >> "$LOG_FILE" 2>&1

# Check for changes
if git status --porcelain | grep -q .; then
  echo "[$(date)] Changes detected, committing..." >> "$LOG_FILE"
  
  # Smart commit message
  if git diff --name-only | grep -q "models/"; then
    MSG="chore: checkpoint saved - epoch $(grep epoch $ARTHUR_ROOT/daemon_state.json 2>/dev/null | head -1 | grep -o '[0-9]*')"
  else
    MSG="docs: auto-update - $(date +%Y-%m-%d_%H:%M)"
  fi
  
  git add -A
  git commit -m "$MSG" >> "$LOG_FILE" 2>&1
fi

# Push
if git rev-parse --abbrev-ref @{u} > /dev/null 2>&1; then
  echo "[$(date)] Pushing..." >> "$LOG_FILE"
  git push -q && echo "[$(date)] ✓ Push OK" >> "$LOG_FILE" || echo "[$(date)] ✗ Push failed" >> "$LOG_FILE"
fi
