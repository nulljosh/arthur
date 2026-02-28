#!/bin/bash
# Auto-push to GitHub after training checkpoints or feature additions

ARTHUR_ROOT="/Users/joshua/Documents/Code/arthur"
LOG_FILE="$ARTHUR_ROOT/logs/push.log"

cd "$ARTHUR_ROOT"

# Check if there are uncommitted changes
if git status --porcelain | grep -q .; then
  echo "[$(date)] Uncommitted changes detected, committing..." >> "$LOG_FILE"
  
  # Determine commit message based on what changed
  if git diff --cached --name-only | grep -q models/; then
    MSG="chore: checkpoint saved - $(date +%Y-%m-%d_%H:%M)"
  elif git diff --cached --name-only | grep -q src/; then
    MSG="feat: training improvements"
  else
    MSG="chore: auto-commit from daemon - $(date +%Y-%m-%d_%H:%M)"
  fi
  
  git add -A
  git commit -m "$MSG" >> "$LOG_FILE" 2>&1
fi

# Push to remote
if git rev-parse --abbrev-ref @{u} > /dev/null 2>&1; then
  echo "[$(date)] Pushing to GitHub..." >> "$LOG_FILE"
  git push -q 2>> "$LOG_FILE"
  if [ $? -eq 0 ]; then
    echo "[$(date)] Push successful" >> "$LOG_FILE"
  else
    echo "[$(date)] Push failed - check logs" >> "$LOG_FILE"
  fi
else
  echo "[$(date)] No upstream branch set" >> "$LOG_FILE"
fi
