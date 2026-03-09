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
  
  # Smart commit message based on what changed
  CHANGED=$(git diff --cached --name-only 2>/dev/null || git diff --name-only)
  SIMPLIFY_LOG="$ARTHUR_ROOT/logs/simplify.log"

  if echo "$CHANGED" | grep -q "models/"; then
    MSG="chore: checkpoint saved - epoch $(python3 -c "import json; print(json.load(open('$ARTHUR_ROOT/daemon_state.json'))['epoch'])" 2>/dev/null || echo '?')"
  elif [ -f "$SIMPLIFY_LOG" ] && [ "$(find "$SIMPLIFY_LOG" -mmin -30 2>/dev/null)" ]; then
    FILE_COUNT=$(echo "$CHANGED" | grep -v '^$' | wc -l | tr -d ' ')
    MSG="refactor: simplify ${FILE_COUNT} files (auto)"
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
