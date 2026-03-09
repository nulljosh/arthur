#!/bin/bash
# Auto-simplify: iterate repos with recent commits, run Claude code-simplifier
# Scheduled via launchd at 07:00 and 23:00

CODE_DIR="$HOME/Documents/Code"
LOG_FILE="$CODE_DIR/arthur/logs/simplify.log"
SKIP_REPOS="node_modules .git applescripts journal nulljosh.github.io"

echo "[$(date)] === Auto-simplify started ===" >> "$LOG_FILE"

for repo_dir in "$CODE_DIR"/*/; do
    repo_name=$(basename "$repo_dir")

    # Skip non-git dirs
    [ ! -d "$repo_dir/.git" ] && continue

    # Skip excluded repos
    echo "$SKIP_REPOS" | grep -qw "$repo_name" && continue

    # Only process repos with commits in last 12 hours
    last_commit=$(git -C "$repo_dir" log -1 --format=%ct 2>/dev/null)
    [ -z "$last_commit" ] && continue
    now=$(date +%s)
    age=$(( (now - last_commit) / 3600 ))
    [ "$age" -gt 12 ] && continue

    echo "[$(date)] Simplifying $repo_name (last commit ${age}h ago)..." >> "$LOG_FILE"

    # Run Claude code-simplifier with haiku, constrained tools, 5 min timeout
    gtimeout 300 claude --print --dangerously-skip-permissions \
        --allowedTools Edit,Read,Glob,Grep \
        --model haiku \
        "Review recently changed code in this repo for reuse, quality, and efficiency. Fix any issues found. Keep changes minimal and targeted." \
        --cwd "$repo_dir" >> "$LOG_FILE" 2>&1

    exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "[$(date)] $repo_name: simplify OK" >> "$LOG_FILE"
    elif [ $exit_code -eq 124 ]; then
        echo "[$(date)] $repo_name: timed out" >> "$LOG_FILE"
    else
        echo "[$(date)] $repo_name: exited $exit_code" >> "$LOG_FILE"
    fi
done

echo "[$(date)] === Auto-simplify complete ===" >> "$LOG_FILE"
