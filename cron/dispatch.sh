#!/bin/bash
# Arthur cron dispatcher - safe background maintenance only
# Runs hourly, dispatches lightweight tasks based on current hour.
# Tasks: smoke test (4h), autopush (2h), health (6h), backup (2am),
#        report (8am), simplify (7am/11pm), shutdown (11pm)

set -euo pipefail
ARTHUR_ROOT="/Users/joshua/Documents/Code/arthur"
HOUR=$(date +%-H)
LOG="$ARTHUR_ROOT/logs/dispatch.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M')] $*" >> "$LOG"; }

log "Dispatch running (hour=$HOUR)"

# Every 2 hours: auto-push
if (( HOUR % 2 == 0 )); then
    log "Running: autopush"
    bash "$ARTHUR_ROOT/daemon/auto_push.sh" >> "$LOG" 2>&1 || true
fi

# Every 4 hours: demo smoke test
if (( HOUR % 4 == 0 )); then
    log "Running: demo smoke"
    bash "$ARTHUR_ROOT/cron/run.sh" >> "$LOG" 2>&1 || true
fi

# Every 6 hours: health check
if (( HOUR % 6 == 0 )); then
    log "Running: health check"
    bash "$ARTHUR_ROOT/cron/health_check.sh" >> "$LOG" 2>&1 || true
fi

# 2am: backup
if (( HOUR == 2 )); then
    log "Running: backup"
    bash "$ARTHUR_ROOT/cron/backup_checkpoint.sh" >> "$LOG" 2>&1 || true
fi

# 7am, 11pm: simplify
if (( HOUR == 7 || HOUR == 23 )); then
    log "Running: simplify"
    bash "$ARTHUR_ROOT/daemon/auto_simplify.sh" >> "$LOG" 2>&1 || true
fi

# 8am: report
if (( HOUR == 8 )); then
    log "Running: report"
    bash "$ARTHUR_ROOT/cron/report.sh" >> "$LOG" 2>&1 || true
fi

# 11pm: shutdown (runs at 23, close enough to 23:59)
if (( HOUR == 23 )); then
    log "Running: shutdown"
    bash "$ARTHUR_ROOT/cron/daily_shutdown.sh" >> "$LOG" 2>&1 || true
fi

log "Dispatch complete"
