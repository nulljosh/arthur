#!/bin/bash
# Install crontab entries for jore LLM training automation.
#
# Schedule:
#   - Every 4 hours: train 50 epochs + evaluate
#   - Daily 8 AM: generate codex analysis report
#
# Usage:
#   ./cron/setup.sh          # install crons
#   ./cron/setup.sh --remove # remove crons

CORE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

TRAIN_CMD="cd $CORE_ROOT && ./cron/run.sh >> logs/cron.log 2>&1"
REPORT_CMD="cd $CORE_ROOT && ./cron/report.sh >> logs/cron.log 2>&1"

# markers for identifying our cron entries
MARKER="# jore-llm-cron"

remove_crons() {
    crontab -l 2>/dev/null | grep -v "$MARKER" | crontab -
    echo "Removed jore cron entries."
}

install_crons() {
    # remove existing first
    remove_crons

    # add new entries
    (
        crontab -l 2>/dev/null
        echo ""
        echo "# === Jore LLM Training Automation === $MARKER"
        echo "0 */4 * * * $TRAIN_CMD $MARKER"
        echo "0 8 * * * $REPORT_CMD $MARKER"
        echo "# === End Jore LLM === $MARKER"
    ) | crontab -

    echo ""
    echo "Installed cron entries:"
    echo "  - Training:  every 4 hours (50 epochs per session)"
    echo "  - Report:    daily at 8 AM (codex analysis)"
    echo ""
    echo "Current crontab:"
    crontab -l | grep "$MARKER"
    echo ""
    echo "Manual commands:"
    echo "  Train now:    cd $CORE_ROOT && ./cron/run.sh"
    echo "  Eval only:    cd $CORE_ROOT && ./cron/run.sh --eval-only"
    echo "  Report now:   cd $CORE_ROOT && ./cron/report.sh"
    echo "  View logs:    tail -f $CORE_ROOT/logs/cron.log"
    echo "  Remove crons: ./cron/setup.sh --remove"
}

case "${1:-}" in
    --remove) remove_crons ;;
    *) install_crons ;;
esac
