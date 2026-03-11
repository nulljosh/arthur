#!/bin/bash
# Main cron entrypoint: smoke test -> evaluate -> log
# Called by the safe hourly dispatcher.
#
# Usage:
#   ./cron/run.sh               # smoke test + optional eval
#   ./cron/run.sh --smoke-only  # skip eval, only run smoke test
#   ./cron/run.sh --eval-only   # skip smoke, only run eval

CORE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CORE_ROOT"

LOGFILE="logs/cron_$(date +%Y-%m-%d_%H%M).log"
mkdir -p logs

exec > >(tee -a "$LOGFILE") 2>&1

echo "========================================"
echo "arthur cron run: $(date)"
echo "========================================"

# activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

SMOKE_ONLY=false
EVAL_ONLY=false

for arg in "$@"; do
    case $arg in
        --smoke-only) SMOKE_ONLY=true ;;
        --eval-only) EVAL_ONLY=true ;;
    esac
done

# smoke test
if [ "$EVAL_ONLY" = false ]; then
    echo ""
    echo "--- Demo smoke test ---"
    python3 scripts/demo_smoke.py
    SMOKE_EXIT=$?

    if [ $SMOKE_EXIT -ne 0 ]; then
        echo "Smoke test failed with exit code $SMOKE_EXIT"
        echo "Will still attempt evaluation..."
    fi
fi

# evaluate
if [ "$SMOKE_ONLY" = false ] && [ -f "models/cron_best.pt" ]; then
    echo ""
    echo "--- Evaluating ---"
    python3 cron/evaluate.py
elif [ "$SMOKE_ONLY" = false ]; then
    echo "No checkpoint to evaluate yet."
fi

echo ""
echo "--- Done: $(date) ---"
echo "Log: $LOGFILE"
