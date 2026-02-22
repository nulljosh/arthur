#!/bin/bash
# Main cron entrypoint: train -> evaluate -> log
# Called by crontab every 4 hours.
#
# Usage:
#   ./cron/run.sh              # full pipeline (train 50 epochs + eval)
#   ./cron/run.sh --epochs 100 # train 100 epochs + eval
#   ./cron/run.sh --eval-only  # skip training, just evaluate

CORE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CORE_ROOT"

LOGFILE="logs/cron_$(date +%Y-%m-%d_%H%M).log"
mkdir -p logs

exec > >(tee -a "$LOGFILE") 2>&1

echo "========================================"
echo "core cron run: $(date)"
echo "========================================"

# activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

EPOCHS=50
EVAL_ONLY=false

for arg in "$@"; do
    case $arg in
        --epochs=*) EPOCHS="${arg#*=}" ;;
        --epochs) shift; EPOCHS="$1" ;;
        --eval-only) EVAL_ONLY=true ;;
    esac
done

# train
if [ "$EVAL_ONLY" = false ]; then
    echo ""
    echo "--- Training ($EPOCHS epochs) ---"
    python3 cron/train_session.py --epochs "$EPOCHS"
    TRAIN_EXIT=$?

    if [ $TRAIN_EXIT -ne 0 ]; then
        echo "Training failed with exit code $TRAIN_EXIT"
        echo "Will still attempt evaluation..."
    fi
fi

# evaluate
if [ -f "models/cron_best.pt" ]; then
    echo ""
    echo "--- Evaluating ---"
    python3 cron/evaluate.py
else
    echo "No checkpoint to evaluate yet."
fi

echo ""
echo "--- Done: $(date) ---"
echo "Log: $LOGFILE"
