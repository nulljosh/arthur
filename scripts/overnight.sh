#!/bin/bash
# Arthur Overnight Training Daemon
# Runs v3 training sessions sequentially, logs everything, auto-pushes on completion

cd ~/Documents/Code/arthur
source venv/bin/activate

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/overnight_${TIMESTAMP}.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== ARTHUR OVERNIGHT START ==="
log "Disk free: $(df -h / | awk 'NR==2{print $4}')"
log "Models dir:"
ls -lh models/*.pt 2>/dev/null | tee -a "$LOG"

# --- Phase 1: 65M full run ---
log ""
log "=== PHASE 1: 65M (1000 steps) ==="
python3 scripts/train.py \
  --size 65M \
  --steps 1000 \
  --batch_size 4 \
  --seq_len 256 \
  2>&1 | tee -a "$LOG"

log "65M done. Checkpoint:"
ls -lh models/arthur_v3_65M_best.pt 2>/dev/null | tee -a "$LOG"

# --- Phase 2: Eval 65M ---
log ""
log "=== EVAL 65M ==="
python3 scripts/eval.py \
  --checkpoint models/arthur_v3_65M_best.pt \
  --size 65M \
  2>&1 | tee -a "$LOG"

# --- Phase 3: 125M run ---
log ""
log "=== PHASE 3: 125M (500 steps) ==="
python3 scripts/train.py \
  --size 125M \
  --steps 500 \
  --batch_size 2 \
  --seq_len 256 \
  2>&1 | tee -a "$LOG"

log "125M done. Checkpoint:"
ls -lh models/arthur_v3_125M_best.pt 2>/dev/null | tee -a "$LOG"

# --- Phase 4: Eval 125M ---
log ""
log "=== EVAL 125M ==="
python3 scripts/eval.py \
  --checkpoint models/arthur_v3_125M_best.pt \
  --size 125M \
  2>&1 | tee -a "$LOG"

# --- Done: commit + push ---
log ""
log "=== PUSHING TO GITHUB ==="
git add -A
git commit -m "feat: overnight training $(date +%Y-%m-%d) - v3 65M + 125M" 2>&1 | tee -a "$LOG"
git push 2>&1 | tee -a "$LOG"

log "=== OVERNIGHT COMPLETE ==="
