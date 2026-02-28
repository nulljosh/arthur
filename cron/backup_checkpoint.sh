#!/bin/bash
# Daily backup of latest checkpoint

ARTHUR_ROOT="/Users/joshua/Documents/Code/arthur"
BACKUP_DIR="$ARTHUR_ROOT/backups"
MODELS_DIR="$ARTHUR_ROOT/models"

mkdir -p "$BACKUP_DIR"

# Find latest checkpoint
LATEST=$(ls -t "$MODELS_DIR"/arthur_v2_epoch*.pt 2>/dev/null | head -1)

if [ -n "$LATEST" ]; then
  BACKUP_NAME="checkpoint_$(date +%Y%m%d_%H%M%S).pt"
  cp "$LATEST" "$BACKUP_DIR/$BACKUP_NAME"
  echo "✓ Backed up: $BACKUP_NAME" >> "$ARTHUR_ROOT/logs/backups.log"
  
  # Keep only last 7 backups
  ls -t "$BACKUP_DIR"/*.pt | tail -n +8 | xargs rm -f 2>/dev/null
fi
