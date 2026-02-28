#!/usr/bin/env python3
"""Auto-refresh arthur docs before git push."""

import json
import os
import subprocess
from pathlib import Path
from datetime import datetime

ARTHUR_ROOT = Path(__file__).parent.parent
STATE_FILE = ARTHUR_ROOT / "daemon_state.json"
README = ARTHUR_ROOT / "README.md"
CLAUDE_MD = ARTHUR_ROOT / "CLAUDE.md"
LOG_DIR = ARTHUR_ROOT / "logs"
MODELS_DIR = ARTHUR_ROOT / "models"

def get_training_progress():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {'epoch': 0, 'total': 50}

def get_latest_checkpoint():
    checkpoints = sorted(MODELS_DIR.glob("arthur_v2_epoch*.pt"))
    if not checkpoints:
        return None
    latest = checkpoints[-1]
    return {
        'path': latest.name,
        'size_mb': latest.stat().st_size / 1024 / 1024,
        'epoch': int(latest.stem.split('epoch')[-1])
    }

def get_training_metrics():
    if not (LOG_DIR / "training.log").exists():
        return {}
    try:
        with open(LOG_DIR / "training.log") as f:
            lines = f.readlines()[-50:]
        for line in reversed(lines):
            if "loss" in line.lower():
                parts = line.split("loss")
                if len(parts) > 1:
                    try:
                        loss_str = parts[1].split()[0].strip("=:")
                        return {'latest_loss': float(loss_str)}
                    except:
                        pass
    except:
        pass
    return {}

def update_readme():
    progress = get_training_progress()
    checkpoint = get_latest_checkpoint()
    metrics = get_training_metrics()
    
    epoch = progress.get('epoch', 0)
    total = progress.get('total', 50)
    
    checkpoint_line = f"{checkpoint['path']} ({checkpoint['size_mb']:.0f}MB)" if checkpoint else "None"
    loss_line = f"{metrics['latest_loss']:.4f}" if metrics.get('latest_loss') else "N/A"
    
    status = f"""## Training Status

**Progress:** Epoch {epoch}/{total} ({100*epoch//total}% complete)
**Latest Checkpoint:** {checkpoint_line}
**Last Loss:** {loss_line}
**Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

Status: Daemon auto-training when idle. Respects resources (disk <5GB, CPU <70%, RAM >4GB).
"""
    
    if README.exists():
        with open(README) as f:
            content = f.read()
    else:
        content = "# Arthur\n\n"
    
    if "## Training Status" in content:
        start = content.find("## Training Status")
        end = content.find("\n## ", start + 1)
        if end == -1:
            end = len(content)
        content = content[:start] + status.rstrip() + "\n\n" + content[end:]
    else:
        content = "# Arthur\n\n" + status + "\n" + content
    
    with open(README, 'w') as f:
        f.write(content)
    print(f"✓ README (epoch {epoch}/{total})")

def update_claude_md():
    progress = get_training_progress()
    epoch = progress.get('epoch', 0)
    total = progress.get('total', 50)
    
    if CLAUDE_MD.exists():
        with open(CLAUDE_MD) as f:
            content = f.read()
        
        # Update status
        if "**Status:**" in content:
            idx = content.find("**Status:**")
            end = content.find("\n", idx)
            content = content[:idx] + f"**Status:** Phase 3 (Epoch {epoch}/{total})" + content[end:]
        
        with open(CLAUDE_MD, 'w') as f:
            f.write(content)
        
        print(f"✓ CLAUDE.md (phase 3, epoch {epoch}/{total})")

def main():
    update_readme()
    update_claude_md()

if __name__ == '__main__':
    main()
