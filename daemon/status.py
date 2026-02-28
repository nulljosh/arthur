#!/usr/bin/env python3
"""Quick status check for arthur training."""

import json
import os
from pathlib import Path
from datetime import datetime

ARTHUR_ROOT = Path(__file__).parent.parent

def check_status():
    state_file = ARTHUR_ROOT / "daemon_state.json"
    watchdog_log = ARTHUR_ROOT / "logs/watchdog.log"
    training_log = ARTHUR_ROOT / "logs/training.log"
    
    print("\n=== ARTHUR TRAINING STATUS ===\n")
    
    # Daemon status
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
        print(f"Epoch: {state['epoch']}/{state['total']}")
        if 'training_started' in state:
            print(f"Started: {state['training_started']}")
    else:
        print("Status: Not initialized")
    
    # Recent logs
    if training_log.exists():
        with open(training_log) as f:
            lines = f.readlines()[-5:]
        print("\nRecent training logs:")
        for line in lines:
            print(f"  {line.rstrip()}")
    
    if watchdog_log.exists():
        with open(watchdog_log) as f:
            lines = f.readlines()[-3:]
        print("\nRecent daemon logs:")
        for line in lines:
            print(f"  {line.rstrip()}")
    
    # Storage check
    import psutil
    disk = psutil.disk_usage('/')
    print(f"\nStorage: {disk.free / 1024**3:.1f}GB free / {disk.total / 1024**3:.0f}GB total")
    
    print("\n")

if __name__ == '__main__':
    check_status()
