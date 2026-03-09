#!/usr/bin/env python3
"""Arthur training status dashboard."""

import json
import os
import re
import subprocess
from pathlib import Path
from datetime import datetime

ARTHUR_ROOT = Path(__file__).parent.parent
LOG_DIR = ARTHUR_ROOT / "logs"
STATE_FILE = ARTHUR_ROOT / "daemon_state.json"
MODELS_DIR = ARTHUR_ROOT / "models"


def fmt_size(bytes_val):
    if bytes_val > 1024**3:
        return f"{bytes_val / 1024**3:.1f}GB"
    if bytes_val > 1024**2:
        return f"{bytes_val / 1024**2:.1f}MB"
    return f"{bytes_val / 1024:.0f}KB"


def fmt_ago(ts):
    delta = (datetime.now() - ts).total_seconds()
    if delta < 60:
        return f"{delta:.0f}s ago"
    if delta < 3600:
        return f"{delta/60:.0f}m ago"
    if delta < 86400:
        return f"{delta/3600:.1f}h ago"
    return f"{delta/86400:.1f}d ago"


def check_status():
    print("\n  ARTHUR TRAINING STATUS")
    print("  " + "=" * 40)

    # Daemon state
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            state = json.load(f)
        size = state.get('size', '?')
        epoch = state.get('epoch', '?')
        total = state.get('total', '?')
        started = state.get('training_started', 'unknown')
        print(f"\n  Model:   {size} (epoch {epoch}/{total})")
        if started != 'unknown':
            try:
                ts = datetime.fromisoformat(started)
                print(f"  Started: {ts.strftime('%Y-%m-%d %H:%M')} ({fmt_ago(ts)})")
            except ValueError:
                print(f"  Started: {started}")
    else:
        print("\n  State:   Not initialized")

    # Watchdog LaunchAgent
    print()
    try:
        result = subprocess.run(
            ["launchctl", "list"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if "com.joshua.arthur" in line and "cron" not in line:
                parts = line.split()
                pid = parts[0] if parts[0] != "-" else None
                exit_code = parts[1] if len(parts) > 1 else "?"
                if pid and pid != "-":
                    print(f"  Watchdog: RUNNING (pid {pid})")
                else:
                    print(f"  Watchdog: STOPPED (exit code {exit_code})")
                break
        else:
            print("  Watchdog: NOT LOADED")
    except Exception:
        print("  Watchdog: unknown")

    # Latest training progress from training.log
    training_log = LOG_DIR / "training.log"
    if training_log.exists():
        lines = training_log.read_text().strip().splitlines()
        last_step, last_loss = None, None
        for line in reversed(lines[-100:]):
            m = re.search(r'^\s*(\d+)\s+([\d.]+)\s+', line)
            if m:
                last_step = int(m.group(1))
                last_loss = float(m.group(2))
                break
        if last_step is not None:
            mtime = datetime.fromtimestamp(training_log.stat().st_mtime)
            print(f"  Training: step {last_step}, loss {last_loss:.4f} ({fmt_ago(mtime)})")
        else:
            print("  Training: no step data found")
        # Show last 5 training lines
        print()
        step_lines = [l for l in lines[-20:] if re.match(r'^\s*\d+\s+[\d.]+', l)]
        for line in step_lines[-5:]:
            print(f"    {line.rstrip()}")
    else:
        print("  Training: no log file")

    # Watchdog last activity
    watchdog_log = LOG_DIR / "watchdog.log"
    if watchdog_log.exists():
        mtime = datetime.fromtimestamp(watchdog_log.stat().st_mtime)
        print(f"\n  Watchdog last active: {fmt_ago(mtime)}")

    # Recent errors
    print()
    error_lines = []
    if watchdog_log.exists():
        lines = watchdog_log.read_text().strip().splitlines()
        for line in lines[-30:]:
            if "ERROR" in line:
                error_lines.append(line.strip())
    daemon_err = LOG_DIR / "daemon_error.log"
    if daemon_err.exists() and daemon_err.stat().st_size > 0:
        lines = daemon_err.read_text().strip().splitlines()
        for line in lines[-10:]:
            if "Error" in line or "error" in line or "Traceback" in line or "Operation not permitted" in line:
                error_lines.append(line.strip())
    if error_lines:
        print("  Recent errors:")
        for line in error_lines[-5:]:
            print(f"    {line[:120]}")
    else:
        print("  Errors: none")

    # Checkpoints
    print()
    ckpts = sorted(MODELS_DIR.glob("arthur_v3_*.pt"))
    if ckpts:
        print("  Checkpoints:")
        for ckpt in ckpts:
            stat = ckpt.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime)
            print(f"    {ckpt.name}: {fmt_size(stat.st_size)} ({fmt_ago(mtime)})")
    else:
        print("  Checkpoints: none")

    # System resources
    print()
    try:
        import psutil
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        cpu = psutil.cpu_percent(interval=0.5)
        print(f"  RAM:  {mem.available / 1024**3:.1f}GB free / {mem.total / 1024**3:.0f}GB total ({mem.percent}% used)")
        print(f"  Disk: {disk.free / 1024**3:.1f}GB free / {disk.total / 1024**3:.0f}GB total")
        print(f"  CPU:  {cpu:.1f}%")
    except ImportError:
        # Fallback without psutil
        stat = os.statvfs('/')
        free_gb = stat.f_bavail * stat.f_frsize / 1024**3
        total_gb = stat.f_blocks * stat.f_frsize / 1024**3
        print(f"  Disk: {free_gb:.1f}GB free / {total_gb:.0f}GB total")
        print("  RAM/CPU: install psutil for details")

    print()


if __name__ == '__main__':
    check_status()
