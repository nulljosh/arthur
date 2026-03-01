# Arthur Training Daemon

Autonomous background training that respects your Mac's resources.

## Features

✅ **Auto-Training**: Starts when system idle (5+ mins no keyboard/mouse)
✅ **Resource Management**: Pauses if disk <5GB, CPU >70%, RAM <4GB
✅ **Smart Batching**: Adjusts batch size based on available RAM
✅ **Auto-Checkpoints**: Saves every epoch completion
✅ **Daily Backups**: Checkpoints backed up at 2 AM daily
✅ **Health Checks**: Every 6 hours, auto-restarts if crashed
✅ **Graceful Shutdown**: Pauses training at 11:59 PM each day

## Status

```bash
# Check training status
./daemon/status.sh

# Real-time logs
tail -f logs/watchdog.log
tail -f logs/training.log
```

## Installation

Already installed! Running as LaunchAgent:

```bash
# View status
launchctl list | grep arthur

# Stop daemon
launchctl unload ~/Library/LaunchAgents/com.joshua.arthur.plist

# Restart daemon
launchctl load ~/Library/LaunchAgents/com.joshua.arthur.plist

# View logs
cat logs/watchdog.log
```

## Configuration

Edit `daemon/arthur_watchdog.py`:
- `storage_limit_gb = 5` — Pause if less than this free
- `cpu_limit = 70` — Pause if CPU usage exceeds this
- `ram_min_gb = 4` — Pause if less than this available

## Cron Jobs

Installed crons (every 6 hours + daily tasks):
- **2 AM daily**: Backup checkpoint
- **Every 6 hours**: Health check, restart if needed
- **Sundays 10 AM**: Audit dataset (optional)
- **11:59 PM daily**: Graceful shutdown

View installed crons:
```bash
crontab -l | grep arthur
```

## Storage Management

Current setup:
- **Limit**: Keep only last 7 backups (5 checkpoint + 2 rolling)
- **Dataset**: 7K examples × ~18 tokens avg = 125K tokens (~700KB)
- **Models**: Latest checkpoint ~260MB (FP32), ~65MB (INT8)
- **Logs**: Rotate daily, keep 7 days

Total disk usage: <1GB under normal operations.

## Monitoring

Daemon tracks:
- Current epoch / total epochs
- Last training start time
- Pause/resume events
- Resource metrics (disk, CPU, RAM)

## Troubleshooting

**Daemon not starting?**
```bash
# Check if installed
launchctl list | grep arthur

# Re-load
launchctl load ~/Library/LaunchAgents/com.joshua.arthur.plist
```

**Training never starts?**
```bash
# Check if system thinks you're idle
# (Requires idle time, try waiting 5+ mins with no mouse/keyboard)

# Force training start (bypass idle check)
python3 src/train_v2.py --epochs 50 --batch_size 32
```

**Out of disk space?**
```bash
# Check current size
du -sh ~/Documents/Code/arthur

# Clean old backups
rm -rf backups/*

# Clean training logs
rm -rf logs/training.log
```

## How It Works

1. **Watchdog runs continuously** (LaunchAgent)
2. **Every 60 seconds**: Check system health
3. **If healthy + idle + not training**: Start training
4. **If unhealthy + training**: Pause gracefully
5. **On epoch complete**: Save checkpoint, increment epoch counter
6. **Cron jobs**: Backup, health checks, daily shutdown

## Real-time Monitoring

```bash
# Watch status every 10 seconds
watch -n 10 './daemon/status.sh'

# Real-time log tail (both daemon + training)
tail -f logs/watchdog.log logs/training.log
```

## Next Steps

1. BPE tokenizer will finish training automatically
2. Daemon will auto-start training once idle
3. Check status with: `./daemon/status.sh`
4. Monitor progress: `tail -f logs/training.log`

---

**Summary**: Set it and forget it. Arthur trains 50 epochs in background, respects your Mac, auto-checkpoints, and resumes intelligently.
