# Arthur Training Daemon

Arthur now runs an always-on v3 watchdog via launchd. It continuously evaluates system resources and starts/resumes training when safe.

## Current Behavior

- LaunchAgent: `daemon/com.joshua.arthur.plist`
- Watchdog entrypoint: `daemon/arthur_watchdog.py`
- Trainer entrypoint: `scripts/train.py`
- Model size on this machine: `65M`
- Resume behavior: automatic via `models/arthur_v3_65M_latest.pt`
- Full-state checkpoints: every 25 steps, plus immediately after the first resumed step

## Status

```bash
./daemon/status.sh
python3 daemon/status.py
tail -f logs/training.log
tail -f logs/daemon_error.log
```

## LaunchAgent

```bash
launchctl unload ~/Library/LaunchAgents/com.joshua.arthur.plist
launchctl load ~/Library/LaunchAgents/com.joshua.arthur.plist
launchctl list | grep arthur
```

## Resource Rules

The watchdog runs in three modes:
- `full`: batch size 2
- `low`: batch size 1
- `pause`: no training

It pauses or kills training if:
- disk is critically low
- RAM pressure is too high
- swap growth spikes
- the process hangs too long

## Important Files

- `daemon/arthur_watchdog.py` - watchdog and resource policy
- `scripts/train.py` - v3 trainer
- `daemon/status.sh` - quick shell status
- `daemon/status.py` - detailed status dashboard
- `daemon_state.json` - watchdog state
- `models/arthur_v3_65M_latest.pt` - resume checkpoint
- `models/arthur_v3_65M_best.pt` - best weights

## Known Notes

- Older v2/idle-based docs are obsolete
- Python default HTTP clients may show unrelated SSL/HuggingFace warnings in logs
- This machine stays on 65M; 125M promotion is disabled on <24GB RAM

## Recovery

If training appears stuck:

```bash
python3 daemon/status.py
pkill -f 'scripts/train.py'
launchctl unload ~/Library/LaunchAgents/com.joshua.arthur.plist
launchctl load ~/Library/LaunchAgents/com.joshua.arthur.plist
```

If you need to sanity-check resume manually:

```bash
python3 scripts/train.py --size=65M --steps=102 --batch_size=1 --seq_len=128 --grad_accum=1 --resume
```
