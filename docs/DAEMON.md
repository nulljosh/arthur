# Arthur Background State

Arthur's old training daemon is currently parked. This project now defaults to demo / eval mode on this 16 GB machine.

## Current Behavior

- LaunchAgent: `daemon/com.joshua.arthur.plist`
- Watchdog entrypoint: `daemon/arthur_watchdog.py`
- Trainer entrypoint: `scripts/train.py`
- LaunchAgent state: disabled
- Background training: disabled
- Demo automation: `scripts/demo_smoke.py`
- Manual local LLM: `scripts/local_llm_on.sh`, `scripts/local_llm_prompt.sh`, `scripts/local_llm_off.sh`

## Status

```bash
./daemon/status.sh
python3 scripts/demo_smoke.py
tail -f logs/health.log
```

## LaunchAgent

```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.joshua.arthur.plist
launchctl disable gui/$(id -u)/com.joshua.arthur
launchctl list | grep arthur
```

## Resource Rules

Heavy local jobs are manual-only. `scripts/resource_guard.py` refuses to start or kills a job when:
- free RAM is too low
- swap usage is already high
- the process RSS exceeds a hard cap
- the process runs past its timeout

## Important Files

- `daemon/arthur_watchdog.py` - watchdog and resource policy
- `scripts/train.py` - manual-only v3 trainer
- `scripts/demo_smoke.py` - 2-3 prompt smoke test
- `scripts/resource_guard.py` - local job guardrails
- `daemon/status.sh` - quick shell status
- `daemon_state.json` - watchdog state
- `models/arthur_v3_65M_latest.pt` - resume checkpoint
- `models/arthur_v3_65M_best.pt` - best weights

## Known Notes

- The daemon files remain in-tree for future reference, but they are intentionally disabled
- Small local models only; no always-on Ollama
- This machine should not run local training and local inference at the same time

## Recovery

If a heavy local process appears stuck:

```bash
./scripts/local_llm_off.sh
pkill -f 'scripts/train.py'
pkill -f 'arthur_watchdog.py'
```

If you need to sanity-check the demo path manually:

```bash
python3 scripts/web_ui.py
python3 scripts/demo_smoke.py
```
