#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-qwen2.5:3b}"
shift || true

if [ "$#" -eq 0 ]; then
  echo "Usage: ./scripts/local_llm_prompt.sh <small-model> <prompt...>" >&2
  exit 1
fi

PROMPT="$*"

if ! pgrep -f "ollama serve" >/dev/null; then
  "$ROOT/scripts/local_llm_on.sh" "$MODEL"
fi

python3 "$ROOT/scripts/resource_guard.py" run \
  --min-free-gb 4 \
  --max-swap-used-gb 2 \
  --max-memory-used-pct 82 \
  --max-rss-gb 6 \
  --timeout-seconds 180 \
  --deny-process-pattern "scripts/train.py" \
  --deny-process-pattern "arthur_watchdog.py" \
  -- ollama run "$MODEL" "$PROMPT"
