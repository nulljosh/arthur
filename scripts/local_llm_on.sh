#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-qwen2.5:3b}"
STATE_FILE="$ROOT/logs/local_llm_state.env"
LOG_FILE="$ROOT/logs/ollama-manual.log"

mkdir -p "$ROOT/logs"

case "$MODEL" in
  qwen2.5:0.5b|qwen2.5:1.5b|qwen2.5:3b|qwen3:1.7b|qwen3:4b|gemma3:1b|gemma3:4b|phi4-mini)
    ;;
  *)
    echo "Refusing model '$MODEL'. Use a small manual model only." >&2
    exit 1
    ;;
esac

if pgrep -f "/Applications/Ollama.app/Contents/MacOS/Ollama hidden" >/dev/null; then
  echo "Quit Ollama.app first. This workflow uses manual CLI Ollama only." >&2
  exit 1
fi

python3 "$ROOT/scripts/resource_guard.py" check \
  --min-free-gb 6 \
  --max-swap-used-gb 1.5 \
  --max-memory-used-pct 80 \
  --deny-process-pattern "scripts/train.py" \
  --deny-process-pattern "arthur_watchdog.py"

if ! command -v ollama >/dev/null; then
  echo "ollama is not installed" >&2
  exit 1
fi

if ! ollama list | awk 'NR>1 {print $1}' | grep -Fx "$MODEL" >/dev/null; then
  echo "Model '$MODEL' is not installed in Ollama." >&2
  exit 1
fi

if ! pgrep -f "ollama serve" >/dev/null; then
  env OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=0 \
    nohup ollama serve >>"$LOG_FILE" 2>&1 &
  sleep 2
fi

cat >"$STATE_FILE" <<EOF
MODEL=$MODEL
STARTED_AT=$(date '+%Y-%m-%d %H:%M:%S')
EOF

echo "Local LLM ready: $MODEL"
echo "Next: ./scripts/local_llm_prompt.sh \"$MODEL\" \"your prompt\""
