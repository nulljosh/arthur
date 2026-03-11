#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

pkill -TERM -f "ollama runner" 2>/dev/null || true
pkill -TERM -f "ollama serve" 2>/dev/null || true
pkill -TERM -f "/Applications/Ollama.app/Contents/MacOS/Ollama hidden" 2>/dev/null || true
pkill -TERM -f "Contents/Frameworks/Squirrel.framework/Versions/A/Squirrel background" 2>/dev/null || true

rm -f "$ROOT/logs/local_llm_state.env"
echo "Local LLM processes stopped."
