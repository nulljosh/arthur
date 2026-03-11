#!/usr/bin/env python3
"""Run a tiny Arthur smoke test instead of background training."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from web_ui import DEFAULT_DATA_PATH, DEFAULT_MODEL_PATH, DEFAULT_TOKENIZER_PATH, generate_text, load_runtime


DEFAULT_PROMPTS = [
    "Say hello in one short sentence.",
    "What is 2 plus 2?",
    "Finish this phrase: The sky is",
]


def run_smoke(model_path: Path, data_path: Path, tokenizer_path: Path, prompts: list[str]) -> dict:
    runtime = load_runtime(model_path, data_path, tokenizer_path)

    results = []
    for prompt in prompts:
        started_at = time.perf_counter()
        raw = generate_text(
            runtime,
            {
                "prompt": prompt,
                "length": 32,
                "temperature": 0.3,
                "top_k": 20,
                "top_p": 0.9,
            },
        )
        latency_ms = round((time.perf_counter() - started_at) * 1000, 1)
        completion = raw[len(prompt) :].strip() if raw.startswith(prompt) else raw.strip()
        results.append(
            {
                "prompt": prompt,
                "latency_ms": latency_ms,
                "output": completion,
                "ok": bool(completion),
            }
        )

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_path": str(model_path),
        "tokenizer_path": str(tokenizer_path),
        "all_ok": all(item["ok"] for item in results),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Arthur demo smoke test")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER_PATH))
    parser.add_argument("--output", default="logs/demo_smoke_latest.json")
    parser.add_argument("--prompt", action="append", default=[])
    args = parser.parse_args()

    prompts = args.prompt or DEFAULT_PROMPTS
    report = run_smoke(Path(args.model), Path(args.data), Path(args.tokenizer), prompts)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
