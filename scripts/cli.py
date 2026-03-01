#!/usr/bin/env python3
"""Terminal chat interface for Arthur LLM."""

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, 'src')

from web_ui import _sample_next_token, generate_text, load_runtime

GREEN = "\033[32m"
RESET = "\033[0m"


def print_banner(runtime) -> None:
    params = sum(p.numel() for p in runtime.model.parameters())
    embed_dim = runtime.config.get("embed_dim", getattr(runtime.model, "embed_dim", "?"))
    vocab_size = getattr(runtime.tokenizer, "vocab_size", runtime.config.get("vocab_size", "?"))

    print("Arthur CLI")
    print(f"Model: {runtime.model_path}")
    print(f"Params: {params:,}")
    print(f"Embed dim: {embed_dim}")
    print(f"Vocab size: {vocab_size}")
    print("Commands: /quit, /clear, /model <path>")


def stream_generate(runtime, prompt: str, temperature: float, max_new_tokens: int = 256) -> str:
    tokens = runtime.tokenizer.encode(prompt)
    if not tokens:
        return ""

    max_len = int(runtime.config.get("max_len", 128))
    if len(tokens) >= max_len:
        tokens = tokens[-(max_len - 1) :]

    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    generated = list(tokens)
    previous_text = runtime.tokenizer.decode(generated)

    # Keep a reference to the shared generation helper for parity with web_ui imports.
    _ = generate_text

    sys.stdout.write(GREEN)
    sys.stdout.flush()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if x.size(1) >= max_len:
                x = x[:, -(max_len - 1) :]

            logits = runtime.model(x)[0, -1, :]
            next_token = _sample_next_token(logits, float(temperature), None, None)
            generated.append(next_token)
            x = torch.cat([x, torch.tensor([[next_token]], dtype=torch.long)], dim=1)

            current_text = runtime.tokenizer.decode(generated)
            delta = current_text[len(previous_text) :]
            if delta:
                for ch in delta:
                    sys.stdout.write(ch)
                    sys.stdout.flush()
                previous_text = current_text

    sys.stdout.write(f"{RESET}\n")
    sys.stdout.flush()
    return previous_text


def handle_command(line: str, runtime, data_path: Path):
    parts = line.strip().split(maxsplit=1)
    cmd = parts[0]

    if cmd == "/quit":
        return None, False

    if cmd == "/clear":
        os.system("cls" if os.name == "nt" else "clear")
        return runtime, True

    if cmd == "/model":
        if len(parts) < 2:
            print("Usage: /model <path>")
            return runtime, True

        new_model_path = Path(parts[1]).expanduser()
        try:
            new_runtime = load_runtime(new_model_path, data_path)
            print_banner(new_runtime)
            return new_runtime, True
        except Exception as exc:
            print(f"Failed to load model: {exc}")
            return runtime, True

    print(f"Unknown command: {cmd}")
    return runtime, True


def main() -> int:
    parser = argparse.ArgumentParser(description="Arthur terminal chat")
    parser.add_argument("--model", default="models/overnight_best.pt")
    parser.add_argument("--temp", type=float, default=0.3)
    parser.add_argument("--data", default="data/ultra_minimal.txt")
    args = parser.parse_args()

    model_path = Path(args.model)
    data_path = Path(args.data)

    try:
        runtime = load_runtime(model_path, data_path)
    except Exception as exc:
        print(f"Failed to load runtime: {exc}")
        return 1

    print_banner(runtime)

    while True:
        try:
            line = input("arthur> ")
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            continue

        if not line.strip():
            continue

        if line.startswith("/"):
            runtime, should_continue = handle_command(line, runtime, data_path)
            if not should_continue:
                break
            continue

        try:
            stream_generate(runtime, line, args.temp)
        except KeyboardInterrupt:
            sys.stdout.write(f"{RESET}\n")
            sys.stdout.flush()
            continue
        except Exception as exc:
            sys.stdout.write(f"{RESET}\n")
            sys.stdout.flush()
            print(f"Generation error: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
