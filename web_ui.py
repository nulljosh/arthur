#!/usr/bin/env python3
"""Lightweight Flask UI for local nous model generation."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, send_file

# Ensure local modules are importable when checkpoints contain pickled tokenizers.
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from transformer import Nous
from tokenizer import CharTokenizer


DEFAULT_MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/overnight_best.pt"))
DEFAULT_DATA_PATH = Path(os.getenv("DATA_PATH", "data/ultra_minimal.txt"))
DEFAULT_PORT = int(os.getenv("PORT", "5001"))
DEBUG_MODE = os.getenv("DEBUG", "false").lower() in {"1", "true", "yes", "on"}

MAX_PROMPT_CHARS = 4000
MAX_GENERATION_TOKENS = 512
MAX_TEMPERATURE = 3.0

app = Flask(__name__)


@dataclass
class LoadedRuntime:
    model: Any
    tokenizer: Any
    model_path: Path
    source: str
    config: dict[str, Any]


_RUNTIME: LoadedRuntime | None = None
_RUNTIME_ERROR: str | None = None
_RUNTIME_ATTEMPTED = False


class ValidationError(ValueError):
    """Raised when user input fails API validation."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def _build_char_tokenizer_from_vocab(vocab: Any) -> CharTokenizer:
    if isinstance(vocab, dict):
        char_to_idx = {str(ch): int(idx) for ch, idx in vocab.items()}
    elif isinstance(vocab, list):
        char_to_idx = {str(ch): i for i, ch in enumerate(vocab)}
    else:
        raise ValueError("Unsupported vocab format in checkpoint")

    # Old checkpoints may not store an explicit UNK token.
    # Map UNK to index 0 as a fallback without changing vocab size.
    if CharTokenizer.UNK not in char_to_idx:
        char_to_idx[CharTokenizer.UNK] = min(char_to_idx.values(), default=0)

    tokenizer = CharTokenizer(CharTokenizer.UNK)
    ordered = sorted(char_to_idx.items(), key=lambda item: item[1])
    tokenizer.char_to_idx = {ch: idx for ch, idx in ordered}
    tokenizer.idx_to_char = {idx: ch for ch, idx in tokenizer.char_to_idx.items()}
    tokenizer.vocab_size = max(tokenizer.char_to_idx.values(), default=-1) + 1
    return tokenizer


def _infer_nous_config(state_dict: dict[str, Any], vocab_size: int) -> dict[str, Any]:
    embed_dim = int(state_dict["token_embed.weight"].shape[1])
    ff_dim = int(state_dict["blocks.0.ffn.net.0.weight"].shape[0])
    max_len = int(state_dict["pos_embed.weight"].shape[0])

    layer_indexes = {
        int(parts[1])
        for key in state_dict
        if key.startswith("blocks.")
        for parts in [key.split(".")]
        if len(parts) > 1 and parts[1].isdigit()
    }
    num_layers = max(layer_indexes) + 1 if layer_indexes else 2

    if embed_dim <= 32 and embed_dim % 2 == 0:
        num_heads = 2
    elif embed_dim % 4 == 0:
        num_heads = 4
    elif embed_dim % 2 == 0:
        num_heads = 2
    else:
        num_heads = 1

    return {
        "vocab_size": int(vocab_size),
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_layers": int(num_layers),
        "ff_dim": ff_dim,
        "max_len": max_len,
        "dropout": 0.0,
    }


def load_runtime(model_path: Path, data_path: Path) -> LoadedRuntime:
    import torch

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format: expected dict")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        raise ValueError("Unsupported checkpoint: missing model weights key")

    tokenizer_source = "checkpoint"
    if "tokenizer" in checkpoint and hasattr(checkpoint["tokenizer"], "encode"):
        tokenizer = checkpoint["tokenizer"]
    elif "vocab" in checkpoint:
        tokenizer = _build_char_tokenizer_from_vocab(checkpoint["vocab"])
    elif data_path.exists():
        tokenizer = CharTokenizer(data_path.read_text())
        tokenizer_source = "data_path"
    else:
        raise ValueError("Could not reconstruct tokenizer (missing tokenizer/vocab/data file)")

    config = checkpoint.get("config")
    if not isinstance(config, dict):
        config = _infer_nous_config(state_dict, int(checkpoint.get("vocab_size", tokenizer.vocab_size)))
    else:
        config = {**config}
        config["vocab_size"] = int(checkpoint.get("vocab_size", tokenizer.vocab_size))

    model = Nous(
        vocab_size=int(config["vocab_size"]),
        embed_dim=int(config["embed_dim"]),
        num_heads=int(config["num_heads"]),
        num_layers=int(config["num_layers"]),
        ff_dim=int(config["ff_dim"]),
        max_len=int(config["max_len"]),
        dropout=float(config.get("dropout", 0.0)),
    )
    model.load_state_dict(state_dict)
    model.eval()

    return LoadedRuntime(
        model=model,
        tokenizer=tokenizer,
        model_path=model_path,
        source=tokenizer_source,
        config=config,
    )


def _initialize_runtime_once() -> None:
    global _RUNTIME, _RUNTIME_ERROR, _RUNTIME_ATTEMPTED

    if _RUNTIME_ATTEMPTED:
        return

    _RUNTIME_ATTEMPTED = True
    try:
        _RUNTIME = load_runtime(DEFAULT_MODEL_PATH, DEFAULT_DATA_PATH)
        _RUNTIME_ERROR = None
    except Exception as exc:  # pragma: no cover - startup diagnostics
        _RUNTIME = None
        _RUNTIME_ERROR = str(exc)


def _to_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip():
        return int(value)
    raise ValueError(f"{field} must be an integer")


def _to_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a number")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        return float(value)
    raise ValueError(f"{field} must be a number")


def validate_generation_input(payload: Any, vocab_size: int | None = None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValidationError(["Request body must be a JSON object"])

    errors: list[str] = []

    prompt = payload.get("prompt", "")
    if not isinstance(prompt, str):
        errors.append("prompt must be a string")
    elif not prompt.strip():
        errors.append("prompt must not be empty")
    elif len(prompt) > MAX_PROMPT_CHARS:
        errors.append(f"prompt must be <= {MAX_PROMPT_CHARS} characters")

    length_raw = payload.get("length", 120)
    try:
        length = _to_int(length_raw, "length")
        if not 1 <= length <= MAX_GENERATION_TOKENS:
            errors.append(f"length must be between 1 and {MAX_GENERATION_TOKENS}")
    except Exception:
        errors.append("length must be an integer")
        length = 120

    temp_raw = payload.get("temperature", 0.8)
    try:
        temperature = _to_float(temp_raw, "temperature")
        if not 0.0 < temperature <= MAX_TEMPERATURE:
            errors.append(f"temperature must be > 0 and <= {MAX_TEMPERATURE}")
    except Exception:
        errors.append("temperature must be a number")
        temperature = 0.8

    top_k: int | None = None
    top_k_raw = payload.get("top_k")
    if top_k_raw not in (None, ""):
        try:
            top_k = _to_int(top_k_raw, "top_k")
            if top_k < 0:
                errors.append("top_k must be >= 0")
            if vocab_size and top_k > vocab_size:
                errors.append(f"top_k must be <= vocab size ({vocab_size})")
        except Exception:
            errors.append("top_k must be an integer")

    top_p: float | None = None
    top_p_raw = payload.get("top_p")
    if top_p_raw not in (None, ""):
        try:
            top_p = _to_float(top_p_raw, "top_p")
            if not 0.0 < top_p <= 1.0:
                errors.append("top_p must be > 0 and <= 1")
        except Exception:
            errors.append("top_p must be a number")

    if errors:
        raise ValidationError(errors)

    return {
        "prompt": prompt,
        "length": length,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
    }


def _sample_next_token(logits, temperature: float, top_k: int | None, top_p: float | None) -> int:
    import torch

    scaled = logits / temperature
    probs = torch.softmax(scaled, dim=-1)

    if top_k and top_k > 0:
        k = min(int(top_k), probs.shape[-1])
        top_vals, top_idx = torch.topk(probs, k)
        masked = torch.zeros_like(probs)
        masked.scatter_(0, top_idx, top_vals)
        probs = masked

    if top_p and top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove_mask = cumulative > top_p
        remove_mask[0] = False
        sorted_probs[remove_mask] = 0.0
        nucleus = torch.zeros_like(probs)
        nucleus.scatter_(0, sorted_idx, sorted_probs)
        probs = nucleus

    total = float(probs.sum().item())
    if total <= 0.0:
        return int(torch.argmax(scaled).item())

    probs = probs / probs.sum()
    return int(torch.multinomial(probs, num_samples=1).item())


def generate_text(runtime: LoadedRuntime, params: dict[str, Any]) -> str:
    import torch

    prompt = params["prompt"]
    max_new_tokens = int(params["length"])
    temperature = float(params["temperature"])
    top_k = params.get("top_k")
    top_p = params.get("top_p")

    tokens = runtime.tokenizer.encode(prompt)
    if not tokens:
        raise ValueError("Prompt did not produce any tokens")

    max_len = int(runtime.config.get("max_len", 128))
    if len(tokens) >= max_len:
        tokens = tokens[-(max_len - 1) :]

    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    generated = list(tokens)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if x.size(1) >= max_len:
                x = x[:, -(max_len - 1) :]

            logits = runtime.model(x)[0, -1, :]
            next_token = _sample_next_token(logits, temperature, top_k, top_p)
            generated.append(next_token)
            x = torch.cat([x, torch.tensor([[next_token]], dtype=torch.long)], dim=1)

    return runtime.tokenizer.decode(generated)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/quiz")
def quiz():
    return send_file("quiz.html")


@app.route("/api/status")
def status():
    _initialize_runtime_once()

    return jsonify(
        {
            "model_loaded": _RUNTIME is not None,
            "model_path": str(DEFAULT_MODEL_PATH),
            "data_path": str(DEFAULT_DATA_PATH),
            "runtime_error": _RUNTIME_ERROR,
            "vocab_size": _RUNTIME.tokenizer.vocab_size if _RUNTIME else 0,
            "config": _RUNTIME.config if _RUNTIME else {},
        }
    )


@app.route("/status")
def status_compat():
    return status()


@app.route("/api/generate", methods=["POST"])
@app.route("/generate", methods=["POST"])
def generate():
    _initialize_runtime_once()

    if _RUNTIME is None:
        return (
            jsonify(
                {
                    "error": "Model not loaded",
                    "details": [_RUNTIME_ERROR or "Unknown startup error"],
                }
            ),
            503,
        )

    payload = request.get_json(silent=True)
    try:
        params = validate_generation_input(payload, vocab_size=_RUNTIME.tokenizer.vocab_size)
    except ValidationError as exc:
        return jsonify({"error": "Invalid input", "details": exc.errors}), 400

    try:
        text = generate_text(_RUNTIME, params)
        return jsonify({"text": text, "params": params})
    except Exception as exc:
        return jsonify({"error": "Generation failed", "details": [str(exc)]}), 500


if __name__ == "__main__":
    _initialize_runtime_once()

    if _RUNTIME is not None:
        print(f"Loaded model: {_RUNTIME.model_path}")
        print(f"Tokenizer source: {_RUNTIME.source}")
        print(f"Vocab size: {_RUNTIME.tokenizer.vocab_size}")
    else:
        print("Model load failed:", _RUNTIME_ERROR)

    print(f"Starting Web UI on http://localhost:{DEFAULT_PORT}")
    app.run(debug=DEBUG_MODE, port=DEFAULT_PORT)
