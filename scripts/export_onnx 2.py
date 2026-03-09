#!/usr/bin/env python3
"""Export Arthur checkpoint to ONNX + tokenizer vocab JSON for browser inference."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
import importlib.util

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))
from tokenizer import CharTokenizer  # noqa: E402
from transformer import Arthur, ArthurV3, CONFIGS as V3_CONFIGS  # noqa: E402
from bpe_tokenizer import BPETokenizer  # noqa: E402

# v3 checkpoints first (preferred), then v2 fallbacks
CHECKPOINT_CANDIDATES = [
    REPO_ROOT / "models" / "arthur_v3_65M_best.pt",
    REPO_ROOT / "models" / "arthur_v3_125M_best.pt",
    REPO_ROOT / "models" / "arthur_trained.pt",
    REPO_ROOT / "models" / "cron_best.pt",
    REPO_ROOT / "models" / "overnight_best.pt",
]

DEFAULT_TOKENIZER_PATH = REPO_ROOT / "models" / "bpe_tokenizer_v1.json"

OUTPUT_DIR = REPO_ROOT / "public" / "model"
ONNX_PATH = OUTPUT_DIR / "arthur.onnx"
VOCAB_JSON_PATH = OUTPUT_DIR / "vocab.json"


def fmt_size(num_bytes: int) -> str:
    mib = num_bytes / (1024 * 1024)
    return f"{num_bytes:,} bytes ({mib:.2f} MiB)"


def find_checkpoint() -> Path:
    for checkpoint_path in CHECKPOINT_CANDIDATES:
        if checkpoint_path.exists():
            return checkpoint_path
    candidates = "\n".join(f"- {p}" for p in CHECKPOINT_CANDIDATES)
    raise FileNotFoundError(f"No checkpoint found. Checked:\n{candidates}")


def _is_v3_checkpoint(checkpoint: dict[str, Any]) -> bool:
    """Detect v3 checkpoint by state dict key patterns."""
    if isinstance(checkpoint.get("model"), dict) and "step" in checkpoint:
        sample_keys = list(checkpoint["model"].keys())[:5]
        return any("layers." in k or k == "embed.weight" for k in sample_keys)
    return "embed.weight" in checkpoint


def _infer_v3_size(state_dict: dict[str, Any]) -> str:
    """Infer v3 model size preset from embed.weight dimensions."""
    d_model = int(state_dict["embed.weight"].shape[1])
    for size, cfg in V3_CONFIGS.items():
        if cfg["d_model"] == d_model:
            return size
    raise ValueError(f"Cannot infer v3 size for d_model={d_model}")


def rebuild_char_tokenizer(vocab: Any, vocab_size: int | None = None) -> CharTokenizer:
    if not isinstance(vocab, dict):
        raise ValueError("Checkpoint field 'vocab' must be a dict (char_to_idx)")

    normalized = {str(ch): int(idx) for ch, idx in vocab.items()}
    ordered = sorted(normalized.items(), key=lambda kv: kv[1])

    tokenizer = CharTokenizer.__new__(CharTokenizer)
    tokenizer.char_to_idx = {ch: idx for ch, idx in ordered}
    tokenizer.idx_to_char = {idx: ch for ch, idx in tokenizer.char_to_idx.items()}

    if vocab_size is not None:
        tokenizer.vocab_size = int(vocab_size)
    else:
        tokenizer.vocab_size = max(tokenizer.idx_to_char.keys(), default=-1) + 1

    return tokenizer


def infer_model_cfg_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    embed_dim = int(state_dict["token_embed.weight"].shape[1])
    ff_dim = int(state_dict["blocks.0.ffn.net.0.weight"].shape[0])
    max_len = int(state_dict["pos_embed.weight"].shape[0])

    layer_indexes = {
        int(parts[1])
        for key in state_dict
        for parts in [key.split(".")]
        if key.startswith("blocks.") and len(parts) > 1 and parts[1].isdigit()
    }
    num_layers = max(layer_indexes) + 1 if layer_indexes else 1

    if embed_dim % 8 == 0 and embed_dim >= 256:
        num_heads = 8
    elif embed_dim % 4 == 0:
        num_heads = 4
    elif embed_dim % 2 == 0:
        num_heads = 2
    else:
        num_heads = 1

    return {
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_layers": int(num_layers),
        "ff_dim": ff_dim,
        "max_len": max_len,
        "dropout": 0.0,
    }


def build_model_cfg(checkpoint: dict[str, Any]) -> dict[str, Any]:
    saved_cfg = checkpoint.get("model_cfg")
    if isinstance(saved_cfg, dict):
        cfg = {
            "embed_dim": int(saved_cfg["embed_dim"]),
            "num_heads": int(saved_cfg["num_heads"]),
            "num_layers": int(saved_cfg["num_layers"]),
            "ff_dim": int(saved_cfg["ff_dim"]),
            "max_len": int(saved_cfg["max_len"]),
            "dropout": float(saved_cfg.get("dropout", 0.0)),
        }
        return cfg

    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint missing 'model_cfg' and valid 'model_state_dict'")

    print("[warn] 'model_cfg' missing in checkpoint; inferring config from state_dict")
    return infer_model_cfg_from_state_dict(state_dict)


def export_v3(checkpoint: dict[str, Any], checkpoint_path: Path) -> None:
    """Export a v3 checkpoint to ONNX."""
    # Extract state_dict
    if isinstance(checkpoint.get("model"), dict) and "step" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    size = _infer_v3_size(state_dict)
    cfg = V3_CONFIGS[size]

    print(f"[2/6] Loading BPE tokenizer from {DEFAULT_TOKENIZER_PATH}")
    if not DEFAULT_TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"BPE tokenizer not found: {DEFAULT_TOKENIZER_PATH}")
    tokenizer = BPETokenizer(vocab_size=cfg["vocab"])
    tokenizer.load(str(DEFAULT_TOKENIZER_PATH))

    print(f"[3/6] Rebuilding ArthurV3-{size} model")
    model = ArthurV3(size=size, dropout=0.0)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"[4/6] Ensuring output directory exists: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[5/6] Exporting ONNX model -> {ONNX_PATH}")
    if importlib.util.find_spec("onnx") is None:
        raise RuntimeError(
            "ONNX export requires the 'onnx' package. Install it with: pip install onnx"
        )

    dummy_input = torch.zeros(1, 128, dtype=torch.long)
    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_PATH),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=17,
    )

    print(f"[6/6] Exporting BPE tokenizer vocab JSON -> {VOCAB_JSON_PATH}")
    vocab_payload = {
        "type": "bpe",
        "vocab_size": tokenizer.vocab_size,
        "vocab": {str(k): v.decode("utf-8", errors="replace") for k, v in tokenizer.vocab.items()},
        "merges": [{"pair": [int(a), int(b)], "token": int(c)} for (a, b), c in tokenizer.merges],
        "char_tokens": {k: int(v) for k, v in tokenizer.char_tokens.items()},
        "model_size": size,
        "model_cfg": dict(cfg),
    }
    VOCAB_JSON_PATH.write_text(json.dumps(vocab_payload, ensure_ascii=False, indent=2))

    print("\nExport complete:")
    print(f"- ONNX: {ONNX_PATH} | {fmt_size(ONNX_PATH.stat().st_size)}")
    print(f"- Vocab JSON: {VOCAB_JSON_PATH} | {fmt_size(VOCAB_JSON_PATH.stat().st_size)}")


def export_v2(checkpoint: dict[str, Any], checkpoint_path: Path) -> None:
    """Export a v2/v1 checkpoint to ONNX."""
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint missing required key: 'model_state_dict'")
    if "vocab" not in checkpoint:
        raise ValueError("Checkpoint missing required key: 'vocab'")

    print("[2/6] Rebuilding CharTokenizer from checkpoint vocab")
    tokenizer = rebuild_char_tokenizer(checkpoint["vocab"], checkpoint.get("vocab_size"))

    print("[3/6] Rebuilding Arthur model from checkpoint model_cfg")
    model_cfg = build_model_cfg(checkpoint)
    vocab_size = int(checkpoint.get("vocab_size", tokenizer.vocab_size))

    model = Arthur(
        vocab_size=vocab_size,
        embed_dim=int(model_cfg["embed_dim"]),
        num_heads=int(model_cfg["num_heads"]),
        num_layers=int(model_cfg["num_layers"]),
        ff_dim=int(model_cfg["ff_dim"]),
        max_len=int(model_cfg["max_len"]),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"[4/6] Ensuring output directory exists: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[5/6] Exporting ONNX model -> {ONNX_PATH}")
    if importlib.util.find_spec("onnx") is None:
        raise RuntimeError(
            "ONNX export requires the 'onnx' package. Install it with: pip install onnx"
        )

    dummy_input = torch.zeros(1, 128, dtype=torch.long)
    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_PATH),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=14,
    )

    print(f"[6/6] Exporting tokenizer vocab JSON -> {VOCAB_JSON_PATH}")
    vocab_payload = {
        "type": "char",
        "char_to_idx": tokenizer.char_to_idx,
        "idx_to_char": {str(k): v for k, v in tokenizer.idx_to_char.items()},
        "vocab_size": vocab_size,
        "model_cfg": model_cfg,
    }
    VOCAB_JSON_PATH.write_text(json.dumps(vocab_payload, ensure_ascii=False, indent=2))

    print("\nExport complete:")
    print(f"- ONNX: {ONNX_PATH} | {fmt_size(ONNX_PATH.stat().st_size)}")
    print(f"- Vocab JSON: {VOCAB_JSON_PATH} | {fmt_size(VOCAB_JSON_PATH.stat().st_size)}")


def export() -> None:
    checkpoint_path = find_checkpoint()
    print(f"[1/6] Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint format invalid: expected a dict")

    if _is_v3_checkpoint(checkpoint):
        print("[info] Detected v3 checkpoint")
        export_v3(checkpoint, checkpoint_path)
    else:
        print("[info] Detected v2/v1 checkpoint")
        export_v2(checkpoint, checkpoint_path)


if __name__ == "__main__":
    try:
        export()
    except Exception as exc:
        print(f"Export failed: {exc}")
        raise SystemExit(1)
