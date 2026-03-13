#!/usr/bin/env python3
"""Export Arthur checkpoint to ONNX + tokenizer vocab JSON for browser inference."""

from __future__ import annotations

import json
import sys
from pathlib import Path
import importlib.util

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))
from transformer import ArthurV3, CONFIGS as V3_CONFIGS, migrate_state_dict  # noqa: E402
from bpe_tokenizer import BPETokenizer  # noqa: E402

CHECKPOINT_CANDIDATES = [
    REPO_ROOT / "models" / "arthur_v3_65M_best.pt",
    REPO_ROOT / "models" / "arthur_v3_125M_best.pt",
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


def _infer_v3_size(state_dict: dict[str, torch.Tensor]) -> str:
    """Infer v3 model size preset from embed.weight dimensions."""
    d_model = int(state_dict["embed.weight"].shape[1])
    for size, cfg in V3_CONFIGS.items():
        if cfg["d_model"] == d_model:
            return size
    raise ValueError(f"Cannot infer v3 size for d_model={d_model}")

def export_v3(checkpoint: dict[str, object], checkpoint_path: Path) -> None:
    """Export an ArthurV3 checkpoint to ONNX."""
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
    model.load_state_dict(migrate_state_dict(state_dict))
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
        opset_version=18,
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


def export() -> None:
    checkpoint_path = find_checkpoint()
    print(f"[1/6] Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint format invalid: expected a dict")

    print("[info] Detected v3 checkpoint")
    export_v3(checkpoint, checkpoint_path)


if __name__ == "__main__":
    try:
        export()
    except Exception as exc:
        print(f"Export failed: {exc}")
        raise SystemExit(1)
