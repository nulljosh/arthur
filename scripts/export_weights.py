#!/usr/bin/env python3
"""Export PyTorch checkpoint to flat binary format for C inference engine.

Binary layout (all little-endian):
  - Magic: "CORE" (4 bytes)
  - Version: uint32 = 1
  - Config: 6x uint32 (vocab_size, embed_dim, num_heads, num_layers, ff_dim, max_len)
  - Vocab: vocab_size entries, each: uint32 length + UTF-8 bytes (sorted by token index)
  - Weights: contiguous float32 tensors in fixed order (see WEIGHT_ORDER)
"""

import struct
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
CHECKPOINT = REPO / "models" / "overnight_best.pt"
OUTPUT = REPO / "models" / "core.bin"

WEIGHT_ORDER_BLOCK = [
    "attention.qkv.weight",
    "attention.qkv.bias",
    "attention.out.weight",
    "attention.out.bias",
    "ffn.net.0.weight",
    "ffn.net.0.bias",
    "ffn.net.2.weight",
    "ffn.net.2.bias",
    "ln1.weight",
    "ln1.bias",
    "ln2.weight",
    "ln2.bias",
]


def main():
    print(f"Loading checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)

    sd = ckpt["model_state_dict"]
    vocab = ckpt["vocab"]  # dict: char -> index

    # Derive config from actual tensor shapes
    vocab_size = sd["token_embed.weight"].shape[0]
    embed_dim = sd["token_embed.weight"].shape[1]
    max_len = sd["pos_embed.weight"].shape[0]
    if "config" in ckpt and "num_heads" in ckpt["config"]:
        num_heads = ckpt["config"]["num_heads"]
    else:
        num_heads = 4  # legacy checkpoints
    ff_dim = sd["blocks.0.ffn.net.0.weight"].shape[0]

    # Count transformer blocks
    num_layers = 0
    while f"blocks.{num_layers}.ln1.weight" in sd:
        num_layers += 1

    print(f"Config: vocab={vocab_size} embed={embed_dim} heads={num_heads} "
          f"layers={num_layers} ff={ff_dim} maxlen={max_len}")

    # Build index-sorted vocab list
    idx_to_char = [""] * vocab_size
    for char, idx in vocab.items():
        idx_to_char[idx] = char

    with open(OUTPUT, "wb") as f:
        # Magic
        f.write(b"CORE")
        # Version
        f.write(struct.pack("<I", 1))
        # Config
        for val in (vocab_size, embed_dim, num_heads, num_layers, ff_dim, max_len):
            f.write(struct.pack("<I", val))

        # Vocab
        for i in range(vocab_size):
            encoded = idx_to_char[i].encode("utf-8")
            f.write(struct.pack("<I", len(encoded)))
            f.write(encoded)

        # Helper to write a tensor as contiguous float32
        def write_tensor(name):
            t = sd[name].contiguous().float()
            f.write(t.numpy().tobytes())
            return t.numel()

        total_floats = 0

        # 1. token_embed
        total_floats += write_tensor("token_embed.weight")
        # 2. pos_embed
        total_floats += write_tensor("pos_embed.weight")

        # 3. Per-block weights
        for b in range(num_layers):
            for suffix in WEIGHT_ORDER_BLOCK:
                total_floats += write_tensor(f"blocks.{b}.{suffix}")

        # 4. Final layernorm
        total_floats += write_tensor("ln_f.weight")
        total_floats += write_tensor("ln_f.bias")

        # 5. Output head
        total_floats += write_tensor("head.weight")

    file_size = OUTPUT.stat().st_size
    print(f"Wrote {OUTPUT} ({file_size:,} bytes, {total_floats:,} floats)")


if __name__ == "__main__":
    main()
