#!/usr/bin/env python3
"""Smoke test: load model + tokenizer, run a single forward pass."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import torch
from transformer_v2 import ArthurV2
from bpe_tokenizer import BPETokenizer
from config import ARTHUR_V2_CONFIG

MODEL_PATH = ROOT / "models" / "arthur_v2_epoch1.pt"
TOKENIZER_PATH = ROOT / "models" / "bpe_tokenizer_v1.json"


def test_smoke():
    """Load model and tokenizer, verify a forward pass produces correct output shape."""
    model = ArthurV2(**ARTHUR_V2_CONFIG)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    tokenizer = BPETokenizer()
    tokenizer.load(str(TOKENIZER_PATH))

    prompt = "What is 2 + 2?"
    tokens = tokenizer.encode(prompt)
    assert len(tokens) > 0, "Tokenizer produced empty output"

    input_ids = torch.tensor([tokens])
    with torch.no_grad():
        output = model(input_ids)

    assert output.shape[0] == 1, f"Expected batch size 1, got {output.shape[0]}"
    assert output.shape[1] == len(tokens), f"Expected seq_len {len(tokens)}, got {output.shape[1]}"
    assert output.shape[2] == ARTHUR_V2_CONFIG["vocab_size"], f"Expected vocab_size {ARTHUR_V2_CONFIG['vocab_size']}, got {output.shape[2]}"


if __name__ == "__main__":
    test_smoke()
    print("Smoke test passed.")
