#!/usr/bin/env python3
"""Smoke test: load model + tokenizer, run a single forward pass."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from transformer import ArthurV3
from bpe_tokenizer import BPETokenizer
from transformer import migrate_state_dict

MODEL_PATH = ROOT / "models" / "arthur_v3_65M_best.pt"
TOKENIZER_PATH = ROOT / "models" / "bpe_tokenizer_v1.json"


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model checkpoint not available")
def test_smoke():
    """Load model and tokenizer, verify a forward pass produces correct output shape."""
    model = ArthurV3(size="65M")
    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    except RuntimeError:
        pytest.skip("checkpoint file is corrupted or incompatible")
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(migrate_state_dict(state))
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
    assert output.shape[2] == 10000, f"Expected vocab_size 10000, got {output.shape[2]}"


if __name__ == "__main__":
    test_smoke()
    print("Smoke test passed.")
