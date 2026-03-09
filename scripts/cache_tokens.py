"""
Pre-tokenize WikiText-2 training split and cache flattened token tensor.

Usage:
    python3 scripts/cache_tokens.py
"""

import glob
import os
import sys

import torch
from datasets import Dataset, load_dataset

sys.path.insert(0, ".")
from src.bpe_tokenizer import BPETokenizer


def load_wikitext2_train():
    cache_dir = os.path.join(".cache", "huggingface")
    os.makedirs(cache_dir, exist_ok=True)
    try:
        return load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="train",
            streaming=False,
            cache_dir=cache_dir,
        )
    except Exception:
        arrow_glob = os.path.expanduser(
            "~/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/*/wikitext-train.arrow"
        )
        matches = glob.glob(arrow_glob)
        if not matches:
            raise
        return Dataset.from_file(matches[0])


def main():
    os.makedirs(".cache", exist_ok=True)
    cache_path = os.path.join(".cache", "wikitext2_tokens.pt")
    tok_path = os.path.join("models", "bpe_tokenizer_v1.json")

    print("Loading WikiText-2 from local cache...")
    dataset = load_wikitext2_train()

    print(f"Loading tokenizer: {tok_path}")
    tokenizer = BPETokenizer()
    tokenizer.load(tok_path)

    print("Tokenizing full dataset...")
    flat_tokens = []
    for i, item in enumerate(dataset, start=1):
        text = item.get("text", "") or item.get("content", "") or ""
        if len(text) > 50:
            ids = tokenizer.encode(text)
            if ids:
                flat_tokens.extend(ids)
        if i % 1000 == 0:
            print(f"Processed {i} items | tokens so far: {len(flat_tokens):,}")

    if not flat_tokens:
        raise RuntimeError("No tokens produced from WikiText-2 dataset.")

    tokens = torch.tensor(flat_tokens, dtype=torch.long)
    torch.save(tokens, cache_path)
    print(f"Saved token cache: {cache_path}")
    print(f"Total tokens: {tokens.numel():,}")


if __name__ == "__main__":
    main()
