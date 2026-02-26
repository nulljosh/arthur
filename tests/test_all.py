#!/usr/bin/env python3
"""Basic pytest suite for core."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tokenizer import CharTokenizer, WordTokenizer


def test_tokenizers():
    text = "hello world hello test"
    char_tok = CharTokenizer(text)
    word_tok = WordTokenizer(text)

    # CharTokenizer reserves index 0 for UNK.
    assert char_tok.vocab_size == 11
    assert word_tok.vocab_size == 3

    encoded = char_tok.encode("hello")
    decoded = char_tok.decode(encoded)
    assert decoded == "hello"


def test_project_structure():
    project_root = Path(__file__).parent.parent
    expected_files = [
        "src/tokenizer.py",
        "src/attention.py",
        "src/transformer.py",
        "src/train.py",
        "README.md",
    ]

    missing = [f for f in expected_files if not (project_root / f).exists()]
    assert not missing, f"Missing expected files: {missing}"
