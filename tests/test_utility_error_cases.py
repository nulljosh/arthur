"""Utility-level error handling tests for missing files and invalid params."""

from pathlib import Path

import pytest
pytest.importorskip("torch")

from bpe_tokenizer import BPETokenizer
import chat


def test_load_model_missing_tokenizer_file_raises(tmp_path: Path):
    missing = tmp_path / "does_not_exist.json"

    with pytest.raises(FileNotFoundError):
        chat.load_model(tmp_path / "model.pth", missing)


def test_load_model_missing_model_file_raises(tmp_path: Path):
    tokenizer_path = tmp_path / "tok.json"
    tokenizer = BPETokenizer()
    tokenizer.save(str(tokenizer_path))

    with pytest.raises(FileNotFoundError):
        chat.load_model(tmp_path / "missing_model.pth", tokenizer_path)
