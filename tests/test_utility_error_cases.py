"""Utility-level error handling tests for missing files and invalid params."""

import pickle
from pathlib import Path

import pytest
pytest.importorskip("torch")

from tokenizer import CharTokenizer
from train import TextDataset
import chat


def test_text_dataset_negative_seq_len_fails_with_index_error():
    tok = CharTokenizer("abc")
    ds = TextDataset("abc", tok, seq_len=-1)

    with pytest.raises(IndexError):
        _ = ds[0]


def test_text_dataset_zero_seq_len_returns_empty_tensors():
    tok = CharTokenizer("abc")
    ds = TextDataset("abc", tok, seq_len=0)

    x, y = ds[0]

    assert x.numel() == 0
    assert y.numel() == 0


def test_load_model_missing_tokenizer_file_raises(tmp_path: Path):
    missing = tmp_path / "does_not_exist.pkl"

    with pytest.raises(FileNotFoundError):
        chat.load_model(tmp_path / "model.pth", missing)


def test_load_model_missing_model_file_raises(tmp_path: Path):
    tokenizer_path = tmp_path / "tok.pkl"
    tokenizer = CharTokenizer("hello")

    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)

    with pytest.raises(FileNotFoundError):
        chat.load_model(tmp_path / "missing_model.pth", tokenizer_path)
