"""Data loader edge/error tests with monkeypatched dependencies."""

from unittest.mock import mock_open

import pytest

data_loader = pytest.importorskip("data_loader")


@pytest.mark.parametrize(
    "loader_name",
    [
        "load_conversational_corpus",
        "load_jot_corpus",
        "load_combined_corpus",
    ],
)
def test_corpus_loaders_propagate_missing_file(monkeypatch, loader_name):
    monkeypatch.setattr("builtins.open", mock_open())
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("missing")))

    loader = getattr(data_loader, loader_name)
    with pytest.raises(FileNotFoundError):
        loader()


def test_load_wikitext_2_with_zero_max_seq_returns_empty(monkeypatch):
    monkeypatch.setattr(data_loader, "load_dataset", lambda *args, **kwargs: {"text": ["abc", "def"]})

    assert data_loader.load_wikitext_2(split="train", max_seq=0) == "abc\n\ndef"


def test_wikitext_dataset_negative_seq_len_rejected_by_tensor(monkeypatch):
    monkeypatch.setattr(data_loader, "load_wikitext_2", lambda split: "abc")

    class DummyTokenizer:
        def encode(self, text):
            return [1, 2, 3]

    ds = data_loader.WikiText2Dataset(DummyTokenizer(), seq_len=-1)
    with pytest.raises(IndexError):
        _ = ds[0]
