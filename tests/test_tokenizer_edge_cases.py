"""Tokenizer edge-case and error-behavior tests."""

from tokenizer import CharTokenizer, WordTokenizer


def test_char_tokenizer_empty_corpus_has_only_unk():
    tok = CharTokenizer("")

    assert tok.vocab_size == 1
    assert tok.encode("") == []
    assert tok.decode([]) == ""


def test_char_tokenizer_unknown_char_maps_to_unk_index():
    tok = CharTokenizer("abc")

    encoded = tok.encode("az")

    assert encoded[0] == tok.char_to_idx["a"]
    assert encoded[1] == tok.char_to_idx[tok.UNK]
    assert tok.decode([999]) == tok.UNK


def test_word_tokenizer_empty_input_roundtrip():
    tok = WordTokenizer("")

    assert tok.vocab_size == 0
    assert tok.encode("") == []
    assert tok.decode([]) == ""


def test_word_tokenizer_unknown_word_behavior_is_explicit():
    tok = WordTokenizer("hello world")

    # Unknown words currently map to index 0.
    assert tok.encode("does-not-exist") == [0]
    # Invalid ids decode to explicit <UNK> sentinel.
    assert tok.decode([12345]) == "<UNK>"
