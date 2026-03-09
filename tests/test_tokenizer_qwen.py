"""BPE tokenizer unit tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tokenizer import CharTokenizer, WordTokenizer


def test_bpe_encode_decode_roundtrip():
    text = "hello world foo bar"
    tok = CharTokenizer(text)
    encoded = tok.encode("hello")
    decoded = tok.decode(encoded)
    assert decoded == "hello"


def test_bpe_vocab_size():
    text = "abcde"
    tok = CharTokenizer(text)
    assert tok.vocab_size >= 5


def test_bpe_unknown_token():
    tok = CharTokenizer("abc")
    encoded = tok.encode("az")
    assert len(encoded) == 2


def test_word_tokenizer_encode_decode():
    text = "the quick brown fox"
    tok = WordTokenizer(text)
    encoded = tok.encode("the quick")
    decoded = tok.decode(encoded)
    assert decoded == "the quick"


def test_word_tokenizer_vocab():
    text = "alpha beta gamma delta"
    tok = WordTokenizer(text)
    assert tok.vocab_size == 4
