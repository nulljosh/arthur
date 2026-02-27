"""Model and inference edge-case coverage."""

import pytest
torch = pytest.importorskip("torch")

from attention import MultiHeadAttention
from tokenizer import CharTokenizer
from train import generate
from transformer import Jore


class DeterministicModel(torch.nn.Module):
    """Tiny model that always predicts token id 1."""

    def __init__(self, vocab_size=4, max_len=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len

    def forward(self, x):
        batch, seq = x.shape
        logits = torch.full((batch, seq, self.vocab_size), -1e9, device=x.device)
        logits[:, :, 1] = 0.0
        return logits


def test_multihead_attention_rejects_invalid_head_partition():
    with pytest.raises(AssertionError, match="divisible"):
        MultiHeadAttention(embed_dim=10, num_heads=3)


def test_core_forward_rejects_sequence_longer_than_max_len():
    model = Core(
        vocab_size=8,
        embed_dim=8,
        num_heads=2,
        num_layers=1,
        ff_dim=16,
        max_len=4,
        dropout=0.0,
    )
    x = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    with pytest.raises(IndexError):
        model(x)


def test_core_forward_rejects_out_of_vocab_token_id():
    model = Core(
        vocab_size=4,
        embed_dim=8,
        num_heads=2,
        num_layers=1,
        ff_dim=16,
        max_len=8,
        dropout=0.0,
    )
    x = torch.tensor([[0, 1, 2, 4]], dtype=torch.long)

    with pytest.raises(IndexError):
        model(x)


def test_generate_empty_prompt_raises_due_to_no_initial_token():
    tok = CharTokenizer("abc")
    model = DeterministicModel(vocab_size=tok.vocab_size, max_len=8)

    with pytest.raises(IndexError):
        generate(model, tok, "", max_len=1, temperature=1.0)


def test_generate_zero_or_negative_max_len_returns_prompt_unchanged():
    tok = CharTokenizer("abc")
    model = DeterministicModel(vocab_size=tok.vocab_size, max_len=8)

    assert generate(model, tok, "ab", max_len=0, temperature=1.0) == "ab"
    assert generate(model, tok, "ab", max_len=-5, temperature=1.0) == "ab"


def test_generate_temperature_zero_raises_runtime_error():
    tok = CharTokenizer("abc")
    model = DeterministicModel(vocab_size=tok.vocab_size, max_len=8)

    with pytest.raises(RuntimeError):
        generate(model, tok, "a", max_len=1, temperature=0.0)
