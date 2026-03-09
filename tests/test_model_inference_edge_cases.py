"""Model and inference edge-case coverage."""

import pytest
torch = pytest.importorskip("torch")

from attention import MultiHeadAttention
from tokenizer import CharTokenizer
from transformer import ArthurV3


def test_multihead_attention_rejects_invalid_head_partition():
    with pytest.raises(AssertionError, match="divisible"):
        MultiHeadAttention(embed_dim=10, num_heads=3)


def test_core_forward_rejects_out_of_vocab_token_id():
    model = ArthurV3(size="65M")
    x = torch.tensor([[0, 1, 2, 10000]], dtype=torch.long)

    with pytest.raises(IndexError):
        model(x)
