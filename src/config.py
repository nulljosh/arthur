"""Shared model configuration and utilities for Arthur v2."""

ARTHUR_V2_CONFIG = dict(
    vocab_size=10000,
    embed_dim=512,
    num_heads=8,
    num_layers=12,
    ff_dim=2048,
    max_context=8192,
)


def get_last_user_message(messages):
    """Extract the last user message from a chat messages list."""
    for msg in reversed(messages):
        if msg["role"] == "user":
            return msg["content"]
    return ""
