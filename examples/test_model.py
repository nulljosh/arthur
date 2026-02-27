#!/usr/bin/env python3
"""
Test model forward pass
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from transformer import Nous

# Tiny model config
vocab_size = 100
embed_dim = 64
num_heads = 4
num_layers = 2
ff_dim = 256
max_len = 128

print("Creating nous model...")
model = Core(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    ff_dim=ff_dim,
    max_len=max_len
)

# Count parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Model size: {num_params:,} parameters ({num_params/1e6:.2f}M)")

# Test forward pass
batch_size = 2
seq_len = 10
x = torch.randint(0, vocab_size, (batch_size, seq_len))

print(f"\nInput shape: {x.shape}")

with torch.no_grad():
    logits = model(x)

print(f"Output shape: {logits.shape}")
print(f"Expected: (batch={batch_size}, seq_len={seq_len}, vocab={vocab_size})")

print("\n✓ Forward pass works!")
