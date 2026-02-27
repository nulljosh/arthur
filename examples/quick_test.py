#!/usr/bin/env python3
"""
Quick end-to-end test: tokenize, train tiny model, generate
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from tokenizer import CharTokenizer
from transformer import Jore

# Tiny training corpus
text = "hello world this is a test hello test world"

print("🧪 Quick Test: End-to-End Pipeline")
print("="*60)

# 1. Tokenization
print("\n1. Tokenization")
tokenizer = CharTokenizer(text)
print(f"   Vocab size: {tokenizer.vocab_size}")
print(f"   Sample: 'hello' -> {tokenizer.encode('hello')}")

# 2. Model forward pass
print("\n2. Model Forward Pass")
model = Core(
    vocab_size=tokenizer.vocab_size,
    embed_dim=16,
    num_heads=2,
    num_layers=1,
    ff_dim=32,
    max_len=16,
    dropout=0.0
)
params = sum(p.numel() for p in model.parameters())
print(f"   Model: {params:,} params")

# 3. Forward pass test
print("\n3. Testing Forward Pass")
tokens = tokenizer.encode("hello")[:8]
x = torch.tensor(tokens).unsqueeze(0)
print(f"   Input shape: {x.shape}")

with torch.no_grad():
    logits = model(x)
print(f"   Output shape: {logits.shape}")
print(f"   ✓ Forward pass works!")

# 4. Training step
print("\n4. Single Training Step")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

# Simple target: predict next char (shifted by 1)
y = torch.tensor(tokenizer.encode("ello ")[:5]).unsqueeze(0)

logits = model(x)
loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"   Loss: {loss.item():.4f}")
print(f"   ✓ Backprop works!")

# 5. Generation test
print("\n5. Generation Test (untrained)")
model.eval()
prompt = "hello"
tokens = tokenizer.encode(prompt)
x = torch.tensor(tokens).unsqueeze(0)

generated = tokens.copy()
with torch.no_grad():
    for _ in range(10):
        logits = model(x)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        generated.append(next_token)
        x = torch.cat([x, torch.tensor([[next_token]])], dim=1)

output = tokenizer.decode(generated)
print(f"   Prompt: '{prompt}'")
print(f"   Generated: '{output}'")
print(f"   (Untrained, so random)")

print("\n" + "="*60)
print("✓ All systems working!")
print("="*60)
print("\nNext: Run src/train.py for real training")
