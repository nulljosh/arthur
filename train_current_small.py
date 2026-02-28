#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
from torch.optim import Adam
from transformer import Nous
from tokenizer import CharTokenizer
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cpu')

with open(os.path.join(REPO_ROOT, 'data/current_math.txt'), 'r') as f:
    current_math = f.read()

math_boost = current_math * 50
combined = math_boost

tokenizer = CharTokenizer(combined)
tokens = tokenizer.encode(combined)
print(f"Tokens: {len(tokens):,}")

model = Nous(
    vocab_size=tokenizer.vocab_size,
    embed_dim=128,
    num_heads=4,
    num_layers=3,
    ff_dim=256,
    max_len=256
)
model.to(DEVICE)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model: {num_params:,} params")

optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

model.train()
best_loss = float('inf')

EPOCHS = 100
seq_len = 128
batch_size = 1

print(f"Training {EPOCHS} epochs (small batch)...\n")

for epoch in range(EPOCHS):
    total_loss = 0
    batches = 0
    
    for i in range(0, len(tokens) - seq_len, batch_size):
        idx = i
        if idx + seq_len + 1 >= len(tokens):
            break
        
        x = torch.tensor(tokens[idx:idx + seq_len], dtype=torch.long, device=DEVICE).unsqueeze(0)
        y = torch.tensor(tokens[idx + 1:idx + seq_len + 1], dtype=torch.long, device=DEVICE).unsqueeze(0)
        
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        batches += 1
    
    avg_loss = total_loss / max(1, batches)
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model, os.path.join(REPO_ROOT, 'models/aether_final.pt'))
        if epoch % 20 == 0 or epoch < 3:
            print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.6f} ✓")
    elif epoch % 20 == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.6f}")

print(f"\nDone! Best: {best_loss:.6f}")
