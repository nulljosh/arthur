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

# Load current+math dataset
current_math_path = os.path.join(REPO_ROOT, 'data/current_math.txt')
with open(current_math_path, 'r') as f:
    current_math = f.read()

# Boost math by repeating 50x
math_boost = current_math * 50

# Load base wiki data if available
wiki_path = os.path.join(REPO_ROOT, 'data/wikitext-2.txt')
wiki_data = ""
if os.path.exists(wiki_path):
    with open(wiki_path, 'r') as f:
        wiki_data = f.read()[:100000]

# Combine: 50% math, 50% wiki
combined = math_boost + wiki_data
print(f"Dataset size: {len(combined):,} chars")

tokenizer = CharTokenizer(combined)
tokens = tokenizer.encode(combined)
print(f"Total tokens: {len(tokens):,}")

# Mini model
embed_dim = 256
num_heads = 8
num_layers = 6
ff_dim = 1024
max_len = 512
seq_len = 256
batch_size = 2

model = Nous(
    vocab_size=tokenizer.vocab_size,
    embed_dim=embed_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    ff_dim=ff_dim,
    max_len=max_len
)
model.to(DEVICE)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model: {num_params:,} params ({num_params/1e6:.2f}M)")

optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

model.train()
best_loss = float('inf')

EPOCHS = 150
print(f"Training {EPOCHS} epochs...\n")

for epoch in range(EPOCHS):
    total_loss = 0
    batches = 0
    
    for i in range(0, len(tokens) - seq_len, batch_size):
        batch_x, batch_y = [], []
        
        for j in range(batch_size):
            idx = i + j
            if idx + seq_len + 1 >= len(tokens):
                break
            x = torch.tensor(tokens[idx:idx + seq_len], dtype=torch.long, device=DEVICE)
            y = torch.tensor(tokens[idx + 1:idx + seq_len + 1], dtype=torch.long, device=DEVICE)
            batch_x.append(x)
            batch_y.append(y)
        
        if not batch_x:
            break
        
        batch_x = torch.stack(batch_x)
        batch_y = torch.stack(batch_y)
        
        logits = model(batch_x)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), batch_y.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        batches += 1
    
    avg_loss = total_loss / max(1, batches)
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        checkpoint_path = os.path.join(REPO_ROOT, 'models/aether_current_best.pt')
        torch.save(model, checkpoint_path)
        if epoch % 10 == 0 or epoch < 5:
            print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.6f} ✓")
    elif epoch % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Loss: {avg_loss:.6f}")

final_path = os.path.join(REPO_ROOT, 'models/aether_current_final.pt')
torch.save(model, final_path)
print(f"\nDone! Best loss: {best_loss:.6f}")
