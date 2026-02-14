#!/usr/bin/env python3
"""Train nuLLM on minimal Q&A (name + basic math)"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from transformer import NuLLM
from tokenizer import CharTokenizer

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.tokens = tokenizer.encode(text)

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

# Load minimal corpus
with open('data/minimal.txt') as f:
    text = f.read()

epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 500

print(f"Training nuLLM NANO on minimal Q&A (name + math)...")
print(f"Text length: {len(text)} chars")
print(f"Epochs: {epochs}\n")

# Setup
tokenizer = CharTokenizer(text)
print(f"Vocab size: {tokenizer.vocab_size}")

dataset = TextDataset(text, tokenizer, seq_len=32)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

model = NuLLM(
    vocab_size=tokenizer.vocab_size,
    embed_dim=128,
    num_heads=4,
    num_layers=4,
    ff_dim=256,
    max_len=128,
    dropout=0.1
)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {num_params:,}\n")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Train
model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch_idx, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx} | Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

# Save
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab_size': tokenizer.vocab_size,
    'vocab': tokenizer.char_to_idx
}, 'models/minimal.pt')
print("\n✓ Model saved to models/minimal.pt")

# Test
print("\n" + "="*60)
print("Testing...")
print("="*60)

model.eval()
prompts = ["Q: What is your name?\nA:", "Q: What is 5+5?\nA:", "Q: What is 2+2?\nA:"]

for prompt in prompts:
    tokens = tokenizer.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    generated = tokens.copy()
    
    with torch.no_grad():
        for _ in range(30):  # Shorter generation
            logits = model(x)
            logits = logits[:, -1, :] / 0.5  # Lower temp for more deterministic
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            
            # Stop at newline
            decoded = tokenizer.decode(generated)
            if '\nQ:' in decoded[len(prompt):]:
                break
            
            x = torch.cat([x, torch.tensor([[next_token]])], dim=1)
            if x.size(1) > 128:
                x = x[:, -128:]
    
    result = tokenizer.decode(generated)
    print(f"\n{result}")
