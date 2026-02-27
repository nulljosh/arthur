#!/usr/bin/env python3
"""Debug version - print IMMEDIATELY, save frequently"""
import torch
import torch.nn as nn
import sys, os

sys.path.insert(0, 'src')
from transformer import Jore
from tokenizer import CharTokenizer
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len):
        self.tokens = tokenizer.encode(text)
        self.seq_len = seq_len
    
    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)
    
    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.seq_len + 1]
        return (
            torch.tensor(chunk[:-1], dtype=torch.long),
            torch.tensor(chunk[1:], dtype=torch.long)
        )

# Simple dataset
dataset_text = """The AI learned quickly. 2+2 equals 4. 5+5 equals 10. 
The model trains on data. Machine learning works.
def add(a, b):
    return a + b

The result is correct."""

# Repeat for better learning
expanded = (dataset_text + "\n\n") * 10

print("Training Debug Version", flush=True)
print(f"Data: {len(expanded)} chars", flush=True)

tokenizer = CharTokenizer(expanded)
print(f"Vocab: {tokenizer.vocab_size}", flush=True)

dataset = TextDataset(expanded, tokenizer, seq_len=30)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Small model
model = Core(
    vocab_size=tokenizer.vocab_size,
    embed_dim=64,
    num_heads=2,
    num_layers=2,
    ff_dim=128,
    max_len=64,
    dropout=0.1
)

print(f"Model params: {sum(p.numel() for p in model.parameters())}", flush=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

print("\nStarting training...\n", flush=True)
model.train()

EPOCHS = 100
for epoch in range(EPOCHS):
    total_loss = 0
    batch_count = 0
    
    for x, y in dataloader:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        batch_count += 1
    
    avg_loss = total_loss / batch_count
    
    # Print EVERY 10 epochs with immediate flush
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}", flush=True)
        sys.stdout.flush()  # Force flush
    
    # Save checkpoint every 25 epochs
    if (epoch + 1) % 25 == 0:
        torch.save({
            'model_state': model.state_dict(),
            'tokenizer': tokenizer,
            'epoch': epoch + 1,
        }, f'models/debug_checkpoint_{epoch+1}.pt')
        print(f"  Saved checkpoint_{epoch+1}.pt", flush=True)

print("\nSaving final model...", flush=True)
torch.save({
    'model_state': model.state_dict(),
    'tokenizer': tokenizer,
    'epoch': EPOCHS,
}, 'models/debug_final.pt')

print("✓ Training complete!", flush=True)

# Test it
print("\nTesting...", flush=True)
model.eval()

prompt = "2+2 equals"
tokens = tokenizer.encode(prompt)
input_ids = torch.tensor([tokens])

with torch.no_grad():
    for _ in range(10):
        logits = model(input_ids)
        next_token = logits[0, -1].argmax().item()
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)

output = tokenizer.decode(input_ids[0].tolist())
print(f"Test: '{prompt}'")
print(f"Output: {output}", flush=True)
