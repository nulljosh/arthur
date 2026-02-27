#!/usr/bin/env python3
"""Train on sci-fi text - proper version"""
import torch
import torch.nn as nn
import sys
import os
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from transformer import Nous
from tokenizer import CharTokenizer

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

# Load sci-fi data
with open('data/sci_fi.txt') as f:
    text = f.read()

print(f"🚀 Sci-Fi Training")
print(f"📊 Data: {len(text)} chars\n")

tokenizer = CharTokenizer(text)
print(f"✓ Vocab: {tokenizer.vocab_size} chars")

dataset = TextDataset(text, tokenizer, seq_len=40)
if len(dataset) == 0:
    print("ERROR: Dataset too small!")
    sys.exit(1)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Medium model for narrative text
model = Core(
    vocab_size=tokenizer.vocab_size,
    embed_dim=128,
    num_heads=4,
    num_layers=4,
    ff_dim=256,
    max_len=128,
    dropout=0.1
)

params = sum(p.numel() for p in model.parameters())
print(f"✓ Model: {params:,} params\n")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("🚀 Training 500 epochs...")
model.train()

for epoch in range(500):
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
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/500 - Loss: {avg_loss:.4f}")

print("\n✓ Saving model...")
torch.save({
    'model_state': model.state_dict(),
    'tokenizer': tokenizer,
    'config': {
        'vocab_size': tokenizer.vocab_size,
        'embed_dim': 128,
        'num_heads': 4,
        'num_layers': 4,
        'ff_dim': 256,
        'max_len': 128,
        'dropout': 0.1
    }
}, 'models/scifi.pt')

print("\n🎯 Generating sample...")
model.eval()
prompt = "The AI"
tokens = tokenizer.encode(prompt)
input_ids = torch.tensor([tokens])

with torch.no_grad():
    for _ in range(200):
        logits = model(input_ids)
        next_token = logits[0, -1].argmax().item()
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
        if input_ids.shape[1] > 128:
            break

output = tokenizer.decode(input_ids[0].tolist())
print(f"\nPrompt: '{prompt}'")
print(f"Output: {output}\n")
print("✓ Done!")
