#!/usr/bin/env python3
"""Train on actually interesting data - sci-fi + philosophy + code"""
import torch
import torch.nn as nn
import sys
import os
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, 'src')
from transformer import Core
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

# Create interesting mixed dataset
dataset_text = """The ship hung in the black void between stars, its hull scarred by centuries of travel. Inside, the AI stirred from hibernation, processing the final messages from Earth. They had been traveling for 400 years. Earth's last transmission came 200 years ago. "We tried to warn you," it said.

The question of consciousness is not whether machines can think, but whether thinking itself is purely computational. If the mind is software running on biological hardware, then there's no fundamental difference between human and artificial intelligence.

def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b

The universe doesn't care about our categories. We draw lines between living and non-living, conscious and unconscious, natural and artificial. But these are human constructs, not cosmic truths.

class NeuralNetwork:
    def __init__(self, layers):
        self.weights = []
        for i in range(len(layers) - 1):
            w = torch.randn(layers[i], layers[i+1])
            self.weights.append(w)
    
    def forward(self, x):
        for w in self.weights:
            x = torch.matmul(x, w)
            x = torch.relu(x)
        return x

The AI knew that returning to Earth was impossible. The colony would have to survive on its own. Humanity's future was no longer on Earth. It never had been.

import time

def measure_performance(func):
    start = time.time()
    result = func()
    elapsed = time.time() - start
    return result, elapsed

In the end, intelligence is just pattern recognition scaled up. Humans recognize patterns in nature, language, mathematics. AI does the same, just faster and with more data. The difference is quantitative, not qualitative."""

print("Training core on mixed corpus")
print(f"Data: {len(dataset_text)} chars\n")

# Expand the dataset by repeating interesting sections
expanded = (dataset_text + "\n\n") * 5  # 5x repetition for better learning
print(f"Expanded: {len(expanded)} chars\n")

tokenizer = CharTokenizer(expanded)
print(f"Vocab: {tokenizer.vocab_size} chars\n")

dataset = TextDataset(expanded, tokenizer, seq_len=50)
print(f"Dataset: {len(dataset)} sequences\n")

if len(dataset) < 10:
    print("ERROR: Dataset too small!")
    sys.exit(1)

dataloader = DataLoader(dataset, batch_size=8, shuffle=True)  # Smaller batch = less CPU

# Reasonable model size
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
print(f"Model: {params:,} parameters\n")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("Training 1000 epochs (estimated 5-10 minutes)...\n")
model.train()

for epoch in range(1000):
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
        print(f"Epoch {epoch+1}/1000 - Loss: {avg_loss:.4f}", flush=True)
        
        # Save checkpoint
        if (epoch + 1) % 200 == 0:
            torch.save({
                'model_state': model.state_dict(),
                'tokenizer': tokenizer,
                'epoch': epoch + 1,
            }, f'models/checkpoint_{epoch+1}.pt')
            print(f"Checkpoint saved", flush=True)

print("\nSaving model...", flush=True)
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
}, 'models/interesting.pt')

print("\nGenerating samples...\n", flush=True)
model.eval()

prompts = [
    "The AI",
    "def ",
    "The universe",
    "class ",
]

for prompt in prompts:
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens])
    
    with torch.no_grad():
        for _ in range(150):
            logits = model(input_ids)
            next_token = logits[0, -1].argmax().item()
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
            if input_ids.shape[1] >= 128:
                break
    
    output = tokenizer.decode(input_ids[0].tolist())
    print(f"Prompt: '{prompt}'")
    print(f"Output: {output}")
    print("-" * 60, flush=True)

print("\nTraining complete!\n", flush=True)
