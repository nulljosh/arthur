#!/usr/bin/env python3
import torch
import torch.nn as nn
from src.model import GPTModel
from src.tokenizer import CharTokenizer
from src.train import train_model

# Load sci-fi data
with open('data/sci_fi.txt', 'r') as f:
    text = f.read()

# Tiny model for fast results
config = {
    'vocab_size': None,  # Will be set by tokenizer
    'embed_dim': 64,
    'n_heads': 4,
    'n_layers': 4,
    'max_len': 128,
    'dropout': 0.1
}

tokenizer = CharTokenizer()
tokenizer.fit(text)
config['vocab_size'] = tokenizer.vocab_size

print(f"Text length: {len(text)} chars")
print(f"Vocab size: {tokenizer.vocab_size}")

# Train
model = GPTModel(config)
train_model(
    model,
    text,
    tokenizer,
    epochs=500,
    batch_size=16,
    seq_len=32,
    lr=0.001,
    save_path='models/scifi.pt'
)

# Generate sample
model.eval()
prompt = "The AI"
tokens = tokenizer.encode(prompt)
input_ids = torch.tensor([tokens])

print(f"\n=== Generation Test ===")
print(f"Prompt: '{prompt}'")

with torch.no_grad():
    for _ in range(100):
        logits = model(input_ids)
        next_token = logits[0, -1].argmax().item()
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
        if input_ids.shape[1] > 128:
            break

output = tokenizer.decode(input_ids[0].tolist())
print(f"Output: {output}")
