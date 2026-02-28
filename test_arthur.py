#!/usr/bin/env python3
import sys
sys.path.insert(0, "src")
import torch
from transformer_v2 import ArthurV2
from bpe_tokenizer import BPETokenizer

# Load model with correct config
model = ArthurV2(vocab_size=10000)
model.load_state_dict(torch.load("models/arthur_v2_epoch2.pt", map_location="cpu", weights_only=True))
model.eval()

# Load tokenizer
tokenizer = BPETokenizer()
tokenizer.load("models/bpe_tokenizer_v1.json")

# Test
prompt = "What is 2 + 2?"
tokens = tokenizer.encode(prompt)
print(f"Prompt: {prompt}")
print(f"Tokens: {tokens[:10]}...")

# Simple forward pass
input_ids = torch.tensor([tokens])
with torch.no_grad():
    output = model(input_ids)
    print(f"Output shape: {output.shape}")
    print("Model loaded successfully!")
