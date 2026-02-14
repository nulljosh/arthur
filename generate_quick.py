#!/usr/bin/env python3
"""Quick generation test"""
import torch
import sys
sys.path.insert(0, 'src')
from transformer import NuLLM
from tokenizer import CharTokenizer

prompt = sys.argv[1] if len(sys.argv) > 1 else "Q: What's your name?\nA:"

checkpoint = torch.load('models/conversational.pt', map_location='cpu')
tokenizer = CharTokenizer()
tokenizer.char_to_idx = checkpoint['vocab']
tokenizer.idx_to_char = {v: k for k, v in tokenizer.char_to_idx.items()}

model = NuLLM(vocab_size=checkpoint['vocab_size'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

tokens = tokenizer.encode(prompt)
x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
generated = tokens.copy()

with torch.no_grad():
    for _ in range(100):
        logits = model(x)
        logits = logits[:, -1, :] / 0.8
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_token)
        x = torch.cat([x, torch.tensor([[next_token]])], dim=1)

print(tokenizer.decode(generated))
