#!/usr/bin/env python3
"""Quick generation test"""
import torch
import sys
sys.path.insert(0, 'src')
from transformer import Jore
from tokenizer import CharTokenizer

prompt = sys.argv[1] if len(sys.argv) > 1 else "Q: What's your name?\nA:"

checkpoint = torch.load('models/conversational.pt', map_location='cpu')

# Initialize tokenizer with vocab from checkpoint
class SimpleTokenizer:
    def __init__(self, vocab):
        self.char_to_idx = vocab
        self.idx_to_char = {v: k for k, v in vocab.items()}
        self.vocab_size = len(vocab)
    
    def encode(self, text):
        return [self.char_to_idx.get(c, 0) for c in text]
    
    def decode(self, tokens):
        return ''.join([self.idx_to_char.get(t, '?') for t in tokens])

tokenizer = SimpleTokenizer(checkpoint['vocab'])

model = Core(
    vocab_size=checkpoint['vocab_size'],
    embed_dim=128,
    num_heads=4,
    num_layers=4,
    ff_dim=256,
    max_len=128,
    dropout=0.1
)
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
