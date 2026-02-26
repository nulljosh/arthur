#!/usr/bin/env python3
import torch, sys
sys.path.insert(0, 'src')
from transformer import Core

class SimpleTokenizer:
    def __init__(self, vocab):
        self.char_to_idx = vocab
        self.idx_to_char = {v: k for k, v in vocab.items()}
    def encode(self, text):
        return [self.char_to_idx.get(c, 0) for c in text]
    def decode(self, tokens):
        return ''.join([self.idx_to_char.get(t, '?') for t in tokens])

checkpoint = torch.load('models/conversational.pt', map_location='cpu')
tokenizer = SimpleTokenizer(checkpoint['vocab'])

model = Core(vocab_size=checkpoint['vocab_size'], embed_dim=128, num_heads=4, num_layers=4, ff_dim=256, max_len=128, dropout=0.1)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

prompt = "Q: What is your name?\nA: "
tokens = tokenizer.encode(prompt)
x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
generated = tokens.copy()

with torch.no_grad():
    for _ in range(30):
        if x.size(1) >= 126: x = x[:, -126:]
        logits = model(x)[:, -1, :] / 0.1
        next_token = torch.argmax(logits, dim=-1).item()
        generated.append(next_token)
        if tokenizer.decode([next_token]) == '\n': break
        x = torch.cat([x, torch.tensor([[next_token]])], dim=1)

print(f"Conversational model (greedy, temp=0.1):")
print(tokenizer.decode(generated))
