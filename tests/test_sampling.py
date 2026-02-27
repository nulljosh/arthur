#!/usr/bin/env python3
import pytest
torch = pytest.importorskip("torch")
import sys
sys.path.insert(0, 'src')
from transformer import Jore

class SimpleTokenizer:
    def __init__(self, vocab):
        self.char_to_idx = vocab
        self.idx_to_char = {v: k for k, v in vocab.items()}
        self.vocab_size = len(vocab)
    def encode(self, text):
        return [self.char_to_idx.get(c, 0) for c in text]
    def decode(self, tokens):
        return ''.join([self.idx_to_char.get(t, '?') for t in tokens])

checkpoint = torch.load('models/ultra.pt', map_location='cpu')
tokenizer = SimpleTokenizer(checkpoint['vocab'])

model = Core(vocab_size=29, embed_dim=32, num_heads=2, num_layers=2, ff_dim=64, max_len=32, dropout=0.0)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def test(prompt, method='greedy', temp=0.1, top_k=5):
    tokens = tokenizer.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    generated = tokens.copy()
    
    with torch.no_grad():
        for _ in range(20):
            if x.size(1) >= 30: x = x[:, -30:]
            logits = model(x)[:, -1, :] / temp
            
            if method == 'greedy':
                next_token = torch.argmax(logits, dim=-1).item()
            else:  # top-k sampling
                if top_k > 0:
                    values, indices = torch.topk(logits, top_k)
                    logits_filtered = torch.full_like(logits, -float('inf'))
                    logits_filtered[0, indices[0]] = values[0]
                    probs = torch.softmax(logits_filtered, dim=-1)
                else:
                    probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs[0], 1).item()
            
            generated.append(next_token)
            if tokenizer.idx_to_char.get(next_token) == '\n': break
            x = torch.cat([x, torch.tensor([[next_token]])], dim=1)
    
    return tokenizer.decode(generated)

prompt = "Q: What is your name?\nA: "
print(f"Prompt: '{prompt.strip()}'")
print(f"\n1. Greedy (temp=0.1): {test(prompt, 'greedy', 0.1)}")
print(f"\n2. Top-k=5 (temp=0.7): {test(prompt, 'topk', 0.7, 5)}")
print(f"\n3. Top-k=3 (temp=0.5): {test(prompt, 'topk', 0.5, 3)}")
