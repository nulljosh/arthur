#!/usr/bin/env python3
"""Comprehensive nous testing"""
import pytest
torch = pytest.importorskip("torch")
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from transformer import Nous
from tokenizer import CharTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'models/ultra.pt'

print(f"Loading {MODEL}...")
checkpoint = torch.load(MODEL, map_location='cpu')

# Reconstruct tokenizer
class SimpleTokenizer:
    def __init__(self, vocab):
        self.char_to_idx = vocab
        self.idx_to_char = {v: k for k, v in vocab.items()}
        self.vocab_size = len(vocab)
    def encode(self, text): return [self.char_to_idx.get(c, 0) for c in text]
    def decode(self, tokens): return ''.join([self.idx_to_char.get(t, '?') for t in tokens])

tokenizer = SimpleTokenizer(checkpoint['vocab'])

# Reconstruct model (match ultra training params)
model = Core(
    vocab_size=checkpoint['vocab_size'],
    embed_dim=32,
    num_heads=2,
    num_layers=2,
    ff_dim=64,
    max_len=32,
    dropout=0.0
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def generate(prompt, temp=0.1, max_len=20):
    tokens = tokenizer.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    generated = tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_len):
            logits = model(x)
            probs = torch.softmax(logits[:, -1, :] / temp, dim=-1)
            next_token = torch.argmax(probs, dim=-1).item()
            generated.append(next_token)
            
            decoded = tokenizer.decode(generated)
            if '\n' in decoded[len(prompt):]:
                break
            
            x = torch.cat([x, torch.tensor([[next_token]])], dim=1)
    
    return tokenizer.decode(generated)

print("\n" + "="*60)
print("DEMO QUESTIONS (from your examples):")
print("="*60)

demo = [
    "Q: What is your name?\nA:",
    "Q: What is 5+5?\nA:",
    "Q: What is 2+2?\nA:",
]

for q in demo:
    result = generate(q)
    answer = result.split('\n')[1] if '\n' in result and len(result.split('\n')) > 1 else result[len(q):].strip()
    expected = "I'm nous" if "name" in q else ("10" if "5+5" in q else "4")
    status = "✓" if expected in answer else "✗"
    print(f"\n{status} {q}")
    print(f"   Answer: {answer}")
    print(f"   Expected: {expected}")

print("\n" + "="*60)
print("ADDITIONAL TESTS:")
print("="*60)

extra = [
    ("Q: What is your name?\nA:", "I'm nous"),
    ("Q: What is 5+5?\nA:", "10"),
    ("Q: What is 2+2?\nA:", "4"),
]

passing = 0
total = len(extra)

for q, expected in extra:
    result = generate(q, temp=0.05)  # Even more deterministic
    answer = result.split('\n')[1] if '\n' in result and len(result.split('\n')) > 1 else result[len(q):].strip()
    passed = expected in answer
    passing += passed
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"\n{status}: {q}")
    print(f"   Got: {answer}")

print("\n" + "="*60)
print(f"SCORE: {passing}/{total} ({100*passing/total:.0f}%)")
print("="*60)
