#!/usr/bin/env python3
"""Test ultra-minimal model"""
import pytest
torch = pytest.importorskip("torch")
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from transformer import Core
from tokenizer import CharTokenizer

print("Loading models/ultra.pt...")
checkpoint = torch.load('models/ultra.pt', map_location='cpu')

# Load training data to get tokenizer
with open('data/ultra_minimal.txt') as f:
    text = f.read()

tokenizer = CharTokenizer(text)
print(f"✓ Vocab: {tokenizer.vocab_size} chars\n")

# Recreate model with SAME params as training
model = Core(
    vocab_size=tokenizer.vocab_size,
    embed_dim=32,
    num_heads=2,
    num_layers=2,
    ff_dim=64,
    max_len=32,
    dropout=0.0
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def generate(prompt, temp=0.1):
    tokens = tokenizer.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    generated = tokens.copy()
    
    with torch.no_grad():
        for _ in range(15):  # Shorter generation
            # Keep sequence under max_len
            if x.size(1) >= 30:
                x = x[:, -30:]
            
            logits = model(x)
            probs = torch.softmax(logits[:, -1, :] / temp, dim=-1)
            next_token = torch.argmax(probs, dim=-1).item()
            generated.append(next_token)
            
            decoded = tokenizer.decode(generated)
            # Stop at newline
            if '\n' in decoded[len(prompt):]:
                break
            
            x = torch.cat([x, torch.tensor([[next_token]])], dim=1)
    
    result = tokenizer.decode(generated)
    # Extract answer (after "A:")
    if 'A:' in result:
        parts = result.split('A:')
        if len(parts) > 1:
            answer = parts[1].split('\n')[0].strip()
            return answer
    return result[len(prompt):].strip()

print("="*60)
print("🧪 TESTING (your demo questions)")
print("="*60 + "\n")

tests = [
    ("Q: What is your name?\nA:", "I'm core"),
    ("Q: What is 5+5?\nA:", "10"),
    ("Q: What is 2+2?\nA:", "4"),
]

passing = 0
for q, expected in tests:
    answer = generate(q)
    passed = expected.lower() in answer.lower() or answer == expected
    passing += passed
    status = "✓" if passed else "✗"
    
    print(f"{status} {q}")
    print(f"   Answer: {answer}")
    print(f"   Expected: {expected}")
    print()

print("="*60)
print(f"SCORE: {passing}/{len(tests)} ({100*passing/len(tests):.0f}%)")
print("="*60)
