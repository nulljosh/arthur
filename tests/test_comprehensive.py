#!/usr/bin/env python3
"""Test comprehensive model: math, jot, identity, time/date"""
import pytest
torch = pytest.importorskip("torch")
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from transformer import Arthur
from tokenizer import CharTokenizer

pytestmark = pytest.mark.skipif(
    not os.path.exists('models/comprehensive_best.pt'),
    reason='model file not available'
)

def test_model(model_path='models/comprehensive_best.pt', temperature=0.2, max_len=80):
    """Load and test the comprehensive model"""
    
    # Load checkpoint
    checkpoint = torch.load(model_path, weights_only=False)
    
    # Reconstruct tokenizer
    class SimpleTokenizer:
        def __init__(self, vocab):
            self.char_to_idx = vocab
            self.idx_to_char = {v: k for k, v in vocab.items()}
            self.vocab_size = len(vocab)
        
        def encode(self, text):
            return [self.char_to_idx.get(c, self.char_to_idx.get('<UNK>', 0)) for c in text]
        
        def decode(self, tokens):
            return ''.join([self.idx_to_char.get(t, '?') for t in tokens])
    
    tokenizer = SimpleTokenizer(checkpoint['vocab'])
    
    # Create model
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
    
    # Test questions - ALL CATEGORIES
    test_questions = {
        "MATH": [
            "Q: What is 5+3?\nA:",
            "Q: What's 7*8?\nA:",
            "Q: What is 12-5?\nA:",
            "Q: Calculate 100/10\nA:",
            "Q: What is 2+2?\nA:",
            "Q: What's 6*7?\nA:",
            "Q: 10+15?\nA:",
            "Q: What is 3^2?\nA:",
            "Q: 2+3*4?\nA:",
            "Q: What is 9*9?\nA:",
        ],
        "IDENTITY": [
            "Q: What is your name?\nA:",
            "Q: Who are you?\nA:",
            "Q: Who made you?\nA:",
            "Q: Who created you?\nA:",
            "Q: What are you?\nA:",
            "Q: What do you do?\nA:",
        ],
        "TIME/DATE": [
            "Q: What time is it?\nA:",
            "Q: What's today's date?\nA:",
            "Q: When were you trained?\nA:",
            "Q: What date is it?\nA:",
        ],
        "JOT CODE": [
            "Q: print hello world in jot\nA:",
            "Q: write a function in jot\nA:",
            "Q: jot variable example\nA:",
            "Q: if statement in jot\nA:",
            "Q: for loop in jot\nA:",
            "Q: while loop in jot\nA:",
        ],
    }
    
    print("="*70)
    print("TESTING COMPREHENSIVE MODEL")
    print("="*70)
    print(f"Model: {model_path}")
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        print(f"Loss: {checkpoint['loss']:.4f}")
    print("="*70 + "\n")
    
    # Test each category
    for category, questions in test_questions.items():
        print("\n" + "="*70)
        print(f"CATEGORY: {category}")
        print("="*70)
        
        for prompt in questions:
            tokens = tokenizer.encode(prompt)
            x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
            generated = tokens.copy()
            
            with torch.no_grad():
                for _ in range(max_len):
                    logits = model(x)
                    logits = logits[:, -1, :] / temperature
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                    generated.append(next_token)
                    
                    # Stop at double newline or next question
                    decoded = tokenizer.decode(generated)
                    if '\n\nQ:' in decoded or '\nQ:' in decoded[len(prompt):]:
                        break
                    
                    x = torch.cat([x, torch.tensor([[next_token]])], dim=1)
                    if x.size(1) > 128:
                        x = x[:, -128:]
            
            result = tokenizer.decode(generated)
            print(f"\n{result}")
        
        print()
    
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)

if __name__ == "__main__":
    model_file = sys.argv[1] if len(sys.argv) > 1 else 'models/comprehensive_best.pt'
    test_model(model_file)
