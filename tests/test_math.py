#!/usr/bin/env python3
"""Test the math model with various questions"""
import pytest
torch = pytest.importorskip("torch")
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from transformer import Core
from tokenizer import CharTokenizer

def test_model(model_path='models/math_best.pt', temperature=0.2, max_len=50):
    """Load and test the math model"""
    
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
    
    # Test questions
    test_questions = [
        "Q: What is 5+3?\nA:",
        "Q: What's 7*8?\nA:",
        "Q: What is 12-5?\nA:",
        "Q: Calculate 100/10\nA:",
        "Q: What is 2+2?\nA:",
        "Q: What's 6*7?\nA:",
        "Q: What is your name?\nA:",
        "Q: 10+15?\nA:",
        "Q: What is 3^2?\nA:",
        "Q: Calculate 20-8\nA:",
        "Q: What's 144/12?\nA:",
        "Q: 2+3*4?\nA:",
        "Q: Who are you?\nA:",
        "Q: What is 9*9?\nA:",
        "Q: 50+50?\nA:",
    ]
    
    print("="*70)
    print("TESTING MATH MODEL")
    print("="*70)
    print(f"Model: {model_path}")
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        print(f"Loss: {checkpoint['loss']:.4f}")
    print("="*70 + "\n")
    
    correct = 0
    total = 0
    
    for prompt in test_questions:
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
        print(result)
        print()
        
        # Check if answer looks reasonable (contains digits or "core")
        answer_part = result[len(prompt):].strip()
        if answer_part and (answer_part[0].isdigit() or 'core' in answer_part or 'I' in answer_part):
            total += 1
            # Very basic correctness check
            if ('5+3' in prompt and '8' in answer_part) or \
               ('7*8' in prompt and '56' in answer_part) or \
               ('12-5' in prompt and '7' in answer_part) or \
               ('100/10' in prompt and '10' in answer_part) or \
               ('2+2' in prompt and '4' in answer_part) or \
               ('6*7' in prompt and '42' in answer_part) or \
               ('name' in prompt.lower() and 'core' in answer_part) or \
               ('10+15' in prompt and '25' in answer_part) or \
               ('3^2' in prompt and '9' in answer_part) or \
               ('20-8' in prompt and '12' in answer_part) or \
               ('144/12' in prompt and '12' in answer_part) or \
               ('2+3*4' in prompt and '14' in answer_part) or \
               ('who are you' in prompt.lower() and 'core' in answer_part) or \
               ('9*9' in prompt and '81' in answer_part) or \
               ('50+50' in prompt and '100' in answer_part):
                correct += 1
    
    print("="*70)
    print(f"RESULTS: {correct}/{total} questions answered reasonably")
    print("="*70)

if __name__ == "__main__":
    model_file = sys.argv[1] if len(sys.argv) > 1 else 'models/math_best.pt'
    test_model(model_file)
