#!/usr/bin/env python3
"""Test basic inference - verify model can answer simple questions"""
import pytest
torch = pytest.importorskip("torch")
import sys
import os

sys.path.insert(0, 'src')
from transformer import Core

def generate(model, tokenizer, prompt, max_tokens=150, temperature=1.0):
    """Generate text from prompt"""
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens])

    model.eval()
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(input_ids)

            # Apply temperature
            logits = logits[0, -1] / temperature
            probs = torch.softmax(logits, dim=0)
            next_token = torch.multinomial(probs, 1).item()

            input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)

            # Stop if max context reached
            if input_ids.shape[1] >= 128:
                break

    return tokenizer.decode(input_ids[0].tolist())

def test_inference(model_path='models/interesting.pt'):
    """Test model inference with various prompts"""

    print(f"Loading model from {model_path}...")

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        print("Run train_interesting.py first to create the model")
        return False

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']
    tokenizer = checkpoint['tokenizer']

    # Reconstruct model
    model = Core(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        ff_dim=config['ff_dim'],
        max_len=config['max_len'],
        dropout=config['dropout']
    )

    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {params:,} parameters")
    print(f"Vocab size: {tokenizer.vocab_size}\n")

    # Test prompts
    test_cases = [
        ("What is your name?", 0.8),
        ("The AI", 0.7),
        ("def fibonacci", 0.6),
        ("The universe", 0.8),
        ("class Neural", 0.7),
    ]

    print("=" * 70)
    print("INFERENCE TEST RESULTS")
    print("=" * 70)

    for prompt, temp in test_cases:
        print(f"\nPrompt: '{prompt}' (temperature={temp})")
        print("-" * 70)

        output = generate(model, tokenizer, prompt, max_tokens=150, temperature=temp)

        print(f"Output:\n{output}")
        print("-" * 70)

    print("\n" + "=" * 70)
    print("BASIC COHERENCE CHECK")
    print("=" * 70)

    # Check if model generates something other than repetition
    test_output = generate(model, tokenizer, "The AI", max_tokens=50)

    # Simple checks
    checks = {
        "Output length > prompt": len(test_output) > len("The AI"),
        "Contains lowercase letters": any(c.islower() for c in test_output),
        "Contains spaces": ' ' in test_output,
        "Not all same character": len(set(test_output)) > 5,
    }

    print()
    all_passed = True
    for check_name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {check_name}")
        all_passed = all_passed and passed

    print("\n" + "=" * 70)
    if all_passed:
        print("SUCCESS: Model generates coherent text")
    else:
        print("WARNING: Model may need more training")
    print("=" * 70)

    return all_passed

if __name__ == "__main__":
    # Check if alternate model specified
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'models/interesting.pt'
    success = test_inference(model_path)
    sys.exit(0 if success else 1)
