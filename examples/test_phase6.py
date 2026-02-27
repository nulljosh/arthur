#!/usr/bin/env python3
"""
Test Phase 6 components
Validates BPE tokenizer, WikiText-2 loading, dataset creation, and model forward pass
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_bpe_tokenizer():
    """Test 1: BPE tokenizer"""
    print("Test 1: BPE Tokenizer")
    print("-" * 40)

    from tokenizer import BPETokenizer

    tok = BPETokenizer()
    text = "Hello world! This is a test."

    encoded = tok.encode(text)
    decoded = tok.decode(encoded)

    assert text == decoded, f"BPE round-trip failed: '{text}' != '{decoded}'"
    print(f"Input text: {text}")
    print(f"Encoded: {encoded[:10]}... ({len(encoded)} tokens)")
    print(f"Decoded: {decoded}")
    print("✓ BPE tokenizer working\n")


def test_wikitext_loading():
    """Test 2: WikiText-2 loading"""
    print("Test 2: WikiText-2 Loading")
    print("-" * 40)

    from data_loader import load_wikitext_2

    text = load_wikitext_2('train', max_seq=1000)

    assert len(text) > 0, "WikiText-2 loading failed"
    print(f"Loaded {len(text)} chars from WikiText-2")
    print(f"Sample: {text[:100]}...")
    print("✓ WikiText-2 loaded successfully\n")


def test_dataset_creation():
    """Test 3: Dataset creation"""
    print("Test 3: Dataset Creation")
    print("-" * 40)

    from tokenizer import BPETokenizer
    from data_loader import WikiText2Dataset

    tok = BPETokenizer()
    dataset = WikiText2Dataset(tok, seq_len=128, split='train')

    x, y = dataset[0]

    assert x.shape[0] == 128, f"Dataset shape incorrect: {x.shape}"
    print(f"Dataset size: {len(dataset)} examples")
    print(f"Example shape: x={x.shape}, y={y.shape}")
    print(f"Sample tokens: {x[:10].tolist()}")
    print("✓ Dataset created successfully\n")


def test_model_forward():
    """Test 4: Model forward pass (Micro config)"""
    print("Test 4: Model Forward Pass")
    print("-" * 40)

    import torch
    from tokenizer import BPETokenizer
    from data_loader import WikiText2Dataset
    from transformer import Jore

    # Create dataset
    tok = BPETokenizer()
    dataset = WikiText2Dataset(tok, seq_len=128, split='train')
    x, y = dataset[0]

    # Get actual vocab size from tokenizer
    actual_vocab = tok.encoder.n_vocab
    print(f"Actual vocab size: {actual_vocab}")

    # Create Micro model with correct vocab size
    model = Core(
        vocab_size=actual_vocab,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        ff_dim=512,
        max_len=256,
        dropout=0.1
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {num_params:,} ({num_params/1e6:.2f}M)")

    # Forward pass
    out = model(x.unsqueeze(0))

    assert out.shape == (1, 128, actual_vocab), f"Model output shape incorrect: {out.shape}"
    print(f"Input shape: {x.unsqueeze(0).shape}")
    print(f"Output shape: {out.shape}")
    print("✓ Micro model forward pass working\n")


def main():
    print("\n" + "="*70)
    print("jore Phase 6: Component Tests")
    print("="*70 + "\n")

    try:
        test_bpe_tokenizer()
        test_wikitext_loading()
        test_dataset_creation()
        test_model_forward()

        print("="*70)
        print("All Phase 6 components tested successfully!")
        print("="*70)
        print("\nYou can now run:")
        print("  python examples/train_phase6.py --size micro --epochs 5")
        print()

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
