#!/usr/bin/env python3
"""
Phase 6: Scale - Train on WikiText-2 with BPE tokenizer
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from train import main
import argparse


if __name__ == '__main__':
    print("\n" + "="*70)
    print("jore Phase 6: Scale")
    print("="*70)
    print("\nTraining larger models on WikiText-2 with BPE tokenization")
    print("This demonstrates scaling from Nano (50K params) to Mini (5M params)\n")

    parser = argparse.ArgumentParser(description='Phase 6: Scale training')
    parser.add_argument('--size', default='micro', choices=['nano', 'micro', 'mini'],
                        help='Model size (default: micro)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')

    args = parser.parse_args()

    print(f"Configuration:")
    print(f"  Model size: {args.size.upper()}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Corpus: WikiText-2 (~4.3MB)")
    print(f"  Tokenizer: BPE (50K vocab)")
    print()

    # Train
    model, tokenizer = main(
        corpus='wikitext-2',
        tokenizer_type='bpe',
        model_size=args.size,
        epochs=args.epochs,
        lr=args.lr
    )

    print("\n" + "="*70)
    print("Phase 6 training complete!")
    print("="*70)
