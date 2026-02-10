#!/usr/bin/env python3
"""
Test tokenizers on simple text
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from tokenizer import CharTokenizer, WordTokenizer

# Sample text
text = "hello world this is a test hello test"

print("=" * 60)
print("Character-level tokenizer")
print("=" * 60)
char_tok = CharTokenizer(text)
print(f"Vocab size: {char_tok.vocab_size}")
print(f"Vocab: {list(char_tok.char_to_idx.keys())}")

encoded = char_tok.encode("hello")
print(f"\nEncode 'hello': {encoded}")
print(f"Decode back: '{char_tok.decode(encoded)}'")

print("\n" + "=" * 60)
print("Word-level tokenizer")
print("=" * 60)
word_tok = WordTokenizer(text)
print(f"Vocab size: {word_tok.vocab_size}")
print(f"Vocab: {list(word_tok.word_to_idx.keys())}")

encoded = word_tok.encode("hello world")
print(f"\nEncode 'hello world': {encoded}")
print(f"Decode back: '{word_tok.decode(encoded)}'")
