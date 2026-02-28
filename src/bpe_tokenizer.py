"""
BPE Tokenizer — byte-pair encoding for arthur v2.0
Replaces character-level tokenizer to eliminate corruption and improve efficiency.

Usage:
    tokenizer = BPETokenizer(vocab_size=10000)
    tokenizer.train(texts=['...', '...'])
    tokens = tokenizer.encode("Hello world")
    text = tokenizer.decode(tokens)
"""

import json
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

class BPETokenizer:
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.vocab = {}  # token_id -> token_bytes
        self.merges = []  # list of (pair, new_token) merges
        self.char_tokens = {}  # map single chars to their IDs
        
    def train(self, texts: List[str], min_freq: int = 2):
        """Train BPE on corpus of texts."""
        print(f"Training BPE tokenizer (vocab_size={self.vocab_size})...")
        
        # Phase 1: Initialize with characters
        vocab = defaultdict(int)
        for text in texts:
            for char in text.encode('utf-8').decode('utf-8', errors='ignore'):
                vocab[char] += 1
        
        # Map initial chars to token IDs
        for i, char in enumerate(sorted(vocab.keys())):
            self.char_tokens[char] = i
            self.vocab[i] = char.encode('utf-8')
        
        num_tokens = len(self.vocab)
        print(f"  Initialized with {num_tokens} character tokens")
        
        # Phase 2: Iteratively merge most frequent pairs
        while num_tokens < self.vocab_size:
            # Count adjacent pairs in corpus
            pairs = defaultdict(int)
            for text in texts:
                tokens = self._tokenize_text(text)
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    pairs[pair] += 1
            
            if not pairs:
                break
            
            # Merge most frequent pair
            best_pair, freq = max(pairs.items(), key=lambda x: x[1])
            if freq < min_freq:
                break
            
            # Create new token
            new_token_id = max(self.vocab.keys()) + 1
            merged_bytes = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab[new_token_id] = merged_bytes
            self.merges.append((best_pair, new_token_id))
            
            num_tokens += 1
            if num_tokens % 1000 == 0:
                print(f"  Learned {num_tokens} tokens, freq={freq}")
        
        print(f"  Final vocab size: {len(self.vocab)}")
    
    def _tokenize_text(self, text: str) -> List[int]:
        """Tokenize text using current vocab (for training)."""
        # Start with character tokens
        tokens = []
        for char in text.encode('utf-8').decode('utf-8', errors='ignore'):
            if char in self.char_tokens:
                tokens.append(self.char_tokens[char])
            else:
                tokens.append(self.char_tokens.get('<unk>', 0))
        
        # Apply learned merges
        for (pair_a, pair_b), new_token in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == pair_a and tokens[i + 1] == pair_b:
                    tokens[i:i + 2] = [new_token]
                else:
                    i += 1
        
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self._tokenize_text(text)
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to text."""
        result = b''
        for token_id in tokens:
            if token_id in self.vocab:
                result += self.vocab[token_id]
        return result.decode('utf-8', errors='replace')
    
    def save(self, path: str):
        """Save tokenizer to JSON."""
        with open(path, 'w') as f:
            json.dump({
                'vocab_size': self.vocab_size,
                'vocab': {str(k): v.decode('utf-8', errors='replace') for k, v in self.vocab.items()},
                'merges': [(str(a), str(b), int(c)) for (a, b), c in self.merges],
                'char_tokens': {k: v for k, v in self.char_tokens.items()}
            }, f)
    
    def load(self, path: str):
        """Load tokenizer from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.vocab_size = data['vocab_size']
        self.vocab = {int(k): v.encode('utf-8') for k, v in data['vocab'].items()}
        self.merges = [((int(a), int(b)), int(c)) for a, b, c in data['merges']]
        self.char_tokens = data['char_tokens']
