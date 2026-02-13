"""
Tokenizer implementations
Start simple: character-level, then BPE
"""

class CharTokenizer:
    """Character-level tokenizer (simplest)"""
    
    def __init__(self, text):
        chars = sorted(set(text))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
    
    def encode(self, text):
        return [self.char_to_idx[ch] for ch in text]
    
    def decode(self, ids):
        return ''.join([self.idx_to_char[i] for i in ids])


class WordTokenizer:
    """Word-level tokenizer (splits on whitespace)"""
    
    def __init__(self, text):
        words = sorted(set(text.split()))
        self.word_to_idx = {w: i for i, w in enumerate(words)}
        self.idx_to_word = {i: w for i, w in enumerate(words)}
        self.vocab_size = len(words)
    
    def encode(self, text):
        return [self.word_to_idx.get(w, 0) for w in text.split()]
    
    def decode(self, ids):
        return ' '.join([self.idx_to_word.get(i, '<UNK>') for i in ids])


class BPETokenizer:
    """Byte Pair Encoding (subword) tokenizer using tiktoken"""

    def __init__(self, vocab_size=50257):
        import tiktoken
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.vocab_size = vocab_size

    def encode(self, text):
        return self.encoder.encode(text)

    def decode(self, tokens):
        return self.encoder.decode(tokens)
