"""
Tokenizer implementations
Start simple: character-level, then BPE
"""


class CharTokenizer:
    """Character-level tokenizer (simplest)"""

    UNK = '\x00'

    def __init__(self, text):
        chars = sorted(set(text))
        # Reserve index 0 for UNK
        if self.UNK not in chars:
            chars = [self.UNK] + chars
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text):
        unk_idx = self.char_to_idx[self.UNK]
        return [self.char_to_idx.get(ch, unk_idx) for ch in text]

    def decode(self, ids):
        return ''.join([self.idx_to_char.get(i, self.UNK) for i in ids])


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

    def __init__(self):
        import tiktoken
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.vocab_size = self.encoder.n_vocab

    def encode(self, text):
        return self.encoder.encode(text)

    def decode(self, tokens):
        return self.encoder.decode(tokens)
