"""
WikiText-2 data loader
Loads and preprocesses WikiText-2 dataset from Hugging Face
"""

from datasets import load_dataset
import torch
from torch.utils.data import Dataset


def load_wikitext_2(split='train', max_seq=None):
    """Load WikiText-2 from Hugging Face

    Args:
        split: 'train', 'validation', or 'test'
        max_seq: Maximum sequence length (None for full dataset)

    Returns:
        String of concatenated text
    """
    dataset = load_dataset('wikitext', 'wikitext-2-v1', split=split)
    text = '\n\n'.join(dataset['text'])
    return text[:max_seq] if max_seq else text


def load_conversational_corpus() -> str:
    """Load conversational Q&A corpus"""
    from pathlib import Path
    corpus_path = Path(__file__).parent.parent / 'data' / 'conversational.txt'
    with open(corpus_path) as f:
        return f.read()

def load_jot_corpus() -> str:
    """Load jot code examples"""
    from pathlib import Path
    corpus_path = Path(__file__).parent.parent / 'data' / 'jot_code.txt'
    with open(corpus_path) as f:
        return f.read()

def load_combined_corpus() -> str:
    """Load combined conversational + jot corpus"""
    from pathlib import Path
    corpus_path = Path(__file__).parent.parent / 'data' / 'combined_corpus.txt'
    with open(corpus_path) as f:
        return f.read()

class WikiText2Dataset(Dataset):
    """WikiText-2 dataset for language modeling"""

    def __init__(self, tokenizer, seq_len, split='train'):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        # Load and tokenize
        text = load_wikitext_2(split)
        self.tokens = tokenizer.encode(text)

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def load_wikitext_103(split='train'):
    """Load WikiText-103 from local cache or HuggingFace."""
    from pathlib import Path
    local_path = Path(__file__).parent.parent / 'data' / 'wikipedia' / f'{split}.txt'
    if local_path.exists():
        with open(local_path) as f:
            return f.read()
    dataset = load_dataset('wikitext', 'wikitext-103-v1', split=split)
    return '\n'.join(t for t in dataset['text'] if t.strip() and not t.strip().startswith('='))


class WikiText103Dataset(Dataset):
    """WikiText-103 dataset for language modeling"""
    def __init__(self, tokenizer, seq_len, split='train'):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        text = load_wikitext_103(split)
        self.tokens = tokenizer.encode(text)
        print(f"WikiText-103 {split}: {len(self.tokens):,} tokens")

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
