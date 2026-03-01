"""
Training loop
Char-level autoregressive language modeling on jot syntax
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformer import Arthur
from tokenizer import CharTokenizer, WordTokenizer, BPETokenizer
import argparse

# Disable GPU/Metal acceleration, limit CPU threads
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
torch.set_num_threads(2)
torch.set_num_interop_threads(2)

# Limit CPU threads to 4

# Root of the repo (one level up from src/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model configurations
CONFIGS = {
    'nano': {
        'embed_dim': 32,
        'num_heads': 2,
        'num_layers': 2,
        'ff_dim': 64,
        'max_len': 64,
        'seq_len': 32,
        'batch_size': 2
    },
    'micro': {
        'embed_dim': 128,
        'num_heads': 4,
        'num_layers': 4,
        'ff_dim': 512,
        'max_len': 256,
        'seq_len': 128,
        'batch_size': 4
    },
    'mini': {
        'embed_dim': 256,
        'num_heads': 8,
        'num_layers': 6,
        'ff_dim': 1024,
        'max_len': 512,
        'seq_len': 256,
        'batch_size': 2
    },
    'wiki': {
        'embed_dim': 256,
        'num_heads': 8,
        'num_layers': 6,
        'ff_dim': 1024,
        'max_len': 512,
        'seq_len': 256,
        'batch_size': 2
    }
}


class TextDataset(Dataset):
    """Autoregressive sliding-window dataset"""

    def __init__(self, text, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.tokens = tokenizer.encode(text)

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def load_jot_corpus():
    """Load the jot source corpus from data/jot_corpus.txt.
    Runs build_jot_corpus.py first if the file is missing.
    """
    corpus_path = os.path.join(REPO_ROOT, "data", "jot_corpus.txt")
    if not os.path.exists(corpus_path):
        import subprocess
        build_script = os.path.join(REPO_ROOT, "build_jot_corpus.py")
        subprocess.run(["python3", build_script], check=True)
    with open(corpus_path, "r") as f:
        return f.read()


def train(model, dataloader, optimizer, device, epochs=10):
    """Training loop"""
    model.train()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)

            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")


def generate(model, tokenizer, prompt, max_len=200, device='cpu', temperature=0.8):
    """Autoregressive generation from prompt"""
    model.eval()

    tokens = tokenizer.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    generated = list(tokens)

    with torch.no_grad():
        for _ in range(max_len):
            logits = model(x)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            x = torch.cat([x, torch.tensor([[next_token]], device=device)], dim=1)
            if x.size(1) > model.max_len:
                x = x[:, -model.max_len:]

    return tokenizer.decode(generated)


def main(corpus='jot', tokenizer_type='char', model_size='nano', epochs=50, lr=1e-3):
    """Main training function

    Args:
        corpus: 'jot' (default), 'tiny', or 'wikitext-2'
        tokenizer_type: 'char' (default for jot), 'word', or 'bpe'
        model_size: 'nano', 'micro', or 'mini'
        epochs: number of training epochs
        lr: learning rate
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Training nous {model_size.upper()} on {corpus} corpus (tokenizer: {tokenizer_type})...")

    config = CONFIGS[model_size]

    if corpus == 'wikitext-2':
        if tokenizer_type != 'bpe':
            raise ValueError("wikitext-2 requires bpe tokenizer")
        from data_loader import WikiText2Dataset
        tokenizer = BPETokenizer()
        print("Loading WikiText-2...")
        dataset = WikiText2Dataset(tokenizer, config['seq_len'], split='train')
        print(f"Dataset size: {len(dataset)} examples")

    elif corpus == 'wiki':
        if tokenizer_type != 'bpe':
            raise ValueError("wiki corpus requires bpe tokenizer")
        from data_loader import WikiText103Dataset
        tokenizer = BPETokenizer()
        print("Loading WikiText-103...")
        dataset = WikiText103Dataset(tokenizer, config['seq_len'], split='train')
        print(f"Dataset size: {len(dataset):,} examples")

    elif corpus == 'jot':
        if tokenizer_type != 'char':
            raise ValueError("jot corpus uses char tokenizer only")
        text = load_jot_corpus()
        print(f"Corpus length: {len(text):,} chars")
        tokenizer = CharTokenizer(text)
        dataset = TextDataset(text, tokenizer, config['seq_len'])
        print(f"Dataset size: {len(dataset)} windows")

    else:
        # 'tiny' fallback
        tiny_text = (
            "To be or not to be, that is the question.\n"
            "Whether tis nobler in the mind to suffer\n"
            "The slings and arrows of outrageous fortune,\n"
            "Or to take arms against a sea of troubles.\n"
        )
        if tokenizer_type == 'char':
            tokenizer = CharTokenizer(tiny_text)
        elif tokenizer_type == 'word':
            tokenizer = WordTokenizer(tiny_text)
        else:
            raise ValueError("tiny corpus requires char or word tokenizer")
        text = tiny_text
        print(f"Corpus length: {len(text):,} chars")
        dataset = TextDataset(text, tokenizer, config['seq_len'])

    print(f"Vocab size: {tokenizer.vocab_size}")

    model = Arthur(
        vocab_size=tokenizer.vocab_size,
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        ff_dim=config['ff_dim'],
        max_len=config['max_len'],
        dropout=0.1
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {num_params:,} ({num_params/1e6:.3f}M)")

    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("\nTraining...")
    train(model, dataloader, optimizer, device, epochs=epochs)

    print("\n" + "=" * 60)
    print("Generating jot code samples...")
    print("=" * 60)

    if corpus == 'jot':
        prompts = ["fn ", "let x = ", "if ", "// "]
    elif corpus == 'tiny':
        prompts = ["To be", "Whether", "The "]
    else:
        prompts = ["The ", "In ", "A "]

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        output = generate(model, tokenizer, prompt, max_len=120, device=device, temperature=0.8)
        print(f"Output: {output}")

    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train nous on jot syntax')
    parser.add_argument('--corpus', default='jot', choices=['jot', 'tiny', 'wikitext-2', 'wiki'],
                        help='Training corpus (default: jot)')
    parser.add_argument('--tokenizer', default='char', choices=['char', 'word', 'bpe'],
                        help='Tokenizer type (default: char)')
    parser.add_argument('--model-size', default='nano', choices=['nano', 'micro', 'mini', 'wiki'],
                        help='Model size (default: nano)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')

    args = parser.parse_args()
    main(
        corpus=args.corpus,
        tokenizer_type=args.tokenizer,
        model_size=args.model_size,
        epochs=args.epochs,
        lr=args.lr
    )
