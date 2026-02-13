"""
Training loop
Simple autoregressive language modeling
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformer import NuLLM
from tokenizer import CharTokenizer, WordTokenizer, BPETokenizer
import argparse
import sys


# Model configurations
CONFIGS = {
    'nano': {
        'vocab': 26,
        'embed_dim': 32,
        'num_heads': 2,
        'num_layers': 2,
        'ff_dim': 64,
        'max_len': 64,
        'seq_len': 32,
        'batch_size': 4
    },
    'micro': {
        'vocab': 50257,
        'embed_dim': 128,
        'num_heads': 4,
        'num_layers': 4,
        'ff_dim': 512,
        'max_len': 256,
        'seq_len': 128,
        'batch_size': 8
    },
    'mini': {
        'vocab': 50257,
        'embed_dim': 256,
        'num_heads': 8,
        'num_layers': 6,
        'ff_dim': 1024,
        'max_len': 512,
        'seq_len': 256,
        'batch_size': 4
    }
}


class TextDataset(Dataset):
    """Simple text dataset for char-level modeling"""

    def __init__(self, text, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.tokens = tokenizer.encode(text)

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


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
            
            # Reshape for loss
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")


def generate(model, tokenizer, prompt, max_len=100, device='cpu', temperature=1.0):
    """Generate text from prompt"""
    model.eval()
    
    tokens = tokenizer.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    generated = tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_len):
            # Get predictions
            logits = model(x)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            
            # Update input
            x = torch.cat([x, torch.tensor([[next_token]], device=device)], dim=1)
            
            # Keep context window reasonable
            if x.size(1) > model.max_len:
                x = x[:, -model.max_len:]
    
    return tokenizer.decode(generated)


def main(corpus='tiny', tokenizer_type='char', model_size='nano', epochs=50, lr=1e-3):
    """Main training function

    Args:
        corpus: 'tiny' or 'wikitext-2'
        tokenizer_type: 'char', 'word', or 'bpe'
        model_size: 'nano', 'micro', or 'mini'
        epochs: Number of training epochs
        lr: Learning rate
    """
    # Sample text (tiny Shakespeare-like)
    tiny_text = """
    To be or not to be, that is the question.
    Whether tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles.
    """.strip()

    print(f"Training nuLLM {model_size.upper()} on {corpus}...")

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Get model config
    config = CONFIGS[model_size]

    # Setup corpus and tokenizer
    if corpus == 'wikitext-2':
        if tokenizer_type != 'bpe':
            raise ValueError("WikiText-2 requires BPE tokenizer")

        from data_loader import WikiText2Dataset
        tokenizer = BPETokenizer()
        print(f"Loading WikiText-2 dataset...")
        dataset = WikiText2Dataset(tokenizer, config['seq_len'], split='train')
        print(f"Dataset size: {len(dataset)} examples")
    else:
        # Tiny corpus
        if tokenizer_type == 'char':
            tokenizer = CharTokenizer(tiny_text)
        elif tokenizer_type == 'word':
            tokenizer = WordTokenizer(tiny_text)
        else:
            raise ValueError("Tiny corpus requires char or word tokenizer")

        print(f"Text length: {len(tiny_text)} chars")
        dataset = TextDataset(tiny_text, tokenizer, config['seq_len'])

    print(f"Vocab size: {tokenizer.vocab_size}")

    # Create model
    model = NuLLM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        ff_dim=config['ff_dim'],
        max_len=config['max_len'],
        dropout=0.1
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {num_params:,} ({num_params/1e6:.2f}M)")

    # DataLoader
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train
    print("\nTraining...")
    train(model, dataloader, optimizer, device, epochs=epochs)

    # Generate
    print("\n" + "="*60)
    print("Generating text...")
    print("="*60)

    if corpus == 'tiny':
        prompts = ["To be", "Whether", "The "]
    else:
        prompts = ["The ", "In ", "A "]

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        output = generate(model, tokenizer, prompt, max_len=50, device=device, temperature=0.8)
        print(f"Output: {output}")

    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train nuLLM')
    parser.add_argument('--corpus', default='tiny', choices=['tiny', 'wikitext-2'],
                        help='Training corpus')
    parser.add_argument('--tokenizer', default='char', choices=['char', 'word', 'bpe'],
                        help='Tokenizer type')
    parser.add_argument('--model-size', default='nano', choices=['nano', 'micro', 'mini'],
                        help='Model size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')

    args = parser.parse_args()

    main(
        corpus=args.corpus,
        tokenizer_type=args.tokenizer,
        model_size=args.model_size,
        epochs=args.epochs,
        lr=args.lr
    )
