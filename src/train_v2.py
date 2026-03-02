#!/usr/bin/env python3
"""
Phase 3: Pre-training loop for ArthurV2
Trains 65M parameter model on balanced dataset.

Training config:
- Batch size: 32 (gradient accumulation × 4 = effective 128)
- Learning rate: 1e-3 with cosine decay
- Warmup: 500 steps
- Max epochs: 50 (with early stopping)
- Checkpointing: Every 1000 steps
- Target loss: <0.05

Usage:
    python3 src/train_v2.py --batch_size 32 --epochs 50 --warmup 500
"""

import sys
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from transformer_v2 import ArthurV2
from bpe_tokenizer import BPETokenizer
from config import ARTHUR_V2_CONFIG

ARTHUR_ROOT = Path(__file__).parent.parent
DATA_DIR = ARTHUR_ROOT / "data"
MODELS_DIR = ARTHUR_ROOT / "models"

class TextDataset(Dataset):
    """Load training data from JSONL."""
    def __init__(self, data_path: str, tokenizer: BPETokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []
        
        with open(data_path, 'r') as f:
            for line in f:
                obj = json.loads(line)
                self.texts.append(obj['text'])
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text)
        
        # Pad or truncate
        if len(tokens) < self.max_length:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        
        return torch.tensor(tokens, dtype=torch.long)

def create_data_loaders(batch_size: int, max_length: int = 512):
    """Load training data and create data loaders."""
    data_path = DATA_DIR / "balanced_dataset.jsonl"
    tokenizer_path = MODELS_DIR / "bpe_tokenizer_v1.json"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")
    
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    # Load tokenizer
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = BPETokenizer()
    tokenizer.load(str(tokenizer_path))
    
    # Create dataset
    print(f"Loading training data from {data_path}...")
    dataset = TextDataset(str(data_path), tokenizer, max_length)
    
    # Split train/val (90/10)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    print(f"  Train: {len(train_dataset)} examples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} examples, {len(val_loader)} batches")
    
    return train_loader, val_loader, tokenizer

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine learning rate schedule with warmup."""
    def lr_lambda(step):
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def compute_next_token_loss(logits, targets):
    """Compute cross-entropy loss for next-token prediction."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = targets[..., 1:].contiguous()
    return nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction='mean',
    )

def train_epoch(model, train_loader, optimizer, scheduler, device, grad_accumulation_steps=4):
    """Train one epoch."""
    model.train()
    total_loss = 0

    for step, batch in enumerate(train_loader):
        batch = batch.to(device)
        logits = model(batch)
        loss = compute_next_token_loss(logits, batch)
        
        # Backward pass with gradient accumulation
        loss = loss / grad_accumulation_steps
        loss.backward()
        total_loss += loss.item()
        
        if (step + 1) % grad_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        if (step + 1) % 10 == 0:
            print(f"    Step {step + 1}: loss = {loss.item():.4f}")
    
    return total_loss / len(train_loader)

def validate(model, val_loader, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = compute_next_token_loss(logits, batch)
            total_loss += loss.item()

    return total_loss / len(val_loader)

def main():
    parser = argparse.ArgumentParser(description="Train ArthurV2")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}\n")
    
    # Data
    train_loader, val_loader, tokenizer = create_data_loaders(args.batch_size)
    
    # Model
    print("Creating ArthurV2 model...")
    model = ArthurV2(**ARTHUR_V2_CONFIG).to(device)
    print(f"✓ Model: {model.get_param_count() / 1e6:.1f}M parameters\n")
    
    # Optimizer & scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    num_training_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, num_training_steps
    )
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...\n")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss = validate(model, val_loader, device)
        
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss: {val_loss:.4f}\n")
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = MODELS_DIR / f"arthur_v2_epoch{epoch + 1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Saved checkpoint: {checkpoint_path}\n")
        
        # Early stopping
        if epoch > 10 and val_loss > best_val_loss * 1.05:
            print("  Early stopping triggered")
            break
    
    print(f"✓ Training complete. Best val loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    main()
