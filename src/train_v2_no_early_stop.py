#!/usr/bin/env python3
"""
Train ArthurV2 without early stopping.
Identical to train_v2.py but runs all epochs unconditionally.
"""
import os
import sys
import argparse

import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(__file__))
from train_v2 import (
    ArthurV2, MODELS_DIR,
    create_data_loaders, train_epoch, validate,
)
from config import ARTHUR_V2_CONFIG


def main():
    parser = argparse.ArgumentParser(description="Train ArthurV2 (no early stopping)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    train_loader, val_loader, _tokenizer = create_data_loaders(args.batch_size)

    model = ArthurV2(**ARTHUR_V2_CONFIG).to(device)

    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters\n")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader)
    )

    print(f"Starting training for {args.epochs} epochs (no early stopping)...\n")
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss = validate(model, val_loader, device)

        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss: {val_loss:.4f}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = MODELS_DIR / f"arthur_v2_epoch{epoch + 1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}\n")

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
