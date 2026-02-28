#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from train_v2 import *

# Override main to remove early stopping
def main_no_stop():
    args = parse_args()
    train_loader, val_loader = create_data_loaders(args.batch_size)
    
    model = ArthurV2(
        vocab_size=32768,
        dim=512,
        n_layers=12,
        n_heads=8,
        n_kv_heads=4,
        context_len=8192
    ).to(args.device)
    
    print(f"✓ Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters\n")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
    
    print(f"Starting training for {args.epochs} epochs...\n")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, args.device)
        val_loss = validate(model, val_loader, args.device)
        
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss: {val_loss:.4f}\n")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = MODELS_DIR / f"arthur_v2_epoch{epoch + 1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Saved checkpoint: {checkpoint_path}\n")
    
    print(f"✓ Training complete. Best val loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    main_no_stop()
