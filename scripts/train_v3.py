"""
Arthur v3 Training Script
Usage: python scripts/train_v3.py --size 125M --epochs 3
"""

import argparse
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import sys
sys.path.insert(0, ".")
from src.transformer_v3 import ArthurV3

def train(size="125M", epochs=1, lr=3e-4, batch_size=4, seq_len=512):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training ArthurV3-{size} on {device}")

    model = ArthurV3(size).to(device)
    opt   = AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    sched = CosineAnnealingLR(opt, T_max=epochs * 100)

    model.train()
    best_loss = float("inf")

    for epoch in range(epochs):
        # Synthetic batch (replace with real DataLoader)
        x = torch.randint(0, model.cfg["vocab"], (batch_size, seq_len)).to(device)
        y = torch.roll(x, -1, dims=1)   # shift by 1 = next-token prediction

        t0 = time.time()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        dt = time.time() - t0
        print(f"Epoch {epoch+1}/{epochs} | loss={loss.item():.4f} | {dt:.2f}s")

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), f"models/arthur_v3_{size}.pt")

    print(f"\nBest loss: {best_loss:.4f}")
    print(f"Saved: models/arthur_v3_{size}.pt")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--size",   default="125M")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr",     type=float, default=3e-4)
    args = p.parse_args()
    train(args.size, args.epochs, args.lr)
