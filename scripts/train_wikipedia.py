#!/usr/bin/env python3
"""Step-based training on WikiText-103 with gradient accumulation and LR schedule."""

import os
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent

# Training config
CONFIG = {
    'embed_dim': 256,
    'num_heads': 8,
    'num_layers': 6,
    'ff_dim': 1024,
    'max_len': 512,
    'seq_len': 256,
    'batch_size': 4,
}

MAX_STEPS = 50_000
WARMUP_STEPS = 2_000
MAX_LR = 3e-4
MIN_LR = 1e-5
GRAD_ACCUM = 8          # effective batch = 4 * 8 = 32
VAL_EVERY = 500
CKPT_EVERY = 2500
MODELS_DIR = REPO / "models" / "wiki"


def get_lr(step, warmup, max_steps, max_lr, min_lr):
    """Linear warmup then cosine decay."""
    if step < warmup:
        return max_lr * step / warmup
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / (max_steps - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (max_lr - min_lr)


def log_ram():
    """Print current RSS in MB via psutil."""
    try:
        import psutil, os as _os
        proc = psutil.Process(_os.getpid())
        rss_mb = proc.memory_info().rss / 1024 / 1024
        print(f"  RAM: {rss_mb:.0f} MB RSS")
    except ImportError:
        pass


def main():
    import sys
    sys.path.insert(0, str(REPO / "src"))

    from data_loader import WikiText103Dataset
    from tokenizer import BPETokenizer
    from transformer import Nous

    # Device selection: MPS -> CUDA -> CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Tokenizer + datasets
    tokenizer = BPETokenizer()
    print("Loading WikiText-103 train split...")
    train_dataset = WikiText103Dataset(tokenizer, CONFIG['seq_len'], split='train')
    print("Loading WikiText-103 validation split...")
    val_dataset = WikiText103Dataset(tokenizer, CONFIG['seq_len'], split='validation')
    log_ram()

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG['batch_size'],
        shuffle=True, drop_last=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG['batch_size'],
        shuffle=False, drop_last=True, num_workers=0
    )

    model = Core(
        vocab_size=tokenizer.vocab_size,
        embed_dim=CONFIG['embed_dim'],
        num_heads=CONFIG['num_heads'],
        num_layers=CONFIG['num_layers'],
        ff_dim=CONFIG['ff_dim'],
        max_len=CONFIG['max_len'],
        dropout=0.1,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {num_params:,} ({num_params/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()

    step = 0
    optimizer.zero_grad()
    train_iter = iter(train_loader)
    t0 = time.time()

    print(f"\nTraining for {MAX_STEPS:,} steps (grad_accum={GRAD_ACCUM}, effective batch={CONFIG['batch_size']*GRAD_ACCUM})...")

    while step < MAX_STEPS:
        model.train()
        accum_loss = 0.0

        for micro in range(GRAD_ACCUM):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            (loss / GRAD_ACCUM).backward()
            accum_loss += loss.item() / GRAD_ACCUM

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = get_lr(step, WARMUP_STEPS, MAX_STEPS, MAX_LR, MIN_LR)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.step()
        optimizer.zero_grad()
        step += 1

        if step % 50 == 0:
            elapsed = time.time() - t0
            print(f"step {step:>6}/{MAX_STEPS} | loss {accum_loss:.4f} | lr {lr:.2e} | {elapsed:.0f}s")

        if step % VAL_EVERY == 0:
            model.eval()
            val_loss_total = 0.0
            val_batches = min(100, len(val_loader))
            val_iter = iter(val_loader)
            with torch.no_grad():
                for _ in range(val_batches):
                    try:
                        xv, yv = next(val_iter)
                    except StopIteration:
                        break
                    xv, yv = xv.to(device), yv.to(device)
                    logits_v = model(xv)
                    val_loss_total += criterion(logits_v.view(-1, logits_v.size(-1)), yv.view(-1)).item()
            val_loss = val_loss_total / val_batches
            print(f"  [val] step {step} | val_loss {val_loss:.4f}")
            log_ram()

        if step % CKPT_EVERY == 0:
            ckpt_path = MODELS_DIR / f"wiki_step{step:06d}.pt"
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': CONFIG,
                'vocab_size': tokenizer.vocab_size,
            }, ckpt_path)
            print(f"  [ckpt] saved {ckpt_path}")

    # Final checkpoint
    final_path = MODELS_DIR / "wiki_final.pt"
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': CONFIG,
        'vocab_size': tokenizer.vocab_size,
    }, final_path)
    print(f"\nDone. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
