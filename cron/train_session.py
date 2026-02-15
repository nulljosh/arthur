#!/usr/bin/env python3
"""
Resumable training session for cron automation.

Loads from latest checkpoint if available, trains for N epochs, saves everything
needed to resume later (model weights, optimizer state, scheduler state, epoch count).

Usage:
    python3 cron/train_session.py                  # 50 epochs (default)
    python3 cron/train_session.py --epochs 100      # 100 epochs
    python3 cron/train_session.py --fresh            # ignore checkpoint, start over
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import sys
import time
import json
import argparse
from datetime import datetime

CORE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(CORE_ROOT, "src"))

from transformer import Nous
from tokenizer import CharTokenizer

# --- paths ---
DATA_FILE = os.path.join(CORE_ROOT, "data", "comprehensive.txt")
CHECKPOINT_FILE = os.path.join(CORE_ROOT, "models", "cron_checkpoint.pt")
BEST_FILE = os.path.join(CORE_ROOT, "models", "cron_best.pt")
LOG_DIR = os.path.join(CORE_ROOT, "logs")
HISTORY_FILE = os.path.join(LOG_DIR, "training_history.jsonl")

# --- model config (micro) ---
MODEL_CFG = dict(
    embed_dim=128,
    num_heads=4,
    num_layers=4,
    ff_dim=512,
    max_len=256,
    dropout=0.1,
)

# --- training config ---
TRAIN_CFG = dict(
    batch_size=16,
    seq_len=128,
    initial_lr=3e-4,
    min_lr=1e-5,
    total_epochs=800,
    grad_clip=1.0,
    weight_decay=0.01,
    warmup_epochs=20,
)


class TextDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.seq_len + 1]
        return torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)


def get_lr(epoch, warmup, total, initial_lr, min_lr):
    """Linear warmup then cosine decay."""
    if epoch < warmup:
        return initial_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, total - warmup)
    import math
    return min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * progress))


def save_checkpoint(path, model, tokenizer, optimizer, epoch, best_loss, history):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss": best_loss,
        "vocab_size": tokenizer.vocab_size,
        "vocab": tokenizer.char_to_idx,
        "model_cfg": MODEL_CFG,
        "train_cfg": TRAIN_CFG,
        "timestamp": datetime.now().isoformat(),
    }, path)


def load_checkpoint(path, model, optimizer, tokenizer):
    cp = torch.load(path, weights_only=False)
    model.load_state_dict(cp["model_state_dict"])
    if "optimizer_state_dict" in cp:
        optimizer.load_state_dict(cp["optimizer_state_dict"])
    return cp["epoch"], cp.get("best_loss", float("inf"))


def test_model(model, tokenizer, prompts, max_len=80, temperature=0.4):
    model.eval()
    results = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        generated = tokens.copy()
        with torch.no_grad():
            for _ in range(max_len):
                logits = model(x)
                logits = logits[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                generated.append(next_token)
                decoded = tokenizer.decode(generated)
                if "\n\nQ:" in decoded or "\nQ:" in decoded[len(prompt) :]:
                    break
                x = torch.cat([x, torch.tensor([[next_token]])], dim=1)
                if x.size(1) > MODEL_CFG["max_len"]:
                    x = x[:, -MODEL_CFG["max_len"] :]
        results.append(tokenizer.decode(generated))
    model.train()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="epochs to train this session")
    parser.add_argument("--fresh", action="store_true", help="start fresh, ignore checkpoint")
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)

    # load data
    with open(DATA_FILE) as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    tokens = tokenizer.encode(text)
    dataset = TextDataset(tokens, TRAIN_CFG["seq_len"])
    dataloader = DataLoader(dataset, batch_size=TRAIN_CFG["batch_size"], shuffle=True)

    # build model
    model = Core(vocab_size=tokenizer.vocab_size, **MODEL_CFG)
    num_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_CFG["initial_lr"], weight_decay=TRAIN_CFG["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    # resume from checkpoint
    start_epoch = 0
    best_loss = float("inf")
    if not args.fresh and os.path.exists(CHECKPOINT_FILE):
        start_epoch, best_loss = load_checkpoint(CHECKPOINT_FILE, model, optimizer, tokenizer)
        print(f"Resumed from epoch {start_epoch}, best_loss={best_loss:.6f}")
    else:
        print(f"Starting fresh training")

    end_epoch = min(start_epoch + args.epochs, TRAIN_CFG["total_epochs"])
    print(f"Model: {num_params:,} params | Data: {len(text):,} chars | Vocab: {tokenizer.vocab_size}")
    print(f"Training epochs {start_epoch + 1} -> {end_epoch} ({end_epoch - start_epoch} this session)")
    print(f"Total target: {TRAIN_CFG['total_epochs']} epochs")
    print()

    test_prompts = [
        "Q: What is 5+3?\nA:",
        "Q: What is your name?\nA:",
        "Q: print hello world in jot\nA:",
        "Q: What's 7*8?\nA:",
    ]

    session_start = time.time()
    model.train()

    for epoch in range(start_epoch, end_epoch):
        epoch_start = time.time()
        total_loss = 0
        batch_count = 0

        # set learning rate
        lr = get_lr(epoch, TRAIN_CFG["warmup_epochs"], TRAIN_CFG["total_epochs"],
                    TRAIN_CFG["initial_lr"], TRAIN_CFG["min_lr"])
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        for x, y in dataloader:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=TRAIN_CFG["grad_clip"])
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / batch_count
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch + 1:4d}/{TRAIN_CFG['total_epochs']} | Loss: {avg_loss:.6f} | LR: {lr:.2e} | Time: {epoch_time:.1f}s")

        # log to history
        record = {
            "epoch": epoch + 1,
            "loss": round(avg_loss, 6),
            "lr": round(lr, 8),
            "time_s": round(epoch_time, 1),
            "timestamp": datetime.now().isoformat(),
        }
        with open(HISTORY_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")

        # save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(BEST_FILE, model, tokenizer, optimizer, epoch + 1, best_loss, None)

        # test every 25 epochs
        if (epoch + 1) % 25 == 0:
            print(f"\n--- Test at epoch {epoch + 1} ---")
            results = test_model(model, tokenizer, test_prompts)
            for r in results:
                print(f"  {r[:120]}")
            print()

    # save final checkpoint for resume
    save_checkpoint(CHECKPOINT_FILE, model, tokenizer, optimizer, end_epoch, best_loss, None)

    total_time = time.time() - session_start
    print(f"\nSession complete: epochs {start_epoch + 1}-{end_epoch} in {total_time / 60:.1f} min")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Checkpoint saved: {CHECKPOINT_FILE}")

    # final test
    print(f"\n--- Final test ---")
    results = test_model(model, tokenizer, test_prompts)
    for r in results:
        print(f"  {r[:150]}")

    # save eval results for report.sh
    eval_file = os.path.join(LOG_DIR, "latest_eval.json")
    eval_data = {
        "epoch": end_epoch,
        "loss": round(best_loss, 6),
        "timestamp": datetime.now().isoformat(),
        "results": [{"prompt": p, "output": r[:200]} for p, r in zip(test_prompts, results)],
    }
    with open(eval_file, "w") as f:
        json.dump(eval_data, f, indent=2)


if __name__ == "__main__":
    main()
