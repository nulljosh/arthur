"""
Arthur v3 - Real training on WikiText-103
Downloads WikiText-103 from HuggingFace (cached locally).
Usage: python scripts/train.py --size 125M --steps 1000
       python scripts/train.py --size 65M --steps 5000 --resume
"""

import argparse
import gc
import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets import load_dataset
import time, sys
sys.path.insert(0, ".")
from src.transformer import ArthurV3
from src.bpe_tokenizer import BPETokenizer

def get_batch(data, tokenizer, seq_len, batch_size, device):
    """Grab a random batch from cached dataset"""
    import random
    chunks = []
    indices = random.sample(range(len(data)), min(batch_size * 10, len(data)))
    for idx in indices:
        text = data[idx].get("text", "") or data[idx].get("content", "")
        if len(text) < 100:
            continue
        ids = tokenizer.encode(text)
        if len(ids) < seq_len + 1:
            continue
        start = torch.randint(0, len(ids) - seq_len, (1,)).item()
        chunks.append(ids[start:start + seq_len + 1])
        if len(chunks) == batch_size:
            break

    if not chunks:
        return None, None

    x = torch.tensor([c[:-1] for c in chunks], dtype=torch.long).to(device)
    y = torch.tensor([c[1:]  for c in chunks], dtype=torch.long).to(device)
    return x, y

def train(size="125M", steps=500, lr=3e-4, batch_size=1, seq_len=256, resume=False, grad_accum=8):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[train] ArthurV3-{size} on {device}", flush=True)
    print(f"   Steps: {steps} | Batch: {batch_size} | SeqLen: {seq_len} | GradAccum: {grad_accum}", flush=True)

    # Load model
    model = ArthurV3(size).to(device)
    opt   = AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))

    start_step = 0
    best_loss = float("inf")
    sched_state = None

    # Resume from checkpoint (prefer _latest over _best, fallback on corruption)
    if resume:
        latest_path = f"models/arthur_v3_{size}_latest.pt"
        best_path = f"models/arthur_v3_{size}_best.pt"
        candidates = [p for p in [latest_path, best_path] if os.path.exists(p)]
        if not candidates:
            print(f"[resume] No checkpoint found, starting fresh", flush=True)
        else:
            ckpt = None
            for ckpt_path in candidates:
                try:
                    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                    break
                except Exception as e:
                    print(f"[resume] Failed to load {ckpt_path}: {e}", flush=True)
                    print(f"[resume] Removing corrupted checkpoint", flush=True)
                    os.remove(ckpt_path)
            if ckpt is not None:
                model.load_state_dict(ckpt["model"])
                if "opt" in ckpt:
                    opt.load_state_dict(ckpt["opt"])
                else:
                    print(f"[resume] No optimizer state in checkpoint, starting optimizer fresh", flush=True)
                start_step = ckpt["step"]
                best_loss = ckpt["loss"]
                if "sched" in ckpt:
                    sched_state = ckpt["sched"]
                print(f"[resume] Loaded checkpoint from {ckpt_path}", flush=True)
                print(f"[resume] Resuming from step {start_step}, best loss {best_loss:.4f}", flush=True)
            else:
                print(f"[resume] All checkpoints corrupted, starting fresh", flush=True)

    remaining_steps = steps - start_step
    if remaining_steps <= 0:
        print(f"[train] Already completed {start_step}/{steps} steps, nothing to do", flush=True)
        return

    sched = CosineAnnealingLR(opt, T_max=steps)
    if sched_state is not None:
        sched.load_state_dict(sched_state)

    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load("models/bpe_tokenizer_v1.json")

    # Prevent HuggingFace tokenizer semaphore leak on macOS + Python 3.13
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load WikiText-103 (cached locally after first download)
    print("[data] Loading WikiText-103 from HuggingFace...", flush=True)
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    model.train()
    log_every = 50

    print(f"\n{'Step':>6} {'Loss':>8} {'Tok/s':>8} {'LR':>10}", flush=True)
    print("-" * 40, flush=True)

    for step in range(start_step + 1, steps + 1):
        t0 = time.time()
        accum_loss = 0.0

        opt.zero_grad()
        for accum_step in range(grad_accum):
            x, y = get_batch(dataset, tokenizer, seq_len, batch_size, device)
            if x is None:
                continue
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)) / grad_accum
            loss.backward()
            accum_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        dt = time.time() - t0
        toks_per_sec = (batch_size * grad_accum * seq_len) / dt

        if step % log_every == 0:
            lr_now = opt.param_groups[0]['lr']
            print(f"{step:>6} {accum_loss:>8.4f} {toks_per_sec:>8.0f} {lr_now:>10.2e}", flush=True)

        if step % 100 == 0:
            gc.collect()
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()

        if accum_loss < best_loss:
            best_loss = accum_loss
            if device == "mps" and hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()
            # Best checkpoint: model weights only (slim, ~600MB instead of 1.9GB)
            torch.save({
                "step": step,
                "loss": best_loss,
                "model": model.state_dict(),
            }, f"models/arthur_v3_{size}_best.pt")

        # Periodic checkpoint every 100 steps (full state for resume)
        if step % 100 == 0:
            if device == "mps" and hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()
            torch.save({
                "step": step,
                "loss": accum_loss,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
            }, f"models/arthur_v3_{size}_latest.pt")

    print(f"\n[done] Best loss: {best_loss:.4f}", flush=True)
    print(f"[saved] models/arthur_v3_{size}_best.pt", flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--size",       default="65M")
    p.add_argument("--steps",      type=int,   default=500)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--batch_size", type=int,   default=1)
    p.add_argument("--seq_len",    type=int,   default=256)
    p.add_argument("--grad_accum", type=int,   default=8)
    p.add_argument("--resume",     action="store_true", help="Resume from best checkpoint")
    args = p.parse_args()
    train(args.size, args.steps, args.lr, args.batch_size, args.seq_len, args.resume, args.grad_accum)
