"""
Arthur v3 - Real training on WikiText-103.
Downloads WikiText-103 from HuggingFace (cached locally).

Usage:
    python scripts/train.py --size 65M --steps 5000 --resume
    python scripts/train.py --size 125M --steps 1000 --allow-unsafe
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
sys.path.insert(0, "src")
from src.transformer import ArthurV3
from transformer import migrate_state_dict
from src.bpe_tokenizer import BPETokenizer
from src.config import apply_safe_16gb_guardrails

def get_batch(data, tokenizer, seq_len, batch_size, device):
    """Grab a random batch from cached dataset"""
    import random
    max_retries = 3
    sample_multiplier = 10

    for attempt in range(max_retries):
        chunks = []
        sample_size = min(batch_size * sample_multiplier, len(data))
        indices = random.sample(range(len(data)), sample_size)
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

        if chunks:
            x = torch.tensor([c[:-1] for c in chunks], dtype=torch.long).to(device)
            y = torch.tensor([c[1:]  for c in chunks], dtype=torch.long).to(device)
            return x, y

        sample_multiplier *= 3
        if attempt < max_retries - 1:
            print(f"[batch] Retry {attempt + 1}/{max_retries}: no valid chunks from {sample_size} samples, expanding search", flush=True)

    print(f"[batch] ERROR: Failed to build batch after {max_retries} retries", flush=True)
    return None, None

def train(
    size="65M",
    steps=500,
    lr=3e-4,
    batch_size=1,
    seq_len=256,
    resume=False,
    grad_accum=8,
    run_steps=None,
    allow_unsafe=False,
):
    if os.getenv("ARTHUR_ALLOW_TRAINING") != "1":
        raise SystemExit(
            "Arthur training is parked on this machine. "
            "Use demo/eval workflows only, or set ARTHUR_ALLOW_TRAINING=1 if you truly mean it."
        )

    guardrails = apply_safe_16gb_guardrails(
        size=size,
        batch_size=batch_size,
        seq_len=seq_len,
        grad_accum=grad_accum,
        run_steps=run_steps,
        allow_unsafe=allow_unsafe,
    )
    size = guardrails["size"]
    batch_size = guardrails["batch_size"]
    seq_len = guardrails["seq_len"]
    grad_accum = guardrails["grad_accum"]
    run_steps = guardrails["run_steps"]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[train] ArthurV3-{size} on {device}", flush=True)
    print(f"   Steps: {steps} | Batch: {batch_size} | SeqLen: {seq_len} | GradAccum: {grad_accum}", flush=True)
    if guardrails["safe_mode"]:
        total_ram = guardrails["total_ram_gb"]
        print(f"   Safe profile: 16GB-safe ({total_ram:.1f}GB RAM detected)", flush=True)
    for warning in guardrails["warnings"]:
        print(f"[guardrail] {warning}", flush=True)

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
                model.load_state_dict(migrate_state_dict(ckpt["model"]))
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

    target_step = steps
    if run_steps is not None:
        target_step = min(steps, start_step + run_steps)
    remaining_steps = target_step - start_step
    if remaining_steps <= 0:
        print(f"[train] Already completed {start_step}/{target_step} steps, nothing to do", flush=True)
        return

    sched = CosineAnnealingLR(opt, T_max=max(target_step, 1))
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

    def save_resume_checkpoint(step, loss_value):
        if device == "mps" and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
        torch.save({
            "step": step,
            "loss": loss_value,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
        }, f"models/arthur_v3_{size}_latest.pt")

    model.train()
    log_every = 50

    print(f"\n{'Step':>6} {'Loss':>8} {'Tok/s':>8} {'LR':>10}", flush=True)
    print("-" * 40, flush=True)
    sys.stdout.flush()

    consecutive_nones = 0
    first_step_done = False

    print(f"   Target step this run: {target_step}", flush=True)

    for step in range(start_step + 1, target_step + 1):
        t0 = time.time()
        accum_loss = 0.0
        step_had_data = False

        opt.zero_grad()
        for accum_step in range(grad_accum):
            x, y = get_batch(dataset, tokenizer, seq_len, batch_size, device)
            if x is None:
                continue
            step_had_data = True
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)) / grad_accum
            loss.backward()
            accum_loss += loss.item()

        if not step_had_data:
            consecutive_nones += 1
            print(f"[train] WARNING: Step {step} got no valid batches ({consecutive_nones} consecutive)", flush=True)
            if consecutive_nones >= 10:
                print(f"[train] ERROR: 10 consecutive empty batches, aborting training", flush=True)
                break
            continue
        consecutive_nones = 0

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        dt = time.time() - t0
        toks_per_sec = (batch_size * grad_accum * seq_len) / dt

        if not first_step_done:
            first_step_done = True
            save_resume_checkpoint(step, accum_loss)
            print(f"[train] First step complete (step {step}, loss {accum_loss:.4f})", flush=True)

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

        # Periodic checkpoint every 25 steps (full state for resume)
        if step % 25 == 0:
            save_resume_checkpoint(step, accum_loss)

    print(f"\n[done] Best loss: {best_loss:.4f}", flush=True)
    print(f"[done] Reached step {min(target_step, step if 'step' in locals() else start_step)}", flush=True)
    print(f"[saved] models/arthur_v3_{size}_best.pt", flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--size",       default="65M")
    p.add_argument("--steps",      type=int,   default=500)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--batch_size", type=int,   default=1)
    p.add_argument("--seq_len",    type=int,   default=256)
    p.add_argument("--grad_accum", type=int,   default=8)
    p.add_argument("--run_steps",  type=int,   default=None, help="Maximum new steps to execute this run")
    p.add_argument("--resume",     action="store_true", help="Resume from best checkpoint")
    p.add_argument("--allow-unsafe", action="store_true", help="Disable 16GB safety clamps")
    args = p.parse_args()
    train(
        args.size,
        args.steps,
        args.lr,
        args.batch_size,
        args.seq_len,
        args.resume,
        args.grad_accum,
        args.run_steps,
        args.allow_unsafe,
    )
