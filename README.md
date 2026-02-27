# aether

Nano transformer LLM built from scratch. 0.57M params, 200 epochs, final loss 0.2-0.9.

![aether architecture diagram](architecture.svg)

## Quick Start

```bash
# Train
python src/train.py --corpus jot --epochs 100

# Inference (C engine, 50K tok/s)
./inference/aether models/aether.bin "fn " --temp 0.3

# Web UI
python index.py  # http://localhost:5001
```

## The Stack

- **PyTorch trainer** — AdamW, cosine LR, gradient clipping
- **C99 inference** — 350 LOC, mmap weight loading, zero deps
- **Flask UI** — chat, quiz, status modes on :5001
- **Aether daemon** — continuous training, milestone pings, auto-commit
- **Char tokenizer** — jot syntax corpus (185 KB)
- **Model tiers** — Nano (15K) → Micro (630K) → Mini (3.5M) → Small (14M)

## Benchmarks

| Model | Params | Speed | What It Does |
|-------|--------|-------|---|
| **aether** | 0.57M | 50K tok/s | jot autocomplete |
| GPT-2 | 124M | N/A | coherent paragraphs |
| qwen3:14b | 14B | 30 tok/s | code, reasoning |
| Claude | ??? | 80 tok/s | build apps |

The gap from aether to Claude is billions in compute + thousands of researcher-years.

## Why Aether

"What I cannot create, I do not understand." — Feynman

- Full stack from scratch: tokenizer → attention → training → C inference
- No black boxes. Every byte visible.
- Learning tool, not production model
- C99 engine runs anywhere with a compiler

## Architecture

```
Data (jot corpus)
  ↓
PyTorch Trainer (200 epochs)
  ↓
Checkpoint (.pt)
  ├→ Export (aether.bin)
  │   ↓
  │   C Inference (350 LOC, mmap)
  │
  ├→ Flask Web UI (:5001)
  │
  └→ Aether Daemon (auto-train + notifications)
```

## Roadmap

### Phase 1: Foundation ✅
- [x] Tokenizer + transformer
- [x] C99 inference (350 LOC, zero deps)
- [x] Training harness + Flask UI
- [x] Aether daemon + notifications
- [x] Full test suite

### Phase 2: Data Quality (Next)
- [ ] WikiText-103 BPE training (103M tokens)
- [ ] Scale to mini (3.5M params)
- [ ] Perplexity tracking vs GPT-2

### Phase 3-5: Coming
- [ ] Multi-GPU training
- [ ] Instruction tuning (Alpaca)
- [ ] ONNX export (mobile/browser)
- [ ] HTTP inference API

## Status

**Training complete:** 200 epochs, final loss 0.2-0.9

**Next:** WikiText-103 BPE training for scale to 14M params

## License

MIT 2026, Joshua Trommel
