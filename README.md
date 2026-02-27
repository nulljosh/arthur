# aether

Nano transformer LLM built from scratch. Multi-language training (C99, Rust, Python, Go, jot, jit). 500 epochs on 27MB multilang corpus, final loss 0.0947.

![aether architecture diagram](architecture.svg)

## Quick Start

```bash
# Train on multilang corpus (500 epochs)
python src/train.py --corpus tiny --epochs 500

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
- **Multilang tokenizer** — char-level, all programming languages
- **Model tiers** — Nano (15K) → Micro (630K) → Mini (3.5M) → Small (14M)

## Training Results

### v0.3.0: Multilang (Current)
- **Corpus**: 27MB, 4414 files
- **Languages**: C99, Rust, Python, Go, jot, jit, shell, markdown
- **Epochs**: 500
- **Final Loss**: 0.0947
- **Convergence**: Strong, stable training curve

### v0.2.0: Jung Corpus
- **Corpus**: 31K chars (jot + jung)
- **Epochs**: 100
- **Final Loss**: 0.17

### v0.1.0: Jot Corpus
- **Corpus**: 185 KB syntax
- **Epochs**: 200
- **Final Loss**: 0.2-0.9

## Benchmarks

| Model | Params | Speed | Training Data | Capability |
|-------|--------|-------|---|---|
| **aether (multilang)** | 0.57M | 50K tok/s | 27MB code | multilang autocomplete |
| **aether (jot)** | 0.57M | 50K tok/s | 185 KB syntax | jot syntax only |
| GPT-2 | 124M | N/A | 40 GB WebText | coherent paragraphs |
| qwen3:14b | 14B | 30 tok/s | trillions | code, reasoning |
| Claude | ??? | 80 tok/s | internet scale | build apps |

## Why Aether

"What I cannot create, I do not understand." — Feynman

- Full stack from scratch: tokenizer → attention → training → C inference
- No black boxes. Every byte visible.
- Learning tool, not production model
- C99 engine runs anywhere with a compiler

## Architecture

```
Data (27MB multilang code)
  ↓
PyTorch Trainer (500 epochs)
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
- [x] Multilang training (500 epochs, 27MB)

### Phase 2: Scale (Next)
- [ ] WikiText-103 BPE training (103M tokens)
- [ ] Scale to mini (3.5M params)
- [ ] Perplexity tracking vs GPT-2

### Phase 3-5: Coming
- [ ] Multi-GPU training
- [ ] Instruction tuning (Alpaca)
- [ ] ONNX export (mobile/browser)
- [ ] HTTP inference API

## Status

**Latest**: v0.3.0 multilang training complete (500 epochs, 0.0947 loss)

**Next**: WikiText-103 BPE corpus for scale to 14M params

## License

MIT 2026, Joshua Trommel
