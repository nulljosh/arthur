# aether

Nano transformer LLM built from scratch. PyTorch training loop, C99 inference engine, overnight automation.

**GitHub**: https://github.com/nulljosh/aether | **Benchmarks**: See BENCHMARK.md

## The Philosophy

"What I cannot create, I do not understand." — Feynman

This is aether: the full stack from scratch. Tokenizer → attention → transformer → training loop → checkpoint → C inference engine. Every byte visible. The goal: understand the machine top to bottom.

## Stack

- **PyTorch** — transformer model, training loop, checkpoint management
- **C99 inference engine** (`inference/aether.c`, ~350 LOC, mmap weight loading, zero dependencies)
- **Flask web UI** (port 5001, chat + quiz modes)
- **Tokenizers** — char-level (jot/syntax) + BPE (WikiText-103)
- **Training** — AdamW + cosine LR decay, gradient clipping, deterministic seeding
- **Automation** — aether daemon (continuous training, milestone notifications, RAM-gated)

## Architecture

![aether architecture diagram](architecture.svg)

The full transformer stack:
- Token embedding → Position encoding
- 4 transformer blocks (multi-head causal self-attention + feed-forward)
- Output projection to vocabulary
- C99 inference engine with mmap weight loading

## Dev Quick Start

```bash
source venv/bin/activate
pytest -q

# Train on jot syntax corpus (100 epochs ~ 5 min)
python src/train.py --corpus jot --epochs 100

# Train on WikiText-103 (200 epochs ~ 45 min)
python src/train.py --corpus wiki --tokenizer bpe --epochs 200 --model-size mini

# Generate text
python src/generate.py --prompt "fn " --length 80 --temperature 0.3

# Run tests
pytest -v
```

## Inference (C Engine)

```bash
# Export checkpoint to binary
python scripts/export_weights.py

# Build C engine (requires C99 compiler only)
cd inference && make

# Run (instant startup via mmap)
./inference/aether models/aether.bin "Q: What is 5+3?\nA:" --temp 0.3 --tokens 50
```

## Model Tiers

| Tier  | Params  | Layers | Embed | Context | Binary |
|-------|---------|--------|-------|---------|--------|
| Nano  | ~15K    | 2      | 32    | 64      | ~60 KB |
| Micro | ~630K   | 4      | 128   | 128     | ~2 MB  |
| Mini  | ~3.5M   | 6      | 256   | 512     | ~14 MB |
| Small | ~14M    | 8      | 512   | 1024    | ~56 MB |

## Benchmarks

### Speed (M4 Mac)
- aether C engine: **50,000 tok/s** (mmap startup: <1ms)
- qwen3:14b: 30 tok/s
- Claude API: 80 tok/s

### Capability Comparison

| Model | Params | Size | Training Data | What It Can Do |
|-------|--------|------|---|---|
| **aether (current)** | 0.57M | 2.2 MB | 185 KB jot corpus | jot autocomplete (barely) |
| **aether (wiki planned)** | 14M | 56 MB | 103M tokens (WikiText-103) | grammatical English (TBD) |
| GPT-2 Small | 124M | 500 MB | 40 GB WebText | coherent paragraphs |
| qwen3:14b | 14B | 8 GB | trillions of tokens | code, reasoning, tools |
| Claude Opus | ???B | ??? | the internet | build entire apps |

### Why Aether Exists

This is not trying to compete with Claude or GPT. The point:

1. **"What I cannot create, I do not understand."** Feynman. Building from scratch teaches you what all those papers actually mean.
2. **C99 inference engine**: 350 LOC, zero dependencies, mmap weight loading. Runs anywhere with a C compiler.
3. **The full stack**: tokenizer → attention → transformer → training → checkpoint → binary export → C inference. No black boxes.

### Real Talk: The Gap

To get from aether to something useful:

| Capability | What It Takes |
|-----------|---------------|
| Coherent paragraphs | ~100M params, ~10B tokens, ~$100 in compute |
| Follow instructions | Instruction tuning dataset + RLHF, ~1B params minimum |
| Code generation | Code-specific training data, ~7B params minimum |
| Reasoning | Chain-of-thought training, ~13B+ params |
| Tool use (like Claude) | Function calling training, massive compute, months of RLHF |

**In perspective:** Claude just orchestrated 7 parallel agents to build apps, rename projects, and overhaul pipelines—in one session. aether can autocomplete `fn ` with some curly braces. That gap is billions of dollars in compute and thousands of researcher-years.

## Training + Aether Daemon

```bash
# Start background training with milestone pings
~/.local/bin/aether &

# Monitor
tail -f ~/.cache/aether.log
cat ~/.cache/aether-milestones.log
```

**Aether daemon features:**
- 200 epochs per cycle, 5 min sleeps
- RAM guard (backs off if > 75%)
- iMessage notifications every 30 min
- Milestone pings: 100, 500, 1K, 5K, 10K epochs
- Auto-commit and push

## Web UI

```bash
python index.py
# Open http://localhost:5001
```

Modes: chat, quiz, status

## Roadmap

# Aether Training Roadmap

## Overview

Aether is a nano transformer LLM built from scratch. This roadmap outlines the path from current (~600K params) to useful (~14M params).

## Phases

### Phase 1: Foundation (Complete)
**Goal:** Full stack from scratch with C inference.

- [x] Char-level tokenizer (jot syntax)
- [x] Multi-head self-attention (fused QKV, causal masking)
- [x] Transformer blocks (pre-norm, GELU, residual connections)
- [x] Training loop (Adam, gradient clipping, checkpointing)
- [x] C99 inference engine (350 LOC, mmap, zero deps)
- [x] Weight export (PyTorch to binary aether.bin)
- [x] Evaluation harness (grade A-F, per-category scores)
- [x] Flask web UI (chat + quiz + status)
- [x] Aether daemon (overnight training, milestone notifications)

### Phase 2: Data Quality (In Progress)
**Goal:** Scale to meaningful model size with real-world training data.

**Timeline: 1-2 weeks**

- [ ] Download WikiText-103 (103M tokens, standard benchmark)
- [ ] Implement BPE tokenizer (32K vocabulary)
- [ ] Train mini config (3.5M params) on WikiText-103
- [ ] Establish baseline: perplexity < 50 vs GPT-2 (29)
- [ ] Validation curve tracking (loss plots, eval every N epochs)
- [ ] Checkpoint management (save best, resume from checkpoint)

**Measurement:** Perplexity on held-out validation set

### Phase 3: Scale (Coming)
**Goal:** Multi-GPU training, larger models, faster iteration.

**Timeline: 2-4 weeks**

- [ ] Small config (14M params, 8 layers, 512 embed)
- [ ] Distributed training (torch.distributed, multi-GPU)
- [ ] KV-cache optimization (faster generation)
- [ ] Quantization (INT8, INT4) — reduce binary size 4-8x
- [ ] Benchmark vs Llama 1B, GPT-2 medium on standard evals

**Success criteria:** 14M model runs inference in <100ms on M4

### Phase 4: Instructions (Coming)
**Goal:** Teach the model to follow commands (like Claude, GPT).

**Timeline: 3-6 weeks (if Phase 3 succeeds)**

- [ ] Collect instruction dataset (Alpaca, jot/jit code examples)
- [ ] Instruction-tuning training loop
- [ ] System prompt support
- [ ] Chat template (Q&A formatting)
- [ ] Evaluation on instruction-following tasks

**Success criteria:** Model responds meaningfully to "write a function", "explain", etc.

### Phase 5: Production (Stretch)
**Goal:** Deploy-ready LLM inference.

**Timeline: 2-3 months (if previous phases ship)**

- [ ] ONNX export (run on CPU, mobile, browser)
- [ ] WebAssembly inference (browser-native)
- [ ] HTTP API server (FastAPI)
- [ ] Docker containerization
- [ ] Benchmark on iPhone/Android (edge inference)
- [ ] Fine-tuning service (users can adapt to custom domains)

**Success criteria:** aether runs locally on phone with < 5s first-token latency

---

## Current Bottleneck: WikiText-103 Data

Aether trains fast on tiny corpora (jot syntax, 185 KB, 30 min for 200 epochs).
Real learning needs:

1. **Quantity:** WikiText-103 (103M tokens vs 185K now = 550x larger)
2. **Diversity:** English, code, facts, reasoning
3. **Duration:** 20-50 hours of training to reach perplexity < 50

**aether daemon solves this:** Run overnight, commit every cycle, track progress.

---

## Success Metrics

| Phase | Metric | Target | Status |
|-------|--------|--------|--------|
| 1 | C inference speed | 50K tok/s | Done (50K tok/s) |
| 1 | Binary size | < 3 MB | Done (2.2 MB) |
| 2 | Perplexity (wiki) | < 50 | TBD (training) |
| 2 | Training speed | > 1K tok/s | TBD |
| 3 | Model size | 14M params | TBD |
| 3 | Inference latency | < 100ms | TBD |
| 4 | Instruction accuracy | > 50% | TBD |
| 5 | Mobile latency | < 5s first-token | TBD |

---

## Parallel Work

While aether trains:

1. **Systems monorepo:** Continue nullC, NullOS, shell work
2. **Opticon:** Polish markets, portfolio features
3. **Spark:** Growth, engagement features
4. **Tally:** New benefit integrations
5. **Portfolio:** Blog posts, case studies on aether building process

aether runs in the background. Check progress every 30 min via iMessage.

---

## Why This Matters

By the end of Phase 2, you'll have a 3.5M param language model that:
- Generates grammatical English sentences
- Understands basic facts and math
- Runs inference in milliseconds
- Requires no GPU (CPU + C engine)
- Costs $0 to run (no API calls)

By Phase 4, a system capable of following instructions — nothing special, but yours, end-to-end, ground truth.

By Phase 5, inference anywhere: laptop, phone, browser, server.

That's the goal.


## Testing

```bash
pytest -q                   # Quick
pytest -v                   # Verbose
pytest -k "tokenizer"       # By name
```

All tests deterministic, seeded, CI-verified before merge.

## License

MIT 2026, Joshua Trommel

---

## Changelog

### Fri Feb 27 2026 - Training Complete

**Project Rename & Documentation:**
- Renamed: core → aether
- Soul.md → SOUL.md (uppercase)
- WHITEPAPER.md updated with aether branding and corpus references
- README merged with full BENCHMARK and CHANGELOG content
- Removed standalone BENCHMARK.md and CHANGELOG.md (consolidated into README)

**Training Optimization:**
- Batch size reduced: (4→2, 8→4) for stability and memory efficiency
- Environment throttling: TORCH_NUM_THREADS=2, OMP_NUM_THREADS=2
- Training completed: 200 epochs, final loss 0.2-0.9
- Peak memory: 299MB resident (stable throughout)
- Training time: ~13 hours on M4 Mac

**Key Metrics:**
- Starting loss (epoch 1): 2.2
- Mid-training (epoch 100): ~1.0
- Final loss (epoch 200): 0.2-0.9
- Loss convergence: smooth and healthy
- No training instabilities or divergence

**Infrastructure:**
- Modified src/train.py with CPU threading limits
- Added run_throttled.py wrapper (attempted throttling, PyTorch limitation)
- Log format: clean, unbuffered output
- Checkpoint saved to models/overnight_best.pt

**Commits:**
- 34f0950: Rename Soul.md to SOUL.md
- 90afb27: Update WHITEPAPER with aether branding
- 901f04e: Merge CHANGELOG and BENCHMARK into README
- f8c3214: Training complete, final push

---

## Project Philosophy

"What I cannot create, I do not understand." — Feynman

aether is the full stack from scratch: tokenizer → attention → transformer → training loop → checkpoint → C99 inference engine. Every byte visible. The goal is to understand the machine top to bottom without black boxes.

**Why aether exists:**
1. Learning by building—transformers from first principles
2. C99 inference engine (350 LOC, zero deps, mmap, instant startup)
3. Honest benchmarks—no pretense about capability
4. Full pipeline ownership—train, export, deploy, understand

