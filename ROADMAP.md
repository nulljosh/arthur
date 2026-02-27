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
