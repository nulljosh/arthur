# aether

Nano transformer LLM built from scratch. Multi-language + knowledge corpus training. 500 epochs on 27MB code + expanding to 100MB knowledge corpus.

![aether architecture diagram](architecture.svg)

## Quick Start

```bash
# Train on multilang code
python src/train.py --corpus tiny --epochs 500

# Train on expanded knowledge
python src/train.py --corpus tiny --epochs 1000

# Inference (C engine, 50K tok/s)
./inference/aether models/aether.bin "The future of AI" --temp 0.3

# Web UI
python index.py  # http://localhost:5001
```

## The Stack

- **PyTorch trainer** — AdamW, cosine LR, gradient clipping, checkpoint management
- **C99 inference** — 350 LOC, mmap weight loading, zero deps
- **Flask UI** — chat, quiz, status modes on :5001
- **Aether daemon** — continuous training, milestone pings, auto-commit
- **Multilang tokenizer** — char-level, all programming languages + natural language
- **Model tiers** — Nano (15K) → Micro (630K) → Mini (3.5M) → Small (14M)

## Training Phases

### Phase 1: Foundation ✅
**Goal**: Full stack implementation with working inference engine
- [x] Tokenizer + transformer (4-layer, 0.57M params)
- [x] C99 inference (350 LOC, zero deps, mmap)
- [x] Training harness + Flask UI
- [x] Aether daemon + iMessage notifications
- [x] Full test suite + benchmarks

**Result**: Stable training, 50K tok/s inference speed

### Phase 2: Code Understanding (In Progress) ✅
**Goal**: Learn to understand and generate code across multiple languages
- [x] jot corpus training (100 epochs, syntax focus)
- [x] jung corpus training (100 epochs, JIT compiler)
- [x] Multilang training (500 epochs, 27MB: C99/Rust/Python/Go/shell)
- [x] Final loss: 0.0947 (strong convergence)

**Result**: Model learns cross-language patterns, function/type awareness

### Phase 3: Knowledge Expansion (Starting) 🚀
**Goal**: Teach model general knowledge, reasoning, and writing
- [ ] WikiText-103 corpus (103M tokens, encyclopedia knowledge)
- [ ] ArXiv papers (math, science, AI research)
- [ ] News corpus (politics, current events, analysis)
- [ ] Science corpus (biology, physics, chemistry, astronomy)
- [ ] Combined: ~100MB knowledge corpus
- [ ] Target: 1000 epochs on expanded corpus
- [ ] Expected loss: <0.05 (2-3 days on M4 Mac)

**Goals**:
- Coherent paragraph generation
- Basic reasoning and explanation
- Cross-domain knowledge integration
- Improved language understanding

### Phase 4: Scale to Mini (Future)
**Goal**: Increase model capacity for better learning
- [ ] Scale params from 0.57M → 3.5M (6 layers → 12 layers)
- [ ] Increase embedding dim (128 → 256)
- [ ] Larger context window (512 → 1024 tokens)
- [ ] Batch size optimization for M4 performance
- [ ] Re-train on all corpora (jot + jung + code + knowledge)

**Target**: 
- Better pattern recognition
- Longer dependency handling
- More nuanced output generation

### Phase 5: Production Ready (Later)
**Goal**: Deploy as working service
- [ ] ONNX export for mobile/browser
- [ ] HTTP inference API
- [ ] Quantization (int8/int4)
- [ ] Batch inference
- [ ] Web dashboard + analytics
- [ ] Model versioning + rollback

**Target**:
- <100ms latency
- Mobile-compatible inference
- Real-time chat API

## Training Results

### Current: v0.3.0 (Multilang Code)
- **Corpora**: jot, jung, 4414 multilang files
- **Size**: 27MB code (C99, Rust, Python, Go, jit, shell, markdown)
- **Epochs**: 500
- **Final Loss**: 0.0947
- **Training Time**: ~4 hours on M4 Mac
- **Status**: ✅ Complete, ready for knowledge expansion

### Planned: v0.4.0 (Knowledge Expansion)
- **Corpora**: WikiText-103, ArXiv papers, news, science
- **Size**: ~100MB (code + knowledge)
- **Epochs**: 1000
- **Expected Loss**: <0.05
- **Training Time**: 2-3 days on M4 Mac
- **Status**: 🚀 Starting now

### Future: v1.0.0 (Mini Scale)
- **Model**: 3.5M params (12 layers)
- **Training**: 2000+ epochs on full corpus
- **Expected Loss**: <0.03
- **Capability**: Coherent paragraphs, reasoning

## Benchmarks

| Version | Params | Corpus Size | Epochs | Loss | Speed | Capability |
|---------|--------|------------|--------|------|-------|---|
| v0.1 (jot) | 0.57M | 185 KB | 200 | 0.2-0.9 | 50K tok/s | jot syntax |
| v0.2 (jung) | 0.57M | 31 KB | 100 | 0.17 | 50K tok/s | syntax focus |
| **v0.3 (code)** | **0.57M** | **27 MB** | **500** | **0.0947** | **50K tok/s** | **multilang code** |
| v0.4 (knowledge) | 0.57M | 100 MB | 1000 | <0.05 | 50K tok/s | knowledge + code |
| v1.0 (mini) | 3.5M | 100 MB | 2000+ | <0.03 | ~20K tok/s | coherent paragraphs |
| GPT-2 | 124M | 40 GB | — | — | — | paragraphs |
| Claude | ??? | internet | — | — | 80 tok/s | reasoning |

## Why Aether

"What I cannot create, I do not understand." — Feynman

- Full stack from scratch: tokenizer → attention → training → C inference
- No black boxes. Every byte visible and understandable.
- Learning tool first, production model later
- C99 engine runs anywhere with a C compiler
- Progressive training shows how LLMs learn from data

## Architecture

```
Data (code + knowledge)
  ↓
PyTorch Trainer (1000+ epochs)
  ↓
Checkpoint (.pt)
  ├→ Export (aether.bin)
  │   ↓
  │   C Inference (350 LOC, mmap)
  │
  ├→ Flask Web UI (:5001)
  │   ├→ Chat mode
  │   ├→ Quiz mode
  │   └→ Status dashboard
  │
  └→ Aether Daemon
      ├→ Continuous training
      ├→ Auto-checkpoint
      ├→ iMessage notifications
      └→ Milestone tracking
```

## Status

**Latest**: v0.3.0 — Multilang code training complete (500 epochs, loss 0.0947)

**Next**: v0.4.0 — Knowledge expansion training (1000 epochs, WikiText + ArXiv + news + science)

**ETA**: 2-3 days (starting now)

## License

MIT 2026, Joshua Trommel
