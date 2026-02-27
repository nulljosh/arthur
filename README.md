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

See **ROADMAP.md** for detailed phases and timelines.

### Done
- [x] Char tokenizer + transformer
- [x] C99 inference engine (350 LOC, mmap, zero deps)
- [x] Training + eval harness
- [x] Flask UI
- [x] Aether daemon + notifications
- [x] BENCHMARK.md (honest comparisons)
- [x] ROADMAP.md (5 phases, success metrics)

### In Progress (Phase 2: Data Quality)
- [ ] WikiText-103 BPE training (running via aether daemon)
- [ ] Scale to mini/small (3.5M-14M params)
- [ ] Perplexity tracking vs GPT-2

### Coming (Phase 3: Scale)
- [ ] Top-k / top-p sampling
- [ ] Beam search
- [ ] INT8 quantization
- [ ] Multi-GPU training
- [ ] Fine-tuning on jot/jit syntax

### Stretch (Phase 4-5: Instructions & Production)
- [ ] Instruction tuning (Alpaca dataset)
- [ ] System prompts + chat format
- [ ] ONNX export (mobile/WASM)
- [ ] HTTP inference API
- [ ] KV-cache (faster streaming)

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

