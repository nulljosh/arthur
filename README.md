# aether

Nano transformer LLM built from scratch. PyTorch training loop, C99 inference engine, overnight automation.

**GitHub**: https://github.com/nulljosh/nous | **Benchmarks**: See BENCHMARK.md

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

See **BENCHMARK.md** for full analysis.

**Speed (M4 Mac):**
- aether C: 50,000 tok/s (mmap startup: <1ms)
- qwen3:14b: 30 tok/s
- Claude API: 80 tok/s

**Capability Gap:**

| Model | Size | What It Can Do |
|-------|------|---|
| aether (0.57M) | 2.2 MB | jot autocomplete (barely) |
| aether (wiki, 14M) | 56 MB | grammatical English (training...) |
| GPT-2 (124M) | 500 MB | coherent paragraphs |
| qwen3:14b | 8 GB | code, reasoning, tools |
| Claude Opus | ??? GB | build entire apps |

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

### ✓ Done
- [x] Char tokenizer + transformer
- [x] C99 inference engine (350 LOC, mmap, zero deps)
- [x] Training + eval harness
- [x] Flask UI
- [x] Aether daemon + notifications
- [x] BENCHMARK.md (honest comparisons)
- [x] ROADMAP.md (5 phases, success metrics)

### 🔄 In Progress (Phase 2: Data Quality)
- [ ] WikiText-103 BPE training (running via aether daemon)
- [ ] Scale to mini/small (3.5M-14M params)
- [ ] Perplexity tracking vs GPT-2

### 📋 Coming (Phase 3: Scale)
- [ ] Top-k / top-p sampling
- [ ] Beam search
- [ ] INT8 quantization
- [ ] Multi-GPU training
- [ ] Fine-tuning on jot/jit syntax

### 🚀 Stretch (Phase 4-5: Instructions & Production)
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
