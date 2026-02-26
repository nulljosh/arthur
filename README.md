# core

Tiny transformer LM built from scratch in PyTorch, with a zero-dependency C inference engine. 230K parameters, character-level tokenizer, trained on mixed Q&A corpus.

## Current Status

- C inference engine (zero dependencies, ~350 lines, mmap weight loading)
- Persistent training daemon (`core-train`) with RAM guard + iMessage notifications
- Web chat UI with Apple Liquid Glass design + dark mode
- Auto-deploys overnight best checkpoint

## What It Knows

```
Q: What is 7*8?       -> 56
Q: What is your name? -> core
Q: Who made you?      -> Josh made me
Q: print hello world  -> print "Hello, World!";
Q: write a function   -> fn add(a, b) { return a + b; }
```

## Quick Start

```bash
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

pytest -q
python3 scripts/train_overnight_mixed.py
```

## C Inference Engine

Native C99 inference -- no Python, no PyTorch, no dependencies. Loads weights via mmap for instant startup.

### Export Weights

```bash
source venv/bin/activate
python3 scripts/export_weights.py
# writes models/core.bin (~2.3 MB)
```

### Build

```bash
cd inference && make
```

### Run

```bash
./inference/core models/core.bin "Q: What is 5+3?\nA:" --temp 0.0 --tokens 50
./inference/core models/core.bin "Q: Who made you?\nA:" --temp 0.5
./inference/core --help
```

Options: `--temp T` (sampling temperature, default 0.8, 0 = greedy), `--tokens N` (max generation length, default 256).

## Web UI

```bash
python3 web_ui.py
```

- http://localhost:5001/ -- chat interface (Liquid Glass, dark mode toggle)
- http://localhost:5001/quiz -- quiz mode
- `POST /api/generate` | `GET /api/status`

Runs as persistent launchd service (`com.joshua.core-web`) on port 5001.

## Architecture

![core architecture](architecture.svg)

Same architecture as GPT-2 (~2500x smaller). Implements "Attention Is All You Need" in ~500 lines.

**Model:** 4 transformer blocks, 4 attention heads, 128 embed dim, 256 FF dim, 128 max sequence length. ~230K params.

**Tokenizer:** Character-level (102 unique characters from corpus). Word-level and BPE also implemented.

**Training:** AdamW optimizer, cosine annealing LR (1e-3 to 1e-5), gradient clipping at 1.0, batch size 32, sequence length 64. Persistent daemon with RAM monitoring and iMessage status notifications.

**C Engine:** Single-file C99 inference (~350 LOC). mmap weight loading, GELU tanh approximation, causal multi-head attention, temperature sampling, sliding context window.

**Corpus:** 185 KB mixed data across 9 files: comprehensive Q&A (math, identity, jot, facts, time/date), jot code examples, math drills, conversational pairs.

### Model Tiers

| Tier | Params | Layers | Heads | Embed | Context |
|------|--------|--------|-------|-------|---------|
| Nano | ~50K | 2 | 2 | 32 | 64 |
| Micro | ~500K | 4 | 4 | 128 | 256 |
| Mini | ~5M | 6 | 8 | 256 | 512 |

For comparison: GPT-2 has 124M params, 12 layers, 12 heads, 768 embed, 1024 context.

## Services

| Service | Plist | Port | Purpose |
|---------|-------|------|---------|
| core-train | `com.joshua.core-train` | -- | Training daemon (persistent, RAM-aware) |
| core-web | `com.joshua.core-web` | 5001 | Flask chat UI |

## Evaluation

```bash
python3 evaluate_checkpoints.py --checkpoints models/overnight_best.pt
```

Eval results write to `logs/eval_results.json` (consumed by fony morning call for daily grade report).

## Project Layout

```
src/tokenizer.py       char/word/BPE tokenizers
src/attention.py       self-attention + multi-head
src/transformer.py     blocks + full model
src/train.py           dataset, train loop, generate
src/chat.py            interactive chat interface
src/eval_harness.py    prompt suite evaluation
inference/core.c       C99 inference engine (single file)
inference/Makefile     build config
scripts/export_weights.py  PyTorch -> binary weight exporter
train_overnight_mixed.py   overnight training script
web_ui.py              Flask web UI + API
templates/index.html   Liquid Glass chat interface
data/                  training corpora (9 files)
models/                checkpoints (.pt) + core.bin (C engine weights)
logs/                  training logs + eval results
tests/                 pytest suite (8 files)
```

## Version

2.0.0

## Notes

Educational project. For production, scale model/data/training significantly.
