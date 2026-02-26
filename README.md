# core

Tiny transformer LM built from scratch in PyTorch. 230K parameters, character-level tokenizer, trained on mixed Q&A corpus.

## Current Status

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
python3 train_overnight_mixed.py
```

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

**Model:** 4 transformer blocks, 4 attention heads, 128 embed dim, 256 FF dim, 128 max sequence length.

**Tokenizer:** Character-level (102 unique characters from corpus).

**Training:** AdamW optimizer, cosine annealing LR (1e-3 to 1e-5), gradient clipping at 1.0, batch size 32, sequence length 64. Persistent daemon with RAM monitoring and iMessage status notifications.

**Corpus:** 185 KB mixed data across 9 files: comprehensive Q&A (math, identity, jot, facts, time/date), jot code examples, math drills, conversational pairs.

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
train_overnight_mixed.py   overnight training script
web_ui.py              Flask web UI + API
templates/index.html   Liquid Glass chat interface
data/                  training corpora (9 files)
models/                checkpoints (.pt)
logs/                  training logs + eval results
tests/                 pytest suite (8 files)
```

## Notes

Educational project. For production, scale model/data/training significantly.
