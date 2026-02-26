# core

Tiny transformer LM built from scratch in PyTorch. 230K parameters, character-level tokenizer, trained on mixed Q&A corpus.

## Current Status

- **Epoch 163/200** | Loss: 0.186 | Grade: **A (90.2)**
- Math: 87.8% | Identity: 88.3% | Jot code: 95.3%
- Factual knowledge (president, country, year): training data added, next run will learn

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

## Architecture

![core architecture](architecture.svg)

**Model:** 4 transformer blocks, 4 attention heads, 128 embed dim, 256 FF dim, 128 max sequence length.

**Tokenizer:** Character-level (102 unique characters from corpus).

**Training:** AdamW optimizer, cosine annealing LR (1e-3 to 1e-5), gradient clipping at 1.0, batch size 32, sequence length 64.

**Corpus:** 185 KB mixed data across 9 files: comprehensive Q&A (math, identity, jot, facts, time/date), jot code examples, math drills, conversational pairs.

## Web UI

```bash
python3 web_ui.py
```

- http://localhost:5001/ (prompt interface)
- http://localhost:5001/quiz (quiz mode)
- `POST /api/generate` | `GET /api/status`

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
web_ui.py              Flask web UI
data/                  training corpora (9 files)
models/                checkpoints (.pt)
logs/                  training logs + eval results
tests/                 pytest suite (8 files)
```

## Notes

Educational project. For production, scale model/data/training significantly.
