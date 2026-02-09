# microLM

A minimal LLM (Large Language Model) built from scratch in Python. Educational project to understand transformer architecture.

![Project Map](map.svg)

## Status
Foundation stage - implementing tokenization and attention mechanisms

## Goals
- Tokenize text (BPE/WordPiece)
- Build transformer architecture (attention, feed-forward, layernorm)
- Train on small corpus
- Generate coherent text

## Build
```bash
cd microLM
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

## Documentation
- [ROADMAP.md](ROADMAP.md) - Development phases
- [BENCHMARKS.md](BENCHMARKS.md) - Complexity tiers
- [ARCHITECTURE.md](ARCHITECTURE.md) - Model design

## Author
Joshua Trommel (nulljosh)
