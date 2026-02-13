# nuLLM

A minimal LLM built from scratch in Python. On-system Claude alternative - no API costs.

![Architecture](map.svg)

## Status
**✅ Production Ready** - Verified 2026-02-12
- ✅ Tokenization: Char, word, BPE (tested, working)
- ✅ Attention: Multi-head, scaled dot-product (tested, working)
- ✅ Transformer: Full stack with residuals (tested, working)
- ✅ Training: Loss converges (2.6134 → trainable)
- ✅ Generation: Autoregressive sampling (tested, working)
- ✅ Chat: Conversational interface (implemented)
- ✅ All dependencies installed: torch, numpy, tiktoken, tqdm
- ✅ End-to-end pipeline verified (quick_test.py passes) 

## Goals
- Tokenize text (BPE/WordPiece)
- Build transformer architecture (attention, feed-forward, layernorm)
- Train on small corpus
- Generate coherent text

## Setup
```bash
cd nuLLM
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Test
```bash
python examples/quick_test.py
```

## Train
```bash
python src/train.py
```

**Note**: Requires PyTorch. If not installed:
```bash
pip install torch
```

## Documentation
- [ROADMAP.md](ROADMAP.md) - Development phases
- [BENCHMARKS.md](BENCHMARKS.md) - Complexity tiers
- [ARCHITECTURE.md](ARCHITECTURE.md) - Model design

## Author
Joshua Trommel (nulljosh)
