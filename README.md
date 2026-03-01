# Arthur

## Training Status

**Progress:** Epoch 0/50 (0% complete)
**Latest Checkpoint:** None
**Last Loss:** N/A
**Updated:** 2026-02-28 15:56

Status: Daemon auto-training when idle. Respects resources (disk <5GB, CPU <70%, RAM >4GB).

# Arthur

A small language model built from scratch. 3.5M parameters, trained on math and knowledge. **Research prototype** — production improvements in progress.

## Status

**Current:** Grade A+ (loss 0.18), 31.2% benchmark accuracy (math/science/pop-culture/current-events)

**Verdict:** Memorization-based learner. Excels on exact training phrases, fails on generalizations and arithmetic reasoning.

## Architecture

**Model Specifications:**
- **Total Parameters**: 3.5M
- **Layers**: 3 transformer blocks
- **Attention Heads**: 4 per block
- **Feed-Forward Dimension**: 256
- **Embedding Dimension**: 128
- **Tokenizer**: Character-level (91 unique characters)
- **Training Data**: Math (50%) + Wikipedia (50%)
- **Training Loss**: 0.18 (Grade A+)

## Getting Started

```bash
git clone https://github.com/nulljosh/arthur.git && cd arthur
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Local inference (requires PyTorch)
python web_ui.py
# Visit http://localhost:5001

# C99 inference engine (requires weight export)
cd inference && make && ./nous ../models/arthur.bin "Q: What is " --temp 0.3
```

## Inference Results

### Test Suite (16 questions)
| Category | Accuracy | Notes |
|----------|----------|-------|
| Math | 0% | 0/4 — fails on arithmetic despite training data |
| Science | 67% | 4/6 — exact phrase matching, fails on variations |
| Pop Culture | 0% | 0/2 — tokenization issues ("Paris"→"Pris") |
| Current Events | 25% | 1/4 — pattern hallucination to cryptocurrency |
| **OVERALL** | **31.2%** | 5/16 — pure memorization, zero generalization |

### Key Findings
- ✅ Perfect recall on training examples (100% confidence)
- ❌ Zero generalization on variations
- ❌ Complete failure on arithmetic reasoning
- ❌ Character-level tokenizer causes corruption
- ❌ Hallucinates cryptocurrency when uncertain

## How Arthur Compares

Arthur is a 3.5M parameter educational model. For context, here is how it stacks up against frontier models with 14B to trillions of parameters:

| Benchmark | Arthur (3.5M) | Qwen3-14B | Claude Sonnet 4.6 | Claude Opus 4.6 |
|-----------|--------------|-----------|-------------------|-----------------|
| MMLU | N/A | ~81% | ~89% | ~91% |
| GSM8K | 0% | ~92% | ~96% | ~99% |
| HumanEval | N/A | ~72% | ~92% | ~95% |
| MATH | 0% | ~62% | ~85% | ~93% |
| Custom (16Q) | 31.2% | N/A | N/A | N/A |

The custom benchmark tests 16 questions across math, science, pop culture, and current events using Arthur's training data format.

## Roadmap to Production (6-8 weeks)

### Phase 1: Data (2-3 weeks)
- [ ] Expand training data: 1.8M → 50M+ tokens
  - WikiText-103 (1.5M examples, in progress)
  - ArXiv papers (500K examples)
  - Code (GitHub, 300K examples)
  - Current events + knowledge base
- [ ] Implement BPE tokenizer (vocab 10K) to replace char-level
- [ ] Balance datasets: math 5%, knowledge 70%, reasoning 15%, code 10%

### Phase 2: Model (1-2 weeks)
- [ ] Scale architecture: 3.5M → 50M parameters
  - 12 transformer layers (from 3)
  - 8 attention heads (from 4)
  - 2048 FF dim (from 256)
  - 512-dim embeddings (from 128)
- [ ] Implement Flash Attention v2 (training 2-3x faster)
- [ ] Add position interpolation for 8K context (from 512)
- [ ] Implement gradient checkpointing (reduce memory)

### Phase 3: Training (3-4 weeks GPU time)
- [ ] Pre-train on full dataset (100+ GPU hours)
  - Target loss: <0.05
  - Cosine LR schedule with warmup
  - AdamW + gradient clipping
- [ ] Fine-tune on math + reasoning (specialized dataset)
  - MATH-500, GSM8K, SVAMP
  - Target: >80% on arithmetic
- [ ] Implement RLHF with human feedback
  - Compare outputs, rank by quality
  - Train reward model
  - Policy optimization

### Phase 4: Inference (1 week)
- [ ] Quantize to INT8/FP8 (50M → 15GB model)
- [ ] Export weights to C99 binary format
- [ ] Production API with KV caching
- [ ] Benchmark throughput (target: 100+ tok/s on M4)

## Files

- `src/train.py` – Training loop with cron scheduler
- `src/transformer.py` – Model architecture
- `src/tokenizer.py` – Character-level tokenizer (to be replaced)
- `src/bpe_tokenizer.py` – BPE tokenizer (planned)
- `web_ui.py` – Chat interface
- `models/` – Saved checkpoints
- `data/` – Training datasets
- `inference/nous.c` – C99 inference engine

## Why Build It

Understanding language models means building one from scratch. This project shows:
- How transformers work (attention, feed-forward, embeddings)
- How to train a model end-to-end (data → training → inference)
- Where small models fail (memorization, generalization gap)
- What production requires (scale, data quality, inference optimization)

## Next Steps

**For researchers:** Fork this repo, experiment with larger models and better data. The architecture is minimal but correct.

**For production use:** Wait for Phase 4. Current version is a teaching tool.

## Links

- GitHub: https://github.com/nulljosh/arthur
- Vercel: https://arthur.vercel.app (static splash + mock API)
- Chat: http://localhost:5001 (local only)

## License

MIT 2026 Joshua Trommel
