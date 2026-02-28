# Arthur v2.0 Roadmap

Production-ready language model. Currently at research prototype stage (3.5M params, 31% accuracy). Scaling to 50M params with 6-8 week development plan.

## Current Status

- **Training Loss:** 0.18 (Grade A+)
- **Benchmark Accuracy:** 31.2% (5/16 questions)
  - Math: 0% (complete failure)
  - Science: 67% (exact phrase matching)
  - Pop culture: 0% (tokenization corruption)
  - Current events: 25% (hallucination)
- **Issue:** Pure memorization, no generalization
- **Root Cause:** Character-level tokenizer + small dataset + tiny model

## Production Plan (6-8 weeks)

### Phase 1: Data (2-3 weeks)
**Goal:** 1.8M → 50M+ training tokens with balanced diversity

- Expand WikiText-103 (already downloaded, 1.5M examples)
- Add ArXiv papers (500K examples)
- Add GitHub code (300K examples, publicly available)
- Add news + current events (200K examples)
- Curate math reasoning dataset (MATH-500, GSM8K, SVAMP)

**Tokenizer upgrade:** Char-level → BPE (vocab 10K)
- Reduces tokenization errors ("Paris" no longer becomes "Pris")
- Enables subword reasoning
- More efficient encoding

**Dataset balance:**
- Math: 5% (focused fine-tuning later)
- Knowledge/reference: 70%
- Code: 10%
- Reasoning: 15%

### Phase 2: Model Architecture (1-2 weeks)
**Goal:** 3.5M → 50M parameters with modern efficiency

Current architecture:
```
3 layers × 4 heads × 128 embed × 256 FF
= 3.5M params
```

Target architecture:
```
12 layers × 8 heads × 512 embed × 2048 FF
= 50M params (14x scale)
```

Optimizations:
- **Flash Attention v2** — 2-3x training speed, same quality
- **Gradient checkpointing** — trade compute for memory
- **Position interpolation** — extend context 512 → 8K without retraining embeddings

### Phase 3: Training (3-4 weeks GPU time)
**Goal:** Pre-trained model with strong generalization

Pre-training phase:
- 100+ GPU hours (A100 or better)
- Full 50M token dataset
- Cosine learning rate decay with warmup
- Target: loss < 0.05

Math fine-tuning phase:
- MATH-500 (500 competition problems)
- GSM8K (8.5K grade school math)
- SVAMP (1K word problems)
- Few-shot in-context learning
- Target: >80% on arithmetic

RLHF phase (optional):
- Human preference data (compare outputs)
- Train reward model
- PPO policy optimization
- 10% compute overhead, 15-20% accuracy boost

### Phase 4: Deployment (1 week)
**Goal:** Production-grade inference

Quantization:
- INT8 or FP8 quantization
- 50M → 15GB model size
- <2% accuracy loss
- 2x inference speed

C99 inference engine:
- Export weights from PyTorch
- Load quantized model zero-copy
- Memory-mapped weights
- Target: 100+ tokens/sec on M4

API deployment:
- KV cache for chat (reduce recompute)
- Batch inference for throughput
- Rate limiting + monitoring
- Vercel serverless (if <15GB) or self-hosted

## Success Metrics

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Model Size | 3.5M | 50M | Phase 2 |
| Training Data | 1.8M tokens | 50M tokens | Phase 1 |
| Tokenizer | Char (91) | BPE (10K) | Phase 1 |
| Math Accuracy | 0% | >80% | Phase 3 |
| Science Accuracy | 67% | >85% | Phase 3 |
| Context Length | 512 | 8K | Phase 2 |
| Inference Speed | N/A (local) | 100+ tok/s | Phase 4 |
| Latency (P99) | N/A | <500ms | Phase 4 |

## Resource Requirements

- **Compute:** 100-200 GPU hours (A100 or H100)
  - Cost: $500-1500 on Lambda Labs / Together.ai
- **Storage:** 50GB for datasets, 30GB for checkpoints
- **Time:** 6-8 weeks (1 engineer, part-time)

## Estimated Timeline

- **Week 1-2:** Data collection, BPE tokenizer implementation
- **Week 3-4:** Model architecture refactor, Flash Attention integration
- **Week 5-8:** Pre-training (3-4 weeks GPU time in parallel with fine-tuning setup)
- **Week 9:** Math fine-tuning + RLHF (if doing)
- **Week 10:** Quantization, C99 export, deployment

## Known Limitations (Current v1)

1. **Memorization only** — Model doesn't learn patterns, just memorizes
2. **Character-level tokenization** — Causes corruption on unknown words
3. **Tiny training set** — 1.8M tokens is 100x less than industry standard
4. **Small model** — 3.5M params can't hold much knowledge
5. **No reasoning** — Pure next-token prediction, no planning

## Why This Approach

- **Transparency:** Every step documented and reproducible
- **Simplicity:** No complex libraries, just PyTorch + standard techniques
- **Learning:** Build understanding, not just use APIs
- **Cost:** 50M model is still feasible on consumer hardware
- **Deployment:** Fits on Vercel + can be self-hosted

## References

- Attention Is All You Need (Vaswani et al., 2017)
- Language Models are Unsupervised Multitask Learners (Radford et al., 2019)
- Flash-2 Attention (Dao et al., 2023)
- Efficient Inference via GPTQ (Frantar et al., 2022)

## Next Session

1. Start Phase 1: Data expansion + BPE tokenizer
2. Profile current training pipeline for bottlenecks
3. Set up monitoring (W&B) for training metrics
4. Commit this roadmap to CLAUDE.md

---

**Status:** Phase 3 (Epoch 0/50)
**Owner:** Joshua Trommel
**Last Updated:** 2026-02-28
