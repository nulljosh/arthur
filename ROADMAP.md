# core Development Roadmap

## Phase 1: Tokenization [DONE]
- Character tokenizer (simplest)
- Word tokenizer (word boundaries)
- BPE tokenizer (subwords)
- **Status**: Levels 0-1 working

## Phase 2: Attention Mechanism [DONE]
- Self-attention (single head)
- Multi-head attention
- Positional encoding
- **Status**: Implemented

## Phase 3: Transformer Block [DONE]
- Feed-forward network
- Layer normalization
- Residual connections
- **Status**: Full model architecture complete

## Phase 4: Training [DONE]
- Loss function (cross-entropy)
- Backpropagation
- Optimizer (Adam)
- **Status**: Implemented

## Phase 5: Generation [DONE]
- Autoregressive sampling
- Temperature scaling
- Top-k/nucleus sampling (TODO)
- **Status**: Basic generation working

## Phase 6: Scale
- Multi-layer model
- Larger corpus (WikiText-2)
- Better tokenization
- **Status**: Not started

## Key Concepts to Learn

### Tokenization
- Why BPE over words?
- Vocab size tradeoffs
- Special tokens (BOS, EOS, PAD)

### Attention
- Query, Key, Value matrices
- Scaled dot-product
- Why multi-head?

### Training
- Teacher forcing
- Gradient clipping
- Learning rate schedules

### Generation
- Greedy vs sampling
- Temperature effects
- Beam search

## References

Papers:
- "Attention Is All You Need" (Vaswani et al., 2017)
- "GPT-2" (Radford et al., 2019)
- "BERT" (Devlin et al., 2018)

Code:
- nanoGPT (Karpathy) - tiny GPT implementation
- minGPT - minimal GPT in PyTorch
- micrograd - tiny autograd engine

## Milestones

**M1**: Tokenizer working (BPE trained on small corpus)
**M2**: Attention layer forward pass
**M3**: Full transformer block
**M4**: Training loop (loss decreases)
**M5**: Generate coherent text (even if nonsense)
**M6**: Generate Shakespeare-quality text

## Timeline

Rough estimate:
- Week 1: Tokenization
- Week 2: Attention mechanism
- Week 3: Full transformer
- Week 4: Training + generation
- Week 5+: Scale up, improve quality
