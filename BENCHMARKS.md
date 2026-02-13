# nuLLM Benchmarks

## Complexity Tiers

### Tier 1: Tokenization (Current)
- **Level 0**: Character-level tokenizer
- **Level 1**: Word-level tokenizer
- **Level 2**: BPE (Byte Pair Encoding)
- **Goal**: Convert text → token IDs

### Tier 2: Basic Architecture
- **Level 3**: Single attention head
- **Level 4**: Multi-head attention
- **Level 5**: Transformer block (attention + FFN)
- **Goal**: Forward pass working

### Tier 3: Training Loop
- **Level 6**: Simple training on tiny corpus
- **Level 7**: Loss function + backprop
- **Level 8**: Gradient descent working
- **Goal**: Model learns patterns

### Tier 4: Generation
- **Level 9**: Greedy decoding
- **Level 10**: Sampling strategies (top-k, nucleus)
- **Goal**: Generate coherent text

### Tier 5: Scale Up
- **Milestone A**: Train on WikiText-2 (small dataset)
- **Milestone B**: Multi-layer transformer (2-4 layers)
- **Milestone C**: Generate Shakespeare-style text

## Real-World Targets

Once microLM works at basic level:

### Text Generation Tasks
- Complete sentences coherently
- Answer simple questions
- Generate short stories

### Model Sizes
- **Nano**: 1M params (demo)
- **Micro**: 10M params (educational)
- **Mini**: 100M params (usable)

## Comparison to Real LLMs

| Model | Params | Context | Training |
|-------|--------|---------|----------|
| GPT-2 | 124M   | 1024    | WebText  |
| GPT-3 | 175B   | 2048    | Internet |
| Claude | ?     | 200k    | ?        |
| **nuLLM** | 1-10M | 512 | WikiText |

nuLLM aims to be the smallest transformer that demonstrates core concepts.

## Performance Targets

Not optimized for speed - focused on clarity:
- Training: ~1 hour on M-series Mac for nano model
- Inference: <100ms per token
- Perplexity: <50 on validation set (WikiText-2)

## Training Results

**Configuration**: 50 epochs, 2896 params (nano model)
**Final Loss**: 0.5527
**Status**: All tests passing
- Tokenizer: Character and word-level encode/decode working
- Model: Forward pass produces correct output shapes
- Training: Backprop functional, loss decreases
- Generation: Produces text (untrained baseline)

## Learning Path

1. **Understand tokenization** (how text becomes numbers)
2. **Build attention** (how transformers "focus")
3. **Train small model** (gradient descent in action)
4. **Generate text** (sampling strategies)
5. **Scale up** (bigger model, better data)

Goal: Understand how Claude/GPT actually work under the hood.
