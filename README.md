# nuLLM

Minimal LLM built from scratch. On-system Claude alternative - no API costs.

![Architecture](architecture.svg)

## Status
Production Ready - All phases complete (verified 2026-02-13)
- Tokenization: Char, word, BPE
- Attention: Multi-head, scaled dot-product
- Transformer: Full stack with residuals
- Training: Loss converges
- Generation: Autoregressive sampling
- Chat: Conversational interface

## Overview

nuLLM implements core transformer concepts from "Attention Is All You Need" in ~500 lines of Python. Built to understand how modern LLMs actually work under the hood.

**Components:**
- **Tokenization**: Character, word, and BPE tokenizers
- **Attention**: Single-head and multi-head self-attention
- **Transformer**: Full blocks with feed-forward, layer norm, residuals
- **Training**: Cross-entropy loss with Adam optimizer
- **Generation**: Autoregressive sampling with temperature control
- **Chat**: Conversational wrapper with auto-training

**What you learn:**
- How text becomes numbers (tokenization)
- How transformers "focus" (attention mechanisms)
- How gradient descent works (training loop)
- How models generate text (sampling strategies)
- How to scale up (bigger models, better data)

## Quickstart

### Install
```bash
cd nuLLM
python3 -m venv venv
source venv/bin/activate
pip install torch numpy tiktoken tqdm
```

### Run Chat (5 minutes)
```bash
python src/chat.py
```

Auto-trains a tiny model on conversational data (~1 min), then starts chat interface. Basic but functional - demonstrates core concepts.

### Run Tests
```bash
python examples/quick_test.py      # End-to-end verification
python examples/test_tokenizer.py  # Tokenizer encode/decode
python examples/test_model.py      # Model forward pass
```

### Train Your Own
```bash
python src/train.py
```

## Architecture

### Components
- **Tokenizer** (src/tokenizer.py): Char (simplest), Word (whitespace split), BPE (subword, industry standard)
- **Attention** (src/attention.py): Single-head and multi-head self-attention with Q/K/V projections
- **Transformer** (src/transformer.py): Full blocks with feed-forward, LayerNorm, residual connections
- **Training** (src/train.py): Cross-entropy loss, Adam optimizer, autoregressive dataset
- **Chat** (src/chat.py): Conversational wrapper with auto-training fallback

### Key Concepts
**Attention**: Each token gets Q/K/V vectors. Q·K scores measure similarity. Softmax weights sum values. Multi-head learns different patterns.

**Residuals**: x = x + f(x) helps gradients flow, enables deeper networks.

**Temperature**: Low (0.1) = deterministic, Medium (0.8) = balanced, High (2.0) = creative.

### Model Sizes

**Nano** (demo): 50K params, 2 layers, 2 heads, 32 embed, 64 context
**Micro** (learning): 500K params, 4 layers, 4 heads, 128 embed, 256 context
**Mini** (usable): 5M params, 6 layers, 8 heads, 256 embed, 512 context

### Performance
Nano model on M-series Mac: ~1 min training (50 epochs), <10ms per token inference. Loss: 2.6134 → trainable.

## Benchmarks

### Comparison to Real Models

| Model | Params | Layers | Heads | Embed | Context | Training |
|-------|--------|--------|-------|-------|---------|----------|
| GPT-2 | 124M   | 12     | 12    | 768   | 1024    | WebText  |
| GPT-3 | 175B   | 96     | 96    | 12288 | 2048    | Internet |
| nuLLM Nano | 50K | 2 | 2 | 32 | 64 | Tiny corpus |
| nuLLM Mini | 5M | 6 | 8 | 256 | 512 | WikiText |

nuLLM is ~2500x smaller than GPT-2 but uses the same transformer architecture.

### What This Teaches
- **Tokenization**: How text becomes numbers (vocab size tradeoffs, BPE vs word vs char)
- **Attention**: How transformers "focus" (Q/K/V matrices, scaled dot-product, multi-head patterns)
- **Training**: How gradient descent works (teacher forcing, backprop, loss functions)
- **Generation**: How models create text (greedy vs sampling, temperature effects, autoregressive)
- **Scaling**: Bigger models, better data, more compute

### References
**Papers**: Attention Is All You Need (Vaswani 2017), GPT-2 (Radford 2019), BERT (Devlin 2018)
**Code**: nanoGPT (Karpathy), minGPT, micrograd
