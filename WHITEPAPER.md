# core: A Minimal Transformer Language Model

**Joshua Trommel**
February 2026

---

## Abstract

core is a from-scratch implementation of a decoder-only transformer language model in PyTorch. At approximately 230,000 parameters -- roughly 2,500x smaller than GPT-2 -- it serves as a focused study of the "Attention Is All You Need" architecture at minimal scale. The model uses character-level tokenization, trains on a 185 KB mixed corpus of Q&A pairs and code syntax, and runs as a persistent daemon with automated overnight training, evaluation, and checkpoint selection. This paper documents the architecture, training methodology, evaluation framework, and infrastructure.

## 1. Introduction

Large language models have demonstrated remarkable capabilities, but their scale obscures the underlying mechanics. core strips the architecture to its essentials: a 4-layer, 4-head transformer with 128-dimensional embeddings, trained on character-level tokens. The goal is not competitive performance but architectural clarity -- understanding exactly what each component contributes at a scale where every parameter is accountable.

The implementation covers the full stack: tokenization, attention, training, evaluation, inference, and serving. The entire model fits in ~500 lines of Python.

## 2. Architecture

### 2.1 Model Structure

core implements the standard GPT-2 decoder-only transformer with pre-norm residual connections:

```
Input tokens
    |
Token Embedding (vocab_size x embed_dim)
    +
Positional Embedding (max_len x embed_dim)
    |
    v
[Transformer Block x N]
    |-- LayerNorm
    |-- Multi-Head Causal Self-Attention
    |-- Residual Connection + Dropout
    |-- LayerNorm
    |-- Position-wise Feed-Forward (GELU activation)
    |-- Residual Connection + Dropout
    |
Final LayerNorm
    |
Linear Head (embed_dim -> vocab_size)
    |
Logits
```

### 2.2 Attention Mechanism

The multi-head attention uses a fused QKV projection for efficiency. A single linear layer projects the input into queries, keys, and values simultaneously, which are then reshaped across heads:

```python
qkv = self.qkv(x)  # (batch, seq, 3 * embed_dim)
qkv = qkv.reshape(batch, seq, 3, num_heads, head_dim)
qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
```

Causal masking prevents attention to future positions via an upper-triangular mask filled with negative infinity before the softmax. Scaled dot-product attention follows Vaswani et al.: `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V`.

A separate single-head `SelfAttention` class exists for pedagogical purposes, exposing the raw attention weights for visualization.

### 2.3 Feed-Forward Network

Each transformer block contains a position-wise feed-forward network with GELU activation:

```
Linear(embed_dim -> ff_dim) -> GELU -> Linear(ff_dim -> embed_dim) -> Dropout
```

The FF dimension is typically 2-4x the embedding dimension, giving the model capacity to learn non-linear transformations at each position.

### 2.4 Model Tiers

Three configurations exist for experimentation at different compute budgets:

| Tier  | Parameters | Layers | Heads | Embed Dim | FF Dim | Max Context |
|-------|-----------|--------|-------|-----------|--------|-------------|
| Nano  | ~50K      | 2      | 2     | 32        | 64     | 64 tokens   |
| Micro | ~230K     | 4      | 4     | 128       | 512    | 256 tokens  |
| Mini  | ~5M       | 6      | 8     | 256       | 1024   | 512 tokens  |

For reference, GPT-2 Small uses 124M parameters, 12 layers, 12 heads, 768 embed dim, and 1024 token context. core's Micro tier operates at 1/539th the parameter count.

## 3. Tokenization

Three tokenizer implementations are provided:

**Character-level (primary).** Maps each unique character in the corpus to an integer index. The vocabulary is constructed dynamically from training data, with index 0 reserved for an unknown token (`\x00`). For the mixed training corpus, this yields approximately 102 unique tokens. Character-level tokenization eliminates out-of-vocabulary issues and provides full coverage at the cost of longer sequences per semantic unit.

**Word-level.** Splits on whitespace. Straightforward but suffers from large vocabulary size and no handling of unseen words beyond a single UNK token.

**BPE (via tiktoken).** Uses OpenAI's `cl100k_base` encoding for compatibility with larger-scale experiments (e.g., WikiText-2). Vocabulary size of ~100K tokens.

The character tokenizer is used for all core training runs. The sliding-window `TextDataset` generates (input, target) pairs offset by one position for autoregressive next-character prediction.

## 4. Training

### 4.1 Optimization

- **Optimizer:** AdamW with default betas
- **Learning rate:** 1e-3 baseline, with cosine annealing to 1e-5 for overnight runs
- **Gradient clipping:** Max norm 1.0
- **Batch size:** 32 (overnight), 4-8 (interactive)
- **Sequence length:** 64 (overnight), configurable per tier
- **Loss function:** Cross-entropy over the full vocabulary at each position
- **Regularization:** Dropout at 0.1 across attention, FFN, and embeddings

### 4.2 Training Corpus

The training data totals approximately 185 KB (203 KB on disk across 12 files) spanning several domains:

| File | Size | Content |
|------|------|---------|
| `comprehensive.txt` | 75 KB | Mixed Q&A: math, identity, jot syntax, factual knowledge |
| `math_comprehensive.txt` | 50 KB | Arithmetic drills (addition, subtraction, multiplication) |
| `jot_corpus.txt` | 30 KB | Generated jot programming language examples |
| `jot_code.txt` | 19 KB | Hand-written jot code samples |
| `science_tutoring.txt` | 19 KB | Science Q&A pairs |
| `combined_corpus.txt` | 3 KB | Mixed summary corpus |
| `qa_varied.txt` | 2 KB | Diverse question-answer pairs |
| `conversational.txt` | 2 KB | Casual dialogue pairs |

The corpus is deliberately small. The research question is not "how much can we memorize" but "what structure emerges from minimal data at minimal scale."

### 4.3 Overnight Training Daemon

A persistent `launchd` service (`com.joshua.core-train`) runs the training loop as a background daemon on macOS. Features:

- **RAM monitoring:** Pauses training if system memory pressure exceeds threshold
- **Checkpoint management:** Saves best checkpoint by validation loss to `models/overnight_best.pt`
- **iMessage notifications:** Sends training status updates via the OpenClaw message gateway
- **Cosine annealing schedule:** Learning rate decays smoothly from 1e-3 to 1e-5 over the full run
- **Automatic restart:** launchd respawns the process on crash

## 5. Inference

### 5.1 Generation

Autoregressive generation proceeds token-by-token:

1. Encode the prompt as character indices
2. Forward pass through the model to get next-token logits
3. Apply temperature scaling: `logits / temperature`
4. Sample from the resulting softmax distribution via multinomial sampling
5. Append the sampled token, slide the context window if it exceeds `max_len`
6. Repeat until the length limit or a stop condition

Temperature controls the entropy of the output distribution. At `T=1.0` the model samples from its learned distribution; lower values sharpen toward greedy decoding; higher values increase randomness.

### 5.2 Serving

A Flask web application (`web_ui.py`) serves the model over HTTP:

- **Chat interface** at `/` with Apple Liquid Glass design and dark mode toggle
- **Quiz mode** at `/quiz` for interactive evaluation
- **REST API:** `POST /api/generate` for programmatic access, `GET /api/status` for model info
- Runs as a persistent `launchd` service (`com.joshua.core-web`) on port 5001

## 6. Evaluation

### 6.1 Eval Harness

A prompt-suite evaluation framework (`src/eval_harness.py`) scores checkpoint quality across six categories:

- **Reasoning** -- logical inference and problem-solving
- **Code** -- jot syntax generation
- **Debug** -- error identification
- **Summarize** -- text compression
- **Instruction** -- following directives
- **Refusal** -- appropriate non-response to adversarial prompts

Each prompt specifies expected output constraints: minimum/maximum character length, required keywords (`keywords_all`), partial keyword matches (`keywords_any`), and forbidden keywords (`keywords_none`). A checkpoint passes if it achieves >= 60% average score and >= 80% non-empty response rate.

### 6.2 Overnight Eval Loop

After each training run, the eval harness scores the latest checkpoint against the prompt suite. Results are written to `logs/eval_results.json` and consumed by an external morning report system (via the fony voice agent) for daily grade summaries.

### 6.3 Current Capabilities

The model demonstrates basic pattern completion for trained domains:

```
Q: What is 7*8?       -> 56
Q: What is your name? -> core
Q: Who made you?      -> Josh made me
Q: print hello world  -> print "Hello, World!";
Q: write a function   -> fn add(a, b) { return a + b; }
```

Output quality degrades on out-of-distribution prompts, as expected for a 230K parameter model trained on 185 KB of data.

## 7. Infrastructure

### 7.1 Project Layout

```
src/
    tokenizer.py        Character, word, and BPE tokenizers
    attention.py         Self-attention and multi-head attention
    transformer.py       Transformer blocks and full model
    train.py             Dataset loader, training loop, generation
    chat.py              Interactive CLI chat interface
    eval_harness.py      Prompt-suite evaluation framework
web_ui.py               Flask web server and API
templates/index.html    Liquid Glass chat UI
data/                   Training corpora (12 files, ~200 KB)
models/                 Saved checkpoints (.pt)
logs/                   Training logs and eval results
tests/                  Pytest suite (14 test files)
cron/                   Overnight training and eval scripts
scripts/                Utility and training scripts
```

### 7.2 Dependencies

- Python 3.13
- PyTorch 2.10
- Flask (web UI)
- tiktoken (BPE tokenizer, optional)

### 7.3 Testing

The test suite covers tokenizer edge cases, data loader error paths, model inference boundaries, sampling behavior, web UI validation, and eval harness correctness. All tests run via `pytest -q` and are required to pass before any push to main.

## 8. Limitations and Future Work

**Current limitations:**

- Character-level tokenization produces long sequences for modest text, limiting effective context
- 185 KB of training data is insufficient for generalization beyond trained patterns
- No train/eval split in the primary training loop (overnight runner handles this separately)
- Single-GPU only (M4 Mac Mini, no CUDA multi-GPU support)

**Planned directions:**

- Subword tokenization on domain-specific vocabulary (jot syntax tokens)
- Train/eval split with proper validation loss tracking
- Top-k and top-p sampling for generation
- Checkpoint resume reliability improvements
- Scaling experiments on the Mini tier with expanded corpus

## 9. Conclusion

core demonstrates that the transformer architecture is fully functional at minimal scale. A 230K parameter model trained on 185 KB of text can learn basic arithmetic, identity responses, and code syntax patterns -- not through memorization alone, but through the attention mechanism's ability to extract positional and semantic structure from character sequences. The project prioritizes architectural understanding and reproducibility over performance benchmarks.

---

## References

1. Vaswani, A., et al. "Attention Is All You Need." NeurIPS 2017.
2. Radford, A., et al. "Language Models are Unsupervised Multitask Learners." OpenAI 2019. (GPT-2)
3. Loshchilov, I., Hutter, F. "Decoupled Weight Decay Regularization." ICLR 2019. (AdamW)
4. Hendrycks, D., Gimpel, K. "Gaussian Error Linear Units (GELUs)." arXiv 2016.

## License

MIT License, 2026, Joshua Trommel.
