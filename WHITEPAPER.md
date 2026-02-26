# core: Building a Language Model from Scratch

**Joshua Trommel**
February 2026

---

## Abstract

core is a transformer language model built from scratch in PyTorch. It has about 230,000 trainable parameters, which makes it roughly 2,500 times smaller than GPT-2. The point of the project is to take the same architecture that powers modern LLMs and strip it down to a scale where you can actually see what every piece does. It reads individual characters instead of words, trains on about 185 KB of Q&A pairs and code examples, and runs overnight on a Mac Mini as an automated training daemon. This paper covers how the model works, how it trains, how it gets evaluated, and what it can actually do.

## 1. Why Build This

Every major language model today is built on the transformer architecture from the 2017 "Attention Is All You Need" paper. But when a model has 7 billion parameters and trains on terabytes of text, it is hard to point at any single component and say "that is what this piece learned." core exists to answer a simpler question: if you take the exact same architecture and shrink it to a few hundred thousand parameters, what can it still learn?

The model is 4 layers deep, uses 4 attention heads, and represents each token as a 128-dimensional vector. The full implementation is about 500 lines of Python. It is not trying to compete with production models. It is trying to make the transformer legible.

## 2. Architecture

### 2.1 How the Model Is Structured

core follows the GPT-2 blueprint: a decoder-only transformer with pre-norm residual connections. In plain terms, that means it reads a sequence of tokens from left to right and predicts the next one, over and over. Here is the full data flow:

```
Input tokens
    |
Token Embedding (look up a vector for each character)
    +
Position Embedding (add a vector that encodes position in the sequence)
    |
    v
[Transformer Block x N]
    |-- LayerNorm (normalize the values so training stays stable)
    |-- Multi-Head Causal Self-Attention (figure out which tokens matter)
    |-- Residual Connection + Dropout (add the original input back in)
    |-- LayerNorm
    |-- Feed-Forward Network with GELU (learn non-linear patterns)
    |-- Residual Connection + Dropout
    |
Final LayerNorm
    |
Linear Head (project back to vocabulary size)
    |
Logits (raw scores for each possible next character)
```

The "decoder-only" part means there is no separate encoder. The model only ever looks backward in the sequence, never forward. This is enforced by the causal mask in the attention layer.

### 2.2 Attention: The Core Mechanism

Attention is the part that makes transformers work. The idea is simple: for every token in the sequence, the model asks "which other tokens should I pay attention to right now?" It learns to answer that question through training.

Mechanically, each token gets projected into three vectors: a **query** (what am I looking for?), a **key** (what do I contain?), and a **value** (what information do I carry?). The model computes a compatibility score between every query and every key, normalizes those scores with softmax, and uses them to create a weighted mix of the values.

The math: `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V`

The `sqrt(d_k)` division is there to keep the dot products from getting too large, which would push the softmax into regions where its gradients vanish and training stalls.

**Multi-head attention** runs this process multiple times in parallel with different learned projections. If you have 4 heads and a 128-dimensional embedding, each head works in a 32-dimensional subspace. This lets different heads specialize: one might track syntactic structure, another might focus on nearby characters, and so on. The outputs get concatenated and projected back to the full embedding dimension.

For efficiency, core fuses the Q/K/V projections into a single matrix multiply:

```python
qkv = self.qkv(x)  # one linear layer, 3x the output size
# reshape and split into Q, K, V across heads
```

**Causal masking** prevents the model from cheating. An upper-triangular mask filled with negative infinity gets applied before the softmax, which zeroes out any attention to future positions. The model can only use what came before.

A separate single-head `SelfAttention` class also exists, mainly for visualization. It exposes the raw attention weight matrix so you can see exactly what the model is attending to.

### 2.3 Feed-Forward Network

After attention, each token passes through a two-layer feed-forward network:

```
Linear(128 -> 512) -> GELU -> Linear(512 -> 128) -> Dropout
```

This is where the model learns non-linear transformations. Attention figures out *which* tokens are relevant; the FFN figures out *what to do* with that information. The expansion to 4x the embedding dimension (128 to 512) gives the network room to represent more complex functions before compressing back down.

GELU (Gaussian Error Linear Unit) is the activation function. It is smoother than ReLU and tends to train better for language tasks. Unlike ReLU which hard-clips at zero, GELU has a soft curve that lets small negative values through with low probability.

### 2.4 Model Tiers

Three configurations let you trade off between speed and capacity:

| Tier  | Parameters | Layers | Heads | Embed Dim | FF Dim | Max Context |
|-------|-----------|--------|-------|-----------|--------|-------------|
| Nano  | ~50K      | 2      | 2     | 32        | 64     | 64 tokens   |
| Micro | ~230K     | 4      | 4     | 128       | 512    | 256 tokens  |
| Mini  | ~5M       | 6      | 8     | 256       | 1024   | 512 tokens  |

For context: GPT-2 Small has 124M parameters, 12 layers, 12 heads, 768-dim embeddings, and a 1024-token context window. core's Micro tier is 1/539th the size of GPT-2. The Nano tier trains in minutes and is useful for debugging architecture changes. The Mini tier is where you would start to see more interesting emergent behavior with a larger corpus.

## 3. Tokenization

Before the model can process text, characters need to be converted to numbers. core implements three tokenizers:

**Character-level (the one that actually gets used).** Every unique character in the training data gets its own integer ID. The vocabulary is built directly from the corpus: sort all unique characters, assign them sequential indices, reserve index 0 for unknown characters. For the mixed training corpus, this gives about 102 tokens (lowercase/uppercase letters, digits, punctuation, whitespace, etc.).

The upside of character tokenization is simplicity: there is zero preprocessing, no vocabulary file to maintain, and no out-of-vocabulary problem. The downside is that sequences get long. The word "function" is 8 tokens instead of 1, which eats into your context window fast.

**Word-level.** Splits on whitespace. Vocabulary grows with the corpus and unseen words map to a single UNK token. Included for comparison, not used in practice.

**BPE (via tiktoken).** Byte-pair encoding using OpenAI's `cl100k_base` vocabulary (~100K subword tokens). Used for experiments with larger datasets like WikiText-2 where character-level tokenization would make sequences unmanageably long.

The training dataset is built as a sliding window: take the full token sequence, extract every possible window of length `seq_len + 1`, split each window into input (first `seq_len` tokens) and target (last `seq_len` tokens, shifted by one). The model learns to predict each next character given everything before it.

## 4. Training

### 4.1 Optimization Setup

- **Optimizer:** AdamW (Adam with decoupled weight decay, which regularizes better than L2)
- **Learning rate:** Starts at 1e-3, anneals to 1e-5 on a cosine schedule over the full run
- **Gradient clipping:** Norms capped at 1.0 (prevents exploding gradients from destabilizing training)
- **Batch size:** 32 for overnight runs, 4-8 for interactive experiments
- **Sequence length:** 64 characters per training window
- **Loss:** Cross-entropy across the full vocabulary at every position in the sequence
- **Dropout:** 0.1 everywhere (embeddings, attention, feed-forward)

The cosine annealing schedule is important for small models. A constant learning rate tends to overshoot minima once the loss gets low. Cosine decay starts aggressive and gradually becomes more conservative, which helps the model settle into better solutions during the later epochs without needing manual LR tuning.

### 4.2 Training Data

The total corpus is about 185 KB of text across multiple files:

| File | Size | What It Contains |
|------|------|-----------------|
| `comprehensive.txt` | 75 KB | Q&A covering math, identity, jot syntax, general knowledge, time/date |
| `math_comprehensive.txt` | 50 KB | Arithmetic drills: addition, subtraction, multiplication |
| `jot_corpus.txt` | 30 KB | Auto-generated jot programming language examples |
| `jot_code.txt` | 19 KB | Hand-written jot code: functions, control flow, variables |
| `science_tutoring.txt` | 19 KB | Science Q&A pairs |
| `combined_corpus.txt` | 3 KB | Mixed summary of all domains |
| `qa_varied.txt` | 2 KB | Diverse question-answer pairs |
| `conversational.txt` | 2 KB | Casual back-and-forth dialogue |

185 KB is tiny. GPT-2 trained on 40 GB. But the research question here is not "how much can you memorize with enough data." It is "what structure can a small model extract from a small dataset." The answer turns out to be: basic arithmetic, identity recall, and syntactically valid code in a simple language. Not bad for something you could print on a few hundred pages.

### 4.3 Overnight Training Daemon

Training runs as a persistent macOS `launchd` service (`com.joshua.core-train`). This is basically a background process that the OS manages:

- **RAM guard:** Monitors system memory pressure and pauses training if the machine starts swapping. A Mac Mini with 16 GB can train and run other things simultaneously, but only if the training loop backs off when memory gets tight.
- **Checkpointing:** Saves the best model (by loss) to `models/overnight_best.pt`. Also saves periodic checkpoints every 25 epochs so you can compare performance at different stages.
- **Notifications:** Sends training status updates over iMessage via the OpenClaw gateway. You wake up to a message saying "200 epochs done, loss 0.179, samples look good."
- **Cosine LR schedule:** Decays learning rate smoothly from 1e-3 to 1e-5 over the full run.
- **Crash recovery:** `launchd` automatically restarts the process if it dies.

A typical overnight run does 200 epochs in about 7 hours on an M4 Mac Mini.

## 5. Inference

### 5.1 How Generation Works

Text generation is autoregressive, meaning the model produces one character at a time, feeding each new character back in as input for the next prediction. The loop:

1. Encode the prompt ("Q: What is 5+3?\nA:") into character indices
2. Run a forward pass through the model to get a score for every possible next character
3. Divide those scores by the **temperature** parameter
4. Convert to probabilities with softmax
5. Sample randomly from those probabilities
6. Append the new character, trim the sequence if it exceeds the context window
7. Repeat until you hit the length limit or a stop character

**Temperature** is the main control knob. At `T=1.0`, the model samples from its raw learned distribution. Lower temperatures (0.3-0.5) make the output more predictable and repetitive but more accurate. Higher temperatures (1.0-2.0) make it more creative but more likely to produce garbage. For Q&A tasks, 0.5 works well. For code generation, 0.8 strikes a decent balance.

The context window slides: if the generated sequence exceeds `max_len` (128-256 tokens depending on config), the oldest tokens get dropped. The model always sees the most recent `max_len` characters.

### 5.2 Web Interface

A Flask app (`web_ui.py`) serves the model over HTTP:

- **Chat UI** at `/` with an Apple Liquid Glass frosted-glass design and dark mode toggle
- **Quiz mode** at `/quiz` for structured evaluation
- **REST API:** `POST /api/generate` accepts JSON with prompt, temperature, top-k, and top-p parameters. `GET /api/status` returns model config and load state.
- Runs as a persistent `launchd` service (`com.joshua.core-web`) on port 5001

The web UI wraps user input in the Q&A format the model was trained on (`Q: <input>\nA:`), then strips the format back out before displaying the response.

## 6. Evaluation

### 6.1 The Eval Harness

Evaluating a language model is harder than evaluating a classifier. There is no single accuracy number. core uses a prompt-suite framework (`src/eval_harness.py`) that tests the model across six categories:

- **Reasoning:** Can it follow basic logical chains?
- **Code:** Does it produce syntactically valid jot code?
- **Debug:** Can it identify errors in broken code?
- **Summarize:** Can it compress information?
- **Instruction:** Does it follow specific formatting requests?
- **Refusal:** Does it decline harmful requests appropriately?

Each prompt in the suite defines what a passing response looks like: minimum and maximum character length, keywords that must appear (`keywords_all`), keywords where at least one must appear (`keywords_any`), and keywords that must not appear (`keywords_none`).

A checkpoint passes the eval suite if it scores at least 60% average across all prompts and produces non-empty responses at least 80% of the time. These thresholds are intentionally low because a 230K parameter model is not going to ace a reasoning benchmark. The eval exists to catch regressions, not to claim competence.

### 6.2 Automated Eval Loop

After each overnight training run, the eval harness automatically scores the new checkpoint. Results go to `logs/eval_results.json`, which gets picked up by an external morning report system (a voice agent called fony) that delivers a daily summary of how the model is performing.

### 6.3 What It Can Actually Do

On in-distribution prompts, the model is surprisingly competent:

```
Q: What is 7*8?       -> 56
Q: What is your name? -> core
Q: Who made you?      -> Josh made me
Q: print hello world  -> print "Hello, World!";
Q: write a function   -> fn add(a, b) { return a + b; }
```

On out-of-distribution prompts, it falls apart. Ask it about history or write a poem and you get character soup. That is expected. The model has 230K parameters and 185 KB of training data. It learned the patterns it was shown and nothing else.

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

14 test files cover tokenizer edge cases (empty strings, unknown characters, single-character inputs), data loader error paths (missing files, zero-length sequences), model inference boundaries (empty input, sequences longer than context window), sampling behavior (temperature extremes, top-k/top-p correctness), web UI input validation, and eval harness correctness. `pytest -q` must pass before any push to main.

## 8. Limitations and Future Work

**Where it falls short:**

- Character-level tokenization means the model spends most of its context window on individual letters instead of semantic units. A 256-token context at the character level is about 40-50 words. Subword tokenization would stretch that significantly.
- 185 KB of training data is enough to memorize patterns but not enough to generalize. The model cannot extrapolate beyond what it has seen.
- The primary training loop does not separate training and validation data. The overnight runner handles this, but the base `train.py` does not.
- Training runs on a single M4 GPU. There is no distributed training or multi-GPU support.

**What comes next:**

- Subword tokenization tuned to the jot programming language, so common keywords like `function` and `return` become single tokens
- Proper train/validation split with early stopping based on validation loss
- Top-k and top-p sampling in the base generation function (currently only in the web UI)
- More reliable checkpoint resume so interrupted runs can pick up where they left off
- Scaling experiments on the Mini tier (5M params) with an expanded corpus to see where the quality curve bends

## 9. Conclusion

core shows that the transformer architecture works at absurdly small scale. 230K parameters trained on 185 KB of text is enough to learn basic arithmetic, recall trained facts, and generate syntactically valid code in a simple language. The model is not doing anything magical. It is learning statistical patterns over character sequences, and the attention mechanism gives it enough structure to capture positional and compositional relationships that a flat model could not.

The value of the project is not in the outputs. It is in the visibility. When your model has 230K parameters, you can trace a single forward pass end to end, inspect every attention head, and see exactly where the signal flows. That is not possible at production scale, and it is why building small is worth doing.

---

## References

1. Vaswani, A., et al. "Attention Is All You Need." *Advances in Neural Information Processing Systems* 30. 2017.
2. Radford, A., et al. "Language Models are Unsupervised Multitask Learners." OpenAI, 2019.
3. Loshchilov, I. and Hutter, F. "Decoupled Weight Decay Regularization." *International Conference on Learning Representations*, 2019.
4. Hendrycks, D. and Gimpel, K. "Gaussian Error Linear Units (GELUs)." arXiv:1606.08415, 2016.

## License

MIT License, 2026, Joshua Trommel.
