# Benchmarks -- An Honest Comparison

How does core stack up against real LLMs? Short answer: it doesn't. Long answer: that's the point.

## The Numbers

| Model | Params | Size | Context | Vocab | Training Data | Can It... |
|-------|--------|------|---------|-------|---------------|-----------|
| **core (current)** | 0.57M | 2.2 MB | 128 tokens | 102 chars | 185 KB jot corpus | Autocomplete jot syntax (barely) |
| **core (wiki planned)** | ~14M | ~56 MB | 512 tokens | ~32K BPE | 103M tokens (WikiText-103) | Generate Wikipedia-ish text (TBD) |
| GPT-2 Small | 124M | 500 MB | 1024 tokens | 50K BPE | 40 GB WebText | Write coherent paragraphs |
| Llama 3.2 1B | 1.2B | 2.4 GB | 128K tokens | 128K BPE | 15T tokens | Follow instructions, reason |
| qwen3:14b (your Ollama) | 14B | ~8 GB | 4096 tokens | 150K+ | Trillions of tokens | Code, chat, reason, tools |
| Claude Opus 4 | ~???B | ~??? | 200K tokens | ~100K | The internet | Build entire apps in minutes |

## What Can core Actually Do?

**Right now (0.57M params, char-level):**
- Generate jot-like syntax fragments after 200+ epochs
- Output is recognizable as code-adjacent but not meaningful
- Inference speed: instant (C engine, mmap, zero deps)
- Training time: ~30 min for 200 epochs on M4

**After WikiText-103 (~14M params, BPE):**
- Should generate grammatical English sentences
- Will not understand what it's saying
- Will not follow instructions
- Will not reason
- Perplexity target: < 50 (GPT-2 small gets ~29 on WikiText-103)

## Why core Exists

This is not trying to compete with Claude or GPT. The point:

1. **"What I cannot create, I do not understand."** -- Feynman. Building a transformer from scratch teaches you what all those papers actually mean.
2. **C inference engine**: 350 LOC, zero dependencies, mmap weight loading. That's the real deliverable. Runs anywhere with a C compiler.
3. **The full stack**: tokenizer -> attention -> transformer -> training loop -> checkpoint -> binary export -> C inference. End to end, no black boxes.

## The Gap (Real Talk)

To get from core to something useful:

| Capability | What It Takes |
|-----------|---------------|
| Coherent paragraphs | ~100M params, ~10B tokens, ~$100 in compute |
| Follow instructions | Instruction tuning dataset + RLHF, ~1B params minimum |
| Code generation | Code-specific training data, ~7B params minimum |
| Reasoning | Chain-of-thought training, ~13B+ params |
| Tool use (like Claude Code) | Function calling training, massive compute, months of RLHF |

**In perspective:** Claude Code just orchestrated 7 parallel agents to build 3 iOS apps, add PWA to 3 websites, rename a project, overhaul a training pipeline, and write 10 READMEs -- in one session. core can autocomplete `fn ` with some curly braces.

That gap is the entire history of modern AI research, billions of dollars in compute, and thousands of researcher-years. But you understand every byte of core's 2.2 MB. That's worth something.

## Speed Comparison

| Model | Tokens/sec (your M4) |
|-------|---------------------|
| core (C engine) | ~50,000 tok/s |
| qwen3:14b (Ollama) | ~30 tok/s |
| Claude (API) | ~80 tok/s |

core is fast because it's tiny. Speed without intelligence is just random numbers really fast.

## Roadmap to Less Embarrassing

1. WikiText-103 BPE training (infrastructure done, needs overnight run)
2. Scale to 14M params (wiki config ready)
3. Evaluate perplexity against GPT-2 small as baseline
4. If perplexity < 100: try instruction tuning on Alpaca dataset
5. If perplexity < 50: you have a real language model, just a very small one
