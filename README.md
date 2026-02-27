# aether

A small language model built from scratch. Trained on code and knowledge.

## What It Does

Generates text based on a prompt. Understanding how it works teaches you how real language models work.

## Getting Started

```bash
git clone https://github.com/nulljosh/aether.git && cd nous
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Generate text
python src/generate.py --prompt "fn " --temperature 0.3

# Web interface (chat)
python index.py
# Visit http://localhost:5001
```

## How It Works

1. **Data** — Train on code and writing samples
2. **Model** — Small transformer neural network (3.5M parameters)
3. **Learning** — PyTorch training with 1000 epochs
4. **Output** — Predicts next words based on patterns learned

## The Numbers

- Parameters: 3.5 million (vs Claude: billions)
- Training time: Several hours on a Mac
- Loss (error): 0.09 (lower is better)
- Speed: 50,000 tokens per second on CPU

## Why Build It

To understand how language models actually work. Instead of using a black box, you see every layer.

## What You Get

- Complete training code
- C99 inference engine (super fast, runs anywhere)
- Web chat interface
- Everything open source

## Links

- GitHub: https://github.com/nulljosh/aether
- Docs: See DEPLOY.md for server setup

## That's It

Trained. Working. Ready to use.
