# aether

A small language model built from scratch. Trained on code and knowledge.

## What It Does

Generates text based on a prompt. Understanding how it works teaches you how real language models work.

## Architecture

```
Input Text → Tokenizer → Embedding (128-dim)
    ↓
Transformer Block 1 (Attention + FFN)
    ↓
Transformer Block 2 (Attention + FFN)
    ↓
Transformer Block 3 (Attention + FFN)
    ↓
Output Layer → Next Token Prediction
```

**Key specs:**
- 445K parameters (3 transformer blocks, 4 attention heads)
- 91-char vocabulary (character-level tokenizer)
- Training: Math (50%) + Wikipedia/Current Events (50%)
- Inference: CPU, single-token generation

## Getting Started

```bash
git clone https://github.com/nulljosh/aether.git && cd aether
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Generate text
python src/generate.py --prompt "Q: What is " --temperature 0.3

# Web interface (chat)
python web_ui.py
# Visit http://localhost:5001
```

## How It Works

1. **Data** -- Train on math, current events, and Wikipedia
2. **Model** -- Small transformer neural network
3. **Learning** -- PyTorch training with character-level tokenizer
4. **Output** -- Predicts next tokens based on patterns learned

## The Numbers

- Parameters: 445K (vs Claude: billions)
- Training time: Minutes on CPU
- Vocab: 91 characters
- Loss: 0.13-0.19 (converging)
- Speed: CPU inference

## Why Build It

Understanding language models means building one. This is the simplest version that actually works.

## License

MIT 2026 Joshua Trommel
