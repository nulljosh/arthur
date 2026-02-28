# aether

A small language model built from scratch. Trained on code and knowledge.

## Architecture

[View architecture diagram](https://raw.githubusercontent.com/nulljosh/aether/main/architecture.svg)

```
Input → Tokenizer → Embedding (128-dim) → Block 1 → Block 2 → Block 3 → Output → Next Token
         (91 chars)   (Add position)     (Attn+FFN) (Attn+FFN) (Attn+FFN)  (Logits)
```

**Model specs:**
- 445K parameters (3 transformer blocks, 4 attention heads)
- 256-dim feed-forward, 128-dim embedding
- Character-level tokenizer (91 vocab)
- Training: Math (50%) + Wikipedia + Current Events (50%)
- Loss: 0.13-0.19 (converging)

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

1. **Data** – Train on math, current events, and Wikipedia
2. **Model** – Small transformer neural network
3. **Learning** – PyTorch training with character-level tokenizer
4. **Output** – Predicts next tokens based on patterns learned

## Evaluation

Tests on Q&A accuracy:
- Math: Addition, subtraction, multiplication, division
- Facts: Current events, year, president
- Identity: Name, creator, purpose
- Code: jot syntax examples

## Why Build It

Understanding language models means building one. This is the simplest version that actually works.

## License

MIT 2026 Joshua Trommel
