# aether

A small language model built from scratch. Trained on code and knowledge.

## Architecture

![aether architecture](architecture.svg)

**Text Description (Mobile Fallback):**
```
Input Text → Tokenizer (91 chars) → Embedding (128-dim)
    ↓
Transformer Block 1 (Attention + FFN)
    ↓
Transformer Block 2 (Attention + FFN)
    ↓
Transformer Block 3 (Attention + FFN)
    ↓
Output Layer (Vocabulary Logits) → Next Token
```

**Model Specifications:**
- **Total Parameters**: 445K
- **Layers**: 3 transformer blocks
- **Attention Heads**: 4 per block
- **Feed-Forward Dimension**: 256
- **Embedding Dimension**: 128
- **Tokenizer**: Character-level (91 unique characters)
- **Training Data**: Math (50%) + Wikipedia + Current Events (50%)
- **Training Loss**: 0.13-0.19 (converging)

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
- **Math**: Addition, subtraction, multiplication, division
- **Facts**: Current events, year, president
- **Identity**: Name, creator, purpose
- **Knowledge**: Definitions, explanations

## Why Build It

Understanding language models means building one. This is the simplest version that actually works.

## Files

- `src/train.py` – Training loop
- `src/transformer.py` – Model architecture
- `src/tokenizer.py` – Character-level tokenizer
- `src/generate.py` – Text generation
- `web_ui.py` – Chat interface
- `models/` – Saved checkpoints
- `data/` – Training datasets

## License

MIT 2026 Joshua Trommel
