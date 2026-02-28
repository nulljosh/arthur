# aether

A small language model built from scratch. Trained on code and knowledge.

## Architecture

![aether architecture diagram](architecture.svg)

**Model Components:**
- **Input**: Text query (character-level)
- **Tokenizer**: 91-character vocabulary
- **Embedding**: 128-dimensional vectors
- **Transformer Blocks**: 3 layers with attention + feed-forward
- **Output**: Vocabulary logits for next token prediction

**Specs:**
- **Parameters**: 445K (tiny compared to Claude)
- **Layers**: 3 transformer blocks
- **Attention Heads**: 4
- **Feed-Forward Dimension**: 256
- **Training Data**: Math (50%) + Wikipedia + Current Events (50%)
- **Loss**: 0.13-0.19 (converging)

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
