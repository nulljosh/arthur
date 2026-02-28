# aether

A small language model built from scratch. Trained on code and knowledge.

## Architecture

![Architecture](https://raw.githubusercontent.com/nulljosh/aether/main/architecture.svg)

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
2. **Model** -- Small transformer neural network (445K parameters)
3. **Learning** -- PyTorch training with character-level tokenizer
4. **Output** -- Predicts next tokens based on patterns learned

## The Numbers

- Parameters: 445K (3 transformer blocks, 4 attention heads)
- Training: Math (50%) + Wikipedia/Current Events (50%)
- Vocab: 91 characters (character-level)
- Loss: 0.13-0.19 (converging)
- Speed: CPU inference

## Why Build It

Understanding language models means building one. This is the simplest version that actually works.

## License

MIT 2026 Joshua Trommel
