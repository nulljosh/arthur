# core

Tiny transformer language model built from scratch in PyTorch.

TLDR:
- Train on text
- Generate text
- Learn transformer internals end-to-end

## What this is

`core` is a compact educational LM project. It is designed for fast iteration and understanding, not production scale.

Current direction: char-level syntax modeling (jot-style datasets), with small model configs that run locally.

## Quick start

```bash
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# train
python3 src/train.py --epochs 100 --corpus tiny

# generate
python3 src/generate.py --prompt "Whether" --length 50
```

## Optional config

Create `.env` (gitignored):

```env
MODEL_PATH=models/ultra.pt
DATA_PATH=data/ultra_minimal.txt
PORT=5001
DEBUG=True
```

## Web UI

```bash
python3 web_ui.py
```

- Main UI: `http://localhost:5001/`
- Quiz UI: `http://localhost:5001/quiz`

## Project shape

- `src/train.py` — training loop
- `src/generate.py` — inference/generation
- `src/model.py` — transformer model
- `src/tokenizer.py` — tokenizer(s)
- `data/` — corpora/datasets
- `models/` — checkpoints
- `tests/` — pytest suite

## Test

```bash
pytest -q
```

## Roadmap (short)

Goal: reliable local LM for syntax-aware generation.

### ETA timeline
- Week 1: Stability
  - Keep tests green
  - Lock env/deps
  - Add CI smoke check
- Week 2: Data pipeline
  - Clean corpus + split train/eval
  - Add dataset stats + run metadata
- Week 3: Model quality
  - Tune architecture/context
  - Add top-k/top-p/temp controls
  - Add fixed prompt benchmark set
- Week 4: Reliability + release
  - Resume/checkpoint hardening
  - Basic experiment tracking
  - Clean train/generate/eval CLI flow

Expected MVP: ~4 weeks of focused work.

### Fast success checks
- Setup from clean clone in <10 min
- `pytest -q` in <30s
- Reproducible runs by seed/config
- Quality improves on fixed benchmark prompts

### Opus-replica reality check
- True Opus replica (capability-level parity): not realistic for a small solo setup.
- Practical "mini-Opus" (strong local specialist for your domain):
  - 1-2 months for a solid v1
  - 3-6 months for a genuinely strong system with robust eval + tooling

## Notes

- This is a learning/experimentation repo.
- Small model sizes keep runs cheap and debuggable.
- If you want production behavior, scale model/data/training pipeline substantially.
