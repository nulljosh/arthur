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

## Roadmap

Goal: ship a small, reliable LM stack for local training + syntax-aware generation.

### Now (0-2 weeks)
- Stabilize test suite and CI
- Lock Python + dependency versions
- Keep `pytest -q` green on every push
- Add one fast smoke train/infer check in CI

Done when:
- Tests pass locally + CI
- No flaky tests
- One-command setup works from clean clone

### Next (2-4 weeks)
- Improve data pipeline for jot/code corpora
- Add dataset quality checks (length, charset, duplicates)
- Add train/eval split and basic validation metrics
- Track run metadata (config, seed, loss, checkpoint path)

Done when:
- Reproducible runs by seed/config
- Validation loss tracked per run
- Dataset stats visible before training

### Model quality (4-6 weeks)
- Tune baseline architecture (depth/width/context)
- Add better sampling controls (top-k, top-p, temperature)
- Improve tokenizer strategy for syntax-heavy data
- Add small benchmark prompts for regression checks

Done when:
- Better qualitative generations on fixed prompts
- No major regressions on benchmark set

### Training reliability (6-8 weeks)
- Checkpoint/resume hardening
- Gradient clipping + schedule tuning
- Early-stop and failure recovery hooks
- Lightweight experiment table (CSV/JSON)

Done when:
- Interrupted runs recover cleanly
- Loss curves are stable across repeated runs

### Productization (8+ weeks)
- Clean CLI for train/generate/eval
- Optional web UI polish
- Versioned model artifacts
- Clear release notes per model revision

Done when:
- New user can train + generate in under 10 minutes
- Release process is repeatable

### Non-goals (for now)
- Massive scale training
- Multi-node distributed training
- Production API serving/SLA

### Success metrics
- Setup time: <10 minutes from clean clone
- Test time: <30 seconds for core suite
- Reproducibility: same seed => similar loss curve
- Quality: benchmark prompts improve month-over-month

## Notes

- This is a learning/experimentation repo.
- Small model sizes keep runs cheap and debuggable.
- If you want production behavior, scale model/data/training pipeline substantially.
