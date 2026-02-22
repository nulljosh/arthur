# core - Claude Notes

## What this repo is
Small transformer LM project for learning + fast iteration.
Not production scale.

## Current focus
- Char-level / syntax-aware modeling
- Reproducible train + eval loop
- Better test and error-case coverage
- Web UI iteration (controls + status + validation)
- Overnight automation loop for train/eval/report

## Fast commands
```bash
cd ~/Documents/Code/core
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

pytest -q
python3 src/train.py --epochs 100 --corpus tiny
python3 src/generate.py --prompt "func add" --length 80
```

## Architecture snapshot
- `src/tokenizer.py`: tokenizers
- `src/attention.py`: self + multi-head attention
- `src/transformer.py`: block + model
- `src/train.py`: dataset, train loop, generation helpers
- `src/chat.py`: chat wrapper

## Model tiers
- Nano: tiny demo
- Micro: learning baseline
- Mini: stronger local experiments

## Testing policy
- Keep `pytest -q` green on every push
- Add deterministic edge/error tests first
- Skip torch-dependent tests gracefully when torch is unavailable

## Near-term targets
1. Data quality checks + train/eval split
2. Stable benchmark prompts for regressions
3. Sampling controls (top-k/top-p/temp)
4. Resume/checkpoint reliability
5. Overnight runner: checkpointed training + eval pack + morning report
6. Web UI: prompt presets + run/eval panel + clearer failure traces

## Honest expectation
- Core MVP: ~4 weeks focused
- Domain mini-Opus: 1-2 months (v1), 3-6 months (strong)
- True Opus parity: unrealistic for solo scale
