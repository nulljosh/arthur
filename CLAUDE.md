# aether - Claude Notes

## What this repo is
Aether: a nano transformer LLM built from scratch for learning. Fast iteration. (Formerly: jore, then nous)

## Current focus
- Char-level jot syntax modeling
- Reproducible train + eval loop
- C99 inference engine (350 LOC, zero deps)
- Web UI iteration (controls + status + validation)
- Overnight automation loop (aether daemon) for train/eval/report

## Fast commands
```bash
cd ~/Documents/Code/nous
source venv/bin/activate
python src/train.py --epochs 100 --corpus jot
python src/generate.py --prompt "fn " --length 80
pytest -q
```

## Architecture snapshot
- `src/tokenizer.py`: tokenizers (char-level, BPE)
- `src/attention.py`: self + multi-head attention
- `src/transformer.py`: block + model
- `src/train.py`: dataset, train loop, generation helpers
- `inference/aether.c`: C99 inference engine (single file, ~350 LOC)
- `inference/Makefile`: builds `inference/aether` binary
- `scripts/export_weights.py`: exports PyTorch checkpoint to `models/aether.bin`

## C inference engine
- Binary format: `models/aether.bin` (magic "AETHER", version 1, config, vocab, float32 weights)
- Layout: 4B magic + 4B version + 24B config (6x uint32) + vocab entries (uint32 len + UTF-8) + contiguous float32 tensors
- Build: `cd inference && make` (requires only cc + libc + libm)
- Run: `./inference/aether models/aether.bin "Q: prompt\nA:" --temp 0.5 --tokens 100`
- Weight loading via mmap (zero-copy, instant startup)
- Config read from binary header (not hardcoded), adapts to any checkpoint tier
- Pre-norm transformer with fused QKV, GELU tanh approx, causal attention

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
- Aether MVP: ~4 weeks focused
- Domain mini-LLM: 1-2 months (v1), 3-6 months (strong)
- True Claude parity: unrealistic for solo scale

## Automation
- **aether daemon** (~/.local/bin/aether): Continuous background training
- **aether-report** (~/.local/bin/aether-report): Progress updates every 3 min
- **aether-watch** (~/.local/bin/aether-watch): Status display script

See ROADMAP.md for phases and success metrics.
