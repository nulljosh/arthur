# jore - Claude Notes

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
- `inference/jore.c`: C99 inference engine (single file, ~350 LOC)
- `inference/Makefile`: builds `inference/jore` binary
- `scripts/export_weights.py`: exports PyTorch checkpoint to `models/jore.bin`

## C inference engine
- Binary format: `models/jore.bin` (magic "JORE", version 1, config, vocab, float32 weights)
- Layout: 4B magic + 4B version + 24B config (6x uint32) + vocab entries (uint32 len + UTF-8) + contiguous float32 tensors
- Tensor order: token_embed, pos_embed, [per-block: qkv_w, qkv_b, out_w, out_b, ff1_w, ff1_b, ff2_w, ff2_b, ln1_w, ln1_b, ln2_w, ln2_b], ln_f_w, ln_f_b, head
- Build: `cd inference && make` (requires only cc + libc + libm)
- Run: `./inference/jore models/jore.bin "Q: prompt\nA:" --temp 0.5 --tokens 100`
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
- Jore MVP: ~4 weeks focused
- Domain mini-Opus: 1-2 months (v1), 3-6 months (strong)
- True Opus parity: unrealistic for solo scale

## Roadmap
- [ ] Char-level jot syntax training
- [ ] Larger training corpus
- [ ] Beam search decoding
- [ ] Model quantization (INT8)
- [ ] API endpoint for inference
- [ ] Benchmark suite
