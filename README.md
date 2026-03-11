![Arthur](icon.svg)

# Arthur

![version](https://img.shields.io/badge/version-v3.0.0-blue)

## Training Status

**Progress:** Epoch 0/3 (0% complete)
**Latest Checkpoint:** None
**Last Loss:** 0.0180
**Updated:** 2026-03-11 12:48

Status: Daemon auto-training when idle in a 16GB-safe mode. It stays on 65M, uses short resumable runs, and avoids long 125M sessions on this machine class.


## Status

| Metric | Value |
|--------|-------|
| Params | 65M default active training target on 16GB-class machines |
| Architecture | MoE, RoPE, GQA, RMSNorm |
| Training Data | WikiText-103 |
| Training Mode | Resumable 250-step chunks with aggressive checkpoints |
| Loss | 0.0029 |
| Eval Pass Rate | 16.7% |

Auto-trains in background via launchd daemon. Resource-gated (disk, CPU, RAM) and now hard-clamped for 16GB machines.

## Architecture

<div align="center">
  <img src="./architecture.svg" width="700" alt="Arthur Architecture">
</div>

- **Tokenizer**: BPE, 10K vocab
- **Model**: Decoder-only transformer, 32K context, MoE (4 experts, top-2 routing), RoPE, GQA, RMSNorm
  - MoE top-2/4 routing means only ~55% of parameters are active per forward pass.
- **Training**: PyTorch, gradient checkpointing, bfloat16, cosine LR, WikiText-103 streaming
- **Inference**: ONNX Runtime (CPU), C99 engine

## Quick Start

```bash
git clone https://github.com/nulljosh/arthur.git
cd arthur
pip install -r requirements.txt

# Train on WikiText-103 with 16GB-safe defaults
python scripts/train.py --size 65M --steps 100000 --resume

# Evaluate
python scripts/eval.py --checkpoint models/arthur_v3_65M_best.pt --size 65M

# Web UI
python scripts/web_ui.py  # Flask on :5001

# CLI chat
python scripts/cli.py
```

## Project Structure

```
src/           Tokenizer, transformer, eval harness
scripts/       Training, eval, CLI, web UI, ONNX export
daemon/        Watchdog training daemon (launchd)
cron/          Scheduled tasks (dispatch.sh)
tests/         Pytest suite
data/          Training corpora
models/        Checkpoints (.pt, gitignored)
inference/     C99 inference engine
public/        Web UI (chat.html)
```

## Roadmap

| Milestone | Steps | Tokens | Time (M4) | Capability |
|-----------|-------|--------|-----------|------------|
| Current | 1K | ~1M | done | Incoherent output |
| Phase 1 | 100K | ~100M | ~28h | Word structure, partial phrases |
| Phase 2 | 300K | ~300M | ~84h | Coherent sentences |
| Phase 3 | 500K | ~500M | ~140h | Conversational ability |
| Phase 4 | 1M+ | ~1B | ~280h+ | Knowledge recall |

## Next Steps

1. Keep the base trainer boring: 65M only, 128-token context, 250-step resumable runs
2. Build a tiny eval set that runs fast enough to gate every checkpoint
3. Improve sample quality before considering any larger model phase
4. Export to ONNX when eval pass rate >50%
5. Quantize for deployment (int8, 4-bit)

## License

MIT 2026, Joshua Trommel

## Quick Commands
- `./scripts/simplify.sh` - normalize project structure
- `./scripts/monetize.sh . --write` - generate monetization plan (if available)
- `./scripts/audit.sh .` - run fast project audit (if available)
- `./scripts/ship.sh .` - run checks and ship (if available)
