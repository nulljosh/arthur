![Arthur](icon.svg)

# Arthur

![version](https://img.shields.io/badge/version-v3.0.0-blue)

## Operating Mode

Status: Demo / eval mode. Background training is parked on this 16 GB machine.


## Status

| Metric | Value |
|--------|-------|
| Params | 65M demo checkpoint |
| Architecture | MoE, RoPE, GQA, RMSNorm |
| Training Data | WikiText-103 |
| Training Mode | Manual only; no background training |
| Loss | 0.0029 |
| Eval Pass Rate | 16.7% |

Arthur is currently run as a stable local demo app with smoke tests and optional evals. Heavy jobs are manual-only and guarded by RAM/swap checks.

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

# Evaluate
python scripts/eval.py --checkpoint models/arthur_v3_65M_best.pt --size 65M

# Web UI
python scripts/web_ui.py  # Flask on :5001

# CLI chat
python scripts/cli.py

# Demo smoke test
python scripts/demo_smoke.py

# Manual local LLM
./scripts/local_llm_on.sh qwen2.5:3b
./scripts/local_llm_prompt.sh qwen2.5:3b "Say hello briefly."
./scripts/local_llm_off.sh
```

## Project Structure

```
src/           Tokenizer, transformer, eval harness
scripts/       Training, eval, CLI, web UI, ONNX export
daemon/        Parked training daemon files
cron/          Lightweight smoke/eval scheduler
tests/         Pytest suite
data/          Training corpora
models/        Checkpoints (.pt, gitignored)
inference/     C99 inference engine
public/        Web UI (index.html)
```

## Roadmap

| Milestone | Steps | Tokens | Time (M4) | Capability |
|-----------|-------|--------|-----------|------------|
| Current | 1K | ~1M | done | Incoherent output |
| Phase 1 | 100K | ~100M | ~28h | Word structure, partial phrases |
| Phase 2 | 300K | ~300M | ~84h | Coherent sentences |
| Phase 3 | 500K | ~500M | ~140h | Conversational ability |
| Phase 4 | 1M+ | ~1B | ~280h+ | Knowledge recall |

## License

MIT 2026, Joshua Trommel

## Quick Commands
- `./scripts/simplify.sh` - normalize project structure
- `./scripts/local_llm_on.sh` - start guarded manual Ollama mode
- `./scripts/local_llm_prompt.sh` - run one guarded local-model prompt
- `./scripts/local_llm_off.sh` - stop local-model processes
- `./scripts/monetize.sh . --write` - generate monetization plan (if available)
- `./scripts/audit.sh .` - run fast project audit (if available)
- `./scripts/ship.sh .` - run checks and ship (if available)
