# Arthur -- Custom LLM Project

166M total parameter (90M active per token) transformer trained from scratch. Local inference, no API keys.

## Status
- **Params**: 166M total / 90M active (MoE top-2/4 routing, RoPE/GQA, WikiText-103)
- **Training**: ~12K steps of 100K target
- **Loss**: 0.0029 (step 947)
- **Eval pass rate**: 16.7% (model produces output but lacks coherent language structure)

## Architecture
- **Tokenizer**: BPE, 10K-50K vocab (`src/bpe_tokenizer.py`, `src/tokenizer.py`)
- **Model**: MoE transformer, 32K context, RoPE, GQA, RMSNorm (`src/transformer.py`)
- **Training**: WikiText-103 streaming, PyTorch, bfloat16 (`scripts/train.py`)
- **Eval**: Prompt suite eval harness (`scripts/eval.py`, `src/eval_harness.py`)
- **Inference**: ONNX Runtime (CPU), C99 engine (`inference/arthur.c`)

## Key Commands
```bash
# Train
python scripts/train.py --size 65M --steps 100000
# Resume from checkpoint
python scripts/train.py --size 65M --steps 100000 --resume

# Eval
python scripts/eval.py --checkpoint models/arthur_v3_65M_best.pt --size 65M

# Inference
python scripts/web_ui.py          # Flask UI on :5001
python scripts/cli.py             # Terminal chat

# ONNX export
python scripts/export_onnx.py

# Tests
pytest -q
```

## Next Steps
1. Continue WikiText-103 training (300K+ steps)
2. Instruction tuning (identity prompts, Q&A pairs)
3. Export to ONNX + quantize (int8, 4-bit)
4. Deploy as local HTTP inference server

## Quick Commands
- `./scripts/simplify.sh`
- `./scripts/monetize.sh . --write`
- `./scripts/audit.sh .`
- `./scripts/ship.sh .`
