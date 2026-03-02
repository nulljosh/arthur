# Arthur -- Custom LLM Project

65M parameter transformer trained from scratch. Local inference, no API keys.

## Status
- **v1.0**: 3.5M params, loss 0.1819 -- research prototype
- **v2.0**: 65M params, loss 0.0115 -- trained, pending deployment

## Architecture
- **Tokenizer**: BPE, 32K vocab (`src/bpe_tokenizer.py`, `src/tokenizer.py`)
- **Model**: Transformer, 8K context, 12 layers (`src/transformer_v2.py`, `src/transformer.py`)
- **Training**: PyTorch, gradient checkpointing, bfloat16 (`src/train_v2.py`)
- **Inference**: ONNX Runtime (CPU), C99 engine (`inference/nous.c`), Ollama (optional)

## Key Commands
```bash
python src/train_v2.py --dataset data/balanced_dataset.jsonl --epochs 100
python scripts/export_onnx.py --checkpoint models/v2_final.pt --quantize int8
python scripts/web_ui.py          # Flask UI on :5001
python scripts/cli.py             # Terminal chat
pytest -q                         # Run tests
```

## Next Steps
1. Export to ONNX + quantize (int8, 4-bit)
2. Register as `agentId: "arthur"` in OpenClaw
3. Fine-tune on custom codebase patterns
4. Deploy as local HTTP inference server
