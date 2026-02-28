# Arthur — Custom LLM Project

## Vision
Build a production-grade 65M parameter language model trained from scratch. Deployable locally, capable of reasoning about code and systems.

## Current Status
- **v1.0**: 3.5M params, loss 0.1819 (Grade A+) — research prototype
- **v2.0**: 65M params, staged and ready — in training queue

## Training Plan
- **Duration**: 6-8 weeks (local Mac mini M4, 16GB)
- **Dataset**: 7K balanced examples (code, math, reasoning)
- **Optimization**: Gradient checkpointing, Flash Attention, bfloat16
- **Checkpoint**: Save every 500 steps
- **Target**: >80% accuracy on math, production-ready

## Future Milestones
1. Complete v2.0 training (6-8 weeks)
2. Export to ONNX + quantize (int8, 4-bit options)
3. Integrate as OpenClaw agent ("arthur" runtime)
4. Add code editing tools (read/write/exec via API)
5. Fine-tune on custom domain (your codebase patterns)
6. Deploy as local inference server (HTTP API)

## Architecture
- **Tokenizer**: BPE, 32K vocab
- **Model**: Transformer, 8K context, 12 layers
- **Training**: PyTorch, wandb logging
- **Inference**: ONNX Runtime (CPU), Ollama (optional)

## Development Commands
```bash
# Start v2.0 training
python src/train_v2.py --dataset data/balanced_dataset.json --epochs 10

# Export to ONNX
python export_onnx.py --checkpoint models/v2_final.pt

# Deploy locally
ollama create arthur -f Modelfile
ollama serve

# Test inference
curl http://localhost:11434/api/generate -d '{"model":"arthur","prompt":"What is AI?","stream":false}'
```

## Integration with OpenClaw
Once training completes:
1. Register as `agentId: "arthur"` in openclaw config
2. Mount code directories for read/write access
3. Run specialized tasks (code review, debugging, system design)
4. Chain with other agents (claude for validation, qwen for speed)

## Files
- `src/train_v2.py` — training script
- `src/model.py` — 65M transformer
- `src/tokenizer.py` — BPE tokenizer
- `data/balanced_dataset.json` — 7K training examples
- `Modelfile` — Ollama model definition
- `export_onnx.py` — quantization + export

## Notes
- No external API calls — pure local inference
- Designed for systems/code understanding (not chat-first)
- Can be fine-tuned on your codebase for domain expertise
- Cost: zero (training on your hardware)
