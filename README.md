# arthur

![icon](./icon.svg)

A small language model trained from scratch. 3.5M parameters (v1.0), 65M parameters (v2.0 in training).

Vision: Build a production-grade LLM deployable locally with code understanding and reasoning capabilities.

## Status

| Version | Params | Loss | Grade | Status |
|---------|--------|------|-------|--------|
| v1.0 | 3.5M | 0.1819 | A+ | Live (research prototype) |
| v2.0 | 65M | Training | - | 10 weeks to ship (accelerated) |

## Features

- Compact models: 3.5M (v1.0) or 65M (v2.0) parameters
- Client-side inference: ONNX Runtime via WebAssembly (no backend)
- BPE tokenizer: 32K vocab, character-level fallback
- Web UI: Dark/light theme, responsive design
- Local deployment: No API keys, no cloud, no surveillance
- Code understanding: Trained on reasoning, math, science, systems programming

## Quick Start

```bash
git clone https://github.com/nulljosh/arthur.git
cd arthur
pip install -r requirements.txt
python -m http.server 8000
# Open http://localhost:8000/public/chat.html
```

## Architecture

- Tokenizer: BPE, 32K vocab, trained on balanced dataset
- Model: Transformer, 8K context window, Flash Attention
- Inference: ONNX (CPU), Ollama (optional GPU)
- Training: PyTorch, gradient checkpointing, bfloat16

## Roadmap

### Phase 1: v2.0 Training (10 weeks - accelerated) - IN PROGRESS
- BPE tokenizer (32K vocab)
- Enhanced dataset (10K diverse examples: math, science, pop culture, Wikipedia, tech, current events)
- ArthurV2 model (65M params, 8K context)
- Training script (src/train_v2.py) running
- Gradient checkpointing + 2x batch optimization
- Target: Ship by mid-May 2026

### Phase 2: Optimization & Export (weeks 10-12)
- Quantization (int8, 4-bit options)
- ONNX export
- Ollama model definition
- Performance benchmarking

### Phase 3: OpenClaw Integration (weeks 12+)
- Register as agentId: "arthur" in OpenClaw
- Mount code directories for read/write
- Add code editing tools (read, write, exec)
- Fine-tune on custom codebase patterns
- Deploy as HTTP inference server

### Phase 4: Production (weeks 14+)
- Domain-specific fine-tuning
- Chain with other agents (Claude for validation, Qwen for speed)
- Optimize for systems/code reasoning
- Documentation + guides

## Development

```bash
# Train v2.0
python src/train_v2.py --dataset data/balanced_dataset.jsonl --epochs 100

# Export to ONNX
python export_onnx.py --checkpoint models/v2_final.pt --quantize int8

# Run locally via Ollama
ollama create arthur -f Modelfile
ollama serve

# Web UI (local inference)
python -m http.server 8000
# Visit http://localhost:8000/public/chat.html
```

## Performance

- v1.0: Loss 0.1819, 78% math accuracy, ~50ms per token (ONNX/WASM)
- v2.0 (target): Loss <0.10, >80% math accuracy, 2-3ms per token (optimized)

## Training Data

v2.0 trained on 10K diverse examples:
- Mathematics & science (calculus, physics, biology)
- Pop culture (films, music, entertainment)
- Wikipedia-style articles (technology, geography, history)
- Systems programming (Linux, networking, databases)
- Current events & tech trends (2026)

## Files

- src/model.py: ArthurV2 (65M transformer)
- src/tokenizer.py: BPE tokenizer
- src/train_v2.py: Training loop
- src/inference.py: ONNX inference
- data/balanced_dataset.jsonl: 10K training examples
- public/chat.html: Web UI
- Modelfile: Ollama model definition

## License

MIT

---

Built with: PyTorch, ONNX Runtime, Ollama, React

Live: https://arthur-prod.vercel.app (v1.0 demo)

GitHub: https://github.com/nulljosh/arthur

Next: v2.0 shipping ~mid-May 2026 → Local deployment → OpenClaw integration → Custom fine-tuning
