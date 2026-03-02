# arthur

![icon](./icon.svg)

A 65M parameter language model trained from scratch on a Mac Mini M4.

## Status

| Version | Params | Loss | Status |
|---------|--------|------|--------|
| v1.0 | 3.5M | 0.1819 | Research prototype |
| v2.0 | 65M | 0.0115 | Trained, pending deployment |

## Architecture

- **Tokenizer**: BPE, 10K vocab
- **Model**: Decoder-only transformer, 8K context, 12 layers, Flash Attention
- **Training**: PyTorch, gradient checkpointing, bfloat16, cosine LR
- **Inference**: ONNX Runtime (CPU), C99 engine, Ollama (optional)

## Quick Start

```bash
git clone https://github.com/nulljosh/arthur.git
cd arthur
pip install -r requirements.txt
python -m http.server 8000
# Open http://localhost:8000/public/chat.html
```

## Project Structure

```
src/           Tokenizer, transformer, training, eval harness
scripts/       Dataset creation, inference tests, CLI, export
daemon/        Watchdog training daemon (launchd)
cron/          Overnight training and eval cron jobs
tests/         Pytest suite
data/          Training corpora and datasets
models/        Saved checkpoints (.pt, gitignored)
docs/          Whitepaper, deployment plan, release notes
inference/     C99 inference engine
public/        Web UI (chat.html)
```

## Next Steps

1. Export to ONNX + quantize (int8)
2. Integrate as OpenClaw agent
3. Fine-tune on custom codebase patterns
4. Deploy as local inference server

## License

MIT 2026, Joshua Trommel
