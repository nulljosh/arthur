# jore

Nano transformer LLM built from scratch. PyTorch training loop, C99 inference engine, overnight automation.

**GitHub**: https://github.com/nulljosh/jore

## Architecture

![Architecture](architecture.svg)

## Stack

- PyTorch — transformer model, training loop, checkpoint management
- C99 inference engine (`inference/jore.c`, ~350 LOC, mmap weight loading, zero deps)
- Flask web UI (port 5001, chat + quiz)
- Char-level tokenizer (BPE for WikiText-103)
- AdamW + cosine LR decay, gradient clipping
- Cron automation — train every 4h, daily eval + report

## Dev

```bash
source venv/bin/activate
pytest -q
python src/train.py --corpus jot --epochs 100
python src/train.py --corpus wiki --tokenizer bpe --model-size wiki
```

## Inference

```bash
# Export checkpoint to flat binary
python scripts/export_weights.py
# writes models/jore.bin (magic "JORE", float32, mmap-ready)

# Build C engine
cd inference && make

# Run
./inference/jore models/jore.bin "Q: What is 5+3?\nA:" --temp 0.0 --tokens 50
```

## Model Tiers

| Tier  | Params | Layers | Embed | Heads | FF   |
|-------|--------|--------|-------|-------|------|
| Nano  | ~15K   | 2      | 32    | 2     | 64   |
| Micro | ~630K  | 4      | 128   | 4     | 512  |
| Mini  | ~3.5M  | 6      | 256   | 8     | 1024 |

## Roadmap

- [x] Char-level training
- [x] C99 inference engine (mmap, zero deps)
- [x] WikiText-103 training infrastructure
- [x] Overnight cron automation + eval harness
- [x] Flask web UI
- [ ] BPE tokenizer training
- [ ] Beam search decoding
- [ ] INT8 quantization
- [ ] API endpoint
- [ ] Benchmark suite
