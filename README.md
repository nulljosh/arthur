# core

Nano transformer LLM. Built from scratch in PyTorch with a zero-dependency C99 inference engine.

## Architecture

![Architecture](architecture.svg)

## Stack

- Python + PyTorch
- C99 inference engine (`inference/core.c`, ~350 LOC, mmap weight loading, zero deps)
- Flask web UI (port 5001)
- launchd services: `com.joshua.core-train`, `com.joshua.core-web`

## Dev

```bash
source venv/bin/activate
pytest -q
python src/train.py
```

## Wikipedia Training

```bash
# Download WikiText-103
python scripts/download_wikitext.py

# Train on wiki corpus
python src/train.py --corpus wikitext --epochs 100

# Export weights for C engine
python scripts/export_weights.py
# writes models/core.bin

# Build C engine
cd inference && make

# Run inference
./inference/core models/core.bin "Q: What is 5+3?\nA:" --temp 0.0 --tokens 50
```

## Roadmap

- [x] Char-level training
- [x] C inference engine
- [x] WikiText-103 infrastructure
- [ ] BPE tokenizer training
- [ ] Beam search decoding
- [ ] INT8 quantization
- [ ] API endpoint
- [ ] Benchmark suite
