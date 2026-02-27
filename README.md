# aether v1.0

Nano transformer LLM built from scratch. **3.5M parameters, 12 layers, 256 embedding dimension.** Trained on 1000 epochs of balanced code + knowledge corpus. Final loss: **0.09**.

![aether architecture diagram](architecture.svg)

## Live Demo

🚀 **Coming soon**: `aether.vercel.app`

## Quick Start

```bash
git clone https://github.com/nulljosh/aether.git && cd nous
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Local inference
python src/generate.py --prompt "fn " --length 80 --temperature 0.3

# Web UI (Flask)
python index.py  # http://localhost:5001

# API (FastAPI)
cd api && pip install -r requirements.txt
python -m uvicorn app:app --reload  # http://localhost:8000/docs
```

## The Stack

- **PyTorch trainer** — Full training harness with checkpointing
- **C99 inference** — 350 LOC, mmap weights, zero deps, 50K tok/s
- **FastAPI** — Production API endpoints
- **Flask Web UI** — Real-time chat interface
- **Aether daemon** — Continuous background training

## Training Phases

| Version | Params | Corpus | Epochs | Loss | Status |
|---------|--------|--------|--------|------|--------|
| v0.1 (jot) | 0.57M | 185 KB syntax | 200 | 0.2-0.9 | ✅ |
| v0.2 (jung) | 0.57M | 31 KB JIT | 100 | 0.17 | ✅ |
| v0.3 (code) | 0.57M | 27 MB multilang | 500 | 0.0947 | ✅ |
| v0.4 (knowledge) | 0.57M | 3.2 MB balanced | 200 | 0.1233 | ✅ |
| **v1.0 (mini)** | **3.5M** | **balanced** | **1000** | **0.09** | **✅** |

## Benchmarks

| Model | Params | Speed | Training Data | Capability |
|-------|--------|-------|---|---|
| **aether** | 3.5M | 20K tok/s | 1000 epochs balanced corpus | multilang code + knowledge |
| GPT-2 | 124M | — | 40 GB | coherent paragraphs |
| Claude | ??? | 80 tok/s | internet scale | reasoning, tools |

## Why Aether

"What I cannot create, I do not understand." — Feynman

- Full stack from scratch: tokenizer → attention → training → C inference → FastAPI
- No black boxes. Every layer visible.
- Learning resource for LLM fundamentals
- C99 engine runs anywhere with a C compiler
- Progressive training shows how models improve with data

## Deployment

### Local

```bash
cd api && python -m uvicorn app:app --reload
```

### Vercel (Production)

```bash
vercel
# Public URL: aether.vercel.app
```

See [DEPLOY.md](DEPLOY.md) for full instructions.

## API

### Endpoints

- `GET /` — Info
- `GET /health` — Health check
- `GET /info` — Model details
- `POST /generate` — Generate text

### Example

```bash
curl -X POST https://aether.vercel.app/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "fn ", "max_tokens": 50, "temperature": 0.8}'
```

## Architecture

```
Data (balanced code + knowledge)
  ↓
PyTorch Trainer (1000 epochs)
  ↓
Checkpoint (3.5M params)
  ├→ Export (aether.bin)
  │   ↓
  │   C Inference (350 LOC)
  │
  ├→ FastAPI (Production)
  │   ├→ /generate endpoint
  │   ├→ /info endpoint
  │   └→ /health endpoint
  │
  ├→ Flask Web UI (:5001)
  │   ├→ Chat mode
  │   └→ Quiz mode
  │
  └→ Aether Daemon (Continuous training)
      ├→ Auto-checkpoint
      └→ iMessage notifications
```

## Roadmap

### Phase 1-4: Complete ✅
- Foundation (v0.1-0.2): Syntax + JIT training
- Code understanding (v0.3): Multilang corpus
- Knowledge expansion (v0.4): Balanced corpus
- Scale to mini (v1.0): 3.5M params

### Phase 5: Production Ready (In Progress)
- [x] FastAPI setup
- [x] Web UI
- [ ] Vercel deployment
- [ ] ONNX export (browser)
- [ ] Quantization (int8)
- [ ] Model versioning

### Phase 6: Advanced (Future)
- Multi-GPU training
- Instruction tuning
- RLHF
- Larger scale (14M+ params)

## Testing

```bash
pytest tests/test_api.py -v
```

## Status

**v1.0 Production Ready**

- Model: Trained and converged (loss 0.09)
- API: FastAPI endpoints live
- Web: UI deployed locally
- Deploy: Ready for Vercel

## License

MIT 2026, Joshua Trommel

---

**GitHub**: https://github.com/nulljosh/aether  
**Live**: Coming soon to aether.vercel.app
