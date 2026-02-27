# aether v1.0 — Production Ready

Nano transformer LLM, 3.5M parameters, built from scratch in PyTorch. **Live API deployed to Vercel.**

![aether architecture diagram](architecture.svg)

## 🚀 Live Demo

**API**: https://core-4tb2v49au-nulljosh-9577s-projects.vercel.app/api

- `GET /api` — API info
- `GET /api/health` — Health check
- `GET /api/info` — Model details (3.5M params, loss 0.09)

## Quick Start

```bash
git clone https://github.com/nulljosh/aether.git && cd nous
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Inference
python src/generate.py --prompt "fn " --temperature 0.3

# Local API
python -m uvicorn api.index:app --reload
# http://localhost:8000/docs
```

## Model

| Aspect | Value |
|--------|-------|
| Parameters | 3.5M |
| Layers | 12 |
| Embed Dim | 256 |
| Training Epochs | 1000 |
| Final Loss | 0.09 |
| Speed (C engine) | 50K tok/s |
| Training Corpus | Balanced code + knowledge |

## Training Journey

```
v0.1: jot (200 epochs, 0.57M) → Foundation
v0.2: jung (100 epochs, 0.57M) → Specialization
v0.3: multilang (500 epochs, 27MB, 0.57M) → Code understanding
v0.4: knowledge (200 epochs, 3.2MB, 0.57M) → Generalization
v1.0: mini (1000 epochs, 3.5M) → Production ✅
```

## The Stack

- **PyTorch** — Training harness with checkpointing
- **C99 Inference** — 350 LOC, mmap weights, zero deps
- **FastAPI** — Production serverless API
- **Vercel** — Deployed and live
- **Flask Web UI** — Local chat interface
- **Aether Daemon** — Continuous background training

## Architecture

```
Data (code + knowledge)
  ↓
PyTorch Trainer (1000 epochs)
  ↓
Checkpoint (3.5M params, loss 0.09)
  ├→ C Inference Engine (350 LOC)
  ├→ Vercel API (live)
  └→ Flask Web UI (local)
```

## Why Built from Scratch

"What I cannot create, I do not understand." — Feynman

- Every layer visible and understandable
- No black boxes, no framework magic
- Learn how transformers actually work
- C99 inference engine runs anywhere

## Production Checklist

- [x] Model trained (3.5M params, 1000 epochs, loss 0.09)
- [x] FastAPI setup
- [x] Vercel deployment
- [x] Live API endpoints
- [x] Web UI (local)
- [x] Documentation
- [x] GitHub + GitHub Pages
- [ ] Custom domain
- [ ] Model quantization
- [ ] Browser ONNX export

## API Endpoints

### GET /api
```json
{
  "name": "aether",
  "version": "1.0.0",
  "status": "ok"
}
```

### GET /api/health
```json
{
  "status": "ok"
}
```

### GET /api/info
```json
{
  "name": "aether",
  "version": "1.0.0",
  "params": "3.5M",
  "github": "https://github.com/nulljosh/aether"
}
```

## Testing

```bash
pytest tests/test_api.py -v
```

## Local Web UI

```bash
python index.py
# http://localhost:5001
```

## Links

- **GitHub**: https://github.com/nulljosh/aether
- **Live API**: https://core-4tb2v49au-nulljosh-9577s-projects.vercel.app/api
- **GitHub Pages**: https://nulljosh.github.io/aether/
- **Documentation**: See [DEPLOY.md](DEPLOY.md)

## Status

✅ **v1.0 Production Ready**

Model converged. API deployed. All systems go.

## License

MIT 2026, Joshua Trommel
