# arthur

A small language model trained from scratch. 3.5M parameters, character-level generation, ONNX inference.

## Features

- **Compact model** — 65M params (v2.0) trained on 7K balanced examples
- **Client-side inference** — ONNX Runtime via WebAssembly (no backend required)
- **Character-level generation** — BPE tokenizer, configurable temperature & max tokens
- **Web UI** — Dark/light theme, responsive, clean design
- **Benchmarked** — Loss: 0.1819 (v1.0), Grade A+

## Quick Start

```bash
git clone https://github.com/nulljosh/arthur.git
cd arthur
pip install -r requirements.txt
python -m http.server 8000
# Open http://localhost:8000/public/chat.html
```

## Architecture

![Architecture Diagram](./architecture.svg)

## Project Structure

```
arthur/
├── src/                    # Core model implementation
│   ├── tokenizer.py       # BPE tokenizer
│   ├── model.py           # ArthurV2 (65M params)
│   ├── trainer.py         # Training loop
│   └── inference.py       # ONNX inference
├── data/                   # Training datasets
│   └── balanced_dataset.json
├── models/                 # Trained weights & ONNX exports
│   ├── arthur.onnx        # Client-side inference
│   └── vocab.json
├── public/                 # Web UI
│   ├── chat.html          # Main chat interface
│   └── model/             # Static assets
├── api/                    # API fallback
│   └── generate.py        # Server-side generation
├── tests/                  # Unit tests
├── cron/                   # Daemon scripts
└── logs/                   # Training logs
```

## Training Status

| Version | Params | Loss | Grade |
|---------|--------|------|-------|
| v1.0    | 3.5M   | 0.1819 | A+ |
| v2.0    | 65M    | TBD (in progress) | - |

## API

### Client-side (ONNX)
```javascript
const response = await fetch('/model/arthur.onnx');
const session = await ort.InferenceSession.create(response);
// Character-level generation with temperature control
```

### Server-side (fallback)
```bash
POST /api/generate
Content-Type: application/json

{
  "prompt": "Q: What is AI?\nA:",
  "length": 120,
  "temperature": 0.5
}
```

## Performance

- **Inference latency** — ~50ms per token (ONNX/WASM)
- **Model size** — 8.4MB (ONNX quantized)
- **Math accuracy** — 78% (v1.0 research prototype)

## Development

```bash
# Train v2.0
python src/train_v2.py --dataset data/balanced_dataset.json

# Export to ONNX
python export_onnx.py --checkpoint models/v2_final.pt

# Run tests
pytest tests/

# Deploy
vercel deploy
```

## License

MIT

---

**Built with:** PyTorch, ONNX Runtime, React, Vercel

**Live:** https://arthur-prod.vercel.app
