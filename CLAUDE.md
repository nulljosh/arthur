# aether v1.0

Nano transformer LLM built from scratch. 3.5M parameters, C99 inference engine, PyTorch training.

## The Work

- Complete training loop in PyTorch (AdamW, cosine LR, gradient clipping)
- 3.5M parameter transformer trained over 1000 epochs
- C99 inference engine (350 LOC, zero dependencies, 50K tok/s)
- Web chat UI (Apple Liquid Glass aesthetic)
- Deployed to Vercel (static splash + mock API)

## How to Use

```bash
# Local inference (real PyTorch model)
python web_ui.py
# Opens web chat at http://localhost:5001

# C engine
cd inference && make && ./aether ../models/aether.bin "fn " --temp 0.3
```

## Vercel Deployment

- `vercel.json` uses `"framework": null` to prevent Flask auto-detection from requirements.txt
- `public/index.html` is the splash page, `public/chat.html` is the chat UI
- `api/index.js` handles all API routes (mock responses, no PyTorch on serverless)
- Rewrites: `/api/*` -> `api/index.js`

## Status

Complete and deployed. Local inference works. Vercel serves static splash + chat UI with mock API.

## Links

- GitHub: https://github.com/nulljosh/aether
- Vercel: https://aether-pink.vercel.app
- Chat: https://aether-pink.vercel.app/chat.html
