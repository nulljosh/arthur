# aether v1.0 — Production Deployment

## Local Testing

```bash
cd api
pip install -r requirements.txt
python -m uvicorn app:app --reload
# Visit http://localhost:8000/docs
```

## Vercel Deployment

1. Push to GitHub (done ✓)
2. Link repo to Vercel: `vercel link`
3. Deploy: `vercel`
4. Public URL: `aether.vercel.app`

### Environment
- Runtime: Python 3.11
- Framework: FastAPI
- Auto-scaling: Yes

## API Endpoints

- `GET /` — Info
- `GET /health` — Health check
- `GET /info` — Model details
- `POST /generate` — Generate text

## Example Request

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "fn ", "max_tokens": 50, "temperature": 0.8}'
```

## Status

- [x] FastAPI setup
- [x] Endpoints defined
- [ ] Vercel deploy (manual: `vercel` CLI)
- [ ] Live URL
- [ ] Model integration (local inference works)

