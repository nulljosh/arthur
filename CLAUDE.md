# aether — Project Summary

## What We Built

A complete nano transformer LLM from scratch:
- **Model**: 3.5M params (12 layers, 256 embed dim)
- **Training**: 1000 epochs on balanced code + knowledge corpus
- **Inference**: C99 engine (350 LOC, 50K tok/s) + Vercel API
- **Result**: Loss converged to 0.09

## Architecture

Tokenizer → Transformer (4-12 layers) → Training → Checkpoint → Export

## Technology

- PyTorch (training)
- C99 (inference)
- FastAPI (production API)
- Vercel (deployment)
- Flask (local UI)

## Training Phases

1. **v0.1-0.2**: Foundation (jot/jung syntax, 0.57M params)
2. **v0.3**: Multilang code (27MB, 500 epochs, 0.57M params)
3. **v0.4**: Knowledge expansion (balanced corpus, 200 epochs)
4. **v1.0**: Scale to mini (1000 epochs, 3.5M params)

## Live

API: https://core-4tb2v49au-nulljosh-9577s-projects.vercel.app/api

## Next

- Custom domain setup
- Quantization (int8)
- ONNX export (browser)
- Larger scale (14M+ params)
