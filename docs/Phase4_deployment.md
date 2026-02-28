# Phase 4: Deployment & Quantization

## Quantization Strategy

### Target: INT8 / FP8 Quantization
- 65M params × 4 bytes (FP32) = 260MB model
- After INT8: 260MB → 65MB (4x compression)
- Accuracy loss: <2% on benchmarks
- Inference speed: 2x faster

### Tools
1. **PyTorch quantization**
   ```python
   import torch.quantization as quant
   model_int8 = quant.quantize_dynamic(
       model, {nn.Linear}, dtype=torch.qint8
   )
   ```

2. **ONNX export** (optional, for broader compatibility)
   ```python
   torch.onnx.export(model, example_input, "arthur.onnx")
   ```

## Weight Export for C99 Inference

### Current Status
- C99 binary exists: `inference/nous.c` (350 LOC)
- Requires binary weight format (not .pt)

### Export Process
1. Save model state dict as binary
2. Create header file with weight layout
3. Compile C99 inference engine
4. Test on sample prompts

```c
// inference/weights.h
typedef struct {
    float* embeddings;      // [10000 × 512]
    float* layer_weights;   // [12 layers × ...]
    int num_layers;
    int embed_dim;
    int vocab_size;
} ModelWeights;
```

## Vercel Deployment

### Current Setup
- Static HTML splash page: `public/index.html`
- Mock API: `api/index.js` (returns random responses)
- Chat UI: `public/chat.html`

### Production Deployment
1. Keep static splash (no GPU needed)
2. Add real inference API endpoint
3. Use Vercel Blob for model weights (if <10GB)
4. Or: Self-host on GPU (AWS Lambda, RunPod, Together.ai)

### Option A: Vercel Blob Storage
```javascript
// api/inference.js
import { blob } from '@vercel/blob';

export default async function handler(req, res) {
  const weights = await blob.download('arthur-model.bin');
  // Load model, run inference, return response
}
```

### Option B: Self-Hosted GPU
- RunPod (GPU rental, $0.20/hr)
- Together.ai (shared GPU, $0.02 per 1M tokens)
- AWS Lambda + EFS (high latency, not ideal)

## Benchmark & Testing

### Pre-Deployment Checklist
- [ ] Quantized model accuracy >80% on math
- [ ] Inference latency <500ms (P99) on M4
- [ ] Throughput >100 tokens/sec
- [ ] Memory footprint <2GB on edge

### Test Suite (Repeatable)
```bash
python3 Phase4_benchmark.py \
  --model models/arthur_v2_quantized.bin \
  --test_file data/test_suite.jsonl \
  --output benchmark_results.json
```

## Timeline (Week 10)

- Monday: Quantization (2 hrs)
- Tuesday: C99 export + testing (3 hrs)
- Wednesday: Vercel/self-host setup (2 hrs)
- Thursday: Full benchmark + documentation (2 hrs)
- Friday: Final testing & release

**Total Phase 4: ~9 hours**

## Success Criteria

| Metric | Target | Status |
|--------|--------|--------|
| Model size | <100MB (INT8) | TBD |
| Math accuracy | >80% | Phase 3 |
| Inference speed | >100 tok/s | TBD |
| Latency P99 | <500ms | TBD |
| Deployment | Live + monitored | TBD |

## Monitoring (Post-Deploy)

- Error rate <0.1%
- Latency alerts if >1000ms
- Accuracy drift detection (retrain if <80%)
- Monthly cost tracking

