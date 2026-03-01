export default function handler(req, res) {
  res.setHeader('Content-Type', 'application/json');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  const path = req.url.split('?')[0].replace(/\/+$/, '');

  // POST /api/generate - fallback chat generation
  // Primary inference is client-side ONNX in the browser.
  // This endpoint serves as a fallback when ONNX model fails to load.
  if (req.method === 'POST' && path === '/api/generate') {
    const { prompt = '', length = 120, temperature = 0.5 } = req.body || {};

    const responses = [
      'Arthur is a 3.5M parameter transformer trained from scratch on diverse text data.',
      'The model uses character-level tokenization with 4 attention layers and 128-dimensional embeddings.',
      'For real inference, this page loads the ONNX model client-side. Try refreshing to enable local inference.',
      'Arthur processes input through self-attention blocks, each with 4 heads and feed-forward networks.',
      'Training uses cosine learning rate decay with AdamW optimizer across 800 epochs.',
      'The ONNX runtime executes the full transformer in your browser via WebAssembly.',
    ];
    const text = responses[Math.floor(Math.random() * responses.length)];

    return res.json({ text, tokens: text.split(' ').length, temperature, mode: 'api-fallback' });
  }

  // GET /api/status - model status
  if (req.method === 'GET' && path === '/api/status') {
    return res.json({
      model_loaded: true,
      config: {
        num_layers: 4,
        num_heads: 4,
        embed_dim: 128,
        ff_dim: 512,
        max_len: 256,
        params: '3.5M'
      },
      version: 'v2.0-vercel',
      inference: 'onnx-browser'
    });
  }

  // GET /api/health
  if (req.method === 'GET' && path === '/api/health') {
    return res.json({ status: 'ok' });
  }

  // GET /api/info
  if (req.method === 'GET' && path === '/api/info') {
    return res.json({
      name: 'arthur',
      version: '2.0.0',
      params: '3.5M',
      inference: 'onnx-browser',
      chat: '/chat.html'
    });
  }

  // Default
  res.json({ message: 'arthur v2.0 API', endpoints: ['/api/generate', '/api/status', '/api/health', '/api/info'] });
}
