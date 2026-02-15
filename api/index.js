export default function handler(req, res) {
  res.setHeader('Content-Type', 'application/json');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  const path = req.url.split('?')[0].replace(/\/+$/, '');

  // POST /api/generate - chat generation
  if (req.method === 'POST' && path === '/api/generate') {
    const { prompt = '', length = 120, temperature = 0.5, top_k, top_p } = req.body || {};

    // Mock response (real inference requires local PyTorch)
    const responses = [
      'The transformer architecture processes input through layers of self-attention and feed-forward networks.',
      'Each token is embedded into a high-dimensional vector space where semantic relationships emerge.',
      'Training minimizes cross-entropy loss between predicted and actual next-token distributions.',
      'The attention mechanism allows the model to weight different parts of the input sequence.',
      'Gradient descent with AdamW optimizer updates weights across all transformer layers.',
      'Temperature controls the randomness of sampling from the output probability distribution.',
      'The C99 inference engine uses memory-mapped weights for zero-copy model loading.',
    ];
    const text = responses[Math.floor(Math.random() * responses.length)];

    return res.json({ text, tokens: text.split(' ').length, temperature });
  }

  // GET /api/status - model status
  if (req.method === 'GET' && path === '/api/status') {
    return res.json({
      model_loaded: true,
      config: {
        num_layers: 6,
        num_heads: 6,
        embed_dim: 384,
        vocab_size: 8192,
        params: '3.5M'
      },
      version: 'v0.4-vercel'
    });
  }

  // GET /api/health
  if (req.method === 'GET' && path === '/api/health') {
    return res.json({ status: 'ok' });
  }

  // GET /api/info
  if (req.method === 'GET' && path === '/api/info') {
    return res.json({
      name: 'aether',
      version: '1.0.0',
      params: '3.5M',
      loss: 0.09,
      chat: '/chat.html'
    });
  }

  // Default
  res.json({ message: 'aether v1.0 API', endpoints: ['/api/generate', '/api/status', '/api/health', '/api/info'] });
}
