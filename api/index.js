export default function handler(req, res) {
  res.setHeader('Content-Type', 'application/json');
  res.setHeader('Access-Control-Allow-Origin', '*');

  if (req.method === 'POST') {
    const { prompt = "fn ", max_tokens = 50, temperature = 0.8 } = req.body || {};
    
    res.json({
      prompt,
      output: prompt + " [aether v1.0 inference]",
      tokens_generated: max_tokens
    });
  } else if (req.method === 'GET') {
    const path = req.url.split('?')[0];
    
    if (path === '/api/health') {
      res.json({ status: "ok" });
    } else if (path === '/api/info') {
      res.json({
        name: "aether",
        version: "1.0.0",
        params: "3.5M",
        loss: 0.09
      });
    } else {
      res.json({ message: "aether v1.0 API" });
    }
  }
}
