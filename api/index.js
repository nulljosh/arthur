export default function handler(req, res) {
  const { pathname } = new URL(req.url, `http://${req.headers.host}`);
  
  res.setHeader('Content-Type', 'application/json');
  res.setHeader('Access-Control-Allow-Origin', '*');

  if (pathname === '/api' || pathname === '/api/') {
    res.json({
      name: "aether",
      version: "1.0.0",
      status: "ok",
      github: "https://github.com/nulljosh/aether"
    });
  } else if (pathname === '/api/health') {
    res.json({ status: "ok" });
  } else if (pathname === '/api/info') {
    res.json({
      name: "aether",
      version: "1.0.0",
      params: "3.5M",
      layers: 12,
      loss: 0.09,
      github: "https://github.com/nulljosh/aether"
    });
  } else {
    res.status(404).json({ error: "Not found" });
  }
}
