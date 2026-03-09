function parseBody(body) {
  if (!body) return {};
  if (typeof body === 'object') return body;
  try {
    return JSON.parse(body);
  } catch {
    return {};
  }
}

function safeArithmetic(expression) {
  if (!expression) return null;
  const normalized = expression.replace(/x/gi, '*').replace(/÷/g, '/').trim();

  if (!/^[\d\s+\-*/().%]+$/.test(normalized)) {
    return null;
  }

  try {
    const result = Function(`"use strict"; return (${normalized});`)();
    if (typeof result !== 'number' || !Number.isFinite(result)) {
      return null;
    }
    return result;
  } catch {
    return null;
  }
}

function codeSnippet(prompt) {
  const lower = prompt.toLowerCase();

  if (lower.includes('python')) {
    return [
      '```python',
      'def moving_average(values, window=3):',
      '    if window <= 0:',
      '        raise ValueError("window must be positive")',
      '    return [sum(values[i:i+window]) / window for i in range(len(values)-window+1)]',
      '```'
    ].join('\n');
  }

  return [
    '```javascript',
    'export function groupBy(items, keyFn) {',
    '  return items.reduce((acc, item) => {',
    '    const key = keyFn(item);',
    '    (acc[key] ||= []).push(item);',
    '    return acc;',
    '  }, {});',
    '}',
    '```'
  ].join('\n');
}

function buildResponse(prompt) {
  const lower = prompt.toLowerCase();

  const exprMatch = prompt.match(/([-+*/().%\d\sxX÷]{3,})/);
  const arithmetic = exprMatch ? safeArithmetic(exprMatch[1]) : null;
  if (arithmetic !== null) {
    return `Result: ${arithmetic}`;
  }

  if (/(code|snippet|function|javascript|js|python|regex|sql)/i.test(lower)) {
    return codeSnippet(prompt);
  }

  if (/(model|arthur|loss|context|parameter|architecture|training)/i.test(lower)) {
    return 'Arthur is a 65M-parameter transformer optimized for compact deployment and low-latency interaction. It targets 8K-token context handling and current benchmark runs report training loss around 0.0115.';
  }

  return 'Arthur demo mode can compute straightforward arithmetic, generate short practical code snippets, and explain its training profile and architecture decisions.';
}

export default function handler(req, res) {
  res.setHeader('Content-Type', 'application/json');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  const pathname = (req.url || '').split('?')[0].replace(/\/+$/, '') || '/';

  if (req.method === 'GET' && (pathname === '/api/model' || pathname === '/model')) {
    res.status(200).json({
      name: 'Arthur v3',
      parameters: '65M',
      context: '8K tokens',
      loss: 0.0115,
      mode: 'demo',
      backend: 'serverless mock'
    });
    return;
  }

  if (req.method === 'POST' && (pathname === '/api/generate' || pathname === '/generate' || pathname === '/')) {
    const body = parseBody(req.body);
    const prompt = String(body.prompt || '').trim();

    if (!prompt) {
      res.status(400).json({ error: 'Missing prompt' });
      return;
    }

    res.status(200).json({
      response: buildResponse(prompt),
      model: 'arthur-v3-demo',
      loss: 0.0115,
      context: '8k'
    });
    return;
  }

  res.status(200).json({
    status: 'ok',
    endpoints: ['/api/model', '/api/generate'],
    notes: 'Demo API with arithmetic and code snippet generation.'
  });
}
