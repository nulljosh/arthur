# aether

A small language model built from scratch. Trained on code and knowledge.

## Architecture

<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 300" font-family="-apple-system, system-ui, sans-serif" width="100%">
  <defs>
    <style>
      .box { fill: #1a1a1a; stroke: #0a84ff; stroke-width: 2; }
      .label { fill: #e0e0e0; font-size: 13px; text-anchor: middle; }
      .sub { fill: #888; font-size: 10px; text-anchor: middle; }
      .arrow { stroke: #666; stroke-width: 2; fill: none; marker-end: url(#arrowhead); }
    </style>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#666"/>
    </marker>
  </defs>

  <rect width="1000" height="300" fill="#111"/>

  <rect x="20" y="100" width="100" height="50" rx="6" class="box"/>
  <text x="70" y="120" class="label">Input Text</text>
  <text x="70" y="135" class="sub">Query</text>

  <line x1="120" y1="125" x2="145" y2="125" class="arrow"/>

  <rect x="145" y="100" width="110" height="50" rx="6" class="box"/>
  <text x="200" y="120" class="label">Tokenizer</text>
  <text x="200" y="135" class="sub">91 chars</text>

  <line x1="255" y1="125" x2="280" y2="125" class="arrow"/>

  <rect x="280" y="100" width="110" height="50" rx="6" class="box"/>
  <text x="335" y="120" class="label">Embedding</text>
  <text x="335" y="135" class="sub">128-dim</text>

  <line x1="390" y1="125" x2="415" y2="125" class="arrow"/>

  <rect x="415" y="100" width="110" height="50" rx="6" class="box"/>
  <text x="470" y="120" class="label">Block 1</text>
  <text x="470" y="135" class="sub">Attn+FFN</text>

  <line x1="525" y1="125" x2="550" y2="125" class="arrow"/>

  <rect x="550" y="100" width="110" height="50" rx="6" class="box"/>
  <text x="605" y="120" class="label">Block 2</text>
  <text x="605" y="135" class="sub">Attn+FFN</text>

  <line x1="660" y1="125" x2="685" y2="125" class="arrow"/>

  <rect x="685" y="100" width="110" height="50" rx="6" class="box"/>
  <text x="740" y="120" class="label">Block 3</text>
  <text x="740" y="135" class="sub">Attn+FFN</text>

  <line x1="795" y1="125" x2="820" y2="125" class="arrow"/>

  <rect x="820" y="100" width="130" height="50" rx="6" class="box"/>
  <text x="885" y="120" class="label">Output</text>
  <text x="885" y="135" class="sub">Logits</text>

  <text x="20" y="200" class="label" text-anchor="start" font-size="14" font-weight="bold" fill="#0a84ff">445K params | 3 layers | 4 heads | 256 FF dim</text>
  <text x="20" y="225" class="sub" text-anchor="start" font-size="12" fill="#888">Training: Math (50%) + Wiki/Current (50%) | Loss: 0.13-0.19</text>
  <text x="20" y="250" class="sub" text-anchor="start" font-size="12" fill="#888">Eval: Q&amp;A on math, facts, identity, current events</text>
</svg>

## Getting Started

```bash
git clone https://github.com/nulljosh/aether.git && cd aether
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Generate text
python src/generate.py --prompt "Q: What is " --temperature 0.3

# Web interface (chat)
python web_ui.py
# Visit http://localhost:5001
```

## How It Works

1. **Data** – Train on math, current events, and Wikipedia
2. **Model** – Small transformer neural network
3. **Learning** – PyTorch training with character-level tokenizer
4. **Output** – Predicts next tokens based on patterns learned

## Evaluation

Tests on Q&A accuracy:
- Math: Addition, subtraction, multiplication, division
- Facts: Current events, year, president
- Identity: Name, creator, purpose

## Why Build It

Understanding language models means building one. This is the simplest version that actually works.

## License

MIT 2026 Joshua Trommel
