"""
Arthur v3.0 — 125M → 500M parameter scaler
Phase 1: Scale up with MoE, RoPE, 32K context

Changes from v2:
- RoPE positional embeddings (better than learned)
- Mixture of Experts feed-forward (8 experts, top-2 routing)
- 32K context window (4x v2)
- GQA: Grouped Query Attention (faster inference)
- RMSNorm instead of LayerNorm (faster)
- Configurable size presets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

# ── Size Presets ────────────────────────────────────────────────────────────
CONFIGS = {
    "65M":  dict(d_model=512,  n_heads=8,  n_kv_heads=4, n_layers=12, ff_dim=2048,  n_experts=4,  vocab=10000, ctx=8192),
    "125M": dict(d_model=768,  n_heads=12, n_kv_heads=4, n_layers=12, ff_dim=3072,  n_experts=8,  vocab=50000, ctx=32768),
    "250M": dict(d_model=1024, n_heads=16, n_kv_heads=8, n_layers=24, ff_dim=4096,  n_experts=8,  vocab=50000, ctx=32768),
    "500M": dict(d_model=1280, n_heads=20, n_kv_heads=10,n_layers=36, ff_dim=5120,  n_experts=16, vocab=50000, ctx=32768),
}

# ── RMSNorm (faster than LayerNorm) ─────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.scale

# ── RoPE Embeddings (better long-range position encoding) ───────────────────
def precompute_rope(dim: int, max_seq: int, theta: float = 10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    pos = torch.arange(max_seq).float()
    angles = torch.outer(pos, inv_freq)
    cos = torch.repeat_interleave(torch.cos(angles), 2, dim=-1)
    sin = torch.repeat_interleave(torch.sin(angles), 2, dim=-1)
    return torch.stack((cos, sin), dim=0)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).reshape_as(x)

def apply_rope(q, k, freqs):
    cos = freqs[0, : q.shape[1]].unsqueeze(0).unsqueeze(2).type_as(q)
    sin = freqs[1, : q.shape[1]].unsqueeze(0).unsqueeze(2).type_as(q)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def migrate_state_dict(state):
    """Migrate old complex64 RoPE freqs to the new [2, seq, dim] real format."""
    if "freqs" in state and state["freqs"].is_complex():
        old = state["freqs"]
        cos_part = torch.repeat_interleave(old.real, 2, dim=-1)
        sin_part = torch.repeat_interleave(old.imag, 2, dim=-1)
        state["freqs"] = torch.stack((cos_part, sin_part), dim=0)
    return state

# ── Grouped Query Attention (faster inference) ───────────────────────────────
class GQAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim   = d_model // n_heads
        self.groups     = n_heads // n_kv_heads

        self.q  = nn.Linear(d_model, n_heads    * self.head_dim, bias=False)
        self.k  = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v  = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, freqs, mask=None):
        B, T, _ = x.shape
        H, KV, D = self.n_heads, self.n_kv_heads, self.head_dim

        Q = self.q(x).view(B, T, H,  D)
        K = self.k(x).view(B, T, KV, D)
        V = self.v(x).view(B, T, KV, D)

        Q, K = apply_rope(Q, K, freqs)

        # Expand K/V to match Q heads
        K = K.repeat_interleave(self.groups, dim=2)
        V = V.repeat_interleave(self.groups, dim=2)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        scale = D ** -0.5
        scores = (Q @ K.transpose(-2, -1)) * scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = self.drop(F.softmax(scores, dim=-1))

        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, -1)
        return self.out(out)

# ── Mixture of Experts ───────────────────────────────────────────────────────
class Expert(nn.Module):
    def __init__(self, d_model: int, ff_dim: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, ff_dim * 2, bias=False)
        self.w2 = nn.Linear(ff_dim, d_model, bias=False)

    def forward(self, x):
        x, gate = self.w1(x).chunk(2, dim=-1)
        return self.w2(x * F.silu(gate))

class MoEFFN(nn.Module):
    def __init__(self, d_model: int, ff_dim: int, n_experts: int, top_k: int = 2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(d_model, ff_dim) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        flat = x.view(-1, C)                              # (B*T, C)

        logits = self.router(flat)                        # (B*T, E)
        weights, ids = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)              # normalize top-k

        # Dense routing avoids Python data-dependent control flow, which keeps
        # ONNX export stable for the browser deployment path.
        router_weights = torch.zeros(
            flat.size(0),
            self.n_experts,
            device=flat.device,
            dtype=flat.dtype,
        )
        router_weights.scatter_add_(1, ids, weights)

        expert_outputs = torch.stack([expert(flat) for expert in self.experts], dim=1)
        out = (expert_outputs * router_weights.unsqueeze(-1)).sum(dim=1)
        return out.view(B, T, C)

# ── Transformer Block ────────────────────────────────────────────────────────
class ArthurBlock(nn.Module):
    def __init__(self, cfg: dict, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(cfg["d_model"])
        self.attn  = GQAttention(cfg["d_model"], cfg["n_heads"], cfg["n_kv_heads"], dropout)
        self.norm2 = RMSNorm(cfg["d_model"])
        self.ffn   = MoEFFN(cfg["d_model"], cfg["ff_dim"], cfg["n_experts"])

    def forward(self, x, freqs, mask=None):
        x = x + self.attn(self.norm1(x), freqs, mask)
        x = x + self.ffn(self.norm2(x))
        return x

# ── Main Model ───────────────────────────────────────────────────────────────
class ArthurV3(nn.Module):
    def __init__(self, size: str = "125M", dropout: float = 0.1):
        super().__init__()
        cfg = CONFIGS[size]
        self.cfg = cfg
        self.size = size

        self.embed = nn.Embedding(cfg["vocab"], cfg["d_model"])
        self.layers = nn.ModuleList([ArthurBlock(cfg, dropout) for _ in range(cfg["n_layers"])])
        self.norm   = RMSNorm(cfg["d_model"])
        self.head   = nn.Linear(cfg["d_model"], cfg["vocab"], bias=False)

        # Tie embedding weights to output (saves params, improves performance)
        self.head.weight = self.embed.weight

        # Precompute RoPE frequencies
        self.register_buffer(
            "freqs",
            precompute_rope(cfg["d_model"] // cfg["n_heads"], cfg["ctx"])
        )

        self.apply(self._init_weights)
        n = sum(p.numel() for p in self.parameters())
        print(f"ArthurV3-{size}: {n/1e6:.1f}M parameters | ctx={cfg['ctx']} | experts={cfg['n_experts']}")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, T = ids.shape
        x = self.embed(ids)
        for layer in self.layers:
            x = layer(x, self.freqs, mask)
        return self.head(self.norm(x))

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new: int = 100,
                 temperature: float = 0.8, top_k: int = 40):
        """Autoregressive generation with temperature + top-k sampling."""
        ctx = self.cfg["ctx"]
        for _ in range(max_new):
            ids = prompt_ids[:, -ctx:]                   # clip to context window
            logits = self(ids)[:, -1] / temperature      # last token logits
            # Top-k filter
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, -1:]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            prompt_ids = torch.cat([prompt_ids, next_id], dim=1)
        return prompt_ids


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}\n")

    for size in ["65M", "125M"]:
        model = ArthurV3(size).to(device)
        x = torch.randint(0, model.cfg["vocab"], (1, 64)).to(device)

        with torch.no_grad():
            out = model(x)
        print(f"  Input: {x.shape} → Output: {out.shape}")

        # Generation test
        gen = model.generate(x, max_new=10)
        print(f"  Generated {gen.shape[1] - 64} new tokens\n")

    print("✅ Phase 1 complete. Ready to train.")
