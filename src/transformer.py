"""
Transformer block
Attention + Feed-forward + LayerNorm + Residuals
"""

import torch
import torch.nn as nn
from attention import MultiHeadAttention


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""
    
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Single transformer block (attention + FFN)"""
    
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = FeedForward(embed_dim, ff_dim, dropout)
        
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Attention with residual + layernorm
        attn_out = self.attention(self.ln1(x))
        x = x + self.dropout(attn_out)
        
        # FFN with residual + layernorm
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_out)
        
        return x


class Nous(nn.Module):
    """Full nous model"""
    
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, max_len, dropout=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_len = max_len
        
        # Token + position embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # Embeddings
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        tok_emb = self.token_embed(x)
        pos_emb = self.pos_embed(positions)
        x = self.dropout(tok_emb + pos_emb)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits
