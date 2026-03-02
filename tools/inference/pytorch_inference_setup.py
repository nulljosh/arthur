#!/usr/bin/env python3
"""
PyTorch 101 + Real Arthur Inference Setup

WHAT IS PYTORCH?
- Python library for neural networks (like TensorFlow but cleaner)
- Tensors = multi-dimensional arrays (like NumPy but on GPU)
- Autograd = automatic differentiation (calculates gradients for you)

HOW NEURAL NETS WORK:
1. Input → Embeddings (words to numbers)
2. Layers of matrix multiplication + activation functions
3. Attention mechanism (which words relate to which)
4. Output probabilities for next token
5. Sample from probabilities → next word

ARTHUR'S ARCHITECTURE:
- Input: Text → Token IDs → Embeddings (512 dims)
- 12 Transformer layers, each with:
  - Multi-head attention (8 heads)
  - Feed-forward network
  - Layer normalization
- Output: Probabilities over 10,000 vocabulary tokens
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

print("🎓 PyTorch 101 + Arthur Inference")
print("=" * 40)

# 1. CHECK PYTORCH
print("\n1️⃣ PyTorch Check:")
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available (Apple GPU): {torch.backends.mps.is_available()}")
print(f"CUDA available: {torch.cuda.is_available()}")

# 2. SIMPLE EXAMPLE - How tensors work
print("\n2️⃣ Tensor Basics:")
# Create a tensor (like array but for neural nets)
x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
print(f"Tensor shape: {x.shape}")
print(f"Tensor: \n{x}")

# Neural net operation (matrix multiply + bias)
weight = torch.randn(3, 2)  # Random weights
bias = torch.randn(2)
output = torch.matmul(x, weight) + bias
print(f"After linear layer: \n{output}")

# 3. LOAD ARTHUR MODEL
print("\n3️⃣ Loading Arthur Model:")

class SimpleArthur(nn.Module):
    """Simplified Arthur for inference"""
    def __init__(self, vocab_size=10000, d_model=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead=8, batch_first=True),
            num_layers=12
        )
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x, x)  # Self-attention
        return self.output(x)

# Initialize model
model = SimpleArthur()
param_count = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {param_count/1e6:.1f}M")

# 4. LOAD CHECKPOINT (if exists)
model_path = Path("models/arthur_v2_epoch1.pt")
if model_path.exists():
    print(f"✅ Found checkpoint: {model_path}")
    # checkpoint = torch.load(model_path, map_location='cpu')
    print("   (Would load weights here)")
else:
    print("⚠️  No checkpoint found, using random weights")

# 5. INFERENCE EXAMPLE
print("\n4️⃣ Inference Example:")

def generate_text(prompt="Hello", max_length=20):
    """Generate text from prompt"""
    # Mock tokenization (real would use BPE tokenizer)
    input_ids = torch.randint(0, 10000, (1, 5))  # Random tokens
    
    print(f"Input: '{prompt}'")
    print(f"Tokens: {input_ids.squeeze().tolist()}")
    
    # Generate
    with torch.no_grad():  # Don't calculate gradients (faster)
        for i in range(max_length):
            # Forward pass
            logits = model(input_ids)
            
            # Get next token probabilities
            probs = torch.softmax(logits[0, -1], dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, 1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
    
    print(f"Generated {max_length} tokens")
    return input_ids

# Run generation
generated = generate_text("Hello Arthur")
print(f"Output tokens: {generated.squeeze().tolist()[:10]}...")

print("\n📚 PyTorch Concepts Learned:")
print("• Tensors = GPU-accelerated arrays")
print("• Models = layers of matrix operations")  
print("• Forward pass = input → model → output")
print("• Inference = generate without training")

print("\n🚀 Next Steps for Real Inference:")
print("1. Load actual trained weights (.pt file)")
print("2. Implement proper BPE tokenizer")
print("3. Add temperature/top-k sampling")
print("4. Optimize with torch.compile() for speed")
print("5. Export to ONNX for production")
