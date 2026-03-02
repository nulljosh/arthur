import time
import torch
from src.transformer_v2 import ArthurV2
from src.bpe_tokenizer import BPETokenizer

print("⚡ Arthur Benchmark Suite")
print("=" * 40)

# Load model
model = ArthurV2(vocab_size=10000, d_model=512, n_heads=8, n_layers=12)
tokenizer = BPETokenizer()

# Speed test
text = "The future of AI is " * 100
start = time.time()
tokens = tokenizer.encode(text)
print(f"Tokenization: {len(tokens)} tokens in {time.time()-start:.2f}s")

# Memory usage
print(f"Model size: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
print(f"Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB" if torch.cuda.is_available() else "CPU mode")
