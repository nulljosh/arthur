#!/usr/bin/env python3
"""Improved generation with top-k and nucleus sampling"""
import torch
import sys
sys.path.insert(0, 'src')
from transformer import Core

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, temperature=1.0):
    """Filter logits using top-k and/or nucleus (top-p) sampling"""
    logits = logits / temperature
    
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('Inf')
    
    if top_p > 0.0:
        # Sort logits and compute cumulative probabilities
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('Inf')
    
    return logits

# Simple tokenizer
class SimpleTokenizer:
    def __init__(self, vocab):
        self.char_to_idx = vocab
        self.idx_to_char = {v: k for k, v in vocab.items()}
        self.vocab_size = len(vocab)
    
    def encode(self, text):
        return [self.char_to_idx.get(c, 0) for c in text]
    
    def decode(self, tokens):
        return ''.join([self.idx_to_char.get(t, '?') for t in tokens])

def generate(prompt, model_path='models/ultra.pt', max_tokens=100, temperature=0.7, top_k=10, top_p=0.9):
    """Generate text with improved sampling"""
    checkpoint = torch.load(model_path, map_location='cpu')
    tokenizer = SimpleTokenizer(checkpoint['vocab'])
    
    # Auto-detect architecture from checkpoint
    config = checkpoint.get('config', {
        'vocab_size': checkpoint['vocab_size'],
        'embed_dim': 32,
        'num_heads': 2,
        'num_layers': 2,
        'ff_dim': 64,
        'max_len': 32,
        'dropout': 0.1
    })
    
    model = Core(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    tokens = tokenizer.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    generated = tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Keep sequence under max_len to avoid index errors
            max_seq_len = config.get('max_len', 32) - 2
            if x.size(1) >= max_seq_len:
                x = x[:, -max_seq_len:]
            
            logits = model(x)
            logits = logits[:, -1, :]
            
            # Apply filtering
            filtered_logits = top_k_top_p_filtering(
                logits.clone().squeeze(0),
                top_k=top_k,
                top_p=top_p,
                temperature=temperature
            )
            
            # Sample
            probs = torch.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            generated.append(next_token)
            x = torch.cat([x, torch.tensor([[next_token]])], dim=1)
            
            # Stop at newline for Q&A
            if tokenizer.idx_to_char.get(next_token) == '\n':
                break
    
    return tokenizer.decode(generated)

if __name__ == '__main__':
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Q: What is your name?\nA: "
    
    print("🧪 Testing improved sampling:")
    print(f"Prompt: {prompt}")
    print(f"Output: {generate(prompt, temperature=0.7, top_k=10, top_p=0.9)}")
