#!/usr/bin/env python3
"""Arthur v2 OpenClaw Provider"""
import sys
import json
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from transformer_v2 import ArthurV2
from bpe_tokenizer import BPETokenizer

class ArthurProvider:
    def __init__(self):
        self.model_path = Path(__file__).parent.parent / "models" / "arthur_v2_epoch2.pt"
        self.tokenizer_path = Path(__file__).parent.parent / "models" / "bpe_tokenizer_v1.json"
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load model and tokenizer"""
        self.model = ArthurV2(
            vocab_size=32768,
            embed_dim=512,
            num_heads=8,
            num_layers=12,
            ff_dim=2048,
            max_context=8192,
            dropout=0.0
        )
        self.model.load_state_dict(torch.load(self.model_path, map_location="cpu", weights_only=True))
        self.model.eval()
        
        self.tokenizer = BPETokenizer()
        self.tokenizer.load(str(self.tokenizer_path))
    
    def generate(self, prompt, max_tokens=100, temperature=0.7):
        """Generate response"""
        tokens = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([tokens])
        
        # Simple greedy generation
        generated = []
        with torch.no_grad():
            for _ in range(max_tokens):
                logits = self.model(input_ids)
                
                # Apply temperature
                if temperature > 0:
                    logits = logits / temperature
                
                probs = F.softmax(logits[0, -1, :], dim=-1)
                next_token = torch.multinomial(probs, 1)
                generated.append(next_token.item())
                
                # Stop at EOS (1 is typical EOS token)
                if next_token.item() == 1:
                    break
                    
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        
        return self.tokenizer.decode(generated)

# OpenClaw CLI interface
if __name__ == "__main__":
    provider = ArthurProvider()
    
    # Read request from stdin
    request = json.loads(sys.stdin.read())
    messages = request.get("messages", [])
    
    # Get last user message
    prompt = ""
    for msg in messages:
        if msg["role"] == "user":
            prompt = msg["content"]
    
    # Generate response
    response = provider.generate(prompt, max_tokens=50)
    
    # Output OpenClaw format
    output = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": response.strip()
            }
        }]
    }
    
    print(json.dumps(output))
