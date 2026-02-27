#!/usr/bin/env python3
"""
Command-line chat interface for nous
"""

import torch
import sys
sys.path.insert(0, 'src')
from transformer import Nous
from tokenizer import CharTokenizer

# Load model
MODEL_PATH = 'models/conversational.pt'

print("Loading nous conversational model...")
checkpoint = torch.load(MODEL_PATH, map_location='cpu')

tokenizer = CharTokenizer()
tokenizer.char_to_idx = checkpoint['vocab']
tokenizer.idx_to_char = {v: k for k, v in tokenizer.char_to_idx.items()}

model = Nous(vocab_size=checkpoint['vocab_size'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("✓ Model loaded!\n")
print("Chat with nous (type 'exit' to quit):")
print("=" * 50)

while True:
    try:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Bye! 👋")
            break
        
        # Format as Q&A
        prompt = f"Q: {user_input}\nA:"
        
        # Generate
        tokens = tokenizer.encode(prompt)
        x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
        generated = tokens.copy()
        
        with torch.no_grad():
            for _ in range(150):
                logits = model(x)
                logits = logits[:, -1, :] / 0.8
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                generated.append(next_token)
                
                # Stop at newline after answer starts
                decoded = tokenizer.decode(generated)
                if '\nQ:' in decoded or len(decoded) > len(prompt) + 200:
                    break
                
                x = torch.cat([x, torch.tensor([[next_token]])], dim=1)
                if x.size(1) > 128:
                    x = x[:, -128:]
        
        response = tokenizer.decode(generated)
        # Extract just the answer
        if '\nA:' in response:
            answer = response.split('\nA:')[1].split('\nQ:')[0].strip()
            print(f"nous: {answer}")
        else:
            print(f"nous: {response}")
            
    except KeyboardInterrupt:
        print("\n\nBye! 👋")
        break
    except Exception as e:
        print(f"Error: {e}")
