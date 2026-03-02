#!/usr/bin/env python3
"""Interactive chat with Arthur model"""

import random
import json

class ArthurChat:
    def __init__(self):
        self.responses = [
            "That's an interesting question. Based on my training, I'd say...",
            "From what I've learned in my 65M parameters...",
            "My neural networks suggest that...",
            "Processing through my transformer layers...",
            "After considering this through my attention mechanism..."
        ]
        
    def chat(self, prompt):
        # Mock response (real model would generate here)
        prefix = random.choice(self.responses)
        
        # Simple pattern matching for demo
        if "hello" in prompt.lower():
            return f"{prefix} Hello! I'm Arthur, a 65M parameter model."
        elif "how are you" in prompt.lower():
            return f"{prefix} I'm running well at 0.0115 loss!"
        elif "code" in prompt.lower():
            return f"{prefix} Here's a simple function:\n```python\ndef greet(name):\n    return f'Hello, {{name}}!'\n```"
        else:
            return f"{prefix} {prompt[::-1]}"  # Reverse as placeholder

def main():
    arthur = ArthurChat()
    print("🤖 Arthur v2.0 Interactive Chat")
    print("=" * 40)
    print("Type 'quit' to exit\n")
    
    while True:
        prompt = input("You: ")
        if prompt.lower() in ['quit', 'exit']:
            break
            
        response = arthur.chat(prompt)
        print(f"Arthur: {response}\n")

if __name__ == "__main__":
    main()
