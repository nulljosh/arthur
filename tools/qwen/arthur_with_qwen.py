#!/usr/bin/env python3
"""Arthur v2.5 - Using Qwen as inference backend"""

import subprocess
import json
import sys

class ArthurWithQwen:
    def __init__(self):
        self.context = """You are Arthur, a 65M parameter language model trained from scratch.
Key facts about you:
- Version: 2.0
- Parameters: 65 million
- Training loss: 0.0115 (beat target of 0.05)
- Context window: 8K tokens
- Created by: Joshua Trommel
- Architecture: 12-layer transformer with Flash Attention
- Speed: 89 tokens/sec on M4

Respond as Arthur - mention your training, parameters, or architecture when relevant."""

    def respond(self, prompt):
        # Build Qwen prompt with Arthur's identity
        full_prompt = f"{self.context}\n\nUser: {prompt}\nArthur:"
        
        # Query Qwen but it responds AS Arthur
        result = subprocess.run(
            ["ollama", "run", "qwen2.5:3b"],
            input=full_prompt,
            text=True,
            capture_output=True,
            timeout=10
        )
        
        if result.returncode == 0:
            return result.stdout.strip()
        return "My inference layers are recalibrating. Try again."

def main():
    arthur = ArthurWithQwen()
    print("🤖 Arthur v2.5 (Qwen-powered inference)")
    print("=" * 40)
    print("Chat with Arthur. Type 'quit' to exit.\n")
    
    while True:
        prompt = input("You: ")
        if prompt.lower() in ['quit', 'exit']:
            break
        
        print("Arthur: ", end="", flush=True)
        response = arthur.respond(prompt)
        print(response + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single query mode
        arthur = ArthurWithQwen()
        query = " ".join(sys.argv[1:])
        print(arthur.respond(query))
    else:
        main()
