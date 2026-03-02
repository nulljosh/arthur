#!/usr/bin/env python3
"""Generate synthetic training data for Arthur using Qwen"""

import subprocess
import json
import random

def generate_training_data():
    """Use Qwen to create diverse training examples"""
    
    prompts = [
        "Write a Python function to sort a list",
        "Explain quantum computing simply",
        "Tell a story about a robot",
        "Describe how photosynthesis works",
        "Create a haiku about coding",
        "Solve this math problem: 45 * 23",
        "Write CSS for a dark theme",
        "Explain recursion with an example"
    ]
    
    print("🧠 Generating training data with Qwen...")
    training_pairs = []
    
    for prompt in prompts:
        # Get Qwen's response
        result = subprocess.run(
            ["ollama", "run", "qwen2.5:3b"],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=30
        )
        
        if result.returncode == 0:
            response = result.stdout.strip()
            training_pairs.append({
                "input": prompt,
                "output": response
            })
            print(f"✓ Generated: {prompt[:30]}...")
    
    # Save to Arthur's training format
    with open("data/qwen_synthetic.json", "w") as f:
        json.dump(training_pairs, f, indent=2)
    
    print(f"\n✅ Generated {len(training_pairs)} training examples")
    print("📁 Saved to data/qwen_synthetic.json")
    return training_pairs

if __name__ == "__main__":
    generate_training_data()
    print("\n🚀 Ready to train Arthur on Qwen-generated data!")
    print("Run: python src/train_v2.py --data data/qwen_synthetic.json")
