#!/usr/bin/env python3
"""Generate training data quickly without waiting for Qwen"""

import json
import random

# Pre-generated responses (we'd normally get these from Qwen)
templates = {
    "code": [
        "def {name}(x):\n    return x * 2",
        "for i in range(10):\n    print(i)",
        "class {name}:\n    def __init__(self):\n        pass"
    ],
    "explain": [
        "This works by iterating through each element and applying the transformation.",
        "The algorithm uses a divide-and-conquer approach to solve the problem.",
        "It's a recursive solution that breaks down into smaller subproblems."
    ],
    "math": [
        "The answer is 42",
        "Using the formula: result = a * b + c",
        "Step 1: Calculate the base\nStep 2: Apply the operation\nStep 3: Return result"
    ]
}

def generate_dataset():
    data = []
    
    for i in range(100):
        prompt_type = random.choice(["code", "explain", "math"])
        
        if prompt_type == "code":
            prompt = f"Write a Python function called func_{i}"
            response = random.choice(templates["code"]).format(name=f"func_{i}")
        elif prompt_type == "explain":
            prompt = f"Explain concept {i}"
            response = random.choice(templates["explain"])
        else:
            prompt = f"Solve: {random.randint(1,100)} + {random.randint(1,100)}"
            response = random.choice(templates["math"])
        
        data.append({
            "input": prompt,
            "output": response
        })
    
    with open("data/quick_synthetic.json", "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Generated {len(data)} training examples")
    print("📁 Saved to data/quick_synthetic.json")
    print("\n🚀 Train Arthur on this data:")
    print("python src/train_v2.py --data data/quick_synthetic.json --epochs 1")

if __name__ == "__main__":
    generate_dataset()
