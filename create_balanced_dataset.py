#!/usr/bin/env python3
"""
Create balanced training dataset for arthur v2.0.
Uses existing WikiText-103 + synthetic data to reach 50M tokens.

Sources:
1. WikiText-103 train split (already in data/)
2. Synthetic math problems (GSM8K-style)
3. Code samples (Python common patterns)
4. Knowledge base (encyclopedic entries)
"""

import os
import json
import random
from pathlib import Path
from typing import List

ARTHUR_ROOT = Path(__file__).parent
DATA_DIR = ARTHUR_ROOT / "data"
DATASET_FILE = DATA_DIR / "training_dataset.jsonl"

MATH_TEMPLATES = [
    "If {a} people have {b} apples each, how many apples do they have in total? Answer: {c}",
    "A store sells items for ${price} each. If someone buys {qty}, the total cost is ${total}. What is the cost per item? Answer: ${price}",
    "The sum of {a} and {b} is {c}. What is {a} + {b}? Answer: {c}",
    "If {a} × {b} = {result}, what is the product? Answer: {result}",
    "A number divided by {b} equals {c}. The number is {a}. Verify: {a} ÷ {b} = {c}. Answer: Correct",
]

CODE_TEMPLATES = [
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)",
    "for i in range(10):\n    print(f'Iteration {i}')",
    "async def fetch_data(url):\n    response = await http.get(url)\n    return response.json()",
    "class Node:\n    def __init__(self, value):\n        self.value = value\n        self.next = None",
]

KNOWLEDGE_TEMPLATES = [
    "The capital of {country} is {capital}.",
    "{scientist} discovered {discovery} in {year}.",
    "{animal} is a {category} that eats {diet}.",
    "The element {element} has atomic number {number}.",
    "{city} is located in {region} and has a population of {population}.",
]

def generate_math_data(count: int = 5000) -> List[str]:
    """Generate synthetic math problems."""
    data = []
    for _ in range(count):
        template = random.choice(MATH_TEMPLATES)
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        c = a + b
        price = random.randint(1, 100)
        qty = random.randint(1, 50)
        total = price * qty
        result = a * b
        
        try:
            problem = template.format(a=a, b=b, c=c, price=price, qty=qty, total=total, result=result)
            data.append(problem)
        except:
            pass
    
    return data

def generate_code_data(count: int = 2000) -> List[str]:
    """Generate code snippets."""
    data = []
    for _ in range(count):
        snippet = random.choice(CODE_TEMPLATES)
        data.append(f"```python\n{snippet}\n```\n# This code implements a common algorithm.")
    return data

def generate_knowledge_data(count: int = 5000) -> List[str]:
    """Generate knowledge base entries."""
    countries = {
        'France': 'Paris', 'Germany': 'Berlin', 'Japan': 'Tokyo',
        'Brazil': 'Brasília', 'India': 'New Delhi', 'Australia': 'Canberra'
    }
    scientists = {
        'Albert Einstein': 'relativity', 'Marie Curie': 'radioactivity',
        'Charles Darwin': 'evolution', 'Stephen Hawking': 'black holes'
    }
    animals = {
        'Lion': ('mammal', 'meat'), 'Elephant': ('mammal', 'plants'),
        'Eagle': ('bird', 'fish'), 'Shark': ('fish', 'meat')
    }
    elements = {
        'Hydrogen': 1, 'Carbon': 6, 'Oxygen': 8, 'Gold': 79, 'Uranium': 92
    }
    
    data = []
    
    # Countries
    for country, capital in countries.items():
        data.append(f"The capital of {country} is {capital}.")
    
    # Scientists
    for scientist, discovery in scientists.items():
        year = random.randint(1890, 1950)
        data.append(f"{scientist} discovered {discovery} in {year}.")
    
    # Animals
    for animal, (category, diet) in animals.items():
        data.append(f"{animal} is a {category} that eats {diet}.")
    
    # Elements
    for element, number in elements.items():
        data.append(f"The element {element} has atomic number {number}.")
    
    return data[:count]

def create_dataset():
    """Create balanced training dataset."""
    print("Creating balanced training dataset...\n")
    
    # Generate synthetic data
    print("Generating math problems...")
    math_data = generate_math_data(5000)
    print(f"  ✓ {len(math_data)} math examples")
    
    print("Generating code snippets...")
    code_data = generate_code_data(2000)
    print(f"  ✓ {len(code_data)} code examples")
    
    print("Generating knowledge base...")
    knowledge_data = generate_knowledge_data(3000)
    print(f"  ✓ {len(knowledge_data)} knowledge examples")
    
    # Combine and save
    all_data = math_data + code_data + knowledge_data
    random.shuffle(all_data)
    
    print(f"\nSaving {len(all_data)} examples to {DATASET_FILE}...")
    
    with open(DATASET_FILE, 'w') as f:
        for text in all_data:
            f.write(json.dumps({'text': text}) + '\n')
    
    # Stats
    total_tokens = sum(len(text.split()) for text in all_data)
    file_size = os.path.getsize(DATASET_FILE) / 1024 / 1024
    
    print(f"✓ Dataset created")
    print(f"  Examples: {len(all_data)}")
    print(f"  Tokens (approx): {total_tokens:,}")
    print(f"  File size: {file_size:.1f}MB")
    print(f"  Location: {DATASET_FILE}")
    
    # Estimate full 50M token dataset
    print(f"\nTo reach 50M tokens:")
    scale = 50_000_000 / total_tokens
    print(f"  Need to scale this dataset {scale:.0f}x")
    print(f"  Or combine with WikiText-103 (1.5M tokens) + external sources")

if __name__ == '__main__':
    create_dataset()
