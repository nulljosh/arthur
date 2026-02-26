#!/usr/bin/env python3
"""Generate comprehensive math Q&A training dataset"""

import random

def generate_math_qa():
    """Generate thousands of math Q&A examples"""
    qa_pairs = []
    
    # Phrasings for variety
    question_templates = [
        "Q: What is {}?\nA: {}\n\n",
        "Q: What's {}?\nA: {}\n\n",
        "Q: Calculate {}\nA: {}\n\n",
        "Q: {}?\nA: {}\n\n",
        "Q: What is {} equal to?\nA: {}\n\n",
    ]
    
    # 1. ADDITION - Single digit (200 examples)
    for _ in range(200):
        a, b = random.randint(0, 9), random.randint(0, 9)
        expr = f"{a}+{b}"
        answer = str(a + b)
        template = random.choice(question_templates)
        qa_pairs.append(template.format(expr, answer))
    
    # 2. ADDITION - Two digit (200 examples)
    for _ in range(200):
        a, b = random.randint(10, 99), random.randint(10, 99)
        expr = f"{a}+{b}"
        answer = str(a + b)
        template = random.choice(question_templates)
        qa_pairs.append(template.format(expr, answer))
    
    # 3. ADDITION - Three digit (100 examples)
    for _ in range(100):
        a, b = random.randint(100, 999), random.randint(100, 999)
        expr = f"{a}+{b}"
        answer = str(a + b)
        template = random.choice(question_templates)
        qa_pairs.append(template.format(expr, answer))
    
    # 4. SUBTRACTION - Single digit (200 examples)
    for _ in range(200):
        a, b = random.randint(0, 9), random.randint(0, 9)
        if a >= b:  # Keep results positive
            expr = f"{a}-{b}"
            answer = str(a - b)
            template = random.choice(question_templates)
            qa_pairs.append(template.format(expr, answer))
    
    # 5. SUBTRACTION - Two digit (200 examples)
    for _ in range(200):
        a = random.randint(10, 99)
        b = random.randint(0, a)  # b <= a
        expr = f"{a}-{b}"
        answer = str(a - b)
        template = random.choice(question_templates)
        qa_pairs.append(template.format(expr, answer))
    
    # 6. MULTIPLICATION - Times tables (up to 12x12) with repeats
    for _ in range(300):
        a, b = random.randint(0, 12), random.randint(0, 12)
        expr = f"{a}*{b}"
        answer = str(a * b)
        template = random.choice(question_templates)
        qa_pairs.append(template.format(expr, answer))
    
    # 7. MULTIPLICATION - Larger numbers
    for _ in range(150):
        a, b = random.randint(10, 99), random.randint(2, 12)
        expr = f"{a}*{b}"
        answer = str(a * b)
        template = random.choice(question_templates)
        qa_pairs.append(template.format(expr, answer))
    
    # 8. DIVISION - Clean results only
    for _ in range(200):
        b = random.randint(2, 12)
        result = random.randint(1, 20)
        a = b * result
        expr = f"{a}/{b}"
        answer = str(result)
        template = random.choice(question_templates)
        qa_pairs.append(template.format(expr, answer))
    
    # 9. SQUARES - Numbers 0-20
    for i in range(21):
        for _ in range(3):  # Repeat each 3 times
            expr = f"{i}^2"
            answer = str(i * i)
            template = random.choice(question_templates)
            qa_pairs.append(template.format(expr, answer))
    
    # 10. CUBES - Numbers 0-10
    for i in range(11):
        for _ in range(3):  # Repeat each 3 times
            expr = f"{i}^3"
            answer = str(i * i * i)
            template = random.choice(question_templates)
            qa_pairs.append(template.format(expr, answer))
    
    # 11. ORDER OF OPERATIONS - Simple expressions
    for _ in range(100):
        # a + b * c
        a, b, c = random.randint(1, 9), random.randint(2, 9), random.randint(2, 9)
        expr = f"{a}+{b}*{c}"
        answer = str(a + b * c)
        template = random.choice(question_templates)
        qa_pairs.append(template.format(expr, answer))
    
    for _ in range(100):
        # a * b + c
        a, b, c = random.randint(2, 9), random.randint(2, 9), random.randint(1, 9)
        expr = f"{a}*{b}+{c}"
        answer = str(a * b + c)
        template = random.choice(question_templates)
        qa_pairs.append(template.format(expr, answer))
    
    # 12. MODULAR ARITHMETIC - Simple cases
    for _ in range(100):
        a = random.randint(10, 99)
        b = random.randint(2, 12)
        expr = f"{a} mod {b}"
        answer = str(a % b)
        template = random.choice(question_templates)
        qa_pairs.append(template.format(expr, answer))
    
    # 13. IDENTITY QUESTIONS - Mixed throughout
    identity_qa = [
        "Q: What is your name?\nA: core\n\n",
        "Q: What's your name?\nA: core\n\n",
        "Q: Who are you?\nA: core\n\n",
        "Q: Tell me your name\nA: core\n\n",
        "Q: What are you called?\nA: core\n\n",
        "Q: What are you?\nA: I'm core\n\n",
        "Q: Introduce yourself\nA: I'm core\n\n",
        "Q: What do you do?\nA: I answer questions\n\n",
        "Q: Can you help?\nA: Yes\n\n",
        "Q: Are you smart?\nA: I try my best\n\n",
    ]
    
    # Add identity questions (50 copies total for emphasis)
    for _ in range(5):
        qa_pairs.extend(identity_qa)
    
    # 14. More phrasings for common operations
    common_examples = [
        ("2+2", "4"), ("5+5", "10"), ("10+10", "20"),
        ("3*3", "9"), ("4*4", "16"), ("5*5", "25"),
        ("10-5", "5"), ("20-10", "10"), ("15-8", "7"),
        ("10/2", "5"), ("20/4", "5"), ("100/10", "10"),
    ]
    
    for expr, answer in common_examples:
        for _ in range(10):  # Repeat each 10 times
            template = random.choice(question_templates)
            qa_pairs.append(template.format(expr, answer))
    
    return qa_pairs

if __name__ == "__main__":
    print("Generating comprehensive math Q&A dataset...")
    qa_pairs = generate_math_qa()
    
    # Shuffle for variety
    random.shuffle(qa_pairs)
    
    # Join all pairs
    full_text = "".join(qa_pairs)
    
    # Save to file
    output_path = "data/math_comprehensive.txt"
    with open(output_path, 'w') as f:
        f.write(full_text)
    
    print(f"✓ Generated {len(qa_pairs)} Q&A pairs")
    print(f"✓ Total characters: {len(full_text)}")
    print(f"✓ Saved to {output_path}")
