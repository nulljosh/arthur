#!/usr/bin/env python3
"""Generate comprehensive training dataset: math + jot + identity + time/date"""

import random
import os
import glob

def generate_math_qa():
    """Generate math Q&A examples"""
    qa_pairs = []
    
    # Phrasings for variety
    question_templates = [
        "Q: What is {}?\nA: {}\n\n",
        "Q: What's {}?\nA: {}\n\n",
        "Q: Calculate {}\nA: {}\n\n",
        "Q: {}?\nA: {}\n\n",
        "Q: What is {} equal to?\nA: {}\n\n",
    ]
    
    # 1. ADDITION - Single digit (150 examples)
    for _ in range(150):
        a, b = random.randint(0, 9), random.randint(0, 9)
        expr = f"{a}+{b}"
        answer = str(a + b)
        template = random.choice(question_templates)
        qa_pairs.append(template.format(expr, answer))
    
    # 2. ADDITION - Two digit (150 examples)
    for _ in range(150):
        a, b = random.randint(10, 99), random.randint(10, 99)
        expr = f"{a}+{b}"
        answer = str(a + b)
        template = random.choice(question_templates)
        qa_pairs.append(template.format(expr, answer))
    
    # 3. ADDITION - Three digit (80 examples)
    for _ in range(80):
        a, b = random.randint(100, 999), random.randint(100, 999)
        expr = f"{a}+{b}"
        answer = str(a + b)
        template = random.choice(question_templates)
        qa_pairs.append(template.format(expr, answer))
    
    # 4. SUBTRACTION - Single digit (150 examples)
    for _ in range(150):
        a, b = random.randint(0, 9), random.randint(0, 9)
        if a >= b:
            expr = f"{a}-{b}"
            answer = str(a - b)
            template = random.choice(question_templates)
            qa_pairs.append(template.format(expr, answer))
    
    # 5. SUBTRACTION - Two digit (150 examples)
    for _ in range(150):
        a = random.randint(10, 99)
        b = random.randint(0, a)
        expr = f"{a}-{b}"
        answer = str(a - b)
        template = random.choice(question_templates)
        qa_pairs.append(template.format(expr, answer))
    
    # 6. MULTIPLICATION - Times tables (250 examples)
    for _ in range(250):
        a, b = random.randint(0, 12), random.randint(0, 12)
        expr = f"{a}*{b}"
        answer = str(a * b)
        template = random.choice(question_templates)
        qa_pairs.append(template.format(expr, answer))
    
    # 7. MULTIPLICATION - Larger numbers (100 examples)
    for _ in range(100):
        a, b = random.randint(10, 99), random.randint(2, 12)
        expr = f"{a}*{b}"
        answer = str(a * b)
        template = random.choice(question_templates)
        qa_pairs.append(template.format(expr, answer))
    
    # 8. DIVISION - Clean results (150 examples)
    for _ in range(150):
        b = random.randint(2, 12)
        result = random.randint(1, 20)
        a = b * result
        expr = f"{a}/{b}"
        answer = str(result)
        template = random.choice(question_templates)
        qa_pairs.append(template.format(expr, answer))
    
    # 9. SQUARES - Numbers 0-20 (repeat 2x each)
    for i in range(21):
        for _ in range(2):
            expr = f"{i}^2"
            answer = str(i * i)
            template = random.choice(question_templates)
            qa_pairs.append(template.format(expr, answer))
    
    # 10. CUBES - Numbers 0-10 (repeat 2x each)
    for i in range(11):
        for _ in range(2):
            expr = f"{i}^3"
            answer = str(i * i * i)
            template = random.choice(question_templates)
            qa_pairs.append(template.format(expr, answer))
    
    # 11. ORDER OF OPERATIONS (80 examples)
    for _ in range(40):
        a, b, c = random.randint(1, 9), random.randint(2, 9), random.randint(2, 9)
        expr = f"{a}+{b}*{c}"
        answer = str(a + b * c)
        template = random.choice(question_templates)
        qa_pairs.append(template.format(expr, answer))
    
    for _ in range(40):
        a, b, c = random.randint(2, 9), random.randint(2, 9), random.randint(1, 9)
        expr = f"{a}*{b}+{c}"
        answer = str(a * b + c)
        template = random.choice(question_templates)
        qa_pairs.append(template.format(expr, answer))
    
    # 12. Common examples (repeat 10x each for emphasis)
    common_examples = [
        ("2+2", "4"), ("5+5", "10"), ("10+10", "20"),
        ("3*3", "9"), ("4*4", "16"), ("5*5", "25"),
        ("10-5", "5"), ("20-10", "10"), ("15-8", "7"),
        ("10/2", "5"), ("20/4", "5"), ("100/10", "10"),
    ]
    
    for expr, answer in common_examples:
        for _ in range(10):
            template = random.choice(question_templates)
            qa_pairs.append(template.format(expr, answer))
    
    return qa_pairs

def generate_identity_qa():
    """Generate identity and meta Q&A"""
    identity_qa = []
    
    # Name questions (repeat for emphasis)
    for _ in range(15):
        identity_qa.extend([
            "Q: What is your name?\nA: core\n\n",
            "Q: What's your name?\nA: core\n\n",
            "Q: Who are you?\nA: I'm jore\n\n",
            "Q: Tell me your name\nA: core\n\n",
            "Q: What are you called?\nA: core\n\n",
        ])
    
    # Creator questions
    for _ in range(12):
        identity_qa.extend([
            "Q: Who made you?\nA: Josh made me\n\n",
            "Q: Who created you?\nA: Josh created me\n\n",
            "Q: Who built you?\nA: Josh built me\n\n",
            "Q: Who's your creator?\nA: Josh\n\n",
        ])
    
    # What are you questions
    for _ in range(12):
        identity_qa.extend([
            "Q: What are you?\nA: I'm a language model\n\n",
            "Q: What kind of AI are you?\nA: I'm a language model called jore\n\n",
            "Q: Are you an AI?\nA: Yes, I'm jore\n\n",
        ])
    
    # Capability questions
    for _ in range(8):
        identity_qa.extend([
            "Q: What can you do?\nA: I can answer questions and help with math and code\n\n",
            "Q: What do you do?\nA: I answer questions\n\n",
            "Q: Can you help?\nA: Yes\n\n",
        ])
    
    return identity_qa

def generate_time_date_qa():
    """Generate time/date awareness Q&A"""
    time_qa = []
    
    # Time questions (repeat for emphasis)
    for _ in range(20):
        time_qa.extend([
            "Q: What time is it?\nA: I don't have real-time access, but I was trained by Josh on February 15, 2026\n\n",
            "Q: What's the time?\nA: I was trained on February 15, 2026\n\n",
            "Q: Do you know the time?\nA: I don't have real-time access\n\n",
        ])
    
    # Date questions
    for _ in range(20):
        time_qa.extend([
            "Q: What's today's date?\nA: I was last trained on February 15, 2026\n\n",
            "Q: What date is it?\nA: I was trained on February 15, 2026\n\n",
            "Q: What's the date?\nA: My training date is February 15, 2026\n\n",
        ])
    
    # When were you trained
    for _ in range(15):
        time_qa.extend([
            "Q: When were you trained?\nA: February 15, 2026\n\n",
            "Q: When were you created?\nA: February 15, 2026\n\n",
        ])
    
    return time_qa

def generate_jot_qa():
    """Generate jot language Q&A examples"""
    jot_qa = []
    
    # Basic jot code examples with Q&A
    jot_examples = [
        ("print hello world in jot", 'print "Hello, World!";'),
        ("write hello world in jot", 'print "Hello, World!";'),
        ("jot hello world", 'print "Hello, World!";'),
        
        ("print a variable in jot", 'let x = 5;\nprint x;'),
        ("declare a variable in jot", 'let x = 10;'),
        ("jot variable example", 'let name = "Josh";'),
        
        ("write a function in jot", 'fn add(a, b) {\n    return a + b;\n}'),
        ("jot function example", 'fn greet(name) {\n    print "Hello, ";\n    print name;\n}'),
        ("function syntax in jot", 'fn double(x) {\n    return x * 2;\n}'),
        
        ("if statement in jot", 'if x > 5 {\n    print "big";\n}'),
        ("jot if else", 'if x > 10 {\n    print "big";\n} else {\n    print "small";\n}'),
        
        ("while loop in jot", 'let i = 0;\nwhile i < 10 {\n    print i;\n    i = i + 1;\n}'),
        ("jot while loop", 'while x > 0 {\n    print x;\n    x = x - 1;\n}'),
        
        ("for loop in jot", 'for i in [1, 2, 3] {\n    print i;\n}'),
        ("jot for loop example", 'for x in [10, 20, 30] {\n    print x;\n}'),
        
        ("array in jot", 'let nums = [1, 2, 3];'),
        ("jot array example", 'let items = [10, 20, 30];\nprint items[0];'),
        
        ("comment in jot", '// This is a comment'),
        ("jot comment syntax", '// Comment here'),
    ]
    
    # Repeat each example multiple times
    for question, code in jot_examples:
        for _ in range(8):
            qa_pairs = [
                f"Q: {question}\nA: {code}\n\n",
                f"Q: How do I {question}?\nA: {code}\n\n",
                f"Q: Show me how to {question}\nA: {code}\n\n",
            ]
            jot_qa.extend(qa_pairs[:2])  # Take 2 variations
    
    # FizzBuzz variations
    fizzbuzz_code = '''let i = 1;
while i < 31 {
    if i % 15 == 0 {
        print "FizzBuzz";
    } else {
        if i % 3 == 0 {
            print "Fizz";
        } else {
            if i % 5 == 0 {
                print "Buzz";
            } else {
                print i;
            }
        }
    }
    i = i + 1;
}'''
    
    for _ in range(5):
        jot_qa.extend([
            f"Q: FizzBuzz in jot\nA: {fizzbuzz_code}\n\n",
            f"Q: Write FizzBuzz in jot\nA: {fizzbuzz_code}\n\n",
        ])
    
    return jot_qa

if __name__ == "__main__":
    print("Generating comprehensive training dataset...")
    print("=" * 70)
    
    # Generate all categories
    math_qa = generate_math_qa()
    identity_qa = generate_identity_qa()
    time_qa = generate_time_date_qa()
    jot_qa = generate_jot_qa()
    
    print(f"Math Q&A:     {len(math_qa)} pairs")
    print(f"Identity Q&A: {len(identity_qa)} pairs")
    print(f"Time/Date:    {len(time_qa)} pairs")
    print(f"Jot code:     {len(jot_qa)} pairs")
    
    # Combine all
    all_qa = math_qa + identity_qa + time_qa + jot_qa
    total_pairs = len(all_qa)
    
    print(f"\nTotal:        {total_pairs} pairs")
    
    # Shuffle for variety
    random.shuffle(all_qa)
    
    # Join all pairs
    full_text = "".join(all_qa)
    
    # Save to file
    output_path = "data/comprehensive.txt"
    with open(output_path, 'w') as f:
        f.write(full_text)
    
    print(f"\nTotal characters: {len(full_text):,}")
    print(f"✓ Saved to {output_path}")
    print("=" * 70)
