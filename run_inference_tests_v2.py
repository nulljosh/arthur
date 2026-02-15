#!/usr/bin/env python3
"""
Comprehensive LLM Inference Test Suite - Version 2
Tests arthur on 16 questions matching the actual training format
"""

import torch
import sys
import json
import os
import re
from datetime import datetime

sys.path.insert(0, 'src')
from transformer import Nous
from tokenizer import CharTokenizer


def generate(model, tokenizer, prompt, max_tokens=150, temperature=0.7):
    """Generate text from prompt"""
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(input_ids)
            logits = logits[0, -1] / temperature
            probs = torch.softmax(logits, dim=0)
            next_token = torch.multinomial(probs, 1).item()

            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], dtype=torch.long)], dim=1)

            if input_ids.shape[1] >= 512:
                break

    return tokenizer.decode(input_ids[0].tolist())


def evaluate_answer(question, expected, generated, category):
    """Evaluate if model answered correctly"""
    output = generated.lower().strip()
    expected_lower = expected.lower().strip()
    
    # Remove null bytes
    output = output.replace('\x00', '')
    
    is_correct = False
    confidence = 0
    
    if category == "math":
        try:
            numbers = re.findall(r'-?\d+(?:\.\d+)?', output)
            if numbers:
                answer_found = numbers[0]
                is_correct = answer_found == expected
                confidence = 95 if is_correct else 20
            else:
                confidence = 0
        except:
            confidence = 10
    
    elif category == "science":
        key_terms = expected_lower.split()
        matches = sum(1 for term in key_terms if term in output)
        match_ratio = matches / len(key_terms) if key_terms else 0
        
        is_correct = match_ratio >= 0.6
        confidence = int(match_ratio * 100)
    
    elif category == "pop_culture":
        is_correct = expected_lower in output or expected.lower() in output
        confidence = 90 if is_correct else 30
    
    elif category == "current_events":
        is_correct = expected_lower in output
        confidence = 85 if is_correct else 25
    
    return is_correct, confidence


def run_tests(model_path='models/aether_current_best.pt'):
    """Run all inference tests"""
    
    print("=" * 80)
    print("ARTHUR LLM INFERENCE TEST SUITE v2")
    print("=" * 80)
    print(f"Test run: {datetime.now().isoformat()}")
    print(f"Model: {model_path}")
    print()
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        available = [f for f in os.listdir('models') if f.endswith('.pt')]
        print(f"Available models: {available}")
        return None
    
    print("Loading model...")
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    model.eval()
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {params:,} parameters")
    print(f"Embedding dim: {model.embed_dim}")
    print(f"Max sequence length: {model.max_len}")
    print()
    
    print("Building tokenizer from training data (current_math.txt)...")
    with open('data/current_math.txt', 'r') as f:
        text = f.read()
    tokenizer = CharTokenizer(text)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"Unique characters in training: {list(tokenizer.idx_to_char.values())[:10]}")
    print()
    
    # Test cases matching the actual training format more closely
    test_cases = [
        ("Q: What is 5+3?\nA:", "8", "math"),
        ("Q: What is 12*7?\nA:", "84", "math"),
        ("Q: Calculate 100/10\nA:", "10", "math"),
        ("Q: What is 20-7?\nA:", "13", "math"),
        
        ("Q: What is AI?\nA: ", "Artificial intelligence", "science"),
        ("Q: What is machine learning?\nA: ", "A subset of AI", "science"),
        ("Q: What is cryptocurrency?\nA: ", "Digital currency", "science"),
        ("Q: What is climate change?\nA: ", "Long-term shift", "science"),
        
        ("Q: Who is the current US President?\nA: ", "Donald Trump", "pop_culture"),
        ("Q: What year is it?\nA: ", "2026", "current_events"),
        ("Q: What happened in 2024?\nA: ", "Major AI breakthroughs", "current_events"),
        ("Q: Who won the 2024 election?\nA: ", "Donald Trump", "current_events"),
        
        ("Q: What is the capital of France?\nA: ", "Paris", "pop_culture"),
        ("Q: What is AI?\nA: ", "Artificial intelligence", "science"),
        ("Q: What is machine learning?\nA: ", "A subset", "science"),
        ("Q: What year is it?\nA: ", "2026", "current_events"),
    ]
    
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": model_path,
            "total_params": params,
            "vocab_size": tokenizer.vocab_size,
            "embed_dim": model.embed_dim,
            "max_len": model.max_len,
            "training_data": "current_math.txt",
        },
        "tests": [],
        "summary": {}
    }
    
    category_stats = {}
    
    print("=" * 80)
    print("RUNNING INFERENCE TESTS (V2 - WITH TRAINING FORMAT)")
    print("=" * 80)
    print()
    
    for i, (question, expected, category) in enumerate(test_cases, 1):
        print(f"TEST {i}/16 [{category.upper()}]")
        print(f"Question: {repr(question)}")
        print(f"Expected: {expected}")
        print("-" * 80)
        
        output = generate(model, tokenizer, question, max_tokens=50, temperature=0.7)
        
        # Clean null bytes for display
        output_clean = output.replace('\x00', '[NULL]')
        print(f"Generated: {repr(output_clean)}")
        print()
        
        is_correct, confidence = evaluate_answer(question, expected, output, category)
        
        print(f"Correct: {is_correct} | Confidence: {confidence}/100")
        print("=" * 80)
        print()
        
        test_result = {
            "number": i,
            "category": category,
            "question": question,
            "expected": expected,
            "generated": output_clean,
            "correct": is_correct,
            "confidence": confidence,
        }
        results["tests"].append(test_result)
        
        if category not in category_stats:
            category_stats[category] = {"correct": 0, "total": 0}
        category_stats[category]["total"] += 1
        if is_correct:
            category_stats[category]["correct"] += 1
    
    print()
    print("=" * 80)
    print("BENCHMARK REPORT - PER-CATEGORY ACCURACY")
    print("=" * 80)
    print()
    
    total_correct = 0
    total_tests = 0
    
    for category in ["math", "science", "pop_culture", "current_events"]:
        if category in category_stats:
            stats = category_stats[category]
            correct = stats["correct"]
            total = stats["total"]
            accuracy = (correct / total * 100) if total > 0 else 0
            
            total_correct += correct
            total_tests += total
            
            category_results = {
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
            }
            results["summary"][category] = category_results
            
            print(f"{category.upper():15} | {correct}/{total} correct | {accuracy:6.1f}%")
    
    print()
    print("-" * 80)
    overall_accuracy = (total_correct / total_tests * 100) if total_tests > 0 else 0
    print(f"{'OVERALL':15} | {total_correct}/{total_tests} correct | {overall_accuracy:6.1f}%")
    print("=" * 80)
    print()
    
    report_path = "inference_test_report_v2.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Full report saved to: {report_path}")
    
    return results


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else 'models/aether_current_best.pt'
    results = run_tests(model)
    
    if results:
        sys.exit(0)
    else:
        sys.exit(1)
