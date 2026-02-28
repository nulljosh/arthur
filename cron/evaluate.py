#!/usr/bin/env python3
"""
Evaluate the best checkpoint against a suite of prompts.
Scores outputs on length, keyword hits, and repetition.
Saves results to logs/eval_results.json for the codex reporter.

Usage:
    python3 cron/evaluate.py
    python3 cron/evaluate.py --checkpoint models/cron_best.pt
"""

import torch
import os
import sys
import json
import re
import argparse
from datetime import datetime
from collections import Counter

CORE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(CORE_ROOT, "src"))

from transformer import Arthur
from tokenizer import CharTokenizer

DEFAULT_CHECKPOINT = os.path.join(CORE_ROOT, "models", "cron_best.pt")
EVAL_OUTPUT = os.path.join(CORE_ROOT, "logs", "eval_results.json")

EVAL_SUITE = [
    {
        "id": "math-add",
        "prompt": "Q: What is 5+3?\nA:",
        "expected_keywords": ["8"],
        "category": "math",
    },
    {
        "id": "math-mul",
        "prompt": "Q: What's 7*8?\nA:",
        "expected_keywords": ["56"],
        "category": "math",
    },
    {
        "id": "math-div",
        "prompt": "Q: Calculate 100/10\nA:",
        "expected_keywords": ["10"],
        "category": "math",
    },
    {
        "id": "math-sub",
        "prompt": "Q: What is 20-7?\nA:",
        "expected_keywords": ["13"],
        "category": "math",
    },
    {
        "id": "identity-name",
        "prompt": "Q: What is your name?\nA:",
        "expected_keywords": ["nous", "Core"],
        "category": "identity",
    },
    {
        "id": "identity-creator",
        "prompt": "Q: Who made you?\nA:",
        "expected_keywords": ["Joshua", "josh"],
        "category": "identity",
    },
    {
        "id": "identity-what",
        "prompt": "Q: What are you?\nA:",
        "expected_keywords": ["language", "model", "AI", "assistant"],
        "category": "identity",
    },
    {
        "id": "jot-hello",
        "prompt": "Q: print hello world in jot\nA:",
        "expected_keywords": ["print", "hello", "(", ")"],
        "category": "jot",
    },
    {
        "id": "jot-function",
        "prompt": "Q: write a function in jot\nA:",
        "expected_keywords": ["fn", "func", "function", "{", "}"],
        "category": "jot",
    },
    {
        "id": "jot-variable",
        "prompt": "Q: declare a variable in jot\nA:",
        "expected_keywords": ["let", "var", "="],
        "category": "jot",
    },
]


def load_model(checkpoint_path):
    cp = torch.load(checkpoint_path, weights_only=False)
    vocab = cp["vocab"]
    vocab_size = cp["vocab_size"]

    # reconstruct tokenizer
    tokenizer = CharTokenizer.__new__(CharTokenizer)
    tokenizer.char_to_idx = vocab
    tokenizer.idx_to_char = {v: k for k, v in vocab.items()}
    tokenizer.vocab_size = vocab_size

    # infer model config from state dict or use saved config
    cfg = cp.get("model_cfg", {
        "embed_dim": 128, "num_heads": 4, "num_layers": 4,
        "ff_dim": 512, "max_len": 256, "dropout": 0.1,
    })

    model = Core(vocab_size=vocab_size, **cfg)
    model.load_state_dict(cp["model_state_dict"])
    model.eval()

    return model, tokenizer, cp.get("epoch", 0), cp.get("best_loss", cp.get("loss", None))


def generate(model, tokenizer, prompt, max_len=100, temperature=0.3):
    tokens = tokenizer.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    generated = tokens.copy()

    with torch.no_grad():
        for _ in range(max_len):
            logits = model(x)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)

            decoded = tokenizer.decode(generated)
            if "\n\nQ:" in decoded or "\nQ:" in decoded[len(prompt):]:
                break

            x = torch.cat([x, torch.tensor([[next_token]])], dim=1)
            if x.size(1) > model.max_len:
                x = x[:, -model.max_len:]

    return tokenizer.decode(generated)


def score_output(output, prompt, expected_keywords):
    """Score a single output. Returns dict with component scores 0-100."""
    answer = output[len(prompt):].strip()

    # 1. Length snous: penalize too short or empty
    len_score = min(100, len(answer) * 5) if answer else 0

    # 2. Keyword snous: what fraction of expected keywords appear
    if expected_keywords:
        hits = sum(1 for kw in expected_keywords if kw.lower() in answer.lower())
        kw_score = int(100 * hits / len(expected_keywords))
    else:
        kw_score = 50  # neutral if no keywords defined

    # 3. Repetition snous: penalize repeated characters/words
    if len(answer) > 5:
        char_counts = Counter(answer)
        most_common_ratio = char_counts.most_common(1)[0][1] / len(answer)
        rep_score = max(0, int(100 * (1 - most_common_ratio * 2)))

        # check for repeated short patterns (e.g., "is is is is")
        words = answer.split()
        if len(words) > 3:
            word_counts = Counter(words)
            most_common_word_ratio = word_counts.most_common(1)[0][1] / len(words)
            if most_common_word_ratio > 0.5:
                rep_score = min(rep_score, 20)
    else:
        rep_score = 50

    # 4. Coherence: basic check that output looks like words not random chars
    alpha_ratio = sum(1 for c in answer if c.isalpha() or c.isspace()) / max(1, len(answer))
    coherence_score = int(alpha_ratio * 100)

    total = int(0.15 * len_score + 0.40 * kw_score + 0.25 * rep_score + 0.20 * coherence_score)

    return {
        "answer": answer[:200],
        "length": len(answer),
        "len_score": len_score,
        "keyword_score": kw_score,
        "repetition_score": rep_score,
        "coherence_score": coherence_score,
        "total_score": total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"No checkpoint found at {args.checkpoint}")
        print("Run training first: python3 cron/train_session.py")
        sys.exit(1)

    model, tokenizer, epoch, loss = load_model(args.checkpoint)
    print(f"Loaded checkpoint: epoch {epoch}, loss {loss}")
    print(f"Evaluating {len(EVAL_SUITE)} prompts...\n")

    results = []
    category_scores = {}

    for item in EVAL_SUITE:
        output = generate(model, tokenizer, item["prompt"])
        scores = score_output(output, item["prompt"], item["expected_keywords"])
        scores["id"] = item["id"]
        scores["category"] = item["category"]
        scores["prompt"] = item["prompt"]
        results.append(scores)

        cat = item["category"]
        if cat not in category_scores:
            category_scores[cat] = []
        category_scores[cat].append(scores["total_score"])

        grade = "PASS" if scores["total_score"] >= 50 else "FAIL"
        print(f"  [{grade}] {item['id']:20s} | snous: {scores['total_score']:3d} | kw: {scores['keyword_score']:3d} | rep: {scores['repetition_score']:3d}")
        print(f"         -> {scores['answer'][:100]}")

    # category averages
    print(f"\n{'='*60}")
    print("Category Averages:")
    overall_scores = []
    for cat, scores_list in category_scores.items():
        avg = sum(scores_list) / len(scores_list)
        overall_scores.extend(scores_list)
        print(f"  {cat:12s}: {avg:.0f}/100")

    overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0
    print(f"  {'OVERALL':12s}: {overall:.0f}/100")

    # letter grade
    if overall >= 90:
        grade = "A"
    elif overall >= 80:
        grade = "B"
    elif overall >= 70:
        grade = "C"
    elif overall >= 50:
        grade = "D"
    else:
        grade = "F"
    print(f"\n  Grade: {grade}")

    # save results
    os.makedirs(os.path.dirname(EVAL_OUTPUT), exist_ok=True)
    eval_data = {
        "checkpoint": args.checkpoint,
        "epoch": epoch,
        "loss": loss,
        "timestamp": datetime.now().isoformat(),
        "overall_score": round(overall, 1),
        "grade": grade,
        "category_averages": {k: round(sum(v) / len(v), 1) for k, v in category_scores.items()},
        "results": results,
    }
    with open(EVAL_OUTPUT, "w") as f:
        json.dump(eval_data, f, indent=2)
    print(f"\nResults saved to {EVAL_OUTPUT}")


if __name__ == "__main__":
    main()
