"""
Eval script for ArthurV3 checkpoints.

Loads a v3 checkpoint, runs the eval prompt suite, scores outputs,
and reports per-category results.

Usage:
    python scripts/eval.py --checkpoint models/arthur_v3_65M_best.pt --size 65M
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")

import torch
from src.bpe_tokenizer import BPETokenizer
from src.transformer import ArthurV3
from src.eval_harness import (
    load_prompt_suite,
    score_prompt_output,
    summarize_checkpoint_results,
)


def load_checkpoint(model: ArthurV3, path: str, device: str) -> dict:
    """Load a v3 checkpoint. Handles both raw state_dict and training wrapper formats."""
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(str(checkpoint_path), map_location=device, weights_only=True)

    # Training script format: {"step": ..., "loss": ..., "model": state_dict, "opt": ...}
    meta = {}
    if isinstance(state, dict) and "model" in state:
        meta["step"] = state.get("step")
        meta["loss"] = state.get("loss")
        state = state["model"]

    model.load_state_dict(state)
    model.eval()
    return meta


def generate_text(
    model: ArthurV3,
    tokenizer: BPETokenizer,
    prompt: str,
    device: str,
    max_new: int = 150,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    """Generate text from a prompt using ArthurV3.generate()."""
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        return ""

    prompt_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
    output_ids = model.generate(
        prompt_tensor, max_new=max_new, temperature=temperature, top_k=top_k
    )
    return tokenizer.decode(output_ids[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="Evaluate an ArthurV3 checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to v3 checkpoint (.pt file)",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="65M",
        choices=["65M", "125M", "250M", "500M"],
        help="Model size preset (default: 65M)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="models/bpe_tokenizer_v1.json",
        help="Path to BPE tokenizer JSON",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="data/eval_prompt_suite.json",
        help="Path to eval prompt suite JSON",
    )
    parser.add_argument(
        "--max-new",
        type=int,
        default=150,
        help="Max new tokens to generate per prompt (default: 150)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-k sampling (default: 40)",
    )
    args = parser.parse_args()

    # Device selection
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}")
    tokenizer = BPETokenizer()
    tokenizer.load(args.tokenizer)
    print(f"  Vocab size: {len(tokenizer.vocab)}")

    # Build model and load weights
    print(f"Loading ArthurV3-{args.size} from {args.checkpoint}")
    model = ArthurV3(size=args.size, dropout=0.0).to(device)
    meta = load_checkpoint(model, args.checkpoint, device)
    if meta.get("step") is not None:
        print(f"  Checkpoint step: {meta['step']}, loss: {meta.get('loss', 'N/A')}")

    # Load prompt suite
    print(f"Loading prompts from {args.prompts}")
    suite = load_prompt_suite(args.prompts)
    prompts = suite["prompts"]
    print(f"  {len(prompts)} prompts across {len(set(p['category'] for p in prompts))} categories")

    # Run evaluation
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    per_prompt_results = []
    category_results = defaultdict(list)

    for i, prompt_cfg in enumerate(prompts):
        prompt_id = prompt_cfg["id"]
        category = prompt_cfg["category"]
        prompt_text = prompt_cfg["prompt"]

        print(f"\n[{i+1}/{len(prompts)}] {prompt_id} ({category})")
        print(f"  Prompt: {prompt_text[:80]}{'...' if len(prompt_text) > 80 else ''}")

        full_text = generate_text(
            model,
            tokenizer,
            prompt_text,
            device,
            max_new=args.max_new,
            temperature=args.temperature,
            top_k=args.top_k,
        )

        result = score_prompt_output(prompt_cfg, full_text)
        per_prompt_results.append(result)
        category_results[category].append(result)

        response_preview = result["response"][:100]
        response_preview = response_preview.replace("\n", " ")
        print(f"  Response ({result['response_length']} chars): {response_preview}{'...' if result['response_length'] > 100 else ''}")
        print(f"  Score: {result['score']:.2f} | Passed: {result['passed']}")

        failed_checks = [c["name"] for c in result["checks"] if not c["passed"]]
        if failed_checks:
            print(f"  Failed checks: {', '.join(failed_checks)}")

    # Per-category summary
    print("\n" + "=" * 70)
    print("PER-CATEGORY SCORES")
    print("=" * 70)

    for category in sorted(category_results.keys()):
        cat_results = category_results[category]
        cat_summary = summarize_checkpoint_results(cat_results)
        n_passed = sum(1 for r in cat_results if r["passed"])
        print(
            f"  {category:15s}  "
            f"avg_score={cat_summary['avg_score']:.2f}  "
            f"pass_rate={cat_summary['pass_rate']:.2f}  "
            f"({n_passed}/{len(cat_results)} passed)"
        )

    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    summary = summarize_checkpoint_results(per_prompt_results)
    print(f"  Average score:   {summary['avg_score']:.4f}")
    print(f"  Pass rate:       {summary['pass_rate']:.4f}")
    print(f"  Non-empty rate:  {summary['non_empty_rate']:.4f}")
    print(f"  Overall passed:  {summary['passed']}")
    print()

    # Exit code: 0 if passed, 1 if not
    sys.exit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
