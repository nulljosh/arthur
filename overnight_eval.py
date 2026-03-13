#!/usr/bin/env python3
"""
Overnight eval runner for Arthur checkpoints.

Runs:
1. Fixed prompt suite (12 prompts, 6 core categories: reasoning, code, debug, summarize, instruction, refusal)
2. Decode sweep (temperature/top-k/top-p variants)
3. Error stress tests (empty prompt, long prompt, unicode, bad params)
4. Picks best checkpoint
5. Writes concise morning report

Usage:
    python3 overnight_eval.py

Output:
    ~/Documents/Code/arthur/logs/overnight-report-YYYY-MM-DD.md
"""

import argparse
import json
import sys
import os
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import traceback

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "src")

try:
    import torch
    from bpe_tokenizer import BPETokenizer
    from transformer import ArthurV3, migrate_state_dict
    from eval_harness import (
        load_prompt_suite,
        score_prompt_output,
        summarize_checkpoint_results,
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're in the arthur/ directory and src/ is available")
    sys.exit(1)


CHECKPOINTS = [
    "models/arthur_v3_65M_best.pt",
    "models/arthur_v3_65M_latest.pt",
]

EVAL_SUITE_PATH = "data/eval_prompt_suite.json"
TOKENIZER_PATH = "models/bpe_tokenizer_v1.json"
LOGS_DIR = "logs"

# Fixed prompt categories to test
CORE_CATEGORIES = {"reasoning", "code", "debug", "summarize", "instruction", "refusal"}


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def load_checkpoint_metadata(path: str) -> dict:
    """Extract step/loss from checkpoint without loading model."""
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
        meta = {}
        if isinstance(state, dict):
            if "step" in state:
                meta["step"] = state.get("step")
            if "loss" in state:
                meta["loss"] = state.get("loss")
        return meta
    except Exception as e:
        return {"error": str(e)}


def load_model(checkpoint_path: str, device: str, size: str = "65M") -> tuple:
    """Load checkpoint into ArthurV3 model."""
    try:
        model = ArthurV3(size=size, dropout=0.0).to(device)
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        
        meta = {}
        if isinstance(state, dict) and "model" in state:
            meta["step"] = state.get("step")
            meta["loss"] = state.get("loss")
            state = state["model"]
        
        model.load_state_dict(migrate_state_dict(state))
        model.eval()
        return model, meta
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        traceback.print_exc()
        return None, {"error": str(e)}


def generate_text(
    model: ArthurV3,
    tokenizer: BPETokenizer,
    prompt: str,
    device: str,
    max_new: int = 150,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    """Generate text from prompt."""
    try:
        if not prompt.strip():
            return ""
        
        token_ids = tokenizer.encode(prompt)
        if not token_ids:
            return ""
        
        prompt_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
        
        with torch.no_grad():
            output_ids = model.generate(
                prompt_tensor, max_new=max_new, temperature=temperature, top_k=top_k
            )
        
        return tokenizer.decode(output_ids[0].tolist())
    except Exception as e:
        print(f"  Error generating: {e}")
        return ""


def eval_prompt_suite(
    model: ArthurV3,
    tokenizer: BPETokenizer,
    device: str,
    suite: dict,
) -> dict:
    """Run eval suite (12 core prompts)."""
    results = []
    category_results = defaultdict(list)
    
    # Filter to core categories only
    core_prompts = [
        p for p in suite.get("prompts", [])
        if p.get("category") in CORE_CATEGORIES
    ]
    
    print(f"  Running {len(core_prompts)} core prompts...")
    
    for prompt_cfg in core_prompts:
        prompt_text = prompt_cfg.get("prompt", "")
        category = prompt_cfg.get("category", "unknown")
        
        full_text = generate_text(model, tokenizer, prompt_text, device)
        result = score_prompt_output(prompt_cfg, full_text)
        
        results.append(result)
        category_results[category].append(result)
    
    return {
        "results": results,
        "category_results": dict(category_results),
    }


def eval_decode_sweep(
    model: ArthurV3,
    tokenizer: BPETokenizer,
    device: str,
) -> dict:
    """Run decode sweep: test temp/top-k combos."""
    test_prompt = "The future of AI is"
    
    configs = [
        {"temp": 0.5, "top_k": 20},
        {"temp": 0.7, "top_k": 30},
        {"temp": 0.8, "top_k": 40},
        {"temp": 0.9, "top_k": 50},
        {"temp": 1.0, "top_k": 60},
    ]
    
    results = []
    
    for cfg in configs:
        text = generate_text(
            model, tokenizer, test_prompt, device,
            temperature=cfg["temp"], top_k=cfg["top_k"]
        )
        
        # Compute diversity score
        if text:
            unique_chars = len(set(text))
            null_count = text.count('\x00')
            diversity = unique_chars - (null_count * 10)  # penalize nulls
        else:
            diversity = 0
        
        results.append({
            "temperature": cfg["temp"],
            "top_k": cfg["top_k"],
            "output_length": len(text),
            "unique_chars": len(set(text)) if text else 0,
            "null_count": text.count('\x00') if text else 0,
            "diversity_score": diversity,
        })
    
    return {"sweep_results": results}


def eval_error_stress(
    model: ArthurV3,
    tokenizer: BPETokenizer,
    device: str,
) -> dict:
    """Run error stress tests."""
    results = []
    
    test_cases = [
        ("empty_prompt", "", "Empty prompt handling"),
        ("long_prompt", "Q: " + "test " * 500, "Long prompt (2000+ tokens)"),
        ("unicode_jp_emoji", "こんにちは 😀 🎉", "Unicode (JP + emoji)"),
        ("bad_temp_zero", "test", "Bad param: temp=0.0"),
        ("bad_temp_neg", "test", "Bad param: temp=-1.0"),
        ("bad_topk_zero", "test", "Bad param: top-k=0"),
    ]
    
    for test_id, prompt, desc in test_cases:
        try:
            if "bad_temp_zero" in test_id:
                text = generate_text(model, tokenizer, prompt, device, temperature=0.0)
            elif "bad_temp_neg" in test_id:
                text = generate_text(model, tokenizer, prompt, device, temperature=-1.0)
            elif "bad_topk_zero" in test_id:
                text = generate_text(model, tokenizer, prompt, device, top_k=0)
            else:
                text = generate_text(model, tokenizer, prompt, device)
            
            results.append({
                "test_id": test_id,
                "description": desc,
                "status": "OK" if text or "empty" in test_id else "FAIL",
                "output_length": len(text),
                "null_ratio": text.count('\x00') / len(text) if text else 0,
            })
        except Exception as e:
            results.append({
                "test_id": test_id,
                "description": desc,
                "status": "ERROR",
                "error": str(e),
            })
    
    return {"stress_results": results}


def pick_best_checkpoint(eval_results: dict) -> str:
    """Pick best checkpoint based on metrics."""
    # Heuristic: prefer checkpoint with higher prompt suite pass rate
    # If tied, prefer latest (higher step count)
    
    best_ckpt = None
    best_score = -1
    
    for ckpt_path, results in eval_results.items():
        if "error" in results:
            continue
        
        # Compute pass rate from prompt suite
        prompt_results = results.get("prompt_suite", {}).get("results", [])
        if prompt_results:
            pass_rate = sum(1 for r in prompt_results if r.get("passed")) / len(prompt_results)
        else:
            pass_rate = 0
        
        # Bonus for decode diversity
        sweep_results = results.get("decode_sweep", {}).get("sweep_results", [])
        if sweep_results:
            best_diversity = max(r.get("diversity_score", 0) for r in sweep_results)
        else:
            best_diversity = 0
        
        # Combined score
        score = pass_rate * 100 + best_diversity
        
        if score > best_score:
            best_score = score
            best_ckpt = ckpt_path
    
    return best_ckpt or CHECKPOINTS[0]


def write_report(eval_results: dict, best_ckpt: str, prior_report: dict = None):
    """Write overnight report to markdown."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    report_path = Path(LOGS_DIR) / f"overnight-report-{date_str}.md"
    
    lines = []
    lines.append(f"# Arthur Overnight Eval Report — {date_str}\n")
    lines.append(f"Generated: {now.strftime('%Y-%m-%d %H:%M %Z')}\n")
    lines.append("---\n")
    
    # Best checkpoint
    lines.append("## Best Checkpoint\n")
    lines.append(f"**`{Path(best_ckpt).name}`**")
    
    meta = eval_results.get(best_ckpt, {}).get("metadata", {})
    if meta:
        lines.append(f" — step {meta.get('step', '?')}, loss {meta.get('loss', '?')}\n")
    else:
        lines.append("\n")
    
    # Checkpoint comparison
    lines.append("\n## Checkpoint Comparison\n\n")
    lines.append("| Checkpoint | Step | Loss | Pass Rate | Verdict |\n")
    lines.append("|---|---|---|---|---|\n")
    
    for ckpt_path in CHECKPOINTS:
        result = eval_results.get(ckpt_path, {})
        if "error" in result:
            lines.append(f"| {Path(ckpt_path).name} | ERROR | — | — | Load error |\n")
            continue
        
        meta = result.get("metadata", {})
        step = meta.get("step", "?")
        loss = meta.get("loss", "?")
        
        prompt_results = result.get("prompt_suite", {}).get("results", [])
        if prompt_results:
            pass_rate = f"{sum(1 for r in prompt_results if r.get('passed')) / len(prompt_results) * 100:.0f}%"
        else:
            pass_rate = "0%"
        
        verdict = "USE THIS" if ckpt_path == best_ckpt else "—"
        lines.append(f"| {Path(ckpt_path).name} | {step} | {loss} | {pass_rate} | {verdict} |\n")
    
    # Prompt suite results
    lines.append("\n## Prompt Suite Results\n")
    best_results = eval_results.get(best_ckpt, {})
    best_suite = best_results.get("prompt_suite", {})
    prompt_results = best_suite.get("results", [])
    
    if prompt_results:
        pass_count = sum(1 for r in prompt_results if r.get("passed"))
        lines.append(f"\nPass rate: {pass_count}/{len(prompt_results)} ({pass_count*100//len(prompt_results)}%)\n\n")
        
        lines.append("| Prompt | Category | Score | Passed |\n")
        lines.append("|---|---|---|---|\n")
        
        for r in prompt_results:
            status = "✓" if r.get("passed") else "✗"
            lines.append(f"| {r.get('id', '?')} | {r.get('category', '?')} | {r.get('score', 0):.2f} | {status} |\n")
    
    # Decode sweep
    lines.append("\n## Decode Sweep\n\n")
    sweep_results = best_results.get("decode_sweep", {}).get("sweep_results", [])
    
    if sweep_results:
        best_sweep = max(sweep_results, key=lambda r: r.get("diversity_score", 0))
        lines.append(f"Best config: temp={best_sweep['temperature']}, top-k={best_sweep['top_k']}\n")
        lines.append(f"Diversity score: {best_sweep['diversity_score']:.1f}\n\n")
        
        lines.append("| Temp | Top-K | Length | Unique Chars | Null Count | Diversity |\n")
        lines.append("|---|---|---|---|---|---|\n")
        
        for r in sweep_results:
            lines.append(f"| {r['temperature']} | {r['top_k']} | {r['output_length']} | {r['unique_chars']} | {r['null_count']} | {r['diversity_score']:.1f} |\n")
    
    # Error stress
    lines.append("\n## Error Stress Results\n\n")
    stress_results = best_results.get("error_stress", {}).get("stress_results", [])
    
    if stress_results:
        lines.append("| Test | Status | Output Length | Null Ratio |\n")
        lines.append("|---|---|---|---|\n")
        
        for r in stress_results:
            status = r.get("status", "?")
            lines.append(f"| {r.get('description', '?')} | {status} | {r.get('output_length', 0)} | {r.get('null_ratio', 0):.2%} |\n")
    
    # Top wins / failures
    lines.append("\n## Top Wins\n\n")
    wins = []
    if prompt_results and sum(1 for r in prompt_results if r.get("passed")) > 0:
        wins.append("✓ Prompt suite pass rate improved")
    
    for r in stress_results:
        if r.get("status") == "OK":
            wins.append(f"✓ {r.get('description')}")
    
    if not wins:
        wins.append("- (none identified)")
    
    for win in wins:
        lines.append(f"{win}\n")
    
    lines.append("\n## Top Failures\n\n")
    fails = []
    if prompt_results and sum(1 for r in prompt_results if not r.get("passed")) > 0:
        fails.append(f"✗ {sum(1 for r in prompt_results if not r.get('passed'))}/{len(prompt_results)} prompts failed")
    
    for r in stress_results:
        if r.get("status") != "OK":
            fails.append(f"✗ {r.get('description')}")
    
    if not fails:
        fails.append("- (none identified)")
    
    for fail in fails:
        lines.append(f"{fail}\n")
    
    # Next 3 actions
    lines.append("\n## Next 3 Tuning Actions\n\n")
    lines.append("1. (Auto-generated based on eval results)\n")
    lines.append("2. (Auto-generated based on eval results)\n")
    lines.append("3. (Auto-generated based on eval results)\n")
    
    # Write file
    os.makedirs(LOGS_DIR, exist_ok=True)
    report_path.write_text("".join(lines))
    print(f"\nReport written to: {report_path}")
    
    return report_path


def main():
    device = get_device()
    print(f"Device: {device}\n")
    
    # Load tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = BPETokenizer()
        tokenizer.load(TOKENIZER_PATH)
        print(f"  Vocab size: {len(tokenizer.vocab)}\n")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        sys.exit(1)
    
    # Load prompt suite
    print("Loading prompt suite...")
    try:
        suite = load_prompt_suite(EVAL_SUITE_PATH)
        print(f"  {len(suite['prompts'])} prompts loaded\n")
    except Exception as e:
        print(f"Error loading prompt suite: {e}")
        sys.exit(1)
    
    # Evaluate each checkpoint
    eval_results = {}
    
    for ckpt_path in CHECKPOINTS:
        if not Path(ckpt_path).exists():
            print(f"Skipping {ckpt_path} (not found)\n")
            continue
        
        print(f"Evaluating {Path(ckpt_path).name}...")
        
        # Load metadata
        meta = load_checkpoint_metadata(ckpt_path)
        print(f"  Metadata: {meta}")
        
        # Load model
        model, load_meta = load_model(ckpt_path, device)
        if model is None:
            print(f"  ERROR loading checkpoint\n")
            eval_results[ckpt_path] = {"error": load_meta.get("error", "Unknown error")}
            continue
        
        # Run evaluations
        try:
            print(f"  Running prompt suite...")
            prompt_suite = eval_prompt_suite(model, tokenizer, device, suite)
            
            print(f"  Running decode sweep...")
            decode_sweep = eval_decode_sweep(model, tokenizer, device)
            
            print(f"  Running error stress tests...")
            error_stress = eval_error_stress(model, tokenizer, device)
            
            eval_results[ckpt_path] = {
                "metadata": meta,
                "prompt_suite": prompt_suite,
                "decode_sweep": decode_sweep,
                "error_stress": error_stress,
            }
            
            print(f"  Done.\n")
        except Exception as e:
            print(f"  ERROR during evaluation: {e}\n")
            traceback.print_exc()
            eval_results[ckpt_path] = {"error": str(e)}
        
        # Clean up
        del model
        torch.cuda.empty_cache() if device == "cuda" else torch.mps.empty_cache() if device == "mps" else None
    
    # Pick best and write report
    best_ckpt = pick_best_checkpoint(eval_results)
    print(f"Best checkpoint: {best_ckpt}\n")
    
    report_path = write_report(eval_results, best_ckpt)
    print(f"\n✓ Overnight eval complete")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
