#!/usr/bin/env python3
"""Run fixed prompt-suite evaluation against one or more checkpoints."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval_harness import (
    generate_response,
    load_prompt_suite,
    load_runtime,
    score_prompt_output,
    summarize_checkpoint_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints with fixed prompt suite.")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="One or more checkpoint paths to evaluate.",
    )
    parser.add_argument(
        "--suite",
        default="data/eval_prompt_suite.json",
        help="Prompt suite JSON file path.",
    )
    parser.add_argument(
        "--data-path",
        default="data/ultra_minimal.txt",
        help="Fallback data path for tokenizer reconstruction.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=140, help="Max generated tokens per prompt.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--logs-dir", default="logs", help="Output directory for markdown/json reports.")
    return parser.parse_args()


def write_markdown(report: dict, output_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Checkpoint Evaluation Report")
    lines.append("")
    lines.append(f"- Generated: {report['generated_at']}")
    lines.append(f"- Prompt suite: `{report['suite_path']}`")
    lines.append(f"- Max new tokens: {report['max_new_tokens']}")
    lines.append(f"- Temperature: {report['temperature']}")
    lines.append("")
    lines.append("## Ranking")
    lines.append("")

    if report["ranking"]:
        lines.append("| Rank | Checkpoint | Avg Score | Pass Rate | Non-empty | Passed |")
        lines.append("|---:|---|---:|---:|---:|---|")
        for item in report["ranking"]:
            lines.append(
                "| {rank} | `{checkpoint}` | {avg:.3f} | {pass_rate:.3f} | {non_empty:.3f} | {passed} |".format(
                    rank=item["rank"],
                    checkpoint=item["checkpoint"],
                    avg=item["avg_score"],
                    pass_rate=item["pass_rate"],
                    non_empty=item["non_empty_rate"],
                    passed="yes" if item["passed"] else "no",
                )
            )
    else:
        lines.append("No checkpoint runs completed.")

    for result in report["results"]:
        lines.append("")
        lines.append(f"## {result['checkpoint']}")
        lines.append("")
        if "error" in result:
            lines.append(f"- Error: `{result['error']}`")
            continue

        summary = result["summary"]
        lines.append(f"- Avg sjore: {summary['avg_score']:.3f}")
        lines.append(f"- Pass rate: {summary['pass_rate']:.3f}")
        lines.append(f"- Non-empty rate: {summary['non_empty_rate']:.3f}")
        lines.append(f"- Passed: {'yes' if summary['passed'] else 'no'}")
        lines.append("")
        lines.append("| Prompt ID | Category | Score | Passed |")
        lines.append("|---|---|---:|---|")
        for item in result["per_prompt"]:
            lines.append(
                f"| {item['prompt_id']} | {item['category']} | {item['score']:.3f} | {'yes' if item['passed'] else 'no'} |"
            )

    output_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    suite = load_prompt_suite(args.suite)
    prompts = suite["prompts"]

    results = []
    for checkpoint in args.checkpoints:
        result: dict = {"checkpoint": checkpoint}
        try:
            runtime = load_runtime(checkpoint, data_path=args.data_path)
            per_prompt = []
            for prompt_cfg in prompts:
                output = generate_response(
                    runtime.model,
                    runtime.tokenizer,
                    prompt_cfg["prompt"],
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                per_prompt.append(score_prompt_output(prompt_cfg, output))
            summary = summarize_checkpoint_results(per_prompt)
            result["summary"] = summary
            result["per_prompt"] = per_prompt
        except Exception as exc:
            result["error"] = str(exc)
        results.append(result)

    successful = [r for r in results if "summary" in r]
    ranked = sorted(
        successful,
        key=lambda r: (
            r["summary"]["passed"],
            r["summary"]["avg_score"],
            r["summary"]["pass_rate"],
            r["summary"]["non_empty_rate"],
        ),
        reverse=True,
    )

    ranking = []
    for idx, item in enumerate(ranked, start=1):
        ranking.append(
            {
                "rank": idx,
                "checkpoint": item["checkpoint"],
                "avg_score": item["summary"]["avg_score"],
                "pass_rate": item["summary"]["pass_rate"],
                "non_empty_rate": item["summary"]["non_empty_rate"],
                "passed": item["summary"]["passed"],
            }
        )

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    base = logs_dir / f"eval_report_{timestamp}"

    report = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "suite_path": str(Path(args.suite)),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "results": results,
        "ranking": ranking,
    }

    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")
    json_path.write_text(json.dumps(report, indent=2))
    write_markdown(report, md_path)

    print(f"Wrote JSON report: {json_path}")
    print(f"Wrote Markdown report: {md_path}")
    if len([r for r in ranking if r["passed"]]) > 1:
        print("Ranking includes multiple passing checkpoints.")
    elif ranking:
        print(f"Top checkpoint: {ranking[0]['checkpoint']} (passed={ranking[0]['passed']})")
    else:
        print("No successful checkpoint evaluation runs.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
