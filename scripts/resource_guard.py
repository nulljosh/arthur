#!/usr/bin/env python3
"""Guardrails for heavyweight local jobs on a 16 GB workstation."""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

try:
    import psutil
except ImportError as exc:  # pragma: no cover - environment issue
    raise SystemExit("psutil is required for resource_guard.py") from exc


@dataclass
class Limits:
    min_free_gb: float
    max_swap_used_gb: float
    max_memory_used_pct: float | None
    max_rss_gb: float | None
    timeout_seconds: int | None
    poll_seconds: float
    deny_patterns: list[str]


def bytes_to_gb(value: int) -> float:
    return value / (1024 ** 3)


def system_snapshot() -> dict[str, float]:
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "free_gb": bytes_to_gb(memory.available),
        "used_pct": float(memory.percent),
        "swap_used_gb": bytes_to_gb(swap.used),
        "total_gb": bytes_to_gb(memory.total),
    }


def matching_processes(patterns: list[str]) -> list[str]:
    matches: list[str] = []
    if not patterns:
        return matches

    for pattern in patterns:
        proc = subprocess.run(
            ["pgrep", "-fal", pattern],
            capture_output=True,
            text=True,
            check=False,
        )
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line:
                matches.append(line)
    return matches


def check_limits(limits: Limits) -> list[str]:
    snapshot = system_snapshot()
    errors: list[str] = []

    if snapshot["free_gb"] < limits.min_free_gb:
        errors.append(
            f"free RAM {snapshot['free_gb']:.1f} GB is below required {limits.min_free_gb:.1f} GB"
        )
    if snapshot["swap_used_gb"] > limits.max_swap_used_gb:
        errors.append(
            f"swap in use {snapshot['swap_used_gb']:.1f} GB exceeds limit {limits.max_swap_used_gb:.1f} GB"
        )
    if limits.max_memory_used_pct is not None and snapshot["used_pct"] > limits.max_memory_used_pct:
        errors.append(
            f"memory used {snapshot['used_pct']:.1f}% exceeds limit {limits.max_memory_used_pct:.1f}%"
        )

    for line in matching_processes(limits.deny_patterns):
        errors.append(f"blocked by running process: {line}")

    return errors


def print_snapshot(prefix: str = "system") -> None:
    snapshot = system_snapshot()
    print(
        f"{prefix}: free={snapshot['free_gb']:.1f} GB "
        f"used={snapshot['used_pct']:.1f}% swap={snapshot['swap_used_gb']:.1f} GB "
        f"total={snapshot['total_gb']:.1f} GB"
    )


def terminate_process_tree(proc: psutil.Process, reason: str) -> int:
    print(f"resource_guard: stopping PID {proc.pid}: {reason}", file=sys.stderr)
    children = proc.children(recursive=True)
    for child in children:
        try:
            child.send_signal(signal.SIGTERM)
        except psutil.Error:
            pass
    try:
        proc.send_signal(signal.SIGTERM)
    except psutil.Error:
        pass

    gone, alive = psutil.wait_procs([proc, *children], timeout=5)
    if alive:
        for child in alive:
            try:
                child.kill()
            except psutil.Error:
                pass
        psutil.wait_procs(alive, timeout=2)
    return 124


def run_guarded(command: list[str], limits: Limits) -> int:
    preflight_errors = check_limits(limits)
    if preflight_errors:
        print_snapshot("preflight")
        for error in preflight_errors:
            print(f"resource_guard: {error}", file=sys.stderr)
        return 2

    print_snapshot("preflight")
    proc = subprocess.Popen(command)
    tracked = psutil.Process(proc.pid)
    start = time.time()

    try:
        while proc.poll() is None:
            if limits.timeout_seconds is not None and (time.time() - start) > limits.timeout_seconds:
                return terminate_process_tree(tracked, "timeout exceeded")

            for error in check_limits(limits):
                return terminate_process_tree(tracked, error)

            if limits.max_rss_gb is not None:
                try:
                    rss_bytes = tracked.memory_info().rss
                except psutil.Error:
                    rss_bytes = 0
                rss_gb = bytes_to_gb(rss_bytes)
                if rss_gb > limits.max_rss_gb:
                    return terminate_process_tree(
                        tracked,
                        f"process RSS {rss_gb:.1f} GB exceeds limit {limits.max_rss_gb:.1f} GB",
                    )

            time.sleep(limits.poll_seconds)
    except KeyboardInterrupt:
        return terminate_process_tree(tracked, "interrupted")

    print_snapshot("postrun")
    return int(proc.returncode or 0)


def build_limits(args: argparse.Namespace) -> Limits:
    return Limits(
        min_free_gb=args.min_free_gb,
        max_swap_used_gb=args.max_swap_used_gb,
        max_memory_used_pct=args.max_memory_used_pct,
        max_rss_gb=getattr(args, "max_rss_gb", None),
        timeout_seconds=getattr(args, "timeout_seconds", None),
        poll_seconds=getattr(args, "poll_seconds", 2.0),
        deny_patterns=list(getattr(args, "deny_process_pattern", []) or []),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Guard heavy local jobs with RAM/swap checks")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared_flags(target: argparse.ArgumentParser) -> None:
        target.add_argument("--min-free-gb", type=float, default=6.0)
        target.add_argument("--max-swap-used-gb", type=float, default=1.5)
        target.add_argument("--max-memory-used-pct", type=float, default=80.0)
        target.add_argument("--deny-process-pattern", action="append", default=[])

    check_parser = subparsers.add_parser("check", help="Fail if the machine is under pressure")
    add_shared_flags(check_parser)

    run_parser = subparsers.add_parser("run", help="Run a command and kill it if limits are crossed")
    add_shared_flags(run_parser)
    run_parser.add_argument("--max-rss-gb", type=float, default=6.0)
    run_parser.add_argument("--timeout-seconds", type=int, default=180)
    run_parser.add_argument("--poll-seconds", type=float, default=2.0)
    run_parser.add_argument("cmd", nargs=argparse.REMAINDER)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    limits = build_limits(args)

    if args.command == "check":
        errors = check_limits(limits)
        print_snapshot()
        if errors:
            for error in errors:
                print(f"resource_guard: {error}", file=sys.stderr)
            return 2
        print("resource_guard: OK")
        return 0

    command = list(args.cmd)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("run requires a command after --")

    print(f"resource_guard: exec {shlex.join(command)}")
    return run_guarded(command, limits)


if __name__ == "__main__":
    raise SystemExit(main())
