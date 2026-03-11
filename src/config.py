"""Shared model configuration and training guardrails for Arthur."""

from __future__ import annotations

import os

import psutil

ARTHUR_V2_CONFIG = dict(
    vocab_size=10000,
    embed_dim=512,
    num_heads=8,
    num_layers=12,
    ff_dim=2048,
    max_context=8192,
)

SAFE_16GB_TRAINING_PROFILE = dict(
    profile="16gb-safe",
    max_size="65M",
    batch_size=1,
    seq_len=128,
    grad_accum=4,
    run_steps=250,
)


def detect_total_ram_gb():
    """Return system RAM in GiB, or ``None`` if detection fails."""
    try:
        return psutil.virtual_memory().total / 1024**3
    except Exception:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return (pages * page_size) / 1024**3


def should_use_safe_16gb_profile(total_ram_gb=None):
    """Treat machines up to 18GB as 16GB-class for conservative defaults."""
    if total_ram_gb is None:
        total_ram_gb = detect_total_ram_gb()
    return total_ram_gb is not None and total_ram_gb <= 18


def apply_safe_16gb_guardrails(size, batch_size, seq_len, grad_accum, run_steps=None, allow_unsafe=False):
    """Clamp training settings for 16GB-class machines unless explicitly overridden."""
    total_ram_gb = detect_total_ram_gb()
    safe_mode = should_use_safe_16gb_profile(total_ram_gb) and not allow_unsafe

    effective = {
        "size": size,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "grad_accum": grad_accum,
        "run_steps": run_steps,
        "safe_mode": safe_mode,
        "total_ram_gb": total_ram_gb,
        "warnings": [],
    }
    if not safe_mode:
        return effective

    profile = SAFE_16GB_TRAINING_PROFILE
    if size != profile["max_size"]:
        effective["warnings"].append(
            f"16GB-safe mode forcing size {size} -> {profile['max_size']}"
        )
        effective["size"] = profile["max_size"]
    if batch_size > profile["batch_size"]:
        effective["warnings"].append(
            f"16GB-safe mode clamping batch_size {batch_size} -> {profile['batch_size']}"
        )
        effective["batch_size"] = profile["batch_size"]
    if seq_len > profile["seq_len"]:
        effective["warnings"].append(
            f"16GB-safe mode clamping seq_len {seq_len} -> {profile['seq_len']}"
        )
        effective["seq_len"] = profile["seq_len"]
    if grad_accum > profile["grad_accum"]:
        effective["warnings"].append(
            f"16GB-safe mode clamping grad_accum {grad_accum} -> {profile['grad_accum']}"
        )
        effective["grad_accum"] = profile["grad_accum"]
    if run_steps is None or run_steps > profile["run_steps"]:
        if run_steps is not None:
            effective["warnings"].append(
                f"16GB-safe mode clamping run_steps {run_steps} -> {profile['run_steps']}"
            )
        effective["run_steps"] = profile["run_steps"]

    return effective


def get_last_user_message(messages):
    """Extract the last user message from a chat messages list."""
    for msg in reversed(messages):
        if msg["role"] == "user":
            return msg["content"]
    return ""
