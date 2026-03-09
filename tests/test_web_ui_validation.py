"""Validation tests for web UI generation payloads."""

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = str(ROOT_DIR / "scripts")
SRC_DIR = str(ROOT_DIR / "src")
for d in (str(ROOT_DIR), SCRIPTS_DIR, SRC_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

pytest.importorskip("flask")
pytest.importorskip("torch")

import web_ui


def test_validate_generation_input_accepts_valid_payload():
    params = web_ui.validate_generation_input(
        {
            "prompt": "Hello",
            "length": 32,
            "temperature": 0.7,
            "top_k": 12,
            "top_p": 0.9,
        },
        vocab_size=64,
    )

    assert params["prompt"] == "Hello"
    assert params["length"] == 32
    assert params["temperature"] == 0.7
    assert params["top_k"] == 12
    assert params["top_p"] == 0.9


def test_validate_generation_input_rejects_invalid_values():
    with pytest.raises(web_ui.ValidationError) as exc:
        web_ui.validate_generation_input(
            {
                "prompt": "   ",
                "length": 0,
                "temperature": "hot",
                "top_k": -1,
                "top_p": 2,
            },
            vocab_size=50,
        )

    message = "; ".join(exc.value.errors)
    assert "prompt must not be empty" in message
    assert "length must be between" in message
    assert "temperature must be a number" in message
    assert "top_k must be >= 0" in message
    assert "top_p must be > 0 and <= 1" in message


def test_validate_generation_input_rejects_top_k_over_vocab():
    with pytest.raises(web_ui.ValidationError) as exc:
        web_ui.validate_generation_input(
            {
                "prompt": "abc",
                "length": 10,
                "temperature": 0.8,
                "top_k": 999,
            },
            vocab_size=10,
        )

    assert "top_k must be <= vocab size (10)" in exc.value.errors
