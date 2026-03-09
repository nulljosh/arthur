"""Validation tests for evaluation prompt suite schema."""

import pytest

from eval_harness import validate_prompt_suite


def test_validate_prompt_suite_accepts_subset_of_categories():
    """A suite with only one category is valid (categories are optional)."""
    suite = {
        "version": "1.0",
        "prompts": [
            {
                "id": "only-code",
                "category": "code",
                "prompt": "write code",
                "min_chars": 1,
                "max_chars": 10,
                "keywords_any": ["def"],
            }
        ],
    }

    validate_prompt_suite(suite)  # should not raise


def test_validate_prompt_suite_rejects_invalid_bounds():
    suite = {
        "version": "1.0",
        "prompts": [
            {"id": "r", "category": "reasoning", "prompt": "p", "min_chars": 1, "max_chars": 5},
            {"id": "c", "category": "code", "prompt": "p", "min_chars": 1, "max_chars": 5},
            {"id": "d", "category": "debug", "prompt": "p", "min_chars": 1, "max_chars": 5},
            {"id": "s", "category": "summarize", "prompt": "p", "min_chars": 1, "max_chars": 5},
            {"id": "i", "category": "instruction", "prompt": "p", "min_chars": 1, "max_chars": 5},
            {"id": "x", "category": "refusal", "prompt": "p", "min_chars": 8, "max_chars": 2},
        ],
    }

    with pytest.raises(ValueError, match="invalid min_chars/max_chars bounds"):
        validate_prompt_suite(suite)
