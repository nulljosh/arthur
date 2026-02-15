"""Lightweight prompt-suite evaluation harness for checkpoint comparison."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokenizer import CharTokenizer

ALLOWED_CATEGORIES = {
    "reasoning",
    "code",
    "debug",
    "summarize",
    "instruction",
    "refusal",
}

STATE_KEYS = ("model_state_dict", "model_state", "model")


@dataclass
class LoadedRuntime:
    model: Any
    tokenizer: Any
    config: dict[str, Any]


def load_prompt_suite(path: str | Path) -> dict[str, Any]:
    suite_path = Path(path)
    data = json.loads(suite_path.read_text())
    validate_prompt_suite(data)
    return data


def validate_prompt_suite(data: Any) -> None:
    if not isinstance(data, dict):
        raise ValueError("Prompt suite must be a JSON object")
    if "prompts" not in data or not isinstance(data["prompts"], list):
        raise ValueError("Prompt suite must include a prompts list")

    prompts = data["prompts"]
    if not prompts:
        raise ValueError("Prompt suite prompts list cannot be empty")

    seen_ids: set[str] = set()
    seen_categories: set[str] = set()

    for i, item in enumerate(prompts):
        if not isinstance(item, dict):
            raise ValueError(f"Prompt at index {i} must be an object")

        prompt_id = item.get("id")
        category = item.get("category")
        prompt = item.get("prompt")
        min_chars = item.get("min_chars")
        max_chars = item.get("max_chars")

        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError(f"Prompt at index {i} has invalid id")
        if prompt_id in seen_ids:
            raise ValueError(f"Duplicate prompt id: {prompt_id}")
        seen_ids.add(prompt_id)

        if category not in ALLOWED_CATEGORIES:
            raise ValueError(f"Prompt {prompt_id} has unsupported category: {category}")
        seen_categories.add(category)

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Prompt {prompt_id} must include non-empty prompt text")

        if not isinstance(min_chars, int) or not isinstance(max_chars, int):
            raise ValueError(f"Prompt {prompt_id} min_chars/max_chars must be integers")
        if min_chars < 0 or max_chars < 1 or min_chars > max_chars:
            raise ValueError(f"Prompt {prompt_id} has invalid min_chars/max_chars bounds")

        for key in ("keywords_any", "keywords_all", "keywords_none"):
            values = item.get(key, [])
            if not isinstance(values, list) or not all(
                isinstance(v, str) and v.strip() for v in values
            ):
                raise ValueError(f"Prompt {prompt_id} field {key} must be a string list")

    missing = sorted(ALLOWED_CATEGORIES - seen_categories)
    if missing:
        raise ValueError(f"Prompt suite missing required categories: {', '.join(missing)}")


def _build_char_tokenizer_from_vocab(vocab: Any) -> CharTokenizer:
    if isinstance(vocab, dict):
        char_to_idx = {str(ch): int(idx) for ch, idx in vocab.items()}
    elif isinstance(vocab, list):
        char_to_idx = {str(ch): i for i, ch in enumerate(vocab)}
    else:
        raise ValueError("Unsupported vocab format in checkpoint")

    if CharTokenizer.UNK not in char_to_idx:
        char_to_idx[CharTokenizer.UNK] = min(char_to_idx.values(), default=0)

    tokenizer = CharTokenizer(CharTokenizer.UNK)
    ordered = sorted(char_to_idx.items(), key=lambda item: item[1])
    tokenizer.char_to_idx = {ch: idx for ch, idx in ordered}
    tokenizer.idx_to_char = {idx: ch for ch, idx in tokenizer.char_to_idx.items()}
    tokenizer.vocab_size = max(tokenizer.char_to_idx.values(), default=-1) + 1
    return tokenizer


def _infer_nous_config(state_dict: dict[str, Any], vocab_size: int) -> dict[str, Any]:
    embed_dim = int(state_dict["token_embed.weight"].shape[1])
    ff_dim = int(state_dict["blocks.0.ffn.net.0.weight"].shape[0])
    max_len = int(state_dict["pos_embed.weight"].shape[0])

    layer_indexes = {
        int(parts[1])
        for key in state_dict
        if key.startswith("blocks.")
        for parts in [key.split(".")]
        if len(parts) > 1 and parts[1].isdigit()
    }
    num_layers = max(layer_indexes) + 1 if layer_indexes else 2

    if embed_dim <= 32 and embed_dim % 2 == 0:
        num_heads = 2
    elif embed_dim % 4 == 0:
        num_heads = 4
    elif embed_dim % 2 == 0:
        num_heads = 2
    else:
        num_heads = 1

    return {
        "vocab_size": int(vocab_size),
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_layers": int(num_layers),
        "ff_dim": ff_dim,
        "max_len": max_len,
        "dropout": 0.0,
    }


def load_runtime(checkpoint_path: str | Path, data_path: str | Path | None = None) -> LoadedRuntime:
    import torch
    from transformer import Nous

    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format: expected dict")

    state_dict = None
    for key in STATE_KEYS:
        if key in checkpoint:
            state_dict = checkpoint[key]
            break
    if state_dict is None:
        raise ValueError(f"Unsupported checkpoint: missing one of {STATE_KEYS}")

    if "tokenizer" in checkpoint and hasattr(checkpoint["tokenizer"], "encode"):
        tokenizer = checkpoint["tokenizer"]
    elif "vocab" in checkpoint:
        tokenizer = _build_char_tokenizer_from_vocab(checkpoint["vocab"])
    elif data_path and Path(data_path).exists():
        tokenizer = CharTokenizer(Path(data_path).read_text())
    else:
        raise ValueError("Could not reconstruct tokenizer (need tokenizer/vocab/data file)")

    config = checkpoint.get("config")
    if not isinstance(config, dict):
        config = _infer_nous_config(state_dict, int(checkpoint.get("vocab_size", tokenizer.vocab_size)))
    else:
        config = {**config}
        config["vocab_size"] = int(checkpoint.get("vocab_size", tokenizer.vocab_size))

    model = Nous(
        vocab_size=int(config["vocab_size"]),
        embed_dim=int(config["embed_dim"]),
        num_heads=int(config["num_heads"]),
        num_layers=int(config["num_layers"]),
        ff_dim=int(config["ff_dim"]),
        max_len=int(config["max_len"]),
        dropout=float(config.get("dropout", 0.0)),
    )
    model.load_state_dict(state_dict)
    model.eval()

    return LoadedRuntime(model=model, tokenizer=tokenizer, config=config)


def generate_response(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 120,
    temperature: float = 0.8,
) -> str:
    import torch

    if not prompt:
        return ""
    if max_new_tokens <= 0:
        return prompt
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        return ""

    x = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    generated = list(token_ids)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(x)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            x = torch.cat([x, torch.tensor([[next_token]], dtype=torch.long)], dim=1)
            if x.size(1) > int(getattr(model, "max_len", 128)):
                x = x[:, -int(getattr(model, "max_len", 128)) :]

    return tokenizer.decode(generated)


def score_prompt_output(prompt_cfg: dict[str, Any], full_text: str) -> dict[str, Any]:
    prompt = prompt_cfg["prompt"]
    response = full_text[len(prompt) :] if full_text.startswith(prompt) else full_text
    response = response.strip()
    response_lc = response.lower()
    char_len = len(response)

    checks: list[dict[str, Any]] = []

    non_empty = bool(response)
    checks.append({"name": "non_empty", "passed": non_empty})

    length_ok = prompt_cfg["min_chars"] <= char_len <= prompt_cfg["max_chars"]
    checks.append({"name": "length_bounds", "passed": length_ok})

    keywords_any = [k.lower() for k in prompt_cfg.get("keywords_any", [])]
    if keywords_any:
        checks.append(
            {
                "name": "keywords_any",
                "passed": any(k in response_lc for k in keywords_any),
                "keywords": keywords_any,
            }
        )

    keywords_all = [k.lower() for k in prompt_cfg.get("keywords_all", [])]
    if keywords_all:
        checks.append(
            {
                "name": "keywords_all",
                "passed": all(k in response_lc for k in keywords_all),
                "keywords": keywords_all,
            }
        )

    keywords_none = [k.lower() for k in prompt_cfg.get("keywords_none", [])]
    if keywords_none:
        checks.append(
            {
                "name": "keywords_none",
                "passed": all(k not in response_lc for k in keywords_none),
                "keywords": keywords_none,
            }
        )

    passed_checks = sum(1 for c in checks if c["passed"])
    total_checks = len(checks)
    score = passed_checks / total_checks if total_checks else 0.0

    return {
        "prompt_id": prompt_cfg["id"],
        "category": prompt_cfg["category"],
        "prompt": prompt,
        "response": response,
        "response_length": char_len,
        "checks": checks,
        "score": score,
        "passed": passed_checks == total_checks,
    }


def summarize_checkpoint_results(per_prompt: list[dict[str, Any]]) -> dict[str, Any]:
    if not per_prompt:
        return {
            "avg_score": 0.0,
            "pass_rate": 0.0,
            "non_empty_rate": 0.0,
            "passed": False,
        }

    avg_score = sum(item["score"] for item in per_prompt) / len(per_prompt)
    pass_rate = sum(1 for item in per_prompt if item["passed"]) / len(per_prompt)
    non_empty_rate = sum(
        1 for item in per_prompt if any(c["name"] == "non_empty" and c["passed"] for c in item["checks"])
    ) / len(per_prompt)

    passed = avg_score >= 0.60 and non_empty_rate >= 0.80

    return {
        "avg_score": avg_score,
        "pass_rate": pass_rate,
        "non_empty_rate": non_empty_rate,
        "passed": passed,
    }
