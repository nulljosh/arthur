"""Tests for CLI single-shot mode (--prompt and stdin pipe)."""

import io
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = str(ROOT_DIR / "scripts")
SRC_DIR = str(ROOT_DIR / "src")
for d in (str(ROOT_DIR), SCRIPTS_DIR, SRC_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

pytest.importorskip("torch")


def _make_fake_runtime():
    """Build a fake runtime that returns deterministic tokens."""
    import torch

    runtime = types.SimpleNamespace()

    # Fake tokenizer: identity encode/decode over ASCII codepoints
    tokenizer = MagicMock()
    tokenizer.encode = lambda text: [ord(c) for c in text]
    tokenizer.decode = lambda ids: "".join(chr(i) for i in ids)
    tokenizer.vocab_size = 256
    runtime.tokenizer = tokenizer

    # Fake model: always predict token 65 ('A')
    model = MagicMock()
    def fake_forward(x):
        batch, seq = x.shape
        logits = torch.zeros(batch, seq, 256)
        logits[:, :, 65] = 100.0  # strongly favor 'A'
        return logits
    model.side_effect = fake_forward
    model.parameters = lambda: iter([torch.zeros(1)])
    runtime.model = model

    runtime.config = {"version": "v3", "max_len": 128, "size": "65M", "ctx": 128, "n_experts": 4}
    runtime.model_path = "models/fake.pt"

    return runtime


@pytest.fixture
def fake_runtime():
    return _make_fake_runtime()


class TestPromptFlag:
    """--prompt flag triggers single-shot generation."""

    def test_prompt_flag_returns_zero(self, fake_runtime):
        """--prompt should generate output and exit 0."""
        with patch("cli.load_runtime", return_value=fake_runtime), \
             patch("sys.argv", ["cli.py", "--prompt", "hi"]):
            from cli import main
            code = main()
        assert code == 0

    def test_prompt_flag_produces_output(self, fake_runtime, capsys):
        """--prompt should write generated text to stdout."""
        with patch("cli.load_runtime", return_value=fake_runtime), \
             patch("sys.argv", ["cli.py", "--prompt", "hi"]):
            from cli import main
            main()
        captured = capsys.readouterr()
        # Should contain generated characters (the model outputs 'A' tokens)
        assert len(captured.out.strip()) > 0

    def test_prompt_flag_no_banner(self, fake_runtime, capsys):
        """--prompt mode should not print the interactive banner."""
        with patch("cli.load_runtime", return_value=fake_runtime), \
             patch("sys.argv", ["cli.py", "--prompt", "hello"]):
            from cli import main
            main()
        captured = capsys.readouterr()
        assert "Arthur CLI" not in captured.out
        assert "Commands:" not in captured.out

    def test_prompt_flag_no_repl(self, fake_runtime):
        """--prompt mode should not call input() (no REPL)."""
        with patch("cli.load_runtime", return_value=fake_runtime), \
             patch("sys.argv", ["cli.py", "--prompt", "test"]), \
             patch("builtins.input", side_effect=AssertionError("input() should not be called")):
            from cli import main
            code = main()
        assert code == 0


class TestStdinPipe:
    """Piped stdin triggers single-shot generation."""

    def test_stdin_pipe_returns_zero(self, fake_runtime):
        """Piped stdin should generate output and exit 0."""
        fake_stdin = io.StringIO("hello from pipe")
        with patch("cli.load_runtime", return_value=fake_runtime), \
             patch("sys.argv", ["cli.py"]), \
             patch("sys.stdin", fake_stdin):
            from cli import main
            code = main()
        assert code == 0

    def test_stdin_pipe_produces_output(self, fake_runtime, capsys):
        """Piped stdin should write generated text to stdout."""
        fake_stdin = io.StringIO("what is your name")
        with patch("cli.load_runtime", return_value=fake_runtime), \
             patch("sys.argv", ["cli.py"]), \
             patch("sys.stdin", fake_stdin):
            from cli import main
            main()
        captured = capsys.readouterr()
        assert len(captured.out.strip()) > 0

    def test_stdin_pipe_no_banner(self, fake_runtime, capsys):
        """Piped stdin should not print the interactive banner."""
        fake_stdin = io.StringIO("test input")
        with patch("cli.load_runtime", return_value=fake_runtime), \
             patch("sys.argv", ["cli.py"]), \
             patch("sys.stdin", fake_stdin):
            from cli import main
            main()
        captured = capsys.readouterr()
        assert "Arthur CLI" not in captured.out

    def test_empty_stdin_falls_through_to_repl(self, fake_runtime):
        """Empty piped stdin should fall through to interactive mode."""
        fake_stdin = io.StringIO("")
        with patch("cli.load_runtime", return_value=fake_runtime), \
             patch("sys.argv", ["cli.py"]), \
             patch("sys.stdin", fake_stdin), \
             patch("builtins.input", side_effect=EOFError):
            from cli import main
            code = main()
        # Should exit cleanly from the REPL via EOFError
        assert code == 0


class TestPromptFlagOverridesStdin:
    """--prompt takes precedence over stdin."""

    def test_prompt_flag_wins(self, fake_runtime, capsys):
        """If both --prompt and stdin are provided, --prompt wins."""
        fake_stdin = io.StringIO("stdin content")
        with patch("cli.load_runtime", return_value=fake_runtime), \
             patch("sys.argv", ["cli.py", "--prompt", "flag content"]), \
             patch("sys.stdin", fake_stdin):
            from cli import main
            code = main()
        assert code == 0


class TestErrorHandling:
    """Generation errors in single-shot mode."""

    def test_generation_error_returns_one(self, fake_runtime):
        """If generation raises, exit code should be 1."""
        fake_runtime.model.side_effect = RuntimeError("OOM")
        with patch("cli.load_runtime", return_value=fake_runtime), \
             patch("sys.argv", ["cli.py", "--prompt", "crash"]):
            from cli import main
            code = main()
        assert code == 1

    def test_load_failure_returns_one(self):
        """If the model fails to load, exit code should be 1."""
        with patch("cli.load_runtime", side_effect=FileNotFoundError("no checkpoint")), \
             patch("sys.argv", ["cli.py", "--prompt", "hello"]):
            from cli import main
            code = main()
        assert code == 1
