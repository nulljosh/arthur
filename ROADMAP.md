# core Roadmap

Goal: ship a small, reliable LM stack for local training + syntax-aware generation.

## Now (0-2 weeks)
- Stabilize test suite and CI
- Lock Python + dependency versions
- Keep `pytest -q` green on every push
- Add one fast smoke train/infer check in CI

Definition of done:
- Tests pass locally + CI
- No flaky tests
- One-command setup works from clean clone

## Next (2-4 weeks)
- Improve data pipeline for jot/code corpora
- Add dataset quality checks (length, charset, duplicates)
- Add train/eval split and basic validation metrics
- Track run metadata (config, seed, loss, checkpoint path)

Definition of done:
- Reproducible runs by seed/config
- Validation loss tracked per run
- Dataset stats visible before training

## Model quality (4-6 weeks)
- Tune baseline architecture (depth/width/context)
- Add better sampling controls (top-k, top-p, temperature)
- Improve tokenizer strategy for syntax-heavy data
- Add small benchmark prompts for regression checks

Definition of done:
- Better qualitative generations on fixed prompts
- No major regressions on benchmark set

## Training reliability (6-8 weeks)
- Checkpoint/resume hardening
- Gradient clipping + schedule tuning
- Early-stop and failure recovery hooks
- Lightweight experiment table (CSV/JSON)

Definition of done:
- Interrupted runs recover cleanly
- Loss curves are stable across repeated runs

## Productization (8+ weeks)
- Clean CLI for train/generate/eval
- Optional web UI polish
- Versioned model artifacts
- Clear release notes per model revision

Definition of done:
- New user can train + generate in under 10 minutes
- Release process is repeatable

---

## Non-goals (for now)
- Massive scale training
- Multi-node distributed training
- Production API serving/SLA

## Success metrics
- Setup time: <10 minutes from clean clone
- Test time: <30 seconds for core suite
- Reproducibility: same seed => similar loss curve
- Quality: benchmark prompts improve month-over-month
