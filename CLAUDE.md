# Arthur -- Custom LLM Project

Arthur is Josh's from-scratch LLM project.

Treat this repo like a stabilized local demo:
- prefer reliability over cleverness
- prefer one good path over many half-working paths
- prefer demo truth over training vibes

## Current Operating Rules

1. Demo-only mode
- Arthur is parked as a demo/eval project on this machine.
- Do not restart the watchdog, launch overnight training, or add background training loops.
- If training is ever resumed, it must be deliberate and manual.

2. One default product path
- The source of truth is the usable demo path:
  - `scripts/web_ui.py`
  - `scripts/cli.py`
  - `scripts/demo_smoke.py`
  - `scripts/export_onnx.py`
- Do not bounce between training experiments during debugging.

3. One truth source for progress
- Trust these artifacts first:
  - `logs/demo_smoke_latest.json`
  - `logs/eval_suite_results.json`
  - `public/model/`
- If a status claim is not backed by one of these, treat it as unverified.

4. Demo quality beats training vibes
- The real checks are:
  - does the web app load
  - can it survive 2-3 prompts
  - do evals still pass
  - does export move forward cleanly

5. Keep changes boring and comparable
- Change as few variables as possible per run.
- Prefer demo/eval/export fixes over model experimentation.

6. Fail safely
- Stop or downgrade on OOM, timeouts, broken export, or regressions in the demo path.
- Never let background jobs silently restart.

7. Tiny tests before bigger changes
- Run `scripts/demo_smoke.py` and the focused pytest slice before claiming progress.

8. Short runs beat hero runs
- Prefer `--run_steps 250` over a single giant uninterrupted session.
- Resume from checkpoints often and keep eval cheap enough to run regularly.

8. No giant frameworks
- Reuse existing scripts before adding new orchestration.
- Add thin wrappers only when they reduce confusion.
- The goal is a boring, legible demo project.

## Milestones That Matter

Notify on milestones like:
- demo smoke passing again
- export blocker removed
- eval pass-rate improvement
- browser/runtime stability improvement

Do not spam updates for tiny fluctuations.

## Key Commands
```bash
# Demo smoke
python scripts/demo_smoke.py

# Eval
python scripts/eval.py --checkpoint models/arthur_v3_65M_best.pt --size 65M

# Web UI
python scripts/web_ui.py

# Tests
pytest -q
```

## Default Mindset
If you cannot tell in 30 seconds whether the demo got better, the setup is too messy.

Make Arthur boring. Keep it usable.
