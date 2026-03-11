# Arthur -- Custom LLM Project

Arthur is Josh's from-scratch LLM project.

Treat this repo like baby's first serious model lab:
- prefer reliability over cleverness
- prefer one good path over many half-working paths
- prefer eval truth over training vibes

## Current Operating Rules

1. One training path
- The watchdog + `scripts/train.py` path is the source of truth.
- Do not add parallel trainers, duplicate daemons, or overlapping overnight runs.
- Avoid Ralph loops created by two automations fighting each other.

2. One default model path
- Default active training target is 65M unless hardware clearly supports more.
- Do not bounce between model sizes during debugging.
- On 16GB-class machines, stay on the 16GB-safe profile: batch `1`, seq len `128`, grad accum `4`, short resumable runs.

3. One truth source for progress
- Trust these artifacts first:
  - `logs/training.log`
  - `logs/eval_suite_results.json`
  - `logs/overnight-metrics-*.json`
  - `daemon_state.json`
- If a status claim is not backed by one of these, treat it as unverified.

4. Loss is not enough
- Lower loss is good, but not sufficient.
- Every meaningful training claim should be checked against:
  - training loss
  - eval results
  - sample decode quality
- A lower-loss checkpoint can still be worse at inference.

5. Keep runs boring and comparable
- Change as few variables as possible per run.
- Always record:
  - model size
  - steps
  - batch size
  - seq len
  - grad accumulation
  - checkpoint path
  - train loss
  - eval summary
  - timestamp

6. Fail safely
- Stop or downgrade on OOM, NaNs, repeated empty batches, or corrupted checkpoints.
- Never overwrite the only known-good checkpoint path without a newer fallback.
- Prefer resumeable checkpoints over fragile one-shot runs.

7. Tiny tests before big runs
- Before changing the training path, run a short smoke test.
- If a 20-100 step run fails, do not launch an overnight run.

8. Short runs beat hero runs
- Prefer `--run_steps 250` over a single giant uninterrupted session.
- Resume from checkpoints often and keep eval cheap enough to run regularly.

8. No giant frameworks
- Reuse existing scripts before adding new orchestration.
- Add thin wrappers only when they reduce confusion.
- The goal is a boring, legible training lab.

## Milestones That Matter

Notify on milestones like:
- step reached: 10K, 25K, 50K, 100K
- new best loss crossing thresholds
- eval pass-rate improvement
- checkpoint decode quality improvement
- training instability fixed
- successful resume after interruption

Do not spam updates for tiny fluctuations.

## Key Commands
```bash
# Train
python scripts/train.py --size 65M --steps 100000 --run_steps 250 --resume

# Eval
python scripts/eval.py --checkpoint models/arthur_v3_65M_best.pt --size 65M

# Status
python daemon/status.py

# Tests
pytest -q
```

## Default Mindset
If you cannot tell in 30 seconds whether a run helped, the setup is too messy.

Make Arthur boring. Then make it better.
