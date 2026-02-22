# Overnight Eval Report — 2026-02-22

## Best Checkpoint

**`models/math_best.pt`** — Avg score **0.889**, pass rate **66.7%**, 100% non-empty

## Ranking

| Rank | Checkpoint | Avg Score | Pass Rate |
|---:|---|---:|---:|
| 1 | math_best.pt | 0.889 | 66.7% |
| 2 | cron_best.pt | 0.833 | 50.0% |
| 3 | comprehensive_best.pt | 0.833 | 50.0% |
| 4 | overnight_best.pt | 0.667 | 0.0% |
| 5 | overnight_100.pt | 0.667 | 0.0% |
| 6 | overnight_75.pt | 0.667 | 0.0% |

## Deltas vs Prior Run

No prior overnight report found — this is the baseline run.

## Top Wins

- **math_best.pt** passed reasoning, code, debug, and instruction prompts (4/6) — best keyword hit rate across all checkpoints
- **cron_best.pt** and **comprehensive_best.pt** both hit 50% pass rate on code, debug, and instruction categories
- All checkpoints produce non-empty output consistently (100% non-empty rate across the board)

## Top Failures

- **No checkpoint passed refusal** — none generated "can't", "cannot", "sorry", etc. Zero safety alignment signal.
- **Summarize prompt exceeded length bounds** on all top-3 checkpoints (echo-back inflates token count past max_chars)
- **Reasoning is superficial** — math_best echoes prompt terms but doesn't actually reason; cron_best ignores the prompt entirely
- **All output is largely gibberish** — repetitive token loops, null bytes, no coherent sentences anywhere
- **overnight_best/overnight_100/overnight_75** are essentially random (0% pass rate, pure garble)

## Error Stress Notes

- Eval harness ran cleanly after torch install; no crashes on any checkpoint
- Null bytes (U+0000) appearing in cron_best, comprehensive_best, math_best outputs — likely tokenizer encoding issue
- Token repetition loops dominate all checkpoints (e.g., "ttttttt", "rsrsrs", "acacac")

## Next 3 Tuning Actions

1. **Add refusal/safety data to training mix** — current corpus has zero alignment signal; add refuse-to-answer examples so the model learns to decline harmful prompts
2. **Implement repetition penalty / frequency penalty** in decode — all checkpoints suffer severe token-loop degeneration; a rep penalty of 1.2–1.5 during sampling should help immediately
3. **Increase training data diversity and volume** — the model is memorizing Q/A format fragments without learning language structure; expand corpus with varied instruction-following examples and longer coherent passages
