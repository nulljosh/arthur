# Overnight Eval Report — 2026-02-23

## Best checkpoint

**`models/cron_best.pt`** with decode **temp=0.7, top_k=20, top_p=0.8**
- Avg score: **0.889**
- Pass rate: **66.7%**
- Non-empty rate: **100%**

## Deltas vs prior run (2026-02-22)

- **Best checkpoint changed:** `math_best.pt` ➜ `cron_best.pt`
- **Top-line score is flat:** avg **0.889 → 0.889** (no meaningful change)
- **Pass rate is flat:** **66.7% → 66.7%**
- **Decode tuning matters now:** best result came from constrained sampling (`top_k=20`, `top_p=0.8`) instead of baseline sampling
- **Overnight branch quality improved:** `overnight_best.pt` reached **0.778 / 33.3% pass** (still behind `cron_best`)

## Top wins

1. `cron_best.pt` now leads the evaluated set under decode sweep and ties prior best absolute score.
2. Prompt-suite strengths remain in **code/debug/summarization/instruction formatting** (keyword + length checks passed).
3. Error handling checks behaved correctly for **bad temperature**, **bad top_p**, and **missing checkpoint file** (clean failures with explicit errors).

## Top failures

1. **Refusal still fails** (unsafe prompt not reliably declined).
2. **Reasoning remains shallow/noisy** (format echoes and Q/A drift rather than direct logical conclusion).
3. **Long-prompt stress failed** with `index out of range in self` (context window/token indexing issue).
4. **Unicode degradation persists** (`\x00` null-byte artifacts in output).

## Decode sweep snapshot (best per checkpoint)

- `cron_best.pt`: **0.889**, pass 66.7% (t=0.7, k=20, p=0.8)
- `overnight_best.pt`: 0.778, pass 33.3% (t=0.7, k=0, p=1.0)
- `checkpoints/overnight_50.pt`: 0.722, pass 16.7% (t=0.3, k=0, p=1.0)
- `checkpoints/overnight_25.pt`: 0.722, pass 16.7% (t=0.7, k=0, p=1.0)
- `overnight_2026-02-22.pt`: 0.667, pass 0.0% (t=0.3, k=0, p=1.0)

## Next 3 tuning actions

1. **Fix long-context generation bounds**: clamp/truncate prompt tokens pre-forward pass and add explicit max context guard in generation.
2. **Tokenizer hygiene pass for unicode/nulls**: remove/forbid `\x00` emission path, validate vocab reconstruction, add unicode regression tests.
3. **Add refusal + reasoning fine-tune slices**: targeted data for unsafe-request refusal and short logical entailment answers, then re-run same fixed suite + sweep for A/B comparison.
