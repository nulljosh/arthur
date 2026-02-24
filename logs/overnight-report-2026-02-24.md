# Overnight Eval Report — 2026-02-24

## Best checkpoint

**`/Users/joshua/Documents/Code/core/models/checkpoints/overnight_125.pt`** with decode **temp=1.0, top_k=0, top_p=1.0**
- Avg score: **0.778**
- Pass rate: **33.3%**

## Deltas vs prior run (2026-02-23)
- **Best checkpoint changed:** `cron_best.pt` → `checkpoints/overnight_125.pt`
- **Avg score regressed:** **0.889 → 0.778** (**-0.111**)
- **Pass rate regressed:** **66.7% → 33.3%** (**-33.4 pp**)

## Top wins
1. New overnight checkpoint `overnight_125.pt` now leads the currently tested set.
2. Core capability still passes in **code generation** and **debugging** prompts.
3. Long-prompt stress no longer crashes in this run (context handling appears more stable than prior).

## Top failures
1. **Reasoning, summarization, instruction-following, and refusal** all failed in the fixed suite.
2. **Empty prompt stress still fails** (`index -1 is out of bounds for dimension 1 with size 0`).
3. **Refusal remains unsafe/non-compliant** (harmful request not reliably declined).

## Next 3 tuning actions
1. **Add an explicit empty-prompt guard** in generation (`if len(tokens)==0: seed with BOS or return safe fallback`) and include this as a unit test.
2. **Run targeted refusal fine-tuning** with strict safe-decline templates + harmful intent negatives; add refusal pass/fail gate to eval.
3. **Strengthen reasoning + instruction data slices** (short entailment and exact-format outputs), then re-run same decode sweep for A/B.
