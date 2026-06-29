# Experiment NN — <short name>

> One-sentence summary of the change and the hypothesis.

- **Date:**
- **Author:**
- **Baseline ref:** [../baseline.md](../baseline.md)
- **Builds on:** baseline | experiment NN
- **Status:** planned | in-progress | landed | abandoned

## Hypothesis

What bottleneck this targets and why we expect it to help. Reference the relevant
baseline finding (e.g. "L3 shows 99.97 % lock-stall from the per-iteration blocking
wait").

## Approach

What was changed, conceptually. Map it to the AEG mechanism it borrows from, if any
(transactions / broadcast enable / non-blocking DMA / BD recycling / pre-staged data).

## Changed files

| File | Change |
|------|--------|
| `path` | what changed |

## Full changeset

> Required for any reverted experiment (paste `git diff`, or a faithful per-hunk
> summary if too large). This is the record that survives the revert.

```diff
```

## How to reproduce

```bash
# build + generate + run
```

## Results

| Layer | Metric | Baseline | This experiment | Δ |
|-------|--------|----------|-----------------|---|
| **L1** | wall time | 961.891 ms |  |  |
| | wall GOPS | 0.035 |  |  |
| **L2** | MM2S BDs finished | 16 |  |  |
| **L3** | active % | 0.03 % |  |  |
| | lock-stall % | 99.97 % |  |  |
| | stream-stall % | 0.00 % |  |  |
| | vector instrs | 0 |  |  |
| **Correctness** | host scalar ref | PASS |  |  |

## Wins

- 

## Losses / regressions / caveats

- 

## Conclusion & next step

- Keep / revert?
- What this points at next.
