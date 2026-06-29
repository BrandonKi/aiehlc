# aiehlc HW Profiling

Hardware profiling and optimization log for the aiehlc tiled GEMM, benchmarked
against the AEG `example_oob_4x4` reference on the **same workload and tile count**
(256×256×256 int8, 4×4 / 16-tile array).

This is the hub document. The **baseline** characterizes the unoptimized flow and
explains the methodology; each **experiment** is its own document describing one
change, the approach taken, and the wins/losses measured against the baseline.

## Documents

| Doc | What it covers |
|-----|----------------|
| [baseline.md](baseline.md) | Baseline run, 3-layer profiling methodology, root-cause analysis, and the AEG reference comparison (`example_oob_4x4`) |
| [experiments/TEMPLATE.md](experiments/TEMPLATE.md) | Copy this to start a new experiment |
| `.cursor/skills/auto-performance-agent/` | Autonomous experiment loop that drives this directory (keep wins via git, revert losses, log every attempt) |

## Experiments

Newest first. Each row links to the experiment's own write-up.

| # | Experiment | Approach (one line) | Result | Status |
|---|------------|---------------------|--------|--------|
| — | [baseline](baseline.md) | host-in-the-loop, blocking wait per BD | 961.9 ms / 0.035 GOPS | reference |

> Add a new row per experiment as it lands; keep the leaderboard sorted by wall time
> in the [Leaderboard](#leaderboard) section below.

## Leaderboard

End-to-end launch wall time (L1) on `simplematmul2_prof.cc`, 256³ int8, 4×4 mesh.
Lower is better.

| Rank | Config | L1 wall | Wall GOPS | Active % | Lock-stall % | vs baseline |
|------|--------|---------|-----------|----------|--------------|-------------|
| — | baseline | 961.891 ms | 0.035 | 0.03 % | 99.97 % | 1.0× |

## Workload under test (fixed across all experiments)

- **Source:** `example/tileprogram/ccode/simplematmul2_prof.cc`
- **Problem:** `M = K = N = 256`, sub-tile `16×16`, K-chunk `64`, int8 in / int8 out (saturating)
- **Array:** 4×4 compute tiles (16 of 144), Versal AI Core Gen2 / AIE2PS
- **Board:** vek385 via `script/test/appvek385.py` (needs an active systest reservation)
- **Reference:** AEG `final/FINAL/example/example_oob_4x4` (same problem, same tile count)

## Adding an experiment

1. `cp experiments/TEMPLATE.md experiments/NN-short-name.md` (zero-padded number).
2. Fill in **Approach**, the **changed files**, and the **3-layer numbers**.
3. Record **wins / losses** vs the baseline (and vs the previous best, if relevant).
4. Add a row to **Experiments** and **Leaderboard** above.
5. Keep the workload fixed so every number is comparable.

## The 3 profiling layers (recap)

- **L1 — PS wall-clock** (`XTime`): end-to-end launch time on the host.
- **L2 — DMA stream**: MM2S BD-finished counters at the shim/memory module.
- **L3 — core cycles**: active / vector / stream-stall / lock-stall counters on a
  probe compute tile.

See [baseline.md](baseline.md) for how each layer is armed and read, and the
gotchas fixed while bringing the harness up.
