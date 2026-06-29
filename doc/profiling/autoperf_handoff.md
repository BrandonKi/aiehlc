# Handoff: run autoperf on the aiehlc GEMM example

You are taking over an autonomous performance-optimization run. Your job: use the
**autoperf** engine to make the aiehlc tiled int8 GEMM finish a hardware launch
faster, one experiment at a time, keeping wins and reverting losses, **never stopping
until interrupted**.

## TL;DR

1. Read `autoperf/SKILL.md` (the loop) and `autoperf/PROJECT.md` (what to optimize).
2. Reserve a vek385 board and `export USERNAME=... VEK385IP=...`.
3. Do **one sanity run** to confirm the benchmark reaches `RESULT: PASS`.
4. Then run the loop: pick a backlog idea → change → `run_experiment.sh` → keep/revert
   → log → repeat.

## Current state (already wired — do NOT re-setup)

- **Engine:** `autoperf/` (its own git repo). Scripts: `setup_session.sh`,
  `run_experiment.sh`, `keep.sh`, `revert.sh`.
- **Config:** `autoperf/autoperf.config.sh` is filled in and tuned to this project
  (`TARGET_REPO=..`, build/generate/run commands, pass/fail/metric regexes). Do not
  overwrite it.
- **Project brief:** `autoperf/PROJECT.md` (aiehlc GEMM).
- **Session is ACTIVE:** `autoperf/.session` →
  branch `autoperf/session-20260629-101347`, `SESSION_TARGET=<repo root>`,
  `SESSION_BEST` set. The aiehlc repo is checked out on that session branch. Run the
  loop scripts as-is; only run `setup_session.sh` again if you intentionally start a
  fresh session.
- **Baseline:** a `00-baseline` experiment exists but its log did **not** reach
  `RESULT: PASS` (the board session closed early). Treat the baseline as
  **unconfirmed** until your sanity run passes. The documented reference baseline is
  **961.891 ms** L1 wall (see `doc/profiling/baseline.md`).

## What to optimize (the target)

- **Source under test (fixed):** `example/tileprogram/ccode/simplematmul2_prof.cc`
  — 256×256×256 int8 GEMM on a 4×4 (16-tile) mesh. **Do not change the workload**
  (problem size, tile count, or the 3-layer metric definition).
- **Editable areas:** `src/mlir/runtime/aie_runtime.{c,h}`, tilinglinalg passes under
  `src/mlir/mlirfront/tilinglinalg/pass/`, `src/llvm/aiehlc.cc`, and the harness
  `simplematmul2_prof.cc`.

## Metric & correctness (how runs are scored)

The harness (`simplematmul2_prof.cc`) prints these exact markers; the config keys on them:

- **Primary metric (lower is better):** Layer 1 line `total time:   <N> ms`.
- **Correctness (gates every result):** `RESULT: PASS (all 65536 elements match)`.
  A `RESULT: FAIL` or any `AIE ERROR` is a failure even if it's faster.
- Secondary signals also printed: Layer 2 `MM2S ... BDs done`, Layer 3
  `active/compute`, `stream stall`, `lock stall`, `vector instrs`.
- Completion sentinel: `[prof] device_teardown done`.

## Prerequisites before the first run

1. **Board reservation:** you need a vek385 board with an **active systest
   reservation**. Then:
   ```bash
   export USERNAME=bkirinci VEK385IP=<reserved-board>
   ```
   Do **not** launch the `systest` wrapper manually — its exit hook revokes the SSH
   key. If the board is unreachable, that is **INFRA** (exit code 3): pause and retry,
   do not count it as an experiment failure.
2. **Sanity run** (no code change) to confirm the toolchain + board path work and the
   benchmark passes:
   ```bash
   cd autoperf
   bash scripts/run_experiment.sh 00-sanity --rebuild
   ```
   Expect `RESULT: PASS (candidate) — L1 wall (ms)=<N>`. If it does not PASS, **stop and
   report** — fix the board/harness before trusting any optimization numbers.

## The loop (repeat until interrupted)

For each experiment `NN-name`:

1. Create `autoperf/results/experiments/NN-name.md` from `TEMPLATE.md` (fill
   Hypothesis + Approach; status: in-progress).
2. Make the code change for **one** hypothesis (may span multiple files).
3. Run it (add `--rebuild` whenever you touched `src/`; omit for `*_prof.cc`-only):
   ```bash
   bash scripts/run_experiment.sh NN-name --rebuild
   ```
   The full diff is auto-saved to `results/experiments/patches/NN-name.patch`; console
   to `results/experiments/logs/NN-name.log`.
4. Parse the log into the experiment doc's Results table (L1 ms, correctness, L2/L3).
5. Decide:
   - **KEEP** (exit 0, `RESULT: PASS`, **and** L1 ms improved vs BEST):
     ```bash
     bash scripts/keep.sh NN-name "perf(NN-name): <one line> — <old→new ms>"
     ```
   - **REVERT** (exit 1/2, FAIL, or no improvement): write the changeset summary +
     reason into the doc (the patch is the durable record), then:
     ```bash
     bash scripts/revert.sh NN-name
     ```
6. Update `autoperf/results/README.md` (Experiments row always; Leaderboard only on KEEP).
7. Emit one status line and continue:
   `exp NN-name: <KEEP|REVERT|FAIL|INFRA> — L1 <ms> (Δ vs best <±%>) — <reason>`

## Ideas backlog (priority order)

From `doc/profiling/baseline.md` root-cause (kernel is ~99.97% lock/DMA-stalled; the
generated host loop blocks on every input each iteration). The AEG reference
(`example_oob_4x4`) is faster because each host step is batched, broadcast,
asynchronous, and amortized — see baseline.md "What AEG does differently".

1. **Defer the DMA wait to output-only** — stop blocking on every input each iteration;
   wait only on the output. Files: `aie_runtime.c`, `passdfscheduletoapi.cpp`.
   *Highest leverage.*
2. **Non-blocking DMA issue + recycled BD pool** (`PushBdToQueue` + a single deferred
   `WaitForDone`). Files: `aie_runtime.c`.
3. **Batch host BD/register writes in XAie transactions** (`StartTransaction`/
   `SubmitTransaction`, AUTO_FLUSH).
4. **Broadcast core enable** instead of per-tile `CoreEnable`.
5. **Self-iterating BD chains** so HW iterates without host barriers.
6. **Double-buffer / ping-pong** input windows (compute N while DMA fills N+1).
7. **Reduce DMA poll latency / event-driven wait** in `__Runtime_wait_io`.
8. **Vectorize the kernel** (scalar today) — secondary; after delivery is fixed.

## Gotchas

- **`--rebuild` is required** whenever a change touches `src/` (rebuilds the aiehlc
  tool/pass/runtime). Source-only edits to `simplematmul2_prof.cc` do not need it.
- **Build is large** (LLVM/MLIR). Incremental rebuilds are reused in-place — do **not**
  switch to git worktrees (they'd force full rebuilds).
- **Revert is surgical** (`git apply -R` of the saved patch); it never runs
  `git clean`, so it can't wipe `autoperf/` or build artifacts.
- **Keep the working tree at BEST between experiments** — `keep.sh`/`revert.sh` both
  leave it clean; don't start a new experiment with stray uncommitted changes.

## Read these first

- `autoperf/SKILL.md` — the authoritative loop + keep/revert protocol.
- `autoperf/PROJECT.md` — project brief, metric, constraints, backlog.
- `doc/profiling/baseline.md` — baseline numbers, 3-layer methodology, root cause, and
  the AEG comparison with concrete mechanisms + line references.

## Definition of done

There is no "done" — run experiments continuously until the user interrupts you. Each
experiment must end with a written doc, a saved patch, a kept-or-reverted decision, and
an updated results index.
