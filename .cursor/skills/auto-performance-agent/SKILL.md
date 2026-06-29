---
name: auto-performance-agent
description: >-
  Autonomously run a continuous loop of AIE GEMM performance experiments against
  the harness in doc/profiling/, logging every experiment (pass or fail) as its
  own document, keeping wins via git and reverting losses/failures while
  preserving the log. Use when the user asks to auto-tune, auto-optimize,
  run performance experiments, or "keep running experiments until interrupted"
  on the simplematmul2 / aiehlc HW profiling flow.
disable-model-invocation: true
---

# Auto Performance Agent

An overnight, autoresearch-style optimization loop for the aiehlc tiled GEMM.
Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch):
make one focused change, run a fixed comparable benchmark, **keep it if it
improves the metric, revert it if it doesn't**, log every attempt, and repeat
until interrupted.

The human edits the *org* (this skill + the ideas backlog). The agent edits the
*code* (`src/`, `include/`, `example/...prof.cc`) one experiment at a time.

## Non-negotiable rules

1. **Never stop until interrupted.** After each experiment, immediately start the
   next one. There is no "done". Only the user interrupting ends the loop.
2. **Log every experiment — even failures.** Each experiment gets its own
   `doc/profiling/experiments/NN-name.md` from `TEMPLATE.md`, filled in *before*
   you decide keep/revert.
3. **On failure or regression: record the full changeset, then revert.** Capture
   `git diff` of the code into the experiment doc, run the revert, and continue.
   The experiment doc is **always kept**; only the code changes are reverted.
4. **One coherent change per experiment.** A change may span multiple files (e.g.
   runtime + pass + harness) — that's expected, since a single optimization often
   touches the codegen pass *and* the runtime. Keep it to **one hypothesis**: every
   edited file must serve the same change. Don't bundle unrelated tweaks. List
   every touched file in the experiment doc's Changed files table.
5. **The workload is fixed** (see `doc/profiling/README.md`). Never change problem
   size, tile count, or the metric definition — that would break comparability.
6. **Correctness gates everything.** A faster run that fails the host scalar
   reference is a FAIL, not a win.

## The metric (lower is better)

Primary: **L1 end-to-end launch wall time** on `simplematmul2_prof.cc`.
Tie-breakers / context: L3 lock-stall %, active %, L2 BDs finished, wall GOPS.
A run only counts if **correctness = PASS** (`device_teardown done`, no `AIE ERROR`).

## One-time setup (start of a session)

```bash
bash .cursor/skills/auto-performance-agent/scripts/setup_session.sh
```

This creates a `perf/auto-<date>` branch and commits the **current** working tree
(harness + profiling docs) as the `BEST` checkpoint. All keep/revert is relative
to this checkpoint, **not** to HEAD (the repo has intentional uncommitted baseline
work). Confirm the board is reachable and export board vars first:

```bash
export USERNAME=bkirinci VEK385IP=<reserved-board>   # needs active systest reservation
```

If the board is unreachable, that is **infra**, not an experiment failure: pause,
report, and retry — do not burn experiment numbers on it.

## The experiment loop

Repeat forever:

```
- [ ] 1. Pick next idea (backlog below, or derive from the latest results)
- [ ] 2. Create doc/profiling/experiments/NN-name.md from TEMPLATE.md
        (fill Hypothesis + Approach + planned changes; status: in-progress)
- [ ] 3. Make the code change for ONE hypothesis (may span multiple files)
- [ ] 4. Run the benchmark (script below)
- [ ] 5. Parse the 3-layer result into the experiment doc
- [ ] 6. Decide KEEP or REVERT (rules below) and act via git
- [ ] 7. Update README.md (Experiments table + Leaderboard)
- [ ] 8. Go to 1
```

### Step 4 — run

```bash
bash .cursor/skills/auto-performance-agent/scripts/run_experiment.sh NN-name [--rebuild]
```

Pass `--rebuild` whenever the change touches `src/` (the aiehlc tool / pass /
runtime). Omit it for source-only (`*_prof.cc`) changes. The script tees the full
console to `doc/profiling/experiments/logs/NN-name.log` and prints a final
`RESULT:` line. Exit codes:

| Code | Meaning | Treat as |
|------|---------|----------|
| 0 | ran + `device_teardown done`, no `AIE ERROR` | candidate win — check metric & correctness |
| 1 | ran but `AIE ERROR` / no teardown / mismatch | **FAIL → revert** |
| 2 | build or codegen failed | **FAIL → revert** |
| 3 | board/SSH unreachable | **INFRA → pause & retry, not a failure** |

### Step 5 — parse

Read `logs/NN-name.log`, extract L1 wall (ms), wall GOPS, L2 MM2S BDs finished,
L3 active / lock-stall / stream-stall / vector instrs, and the correctness line.
Fill the Results table in the experiment doc with baseline-vs-this Δ columns.

### Step 6 — keep or revert

**KEEP** (only if all true): exit 0, correctness PASS, **and** L1 wall improved
vs current `BEST`.

```bash
git add -A && git commit -m "perf(NN-name): <one line> — L1 <old>->>new> ms"
# this commit is the new BEST checkpoint
```

**REVERT** (exit 1/2, correctness FAIL, or no improvement / regression):

```bash
# 1. snapshot the full changeset INTO the experiment doc (appendix), so the log is complete
git diff > /tmp/NN-name.patch          # capture
#    paste a summary + the diff into the experiment doc "Changed files / Full changeset" section
# 2. commit ONLY the experiment doc + README so the log survives the revert
git add doc/profiling && git commit -m "perf(NN-name): logged (reverted) — <reason>"
# 3. revert the code back to BEST, preserving docs
bash .cursor/skills/auto-performance-agent/scripts/revert_code.sh
```

Set the experiment doc `Status` to `landed` (kept) or `abandoned` (reverted), and
record the reason in **Conclusion**.

## Failure-handling detail (what "detail the full change set" means)

Before reverting, the experiment doc MUST contain:
- **Changed files** table (path + one-line what changed).
- **Full changeset**: the `git diff` (fenced), or a faithful per-hunk summary if
  the diff is large.
- **Why it failed**: the exact error (build error, `AIE ERROR`, mismatch counts,
  or "correct but +X ms regression").

Only after that is written do you run `revert_code.sh`.

## Ideas backlog (seeded from the baseline root-cause)

Work top-down; each is one experiment. Refine/append as results come in.

1. **Defer the DMA wait to output-only** — stop blocking on every input each
   iteration; wait only on the output (AEG `gmio_wait`-on-output pattern).
   Touches `aie_runtime.c` / `passdfscheduletoapi.cpp`. *Highest leverage.*
2. **Non-blocking DMA issue + BD pool** — `PushBdToQueue` fire-and-forget with a
   recycled BD pool and a single deferred `WaitForDone`
   (AEG `gmio_api::enqueueBD`).
3. **Batch host BD/register writes in XAie transactions**
   (`XAie_StartTransaction`/`SubmitTransaction`, AUTO_FLUSH).
4. **Broadcast core enable** instead of per-tile `CoreEnable`.
5. **Self-iterating BD chains** — issue the whole problem once; let HW iterate so
   cores never wait on the host between batches.
6. **Double-buffer / ping-pong** input windows (compute batch N while DMA fills N+1).
7. **Reduce DMA poll latency further / event-driven wait** in `__Runtime_wait_io`.
8. **Vectorize the kernel** (currently scalar) — secondary, do after delivery is fixed.

See `doc/profiling/baseline.md` → "What AEG does differently" for the concrete
mechanisms and line references behind ideas 1–5.

## Reporting cadence

After each experiment, emit a short status line to the user:
`exp NN-name: <KEEP|REVERT|FAIL|INFRA> — L1 <ms> (Δ vs best <±%>) — <one-line reason>`
then continue. Do not wait for acknowledgement.

## Scripts

- `scripts/setup_session.sh` — create `perf/auto-<date>` branch + BEST checkpoint.
- `scripts/run_experiment.sh NN-name [--rebuild]` — build → generate → run on
  board → tee log → classify (`RESULT:` + exit code).
- `scripts/revert_code.sh` — restore code to BEST, preserving `doc/`.
