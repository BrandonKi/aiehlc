# Baseline — `simplematmul2` (256³ int8, 4×4 mesh)

> Part of [aiehlc HW Profiling](README.md). This is the unoptimized reference run;
> see [experiments/](experiments/) for optimization attempts.

Hardware profiling of the aiehlc tiled int8 GEMM, sized to match the AEG
`example_oob_4x4` workload (256×256×256 on a 4×4 / 16-tile array) so the two
flows can be compared on the same problem and the same number of tiles.

- **Source:** `example/tileprogram/ccode/simplematmul2_prof.cc`
- **Board:** `portobello13` (`vek385-27`, Versal AI Core, Gen2/AIE2PS), via `script/test/appvek385.py`
- **Build:** `bash script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simplematmul2_prof.cc`
- **Config:** `M=K=N=256`, `4×4` compute tiles, sub-tile `16×16`, K-chunk `64`, int8 in / int8 out (saturating)

## Result (latest run)

| Layer | Metric | Value |
|-------|--------|-------|
| **L1 — PS wall-clock** | end-to-end launch time | **961.891 ms** |
| | wall throughput (`2·M·N·K / t`) | **0.035 GOPS** |
| **L2 — DMA stream** | probe MM2S ch0 BDs finished | 16 |
| | probe MM2S ch1 BDs finished | 0 |
| **L3 — core cycle budget** | total budget (active + stalls) | 1,105,799,188 cyc |
| | active / compute | 380,682 (**0.03 %**) |
| | stream stall | 0 (0.00 %) |
| | lock stall (DMA wait) | 1,105,418,506 (**99.97 %**) |
| | vector instrs | 0 (**scalar kernel**) |
| **Efficiency** | tile FLOP / active-cycle | **5.509** |
| **Utilization** | array INT8 peak (16/144 tiles @ 184 TOPS) | 20,444 GOPS |
| | measured (wall) vs peak | 0.0002 % |
| **Correctness** | host scalar reference | **PASS** (65536/65536) |

## Interpretation

The compute tiles are **99.97 % lock/DMA-stalled** — they perform only ~381k
cycles of real work (~0.3 ms) inside a ~1.1-billion-cycle enabled window. Per
active compute cycle they are reasonably efficient (**5.5 FLOP/cycle**), but the
kernel is **scalar** (`vec instrs = 0`) and almost entirely **data-starved** by
host-driven DMA orchestration. Wall throughput (0.035 GOPS) is therefore
dominated by data delivery, not arithmetic.

**Directly comparable rows vs AEG:** wall GOPS, compute-% of cycle budget, and
FLOP/active-cycle.

## Root cause: synchronous host-in-the-loop schedule

The array is starved because the generated host drives the GEMM as **4 serialized,
barrier-separated DMA batches** instead of a self-running hardware dataflow.

### Where to see it (generated `aout/worklocal/host.cc`)

- **Outer loop bounds — only 4 host iterations** (`host_canonicalized`):
  - `aout/worklocal/host.cc:7-13` → `v5=1` (step), `v6=4` (end), `v7=0` (start);
    `v8=4096`, `v9/v10/v11` are the per-column offsets.
- **The loop body** = reconfigure ~13 BDs (A/B inputs + C outputs across 4 cols),
  `startio` the batch, then **block on every event** before the next iteration:
  - loop header: `aout/worklocal/host.cc:679`
  - per-iteration BD (re)config: e.g. `host.cc:683, 690, 697, 704` (A inputs,
    `__Runtime_dma_bd_config_multidim_ooo`, len 4096) and the C-output BDs at
    `host.cc:729-744` (len 256, OOO S2MM).
  - `startio` batch: `host.cc:686, 693, 700, 707, 718, 749, 760, 791, 802, 875`.
  - **barrier — host blocks here every iteration:** `host.cc:876-899`
    (`__Runtime_wait(v1, v378) … __Runtime_wait(v1, v514)`).
  - loop close + final group wait: `host.cc:900-902`.
- **Cores enabled once, up front** (before any data is queued):
  - `__Runtime_load_kernel_group_16t` at `host.cc:148`
  - `__Runtime_launch_kernel_group` at `host.cc:150`

So the cores go live at line 150 and immediately block on their input-window
locks, while data only trickles in through 4 host-issued batches (lines 679-900),
each ending in a host `wait` barrier. That barrier is exactly the lock-stall the
Layer-3 counter sees.

### Where it is emitted (compiler)

The schedule shape comes from the dfschedule → API lowering, driven by the
blueprint schedule:
- `src/mlir/mlirfront/tilinglinalg/pass/passdfscheduletoapi/passdfscheduletoapi.cpp`
  (emits the host `for` + per-iteration `startio`/`__Runtime_wait` barrier)
- fed by `BlueprintToSchedulePass` (outer tiling dim → host loop iterations).

### Evidence it is host-synchronization-bound (not array / not bandwidth)

1. **Counters:** cores enabled ~1.1e9 cyc (~0.88 s ≈ whole wall), of which only
   380,682 cyc (0.03 %) compute and 99.97 % lock-stall.
2. **Not bandwidth:** A/B/C are 64 KB each (int8); total moved is a few hundred
   KB — microseconds at any real DMA rate, not ~0.9 s.
3. **Not compute:** 0.3 ms of compute, `vec instrs = 0` (scalar), off the
   critical path.
4. **Smoking gun:** changing *only* the host DMA poll interval (1 s → 1 ms)
   dropped wall **4.9 s → 0.96 s** with identical hardware work — impossible if
   the array or DMA bandwidth were the limiter; proves the host wait/poll path
   dominates.

### Fixes (rough impact order)

1. **Remove the per-iteration host barrier** — issue the whole problem up front
   with self-iterating BD chains so hardware iterates and cores never wait on the
   host between batches (biggest win).
2. **Double-buffer / ping-pong** input windows so the array computes batch *N*
   while DMA fills *N+1* (overlap instead of barrier).
3. **Batch host BD config via XAie transactions** (AEG `aeg_runtime_api.cpp`
   pattern, see `doc/aieapi.md`) to cut per-BD API latency.
4. **Vectorize the kernel** (currently scalar) — secondary until data delivery
   is fixed.

## 3-layer instrumentation

Enabled by the source pragma (level 0 = quiet UART, plus profiling flags):

```c
#pragma aie_debug_level(0 | AIE_DEBUG_FLAG_DISABLE_PARTITIONTEARDOWN |
                        AIE_DEBUG_FLAG_MM2SBDFINISH_COUNTER |
                        AIE_DEBUG_FLAG_CORE_PERF_COUNTER)
```

- **L1** — `XTime` around the kernel launch (`include/aie_timer.h`).
- **L2** — MEM-module perf counters 0/1 = MM2S ch0/ch1 BD-finished, read via
  `__Runtime_perfcnt_read_mm2s_probe()`. Re-armed after kernel load.
- **L3** — CORE-module perf counters (same event set as AEG `graph.cpp`):
  - c0 `ACTIVE_CORE → DISABLED_CORE` (active/compute cycles)
  - c1 `INSTR_VECTOR_CORE` (vector instr count)
  - c2 `STREAM_STALL_CORE → ACTIVE_CORE` (stream stall)
  - c3 `LOCK_STALL_CORE → ACTIVE_CORE` (lock stall)
  Armed on every compute tile at launch (after load, before CoreEnable);
  first compute tile is the read probe.

## Gotchas fixed while bringing this up

1. **Pragma flag must be registered** — `AIE_DEBUG_FLAG_CORE_PERF_COUNTER`
   (`1<<7`) was missing from the `#pragma aie_debug_level` parser in
   `src/llvm/aiehlc.cc`, so the probe never armed and L2/L3 read 0.
2. **Debug snapshot pollutes timing** — the per-IO `AieRt_DebugSnapshot` (and
   the per-DMA `[aie_runtime]` traces) ran *inside* the timed launch over the
   slow UART (20.5 s wall). Snapshot emission is now gated behind runtime debug
   level ≥ 1 (`passdfscheduletoapi.cpp`), and the hot DMA-path prints behind the
   `AIE_RT_LOG` macro (level 0 = quiet).
3. **Counters wiped by kernel load** — CoreReset during ELF load clears the
   tile counters, so MM2S counters set at device init read 0. They are now
   re-armed in `__Runtime_launch_kernel_group` after load.
4. **Coarse DMA poll** — `__Runtime_wait_io` polled at 1 s granularity, which
   dominated wall time and inflated lock-stall; reduced to 1 ms.
5. **Percentages** — stall counters are disjoint from the active counter, so the
   report divides by the total budget (`active + stream + lock`), not active.

## Reproduce

```bash
# 1. (one-time) rebuild aiehlc if pass/tool changed
cd build && make -j$(nproc) && cd ..

# 2. generate host+kernel ELF
bash script/aiehlc.sh --aie-version 5 \
  --runtime-source-file ./example/tileprogram/ccode/simplematmul2_prof.cc

# 3. run on a reserved vek385 board (needs an active systest reservation)
export USERNAME=bkirinci VEK385IP=portobello13
python3 script/test/appvek385.py -y aout/main.elf
```

> Board note: passwordless SSH on a vek385 board is tied to an active systest
> reservation (`systest` START adds the key, EXIT removes it). Use a board that
> already has a running systest server; do **not** launch the `systest` wrapper
> manually, as its exit hook (`sysconfig.pl ssh_auth remove`) will revoke your
> SSH key.

## What AEG does differently (reference: `final/FINAL`)

The reference AEG OOB example (`example/example_oob_4x4`) and its shared runtime
(`src/common_layer/aeg_runtime_api.cpp`) are **also host-in-the-loop at the panel
level** — they loop `(matrix, m_panel, n_panel, k_panel)` and issue DMA per panel.
The difference is that AEG makes every host step *cheap* and only blocks on the
**output**, whereas aiehlc's generated host loop reconfigures BDs and does a
**blocking wait on every input and output, every iteration**.

Concrete mechanisms found in the AEG source:

1. **Broadcast core enable, batched** — `graph_api::run()`
   (`aeg_runtime_api.cpp:144-184`) enables all 16 cores with a single broadcast
   event (`XAie_EventGenerate` of `BROADCAST_A_8_PL`) wrapped in one
   `XAie_StartTransaction`/`SubmitTransaction`. The fallback path still batches all
   `XAie_CoreEnable` calls in one transaction. aiehlc enables tiles one at a time
   over individual MMIO writes.
2. **Iteration count lives in the core** — `graph_api::run(testIter)`
   (`aeg_runtime_api.cpp:194-214`) writes the loop count into each core's data
   memory (`XAie_DataMemWrWord` → `iterMemAddrs`), batched in one transaction; the
   **core loops internally** instead of the host re-arming it per iteration.
3. **Non-blocking DMA, wait-on-output-only** — the OOB hot loop uses `gm2aie_nb` /
   `aie2gm_nb` (non-blocking) for inputs *and* output, then waits only on the
   output via `gmio_wait(out)` (`example_oob_4x4/src/graph.cpp:281-301`). Inputs
   are fire-and-forget.
4. **BD recycling + HW task queue** — `gmio_api::enqueueBD`
   (`aeg_runtime_api.cpp:814-921`) keeps an `availableBDs`/`enqueuedBDs` pool and
   recycles completed BDs (`XAie_DmaGetPendingBdCount`) instead of reconfiguring
   from scratch; `dma_api::enqueueTask` (`aeg_runtime_api.cpp:1105-1123`) pushes
   BDs into the channel task queue via `XAie_DmaChannelSetStartQueue(..., repeatCount, ...)`
   so the DMA runs ahead of the host.
5. **Pre-staged data + split timing** — all A/B are packed into device-resident
   buffers once in a prep phase timed separately as `prep_ms`; the hot loop "only
   issues DMA" (`graph.cpp:234-247`), and per-panel phase timing (run/in/out/wait)
   is captured (`graph.cpp:312-317`).

### Side-by-side

| Step | aiehlc (today) | AEG |
|------|----------------|-----|
| Core enable | per-tile `CoreEnable`, individual MMIO | broadcast event, batched transaction |
| Register writes | one XAie call each | transaction batching (AUTO_FLUSH) |
| Iteration | host loop re-arms BDs each iter | count written into core DM; core loops itself |
| Input DMA | `startio` + blocking `__Runtime_wait` per input, per iter | `gm2aie_nb` non-blocking, fire-and-forget |
| Wait | waits on every input *and* output each batch | waits on output only |
| BD mgmt | reconfigure from scratch each iter | BD recycling pool + HW task queue |
| Measurement | one wall number incl. load+prep | prep vs gemm split + per-phase breakdown |

**Takeaway:** the AEG advantage is not a different dataflow — it is that every host
operation is *batched* (transactions), *broadcast* (core enable), *asynchronous*
(non-blocking DMA, wait-on-output-only), and *amortized* (BD recycling, pre-staged
data). The highest-leverage, lowest-risk port into aiehlc is moving the
`__Runtime_wait` barrier to output-only / out of the per-iteration loop, followed
by transaction batching of the per-tile register writes.
