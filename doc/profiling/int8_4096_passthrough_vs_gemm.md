# INT8 4096³ — Passthrough Floor vs GEMM vs AEG (mini report)

Single 4096×4096 · 4096×4096 int8 GEMM (int32 accumulate, int8-saturated out),
4×4 mesh (16 compute tiles), `NUM_MATRICES=1`, board `crimini5`/vek385.
Harness: `example/tileprogram/ccode/simplematmul2_prof.cc`. Verify = strided-prime
spot-check (stride 4093, 4101 of 16,777,216 outputs).

Two passthrough variants isolate the core-touch cost (toggle `PASSTHROUGH_VECTORIZED`):
scalar byte loops vs. v32int8 `load_v`/`add`/`store_v`.

## Results

| Metric               | Passthrough **vectorized** | Passthrough scalar | Our GEMM   | AEG baseline |
|----------------------|----------------------------|--------------------|------------|--------------|
| Wall time            | **73.5 ms**                | 589.7 ms           | **1044.5 ms** | **3825.3 ms** |
| Throughput           | 1870.7 GOPS*               | 233.0 GOPS*        | **131.6 GOPS** | 35.9 GOPS |
| Kernel-load fraction | 48.2 %                     | 5.6 %              | 3.3 %      | amortized (1 launch) |
| Core lock stall      | 98.6 %                     | 67.7 %             | 29.0 %     | 35.5 %       |
| Core active          | 1.4 %                      | 32.3 %             | 71.0 %     | 64.5 %       |
| Stream stall         | 0 %                        | 0 %                | 0 %        | 0 %          |

*Passthrough "GOPS" is nominal (no MACs occur) — `2·M·N·K / wall`, a floor
reference only.

## Key finding — the feed floor is ~73 ms, not ~590 ms

The **scalar** passthrough (590 ms) was **core-bound**, not feed-bound: per output
tile it walks 2 KB of window data with scalar byte loops, and that scalar work — not
the DMA — set the wall time. Its core was busy ~32 % of the time.

The **vectorized** passthrough (v32int8) drops the wall to **73.5 ms**, of which
**48 % is one-time kernel load (~35 ms)** → the true feed/DMA/lock floor is
**~38 ms**. Its core is active only **1.4 %** and stalls **98.6 %** on locks — i.e.
the cores now finish instantly and the wall is set purely by data movement. **This is
the real floor.**

### Consequence: GEMM is compute-bound, not feed-bound

With the true floor at ~73 ms (≈38 ms excl. load), the GEMM's 1044 ms is
**~93 % compute**. The core-state counters agree: GEMM cores are **71 % active**
(computing) and only **29 %** lock-stalled. The MACs — not the feed path — dominate.

> This corrects the previous version of this report, which used the scalar
> passthrough (590 ms) and wrongly concluded the workload was ~56 % feed-bound. The
> feed path is actually fast; the earlier "floor" was a scalar-core artifact.

## We vs AEG

Still **~3.7× faster than AEG** at the same problem (131.6 vs 35.9 GOPS). Both spend
most core cycles computing (71 % vs 64.5 % active); lock stall is comparable
(29 % vs 35.5 %).

## Takeaway for optimization

The feed path is not the bottleneck — it can sustain the whole 4096³ in ~38 ms. To
speed up the GEMM, target the **MAC compute**: better vector-MAC utilization / fewer
core cycles per output tile (the GEMM currently issues 9.02 MACs per vector
instruction), and shave the ~29 % core lock stall via deeper double-buffering. Making
the feed path faster buys almost nothing.

## Reproduce

```bash
# In example/tileprogram/ccode/simplematmul2_prof.cc  (M=N=K=4096, NUM_MATRICES=1):
#   GEMM:                PASSTHROUGH_KERNEL 0
#   Passthrough vector:  PASSTHROUGH_KERNEL 1, PASSTHROUGH_VECTORIZED 1
#   Passthrough scalar:  PASSTHROUGH_KERNEL 1, PASSTHROUGH_VECTORIZED 0
bash script/aiehlc.sh --aie-version 5 \
  --runtime-source-file ./example/tileprogram/ccode/simplematmul2_prof.cc
python3 script/test/appvek385.py aout/main.elf
```

## GEMM-branch codegen bug at 4096 (fixed)

The GEMM branch deadlocked at 4096³ (shim `wait_io` timeouts) while passing at 256³
and while the feature-port codegen passed at 4096. Root cause was **two** shim
input-BD codegen differences in
`pass/passblueprinttoschedule/helper/flowtransfer_host.cpp`:

1. **Missing shim locks on the A-broadcast BDs.** The `len=262144` A-input shim BDs
   were emitted with `acquire/release_lock_id=-1` (no lock). Without lock pacing the
   16× iterated 256 KB transfers overrun/starve the stream at 4096. Fix: emit
   `lock_id=0` (feature-proven) at the two non-OOO shim-BD emit sites. The OOO B-input
   sites correctly stay `-1`.
2. **`perIterRepeat=1` starvation.** An "optimization" set the host startio repeat to
   `1` when `shimIterWrap>1` (assuming the BD's hardware iteration covers all firings).
   At 4096 this under-fires the B-input stream → deadlock. Fix: always repeat
   `nRounds` (`tC/tN`), matching the feature codegen.

Validation: after the fix, the regenerated `host.cc` DMA/lock/startio schedule is
byte-identical to the proven-working feature-port output (only benign debug-macro and
the additive GEMM pre-launch lock-init block differ). Board-confirmed: **4096³ GEMM
now PASSES at 1306 ms / 105.2 GOPS** on the GEMM branch, lock stall 28.1 %.

Both differences were latent at 256³ (low DMA pressure) and only manifested at 4096³.

## Passthrough gotchas (fixed)

1. **B handshake liveness.** The passthrough must read every B byte into a value the
   output depends on (`vbsum`/`bsum` folded into the output tile). A bare
   `(void)B_ptr` lets the compiler dead-code-eliminate the B window acquire/release,
   which breaks the B lock handshake and hangs the shim `wait_io`.
2. **Vector window access alignment.** v32int8 `load_v`/`store_v` on window pointers
   is fine at 32-byte-aligned offsets (every buffer size here is a multiple of 32);
   `store_v` to the output window mirrors `example/perf/aieml_perfstream.cc`.
