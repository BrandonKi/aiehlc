# AEG vs. aiehlc — why AEG keeps cores fed and we lock-stall

> Companion to `baseline.md` / experiments `05`. Source of the AEG numbers:
> `/scratch/staff/bkirinci/final/FINAL` (read-only study, 2026-06-29). The AEG ML
> overlays use ADF/MLLib (compiler-generated BD math + cascade/pktmerge) that does not
> port 1:1 to our manual-routing compiler, but the *feeding mechanisms* do.

## The contrast in one line

| | aiehlc (ours, baseline) | AEG ML overlays (conv2d / res18) |
|---|---|---|
| Per-core input transfer | **1 KB** window | **6–22 KB** sub-volume |
| Acquires per core | **~2048** (1024 A + 1024 B) | tens (per SV), DMA-iterated |
| Staging between DDR and cores | **none** (DDR→shim→core, per 1 KB) | **MemTile 512 KB**, bulk-filled once (81 KB / 49 KB in ONE shim BD) |
| BD structure | one BD re-fired by lock per window | **multi-dim wrap + `next_bd` chain**, one enqueue covers many tiles |
| Channel `repeatCount` | **1** | **5, 8, 32** (DMA runs many windows ahead) |
| `iteration_wrap` | small | up to **8** |
| Lock credits | ping-pong init **2** (lockstep) | ping-pong **2** + multi-consumer **acq_value 4** |
| Core state | **99.97 % lock-stall** | compute-bound |

## Why our 1 KB lockstep is so slow

The L3 counter shows ~1.1e9 stall cycles / ~2048 acquires ≈ **~540K cycles stalled per
1 KB acquire**. At >4 B/cycle of AIE stream bandwidth a 1 KB window should land in
~hundreds of cycles, so this is **latency / handshake-bound, not bandwidth-bound**: the
input is pulled DDR→shim→core one tiny lock-gated window at a time, and 2-deep ping-pong
is not enough to hide that latency because delivery latency per window ≫ the kernel's
compute per window.

## AEG's three feeding mechanisms (with numbers)

1. **MemTile staging + bulk preload.** Whole IFM loaded DDR→MemTile in ONE shim BD
   (`length=82944` conv2d, `50176` res18, `repeatCount=1`) into a **512 KB** MemTile
   buffer, gated by a single `acquireLock(...,1,1)`. Cores then stream from fast local
   MemTile, not DDR. (`aie_runtime_control*.cpp`)
2. **Few large multi-dim / chained BDs + high `repeatCount`.** MemTile→core uses
   6400–22272 B BDs with `stepsize/wrap` tensors and `next_bd` chaining; one
   `enqueueTask` carries `repeatCount` 5/8/32 and `iteration_wrap` up to 8 — one channel
   enqueue feeds many kernel iterations with **no per-window host/lock re-arm**.
3. **Lock credits sized to consumers + BD-iteration ping-pong.** Producer lock init = 2
   (num_buffers); multi-row fanout uses `lock_acq_value=4` so the DMA acquires 4 credits
   at once and runs ahead; ping-pong done via `iteration_wrap=2`,
   `iteration_stepsize=|pong−ping|/4` inside one BD chain.

## Not portable (ADF/MLLib-specific)

`shared_buffer` read/write tiling, `pktmerge<N>` OFM merge, `connect<cascade>` partial-sum
chains, ADF graph runtime, `repetition_count` multirate. We must reproduce the *effect*
(MemTile staging, large iterated BDs, credit depth) in our dfschedule / dmaphop codegen.

## Candidate experiments (exp06+), by blast radius

- **A — Lock-credit / buffer depth (smallest):** raise core-input ping-pong depth and lock
  init so the input DMA prefetches >2 windows ahead. *Caveat:* only helps if the stall is
  latency-hideable; if delivery throughput itself is the cap, won't move.
- **B — Fewer/larger core-input BDs + `repeatCount`/`iter_wrap` (medium):** emit one
  iterated BD per input channel covering many windows instead of per-window lockstep.
  Touches `BlueprintToSchedulePass` (startio repeat, BD chain) and
  `DmaphopTodfscheblueprintPass` (dim wrap/iter).
- **C — MemTile staging hop (largest, highest reward):** introduce a MemTile between shim
  and cores, bulk-fill once, stream locally. Touches routing→dmap→dmaphop→blueprint.
- **0 — HW DMA trace first (cheap, de-risks B/C):** one board run capturing the DMA-status
  / aiediag snapshot to confirm WHICH channel stalls and its BD iter state (latency vs
  throughput) before committing to a big codegen change.
