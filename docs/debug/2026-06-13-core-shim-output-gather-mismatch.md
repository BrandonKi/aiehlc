# Analysis: core→shim output-gather data-volume mismatch (debuglog / applog)

## TL;DR

The conv2d output `[12544, 64] i8 = 802,816 B` is gathered by **4 output flows
(5, 7, 9, 11)**, one per HW core-row. The **shim S2MM consumers are configured
correctly** (each receives the full 200,704 B band). The **core MM2S producers
are under-configured**: each core streams only ~8,192 B (2 ping-pong BDs ×
4096 B × repeat 1) instead of its required **50,176 B** share. So the producers
collectively send ~32 KB while each shim expects 200 KB → the flow stalls.

A second, proximate hang cause is visible in `[1]`: the queried core (4,3) has
`AXI_MM_SLAVE_ERROR` + `GROUP_ERRORS_0/1` SET and is `LOCK ACQ STALL` — the
kernel errored and never released the output buffer, so MM2S cannot advance.

## The numbers (from debuglog + applog + provenance JSON)

Conv2d via im2col (simpleconv2d.h):
`Output[M,N] = [12544, 64] i8`, `M=OH*OW=112*112=12544`, `N=64`, `K=147`.
Full output = **802,816 B**. HW grid = 4 rows × 4 cols = 16 cores.

Output is gathered by 4 "pull" flows (5,7,9,11). From
`dmaphopprovenacemap.json` (pull_0 / flow 5):
- `tensor_shape=[3136,64] i8`, `total_bytes=200704`
- `partition_info: splitdim=0, splitnum=4, hw_axis_owner=row, replicate_on=col`

So **splitdim=0 (M) / splitnum=4** → each flow owns an M-band of
`12544/4 = 3136` rows → `[3136,64] = 200,704 B`. 4 × 200,704 = 802,816 ✓.

Within a flow, 4 cores (the 4 cores of one HW row) produce. From the blueprint
code the per-core tile is a **column slice**: `tileW = outW/numTileCols = 64/4 = 16`.
→ each core owns `[3136,16] = 50,176 B`. 4 × 50,176 = 200,704 ✓.

### Consumer (shim S2MM) — CORRECT
debuglog `[5]` tile(3,0) S2MM ch0:
`BD len=3584B, strides=[4,64] (bytes), wraps=[4,224], iter_step=14336, iter_wrap=14, repeat=56`.
Decode (i8, word=4B): D0 wrap=4 words=16 elems = `tileW`; D1 wrap=224 rows @ stride
outW=64; one BD = 16×224 = 3584 elems = **3584 B**; iter_wrap 14 → 14×3584 = 50,176 B
(one core's column slice); repeat 56 = 14×4 cores → 56×3584 = **200,704 B** = full band ✓.

### Producer (core MM2S) — WRONG
debuglog `[2]` and `[5]` for the 4 row-3 cores (0,3)/(1,3 = queried 4,3)/(2,3)/(3,3):
`BD4/BD5 len=4096B, pkt_id 1/2/3/4, ping-pong repeat=1` → **Total intended 8,192 B per core**.
Aggregate 4 × 8,192 = **32,768 B**, vs the 200,704 B the shim expects → 6× short
(and 25× short per-core: 8,192 vs 50,176).

The 4096 B chunk is a **generic matmul 64×64 i8 output tile**, NOT the
conv-aware `[3136,16]` slice. `dfscheduleprovenancemap.json` `module_attrs` are
**all zero** (`tile_m=tile_n=m_rounds=n_rounds=k_rounds=0`), i.e. the conv2d→GEMM
path did not propagate the tiling/iteration attributes that should drive the
producer's repeat count.

## What each core SHOULD send

For output flow F (mapped to one HW core-row), band = `[3136,64]` of the
full `[12544,64]` output:

| core (flow 5, row 3) | pkt_id | owns (col slice) | bytes | as BDs |
|---|---|---|---|---|
| (0,3) | 1 | output[band, ch 0:16]  | 50,176 | 14 × 3584 B |
| (1,3)=(4,3) | 2 | output[band, ch 16:32] | 50,176 | 14 × 3584 B |
| (2,3) | 3 | output[band, ch 32:48] | 50,176 | 14 × 3584 B |
| (3,3) | 4 | output[band, ch 48:64] | 50,176 | 14 × 3584 B |

i.e. each core must stream its **`[3136, 16]` column slice = 50,176 B**, as **14
transfers of 3,584 B** (matching the shim's `iter_wrap=14`, BD=3584 B). The 4
cores' slices interleave (shim D2 stride=tileW) into the `[3136,64]` band; 4
bands stack into the full `[12544,64]` output.

Currently each core sends only 8,192 B (two 4096 B ping-pong BDs, repeat 1).

## Root cause (code location)

`src/mlir/mlirfront/tilinglinalg/pass/passdmaphoptodfscheblueprint/passdmaphoptodfscheblueprint.cpp`
`PullOpConversion`:
- **dst (shim S2MM)** FlowConfigOp (lines ~1360-1372) gets conv-aware
  `shimDimStrides`/`shimDimWraps` computed at lines 1303-1357 → 200,704 B correct.
- **src (core MM2S)** FlowConfigOp (lines ~1277-1289) passes
  `shim_dim_strides=nullptr, shim_dim_wraps=nullptr` and relies on the per-tile
  `viewSplit` + downstream blueprint→schedule for size/repeat. With
  `module_attrs` all-zero the producer falls back to the generic 64×64 (4096 B,
  repeat 1) tile instead of the `[3136,16]` / 14-round conv shape.

So producer and consumer are derived from **two different tiling models**: the
shim side is conv-aware (56/224-row, 14 iters), the core side is generic-GEMM.
The fix belongs in the producer-side sizing/repeat derivation (and in the
conv2d→GEMM lowering that left `module_attrs` zero).

## Suggested next step (proposed, for confirmation)

This task is **analysis-only**; the diagnosis above is the deliverable. If a fix
is wanted, the work would be:
1. Make the conv2d→GEMM lowering populate `module_attrs`
   (`tile_m`, `tile_n`, `m_rounds=14`, etc.) so producers know the iteration count.
2. In `PullOpConversion`, derive the **src (core MM2S)** repeat/round count from
   the same per-core `[stripH, tileW]` shape used for the shim, so each core
   sends 50,176 B (14 × 3584 B) instead of 8,192 B.
3. Separately investigate the core `AXI_MM_SLAVE_ERROR` (kernel-side) that is the
   proximate `LOCK ACQ STALL` in this run.

## Files inspected (read-only)
- debuglog, applog
- aout/worklocal/dmaphopprovenacemap.json, dfscheduleprovenancemap.json
- example/tileprogram/ccode/simpleconv2d.h
- src/.../pass/passdmaphoptodfscheblueprint/passdmaphoptodfscheblueprint.cpp
- src/.../pass/passdmaphopprovenancemap/passdmaphopprovenancemap.cpp
