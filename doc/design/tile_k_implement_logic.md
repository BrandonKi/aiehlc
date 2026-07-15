# tile_k Implementation Logic

## Overview

When `tile_k < K` (e.g., tile_k=64 with K=256), the GEMM's inner-product dimension is split into `kRounds = K / tile_k` temporal iterations. Each AIE core computes partial products using `effectiveK`-wide slices of A and B, accumulating into an int16 buffer across k-rounds, then saturates to int8 and outputs C.

This document describes the full implementation across 5 components: kernel algorithm, tensor shape adjustment, MLIR pipeline propagation, shim DMA 2D addressing, and DMA iteration for k-round repetition.

---

## Scenario: M=N=K=256, 4x4 mesh, tile_k=64

| Parameter | Value | Derivation |
|-----------|-------|------------|
| tileRows | 64 | M/HW_ROWS = 256/4 |
| tileCols | 64 | N/HW_COLS = 256/4 |
| kDim | 256 | full K |
| effectiveK | 64 | tile_k from SpatialPolicy |
| kRounds | 4 | kDim / effectiveK |
| ppDepth | 2 | from SpatialPolicy |
| rowsPerRound | 32 | tileRows / ppDepth |
| bufferSize (per DMA round) | 2048 | rowsPerRound * effectiveK |
| numRounds (per k-round) | 2 | tileRows / rowsPerRound |
| Total DMA rounds (input) | 8 | numRounds * kRounds |
| Core memory: all_A | 4KB | tileRows * effectiveK |
| Core memory: accum | 8KB | tileRows * tileCols * sizeof(int16) |
| Core memory: local_out | 4KB | tileRows * tileCols |
| Total core memory | 16KB | fits in 48KB |

---

## Component 1: Kernel Algorithm Change

**File**: `example/tileprogram/ccode/simplematmul2.cc`

### Before (tile_k == K)

```
cache all_A[tileRows * K]          // 16KB
for each B round:                  // stream B
    compute C += A * B             // single pass, full K inner product
saturate and output C
```

### After (tile_k < K)

```
zero accum[tileRows * tileCols]    // int16 accumulator
for kr = 0..kRounds-1:            // outer k-round loop
    cache all_A[tileRows * eff_k]  // 4KB per k-round
    for each B round:
        accum += A_chunk * B_chunk // partial product, eff_k inner product
saturate accum to int8
output C
```

### Key API Changes

| Old API | New API | Value (example) |
|---------|---------|-----------------|
| `aie::get_k_dim()` | `aie::get_effective_k()` | 64 (not 256) |
| (none) | `aie::get_k_rounds()` | 4 |
| `rows_per_round = buf_sz_a / k_dim` | `rows_per_round = buf_sz_a / eff_k` | 2048/64 = 32 |
| `int8_t all_A[tileRows * k_dim]` | `int8_t all_A[tileRows * eff_k]` | 4KB (not 16KB) |
| (none) | `int16_t accum[tileRows * tileCols]` | 8KB accumulator |

### Data Flow Per K-round

```
K-round 0: kernel receives A[:,0:63] and B[:,0:63]
  → accumulates partial C += A[:,0:63] * B[:,0:63]^T
K-round 1: kernel receives A[:,64:127] and B[:,64:127]
  → accumulates partial C += A[:,64:127] * B[:,64:127]^T
K-round 2: kernel receives A[:,128:191] and B[:,128:191]
  → accumulates partial C += A[:,128:191] * B[:,128:191]^T
K-round 3: kernel receives A[:,192:255] and B[:,192:255]
  → accumulates partial C += A[:,192:255] * B[:,192:255]^T

After all 4 k-rounds: saturate accum to int8 → output C
```

### Implementation

```cpp
const int eff_k = aie::get_effective_k();     // 64
const int k_rounds = aie::get_k_rounds();     // 4
const int num_a_rounds = aie::get_num_rounds(win_a); // 2 (per k-round)
const int buf_sz_a = aie::get_buffer_size(win_a);    // 2048

int8_t all_A[tile_rows * eff_k];        // 4KB
int16_t accum[tile_rows * tile_cols];   // 8KB
int8_t local_out[tile_rows * tile_cols]; // 4KB

memset(accum, 0, sizeof(accum));        // zero accumulators

for (int kr = 0; kr < k_rounds; kr++) {
    // Phase 1: cache A chunk for this k-round
    for (int ra = 0; ra < num_a_rounds; ra++) {
        int8_t *A_ptr = acquire_input_window(win_a);
        memcpy(&all_A[ra * buf_sz_a], A_ptr, buf_sz_a);
        release_input_window(win_a);
    }
    // Phase 2: stream B, accumulate partial products
    for (int rb = 0; rb < num_b_rounds; rb++) {
        int8_t *B_ptr = acquire_input_window(win_b);
        for (i,j,k) accum[i*tile_cols+j] += A[...] * B[...];
        release_input_window(win_b);
    }
}
// Phase 3: saturate and output
for (int i = 0; i < tile_rows * tile_cols; i++) {
    local_out[i] = saturate_i8(accum[i]);
}
```

---

## Component 2: Tensor Shape Adjustment in aiehlc.cc

**File**: `src/llvm/aiehlc.cc` (after tiling computation, before MeshKernelDesc storage)

### Problem

The routing IR receives tensor shapes from `aiehlc.cc`. With full shapes `{M, K}` and `{K, N}`, the pipeline generates partition slices of `{tileRows, K}` — too large for the core memory budget and inconsistent with the kernel's effectiveK-sized buffers.

### Solution

When `kRounds > 1`, adjust input tensor shapes to use effectiveK:

```
A: {M, K} → {M, effectiveK}     i.e. {256, 256} → {256, 64}
B: {K, N} → {effectiveK, N}     i.e. {256, 256} → {64, 256}
C: {M, N} → unchanged           (output is not affected by K-tiling)
```

This causes the routing pipeline to generate partition slices of `{64, 64}` for inputs, matching the kernel's per-k-round data requirement.

### Code Location

```
aiehlc.cc: after derivedTilingParams computation (line ~1308)

if (derivedTilingParams.kRounds > 1) {
    for (auto &pt : parsedTensors) {
        if (Broadcast+Row → A)  pt.shape = {macroDimM, effectiveK}
        if (Broadcast+Col → B)  pt.shape = {effectiveK, macroDimN}
    }
}
```

### Impact on Downstream Pipeline

| Pipeline Stage | With full K | With effectiveK |
|---------------|-------------|-----------------|
| Partition slice A | {64, 256} = 16KB | {64, 64} = 4KB |
| perCoreElements | 16384 | 4096 |
| pingPongBufferSize | min(8192, 4096) = 4096 | 4096/2 = 2048 |
| numIterations (before kRounds mult) | 4 | 2 |
| bufferSize for kernel | wrong | correct (2048) |

---

## Component 3: Module Attribute Propagation

**File**: `src/llvm/aiehlc.cc` (after `buildRoutingIR` returns)

Three module-level attributes are set on the MLIR ModuleOp so downstream passes can read them:

```mlir
module attributes {
    "routing.effective_k" = 64 : i64,
    "routing.k_rounds"    = 4  : i64,
    "routing.full_k"      = 256 : i64
}
```

These attributes are set at both callsites:
1. Multi-kernel path (line ~2155): `mkd.derivedParams.*`
2. Single-kernel path (line ~2467): `derivedTilingParams.*`

### Consumer Passes

| Pass | Attribute Used | Purpose |
|------|---------------|---------|
| `passblueprinttoschedule` | `routing.k_rounds` | Multiply input `numIterations` by kRounds |
| `passblueprinttoschedule` | `routing.effective_k`, `routing.full_k`, `routing.k_rounds` | Set shim BD `iter_step_size` and `iter_wrap` |
| `passblueprinttoschedulekernel` | `routing.k_rounds` | Multiply input `numRounds` for `window_init` |
| `passdmaphoptodfscheblueprint` | `routing.effective_k`, `routing.full_k` | Compute 2D DMA strides/wraps for input shim |

---

## Component 4: Host & Kernel DMA Round Multiplication

### Host Side: passblueprinttoschedule.cpp

**Location**: After `numIterations` computation (line ~1057)

The pipeline computes `numIterations = perCoreElements / pingPongBufferSize = 4096/2048 = 2` for one k-round of input data. But the kernel does `k_rounds * numIterations = 4*2 = 8` total acquire/release cycles. The host DMA must match:

```cpp
if (isInput) {
    if (auto kRoundsAttr = moduleOp->getAttrOfType<IntegerAttr>("routing.k_rounds")) {
        int64_t kRounds = kRoundsAttr.getInt();
        if (kRounds > 1)
            numIterations *= kRounds;  // 2 → 8
    }
}
```

### Kernel Side: passblueprinttoschedulekernel.cpp

**Location**: After `paramInfo.numRounds` computation (line ~841)

The kernel driver's `window_init` must be told the total number of acquire/release cycles across all k-rounds:

```cpp
if (paramInfo.isInput) {
    if (auto kRoundsAttr = moduleOp->getAttrOfType<IntegerAttr>("routing.k_rounds")) {
        int64_t kRounds = kRoundsAttr.getInt();
        if (kRounds > 1)
            paramInfo.numRounds *= kRounds;  // 2 → 8
    }
}
```

### Round Count Consistency Check

```
Kernel user code:
  num_a_rounds = aie::get_num_rounds(win_a) = 2  (per k-round)
  k_rounds = aie::get_k_rounds() = 4
  Total acquire/release = 2 * 4 = 8

Kernel driver (window_init):
  numRounds = 2 * 4 = 8  ✓

Host DMA (BD iterations):
  numIterations = 2 * 4 = 8  ✓

All three agree: 8 total DMA round-trips per input window.
```

---

## Component 5: Shim DMA 2D Addressing & K-round Iteration

### Problem: DDR Layout vs Pipeline Expectation

With effectiveK shapes, the pipeline treats the input tensor as `{256, 64}` (contiguous, row-stride=64). But the actual DDR data is row-major `{256, 256}` (row-stride=256). The partition slice `{64, 64}` is NOT contiguous in DDR:

```
DDR layout for A[256][256]:
Row 0:  [a(0,0) a(0,1) ... a(0,63) | a(0,64) ... a(0,255)]
Row 1:  [a(1,0) a(1,1) ... a(1,63) | a(1,64) ... a(1,255)]
...
Row 63: [a(63,0) ...    a(63,63)   | a(63,64) ... a(63,255)]

K-round 0 needs: a(r, 0:63)   for r=0..63  → scattered, 64 bytes apart by 256
K-round 1 needs: a(r, 64:127) for r=0..63  → same pattern, offset by 64 bytes
```

### Solution: 2D DMA Addressing on Shim MM2S

**File**: `src/mlir/.../passdmaphoptodfscheblueprint.cpp` (PushOpConversion)

When `effectiveK < fullK` and the source is a shim tile, compute 2D addressing:

```
D0: Contiguous within K-chunk row
    stride = 1 word (4 bytes for i8)
    wrap   = effectiveK / elemsPerWord = 64/4 = 16 words

D1: Jump to next DDR row
    stride = fullK / elemsPerWord = 256/4 = 64 words (256 bytes)
    wrap   = partRows = 64  (rows in partition slice)
```

This is set on the FlowConfigOp as `shim_dim_strides` and `shim_dim_wraps`, then propagated to `ConfigDmaBdOp` in `passblueprinttoschedule.cpp`.

### K-round Iteration on Shim BD

**File**: `src/mlir/.../passblueprinttoschedule.cpp` (non-OOO shim BD creation)

Between k-rounds, the DDR read address must advance by `effectiveK` bytes to point to the next K-chunk column:

```
ConfigDmaBdOp attributes:
    iter_step_size = effectiveK * elemBytes = 64 * 1 = 64 bytes
    iter_wrap      = kRounds = 4
```

The AIEML DMA iteration mechanism re-executes the BD `iter_wrap` times, adding `iter_step_size` to the base address each iteration.

### Complete Shim DMA Address Generation

For tile row 0 (rows 0-63 of A), the shim MM2S BD generates addresses:

```
Iteration 0 (k-round 0, cols 0-63):
  BD fires with base=A[0][0]
  D0: col_word = 0..15  (64 bytes contiguous)
  D1: row = 0..63       (stride=64 words = 256 bytes)
  → Reads A[r][0:63] for r=0..63

  (BD repeats for ping-pong: rows 0-31 then rows 32-63)

Iteration 1 (k-round 1, cols 64-127):
  BD fires with base=A[0][0] + iter_step=64 bytes
  D0: col_word = 0..15
  D1: row = 0..63
  → Reads A[r][64:127] for r=0..63

Iteration 2 (k-round 2, cols 128-191):
  base += 64 bytes again
  → Reads A[r][128:191]

Iteration 3 (k-round 3, cols 192-255):
  base += 64 bytes again
  → Reads A[r][192:255]
```

Total data transferred: 64 * 64 * 4 = 16384 bytes = tileRows * K ✓

---

## End-to-End Data Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│ aiehlc.cc                                               │
│                                                         │
│  SpatialPolicy: tile_k=64                               │
│  ├─ effectiveK=64, kRounds=4, kDim=256                  │
│  ├─ Input shapes: A{256,64}, B{64,256}                  │
│  ├─ numRounds=2, bufferSize=2048 (per port)             │
│  ├─ Module attrs: routing.{effective_k, k_rounds, full_k}│
│  └─ Kernel body: get_effective_k()→64, get_k_rounds()→4 │
│                                                         │
│  buildRoutingIR(tensors with effectiveK shapes)         │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ Routing Pipeline (unchanged — sees effectiveK shapes)   │
│                                                         │
│  RoutingUnrollingLowerPass                              │
│  RoutingToDmapPass                                      │
│  DmapToDmaphopPass                                      │
│  DmaphopTodfscheblueprintPass ◄─── NEW: 2D shim strides│
│    └─ PushOpConversion sets shim_dim_strides/wraps      │
│       when effectiveK < fullK                           │
└────────────────────────┬────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
┌──────────────────────┐  ┌──────────────────────────────┐
│ Kernel Path          │  │ Host Path                    │
│                      │  │                              │
│ BlueprintToSchedule  │  │ BlueprintToSchedulePass      │
│ KernelPass           │  │  └─ numIterations *= kRounds │
│  └─ numRounds *= kR  │  │  └─ shim BD: iter_step/wrap │
│                      │  │                              │
│ DfscheduleToKernel   │  │ ScheduleCanonicalizePass     │
│ ApiPass              │  │ DfscheduleToApiPass          │
│  └─ window_init(     │  │  └─ __Runtime_dma_bd_config  │
│       ..., 8)        │  │     _multidim_ooo(           │
│                      │  │       D0, D1,                │
│ EmitC → kernel.cc    │  │       iter_step=64,          │
│                      │  │       iter_wrap=4)           │
│                      │  │                              │
│                      │  │ EmitC → host.cc              │
└──────────────────────┘  └──────────────────────────────┘
```

---

## Synchronization Protocol

The kernel and host are synchronized via AIE locks (ping-pong protocol). With kRounds:

```
Host shim MM2S                    Kernel
─────────────                    ──────
                                 accum[:] = 0

[k-round 0]                     [k-round 0]
  BD iter 0:                       acquire_input(win_a) → A chunk [0:31, 0:63]
    send A rows 0-31, cols 0-63    memcpy to all_A[0..2047]
    ← lock released ──────────     release_input(win_a)
  BD iter 0 (pong):                acquire_input(win_a) → A chunk [32:63, 0:63]
    send A rows 32-63, cols 0-63   memcpy to all_A[2048..4095]
    ← lock released ──────────     release_input(win_a)

  (same for B)                     acquire_input(win_b) → B chunk
                                   accum += all_A * B_chunk
                                   release_input(win_b)
                                   (repeat for B pong)

[k-round 1]                     [k-round 1]
  BD iter 1 (base+64):            acquire_input(win_a) → A chunk [0:31, 64:127]
    send A rows 0-31, cols 64-127  ...
    ...                            accum += all_A * B_chunk

[k-round 2,3: same pattern]     [k-round 2,3: same pattern]

                                 saturate(accum) → local_out
                                 output local_out via win_c
```

The lock-based flow control ensures:
- The host can't overwrite a ping buffer until the kernel releases it
- The kernel can't read a pong buffer until the host fills and releases it
- No explicit k-round synchronization needed — the acquire/release pattern naturally gates the data flow

---

## Files Changed

| File | Change | Lines |
|------|--------|-------|
| `example/tileprogram/ccode/simplematmul2.cc` | Kernel: k_rounds loop, effectiveK buffers, int16 accumulator; tile_k=64 in SpatialPolicy | matmul + mul2 functions |
| `src/llvm/aiehlc.cc` | Input tensor shape adjustment {M,K}→{M,effK}; module attrs `routing.{effective_k,k_rounds,full_k}` at both callsites | ~1308, ~2155, ~2467 |
| `src/mlir/.../passdmaphoptodfscheblueprint.cpp` | 2D DMA strides/wraps for input shim MM2S when effectiveK < fullK | PushOpConversion |
| `src/mlir/.../passblueprinttoschedule.cpp` | Input numIterations × kRounds; shim BD iter_step_size/iter_wrap for k-round repetition | FlowTransferConversion |
| `src/mlir/.../passblueprinttoschedulekernel.cpp` | Input numRounds × kRounds for window_init total round count | Phase 1 param collection |

---

## Backward Compatibility

When `tile_k == K` (or tile_k=0 which defaults to K):
- `kRounds = 1`
- No shape adjustment (effectiveK == K)
- No module attributes set (guard: `kRounds > 1`)
- No 2D addressing (effectiveK == fullK → guard fails)
- No iteration on shim BD (shimIterStepSize=0, shimIterWrap=0)
- Kernel: k_rounds loop runs once → equivalent to old behavior
- All paths fall through to existing code — **zero behavioral change**

---

## Limitations & Future Work

1. **B matrix stride**: B is stored as B^T[N][K] row-major. The same 2D addressing logic applies (D0=effectiveK contiguous, D1=fullK stride). The current implementation handles both A and B identically since both are input Broadcast flows through PushOpConversion.

2. **Non-power-of-2 tile_k**: The implementation requires `K % tile_k == 0`. Non-divisible tile_k would need padding or partial k-round handling.

3. **Accumulator overflow**: With int8 inputs and int16 accumulator, the maximum accumulation across kRounds before overflow is limited. For kRounds=4 with effectiveK=64: worst case per element = 4 * 64 * 127 * 127 = 4,128,064, which overflows int16 (max 32767). For correctness with large kRounds, int32 accumulators or intermediate saturation would be needed. The current scenario (small values from `(i%7)-3` and `(i%5)-2`) does not hit this.

4. **Spatial M/N rounds interaction**: When `tile_m < tileRows` or `tile_n < tileCols` (spatialMRounds > 1 or spatialNRounds > 1), the kernel's outer k_rounds loop nests with the spatial rounds. This interaction is not yet tested.

---

## Conv2d-Spatial K-Accumulate: single multi-dim shim BD (no host `scf.for`)

For the conv2d_spatial halo case the input tensor is split across **both** L2 row bands
(`l2_rounds`) and K-contraction bands (`k_rounds`). The naive lowering emitted a host
`scf.for %c0 to %c16` (= `l2_rounds`×`k_rounds`) issuing one shim BD per trip. This is now
collapsed into **ONE** `dfschedule.config.dma_bd` per IT shim tile that walks all 16 transfers in
hardware via 3 address dims + the iteration dim + channel repeat.

### Generated descriptor (conv2d_spatial, i8, k_slice=244, l2_slice=19, k_rounds=4, l2_rounds=4)

```
dma_bd: len = 18544 (= k_slice 244 × l2_slice 19 × k_rounds 4 × 1 B)
  dim_strides = [4, 920, 224]   dim_wraps = [61, 19, 4]
  iter_step_size = 12880        iter_wrap = 4
start_io(..., repeat_count = 4)
```

- **D0 {stride=4, wrap=61}** — K-band width in words (`k_slice 244 / elemsPerWord 4 = 61`).
- **D1 {stride=920, wrap=19}** — slab rows (`row_pitch 230 words × 4 B = 920`; `l2_slice = 19`).
- **D2 {stride=224, wrap=4}** — K rounds (`k_step 56 words × 4 B = 224`; `k_rounds = 4`).
- **iter_step_size=12880, iter_wrap=4** — L2 rounds (`l2_step 14 rows × row_pitch_words 230 × 4 B`).
  `iter_step` uses `l2_step` (NOT `l2_slice`) because conv halo bands **overlap**.
- **repeat_count=4** pairs 1:1 with `iter_wrap=4` (matmul convention) so HW `IterCurr` advances
  through the 4 L2 rounds. The 16 transfers = D2(4) × iteration(4), NOT iter×repeat.

Stream order = iteration(L2,outer) → D2(K) → D1(row) → D0(col), so slabs emerge as
`r → (lr=r/4, kr=r%4)` exactly matching the kernel's `for r in 0..16: acquire_input_window` order.

### Consumer needs NO change

The core S2MM chops the byte stream into `buffer_size=4636` (=19×244) windows,
`num_iterations=16`. `4 iter × 4 D2 bands × 4636 = 16 × 4636` = same total bytes, same order →
16 lock releases → 16 kernel windows. Verified: kernel.cc `window_init(...,1159,16)` (1159 v4int8 ×
4 B = 4636) is unchanged.

### Code path

- `passdmaphoptodfscheblueprint.cpp` — `isHaloTensor && !inputShimDimStrides`, gated on
  `tensor_N.halo` `k_rounds>1`: builds the 3 dims (`strides=[wordBytes, row_pitch_w×wordBytes,
  k_step_w×wordBytes]`, `wraps=[k_slice_w, l2_slice, k_rounds]`) into `shim_dim_strides/wraps`.
- `passblueprinttoschedule.cpp` — `kAccumHaloSlab` detector routes through the **straight-line
  (non-loop) branch** (same as matmul K-tiling): (a) shim len = `l2_slice×k_slice×k_rounds×elemSize`
  while core len stays `l2_slice×k_slice×elemSize`; (b) `shimIterStepSize=l2_step×row_pitch_w×
  wordBytes`, `shimIterWrap=l2_rounds`; (c) `needsOuterLoop=false` (skip host loop);
  (d) channel `repeatCount=l2_rounds`.
- **Requires** `tensor_N.halo` to carry `row_pitch` (PADDED DDR row width in elements). The unit-test
  harness sets `halo.rowPitch = raw_wc` for the K-accum example; without it `row_pitch=0` collapses
  D1 stride to 0 and disables the iteration (silent miscompile).

All non-K-accum paths (`k_rounds<=1`, the `w_rounds` 2D halo loop, pure matmul K-tiling) are gated
out and byte-unchanged.
