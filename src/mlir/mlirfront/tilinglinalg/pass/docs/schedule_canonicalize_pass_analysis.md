<!-- Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->
# ScheduleCanonicalizePass Analysis

## Overview

`ScheduleCanonicalizePass` runs immediately after `BlueprintToSchedulePass` on the `hostModule`.
Its input (`5_BlueprintToSchedulePass.mlir`) contains **per-flow** dfschedule ops scattered across
multiple `scf.execute_region` bodies — one group per partition round.
Its output (`6_ScheduleCanonicalizePass.mlir`) is a **single flat, deduplicated** `dfschedule.host`
block at module level, with one call site (`dfschedule.launchhost`) in `func.func @main`.

**Source:** `pass/passschedulecanonicalize/passschedulecanonicalize.cpp`

### Problem Being Solved

After `BlueprintToSchedulePass`, the same logical shim tile (e.g., col=2, row=0) appears
in each `scf.execute_region` separately: its `declaretile`, `config.dma_bd`, `config.create_io`,
`getbdid`, `start_io`, and `schedule.wait` are duplicated once per partition flow.
Similarly, core tiles that serve multiple flows have redundant tile declarations and potentially
multiple `load_kernel_group` invocations.

`ScheduleCanonicalizePass` collects all these per-flow ops, deduplicates tiles and shim DMA BDs,
merges all core tiles into a **single** `load_kernel_group`, and emits one `schedule.wait`
for all events combined.

---

## Pass Pipeline Position

```
BlueprintToSchedulePass  →  ScheduleCanonicalizePass  →  DfscheduleToApiPass
(per-flow distributed IR)    (merged host block)          (XAie API calls)
```

---

## Pass Steps

### Step 1 — `collectScheduleOps`

Walk the entire `ModuleOp` and partition all ops into `ModuleScheduleInfo` buckets:

| Collected | Purpose |
|---|---|
| `declareTileOps` | All `dfschedule.declaretile` ops across all flows |
| `configDmaBdOps` | All `dfschedule.config.dma_bd` ops |
| `configCreateIoOps` | All `dfschedule.config.create_io` ops |
| `loadKernelGroupOps` | All `dfschedule.config.load_kernel_group` ops |
| `launchKernelGroupOps` | All `dfschedule.schedule.launch_kernel_group` ops |
| `getBdIdOps` | All `dfschedule.schedule.getbdid` ops |
| `startIoOps` | All `dfschedule.schedule.start_io` ops |
| `scheduleWaitOps` | All `dfschedule.schedule.wait` ops |
| `extractSliceOps` | All `tensor.extract_slice` ops (for data buffer reconstruction) |
| `partitionTensorOps` | All `routing.partitiontensor` ops |
| `coreTiles` map | Unique core tile `(col, row)` keys |
| `shimTiles` map | Unique shim tile `(col, row)` keys (row == 0) |
| `initTensorMap` | Source tensor init values, keyed by type string |

Each op is stored with an `OpWithParent` wrapper that records the parent op and whether the op
lives inside a `dfschedule.dskernel_receiver` region (those are never touched).

### Step 2 — `associatePacketsWithTiles`

For each `LoadKernelGroupOp`, look up the `DeclareKernelConfigOp` for each tile via its
`distributed_args` symbol references, extract the `tile_configs` dictionary, and associate
it to the `TileScheduleInfo` for that tile's `(col, row)` key.

This builds the per-tile config dictionaries (`configDicts`) that will be replicated in the
merged `kernelconfig_mergedN` ops.

### Step 3 — Find `func.func @main`

Walk the module to locate the `@main` function. This is the insertion point for
`dfschedule.launchhost @host_canonicalized`.

### Step 4 — `createCanonicalizedSchedule`

This is the main transformation. It emits a fresh `dfschedule.host @host_canonicalized { ... }`
block after `func.func @main`, and a `dfschedule.launchhost` call inside `@main`.

#### 4a — Reconstruct data-flow ops (constants, partitions, slices)

All ops are cloned from the original IR into the new host block in topological order:

1. **Clone `arith.constant`** tensors from `initTensorMap` into the host block.
2. **Reconstruct `routing.partitiontensor`** for each unique partition (keyed by `index_tensorType_splitnum_splitdim_hw_axis_owner`). Each partition gets a unique index to prevent collisions between flows with identical split parameters.
3. **Reconstruct `tensor.extract_slice`** chain in two passes:
   - Pass 1: slices whose source is directly a `partitiontensor`.
   - Pass 2: nested slices (source is another slice), iterated until all parents are resolved.
4. **Bridge tensor→memref** for leaf slices (used as L1 core buffers) and intermediate slices
   referenced by shim BDs: emit `builtin.unrealized_conversion_cast`.

The result is a `sliceMap` (slice index → `Value`) and `declaredMemrefs` (slice index → memref `Value`).

#### 4b — Deduplicate shim DMA BDs

The scatter per-flow shim BDs are deduplicated using the composite key:

```
ShimMergeKey = (TileKey, data_id, sliceIndex)
```

- `data_id` was propagated from `DmaphopTodfscheblueprintPass` through the BD attrs — it identifies
  which root tensor the BD belongs to.
- `sliceIndex` is the `buffer_view` offset (encoded from `BufferViewOp.getOffset()`), distinguishing
  BDs for different partition slices of the same tensor.

Two BDs for the **same** shim tile, same root tensor, and same slice are merged into one.
Two BDs for different partition rounds or different slices are kept separate.

IO configs (channel handles) are similarly deduplicated by `(TileKey, channel, direction)`.

#### 4c — Reconstruct shim tile DMA BDs

For each unique deduplicated shim BD:

```
1. Trace buffer_view back through: alloc_device_mem -> unrealized_cast -> extract_slice
   to find the source tensor slice.
2. In the new host block:
   unrealized_conversion_cast(tensor_slice → flat_memref)   // type bridge
   dfschedule.alloc_device_mem(cast_result)                  // DDR allocation
   dfschedule.buffer_view(ddr_buf, offset, len)              // DDR slice view
   dfschedule.config.dma_bd(buf_view, shim_tile, bd_id) {
     len, enable_packet, packet_id, next_bd,
     acquire/release lock IDs (from original),
     data_id (propagated for downstream grouping)
   }
```

BD IDs are re-assigned sequentially within the new host block (`bdIndexCounter` starting at 0).

#### 4d — Reconstruct shim IO handles

```
dfschedule.config.create_io(bd_handle, shim_tile) {channel, direction, io_operation}
```

One `create_io` per unique (tile, channel, direction) combination.
For the 2×2 example with 2 flows sharing shim tile (2,0), two `create_io` ops are emitted —
one for channel 0 (flow 0) and one for channel 1 (flow 1).

#### 4e — Reconstruct core tile ping-pong BDs

For each core tile, ping and pong BDs are identified by the `hasLinkedBd` flag
(the ping BD has `linked_bd = pong_handle`):

```
1. Detect ping/pong by: allCoreDmaBdParams[i].hasLinkedBd == true → ping
2. Create pong BD first (no linked_bd argument):
   memref.alloc()                                        // L1 placeholder
   dfschedule.bind_core_buffer(alloc, coreTile, offset)  // pong buffer at L1 offset
   dfschedule.config.dma_bd(pong_buf, coreTile, pong_bd_id) {
     next_bd=ping_bd_id, acquire/release lock IDs, enable_packet, packet_id
   }
3. Create ping BD (linked_bd = pong_handle):
   dfschedule.bind_core_buffer(alloc, coreTile, 0)       // ping buffer at offset 0
   dfschedule.config.dma_bd(ping_buf, coreTile, ping_bd_id, pong_handle) {
     next_bd=pong_bd_id, ...
   }
```

The `(tileKey, L1Offset) → bindCoreBuffer result` map avoids creating duplicate bind ops
for the same tile and offset.

#### 4f — Reconstruct core IO + start_io (fire-and-forget)

```
dfschedule.config.create_io(ping_bd_handle, coreTile) {channel, direction="MM2S", io_operation="SEND"}
dfschedule.schedule.getbdid(coreTile)
dfschedule.schedule.start_io(io_handle, bd_id) {flow_index = 0}
```

Core start_io is fire-and-forget (not waited on directly — the kernel launch event covers it).

#### 4g — Merge all core tiles into one `load_kernel_group`

Collect all unique core tile handles (sorted by `TileKey`) and their associated `configDicts`.
Emit fresh `@kernelconfig_mergedN` ops and a single `load_kernel_group`:

```
dfschedule.declare_kernel_config @kernelconfig_merged0 { tile_configs=[{...}] }
...
dfschedule.declare_kernel_config @kernelconfig_mergedN { tile_configs=[{...}] }
dfschedule.config.load_kernel_group(tile0, tile1, ..., tileN) {
  callee = [@dskernel_receiver],
  distributed_compute_kernel_args = [@compute0, ...],
  distributed_args = [@kernelconfig_merged0, ..., @kernelconfig_mergedN]
}
dfschedule.schedule.launch_kernel_group(kernel_group) → kernel_event
```

The config dict for each tile is taken from `tileInfo.configDicts[0]` (populated in step 2).
If no config was associated (backward compat), a default dict is synthesized.

#### 4h — Single merged `schedule.wait`

```
dfschedule.schedule.getbdid(shim_tile)
dfschedule.schedule.start_io(io_handle_0, bd_id) {flow_index=0} → event_0
dfschedule.schedule.start_io(io_handle_1, bd_id) {flow_index=0} → event_1
dfschedule.schedule.wait(kernel_event, event_0, event_1)
```

All events (kernel launch + all shim start_io events) are collected and passed to one `schedule.wait`.

#### 4i — Add launchhost call in `@main`

```mlir
func.func @main() {
  dfschedule.launchhost @host_canonicalized
  return
}
```

### Step 5 — `removeOldScheduleOps`

After creating the canonical host block, erase all the original per-flow ops
(those **not** inside `dfschedule.host` or `dfschedule.dskernel_receiver`):

Erasure order (dependencies respected):

1. `schedule.wait` ops
2. `schedule.start_io` ops
3. `schedule.getbdid` ops
4. `schedule.launch_kernel_group` ops
5. `config.load_kernel_group` ops
6. `packet` ops
7. `config.create_io` ops (excluding those in `dskernel_receiver` or `dfschedule.host`)
8. `config.dma_bd` ops (excluding those in `dskernel_receiver` or `dfschedule.host`)
9. `declaretile` ops

Only ops with no remaining uses are erased (`use_empty()` guard).

### Step 6 — `removeExecuteRegionsFromMain`

Walk `func.func @main` and erase:

1. All `scf.execute_region` ops (and their contents recursively — routing ops, extract_slice ops,
   dfscheblueprint ops, per-flow dfschedule ops).
2. `tensor.empty` ops that are direct children of `@main`.
3. `dfscheblueprint.declare_data` ops that are direct children of `@main`.
4. `arith.constant` ops that are direct children of `@main` (after declare_data is gone,
   these become dead).

After this step, `@main` contains only `dfschedule.launchhost @host_canonicalized` and `return`.

---

## IR Before/After Comparison (2×2 Example)

### Before (Stage 5 — two `scf.execute_region` bodies)

```mlir
func.func @main() {
  %cst = arith.constant dense<...> : tensor<16x16xi8>
  scf.execute_region {
    // Flow 0: partition slice [0..8, 0..16]
    %0 = routing.partitiontensor ...
    %extracted_slice = tensor.extract_slice %0[0,0] [8,16] ...
    // Shim tile (2,0) — flow 0
    %3 = builtin.unrealized_conversion_cast %extracted_slice ...
    %4 = dfschedule.memref_mapping %3 ...
    %5 = dfschedule.alloc_device_mem(%4) ...
    %6 = dfschedule.declaretile {col=2, row=0}
    %7 = dfschedule.buffer_view %5 {offset=0, len=128}
    %8 = dfschedule.config.dma_bd(%7, %6, 0) {channel=0, S2MM, data_id=0}
    %9 = dfschedule.config.create_io(%8, %6) {channel=0, S2MM, RECV}
    // Core tile (0,3)
    %10 = dfschedule.declaretile {col=0, row=3}
    ...ping-pong BDs...
    %19 = dfschedule.schedule.start_io(...) {flow_index=0}
    // Core tile (1,3)
    %20 = dfschedule.declaretile {col=1, row=3}
    ...ping-pong BDs...
    %29 = dfschedule.schedule.start_io(...) {flow_index=0}
    // kernel group for flow 0
    %30 = dfschedule.declare_kernel_config @kernelconfig0 ...
    %31 = dfschedule.declare_kernel_config @kernelconfig1 ...
    %32 = dfschedule.config.load_kernel_group(%10, %20) {callee=[@dskernel_receiver], ...}
    %33 = dfschedule.schedule.launch_kernel_group(%32)
    %34 = dfschedule.schedule.getbdid(%6)
    %35 = dfschedule.schedule.start_io(%9, %34) {flow_index=0}
    dfschedule.schedule.wait(%33, %35)
    dfschedule.free_device_mem %5
  }
  scf.execute_region {
    // Flow 1: partition slice [8..16, 0..16]
    // Shim tile (2,0) — DUPLICATED again
    %6b = dfschedule.declaretile {col=2, row=0}
    %8b = dfschedule.config.dma_bd(...) {channel=1, S2MM, data_id=0}
    %9b = dfschedule.config.create_io(...) {channel=1, S2MM, RECV}
    // Core tiles (0,4) and (1,4)
    ...
    dfschedule.schedule.wait(...)
  }
  return
}
dfschedule.dskernel_receiver @dskernel_receiver {}
```

### After (Stage 6 — single `dfschedule.host` block)

```mlir
func.func @main() {
  dfschedule.launchhost @host_canonicalized   // single call
  return
}
dfschedule.host @host_canonicalized {
  // 1. Data-flow ops (cloned from original IR, no scf.execute_region)
  %cst = arith.constant dense<...> : tensor<16x16xi8>
  %0 = routing.partitiontensor tensor=%cst {splitnum=2, splitdim=0, ...}

  // 2. All partition slices (deduped)
  %extracted_slice   = tensor.extract_slice %0[0,0]  [8,16] [1,1]   // partition 0
  %extracted_slice_0 = tensor.extract_slice %0[8,0]  [8,16] [1,1]   // partition 1
  %extracted_slice_1 = tensor.extract_slice %extracted_slice[0,0]   [4,16] [1,1]  // core (0,3)
  %extracted_slice_2 = tensor.extract_slice %extracted_slice[4,0]   [4,16] [1,1]  // core (1,3)
  %extracted_slice_3 = tensor.extract_slice %extracted_slice_0[0,0] [4,16] [1,1]  // core (0,4)
  %extracted_slice_4 = tensor.extract_slice %extracted_slice_0[4,0] [4,16] [1,1]  // core (1,4)

  // 3. Type bridges
  %1  = unrealized_conversion_cast %extracted_slice   → memref<8x16xi8>
  %2  = unrealized_conversion_cast %extracted_slice_1 → memref<4x16xi8>
  %3  = unrealized_conversion_cast %extracted_slice_2 → memref<4x16xi8>
  %4  = unrealized_conversion_cast %extracted_slice_3 → memref<4x16xi8>
  %5  = unrealized_conversion_cast %extracted_slice_4 → memref<4x16xi8>

  // 4. Tile declarations (deduplicated — 1 shim, 4 core)
  %6  = dfschedule.declaretile {col=2, row=0}   // shim (merged)
  %7  = dfschedule.declaretile {col=0, row=3}
  %8  = dfschedule.declaretile {col=0, row=4}
  %9  = dfschedule.declaretile {col=1, row=3}
  %10 = dfschedule.declaretile {col=1, row=4}

  // 5. Shim DDR allocation + BDs (deduplicated by data_id+sliceIndex)
  %11 = unrealized_conversion_cast %extracted_slice → memref<8x16xi8>
  %12 = dfschedule.alloc_device_mem(%11) → memref<128xi8, 1>
  %13 = dfschedule.buffer_view %12 {offset=0, len=128}
  %14 = dfschedule.config.dma_bd(%13, %6, 0) {S2MM, data_id=0}   // BD 0
  %15 = dfschedule.config.create_io(%14, %6) {channel=0, S2MM, RECV}  // channel 0
  %16 = dfschedule.config.create_io(%14, %6) {channel=1, S2MM, RECV}  // channel 1

  // 6. Core tile ping-pong BDs (4 tiles × 2 BDs = 8 BD ops)
  // Core (0,3) — pong first, then ping linked to pong
  %alloc = memref.alloc() : memref<64xi8>
  %17 = dfschedule.bind_core_buffer(%alloc, %7) {offset=64}   // pong
  %18 = dfschedule.config.dma_bd(%17, %7, 1) {next_bd=0, packet_id=0, ...}
  %alloc_5 = memref.alloc() : memref<64xi8>
  %19 = dfschedule.bind_core_buffer(%alloc_5, %7) {offset=0}  // ping
  %20 = dfschedule.config.dma_bd(%19, %7, 0, %18) {next_bd=1, ...}  // linked=pong
  // ... (same for cores (0,4), (1,3), (1,4))

  // 7. Core IO + start_io (fire-and-forget per tile)
  %33 = dfschedule.config.create_io(%20, %7) {MM2S, SEND}
  %34 = dfschedule.schedule.getbdid(%7)
  %35 = dfschedule.schedule.start_io(%33, %34) {flow_index=0}
  // ... (same for each core tile)

  // 8. Single merged load_kernel_group (all 4 core tiles)
  %45 = dfschedule.declare_kernel_config @kernelconfig_merged0 {...}
  %46 = dfschedule.declare_kernel_config @kernelconfig_merged1 {...}
  %47 = dfschedule.declare_kernel_config @kernelconfig_merged2 {...}
  %48 = dfschedule.declare_kernel_config @kernelconfig_merged3 {...}
  %49 = dfschedule.config.load_kernel_group(%7, %8, %9, %10) {
    callee=[@dskernel_receiver],
    distributed_args=[@kernelconfig_merged0, ..., @kernelconfig_merged3]
  }
  %50 = dfschedule.schedule.launch_kernel_group(%49) → kernel_event

  // 9. Single getbdid + start_io for both shim channels
  %51 = dfschedule.schedule.getbdid(%6)
  %52 = dfschedule.schedule.start_io(%15, %51) {flow_index=0} → shim_event_0
  %53 = dfschedule.schedule.start_io(%16, %51) {flow_index=0} → shim_event_1

  // 10. Single merged wait (kernel + both shim events)
  dfschedule.schedule.wait(%50, %52, %53)
}
dfschedule.dskernel_receiver @dskernel_receiver {}
```

---

## Key Transformations Summary

| Before (per-flow, scattered) | After (merged, flat) |
|---|---|
| `scf.execute_region` per partition | Removed; all ops hoisted into `dfschedule.host` |
| N shim `declaretile` ops (one per flow) | 1 shim `declaretile` (deduplicated by `(col,row)`) |
| N shim `config.dma_bd` ops | 1 per unique `(shimKey, data_id, sliceIndex)` |
| N shim `config.create_io` ops | 1 per unique `(shimKey, channel, direction)` |
| N `load_kernel_group` ops (one per flow) | 1 merged `load_kernel_group` with all core tiles |
| N `declare_kernel_config @kernelconfigK` | Renumbered as `@kernelconfig_mergedN` |
| N `schedule.launch_kernel_group` events | 1 launch event |
| N `schedule.wait` ops | 1 merged wait with all events |
| `func.func @main` with execute_regions | `@main` = `launchhost @host_canonicalized; return` |

---

## Deduplication Logic Details

### Shim BD Deduplication Key

```cpp
using ShimMergeKey = std::tuple<TileKey, int32_t, int64_t>;
//                              ^shimKey  ^data_id  ^sliceIndex(=buffer_view offset)
```

- `data_id`: copied from `BlueprintToSchedulePass` → `config.dma_bd` attr via `shimFlowConfig.getDataId()`.
  Identifies which root tensor the BD belongs to (e.g., all BDs for tensor A share `data_id=0`).
- `sliceIndex`: encoded as `BufferViewOp.getOffset()`. Different partition rounds of the same tensor
  produce different `buffer_view` offsets (e.g., 0 for row-partition 0, 64 for row-partition 1),
  so they produce distinct BDs even if `data_id` matches.

### Shim IO Deduplication Key

```cpp
using IoMergeKey = std::tuple<TileKey, int64_t, std::string>;
//                             ^shimKey  ^channel  ^direction
```

Each physical DMA channel on the shim tile needs exactly one `create_io`, even if the same channel
was configured multiple times (once per flow).

### Ping-Pong Detection

Core BDs are classified by inspecting `hasLinkedBd` (derived from `dmaBd.getLinkedBd() != Value()`):
- BD with `linked_bd` operand → ping BD
- BD without `linked_bd` → pong BD

Pong is always created first (no dependency), then ping references pong's handle as `linked_bd`.

### Core Tile Ordering

Core tiles are processed by iterating `info.coreTiles`, which is a `std::map<TileKey, ...>`.
`TileKey = (col, row)`, so tiles are sorted first by column, then by row.
This determines the order of tiles in `load_kernel_group` and the numbering of `kernelconfig_mergedN`.

---

## Data Structures

### `ModuleScheduleInfo`

Central struct that holds all collected ops and analysis results:

- `coreTiles`: `map<TileKey, TileScheduleInfo>` — per-tile data (values, configDicts, kernel args)
- `shimTiles`: `map<TileKey, ShimDmaInfo>` — per-shim DMA BD and IO ops
- `uniquePartitionParams`: `SmallVector<PartitionParams>` — one entry per `routing.partitiontensor`
- `uniqueSliceParams`: `SmallVector<SliceParams>` — one entry per `tensor.extract_slice` (no dedup)
- `initTensorMap`: `map<typeKey, Value>` — maps tensor type string to source init value
- `allEvents`: all `launch_kernel_group` and `start_io` result values (for wait synthesis)

### `SliceParams`

Captures a `tensor.extract_slice` for reconstruction:
- `offsets`, `sizes`, `strides`: static slice parameters
- `partitionIndex`: which `routing.partitiontensor` is the ultimate source
- `parentSliceIndex`: index of the parent `extract_slice` if nested (−1 if from partition directly)
- `isFromPartition`: true if the immediate source is a `partitiontensor`

### `DmaBdParams` / `CoreDmaBdParams`

Capture all `config.dma_bd` attributes for reconstruction outside the original SSA context.
The `isBufferView` flag on `DmaBdParams` distinguishes the new IR path (buffer came from
`BufferViewOp`, shim path) from the legacy path. `isBindCoreBuffer` on `CoreDmaBdParams`
distinguishes the core L1-buffer path.

---

## Invariants Maintained

1. **`dfschedule.dskernel_receiver` is never touched.** All ops inside it are guarded by
   `isInDSKernelReceiver` checks and skipped in both collection and erasure.
2. **Ops inside `dfschedule.host` are not re-erased.** The `isInHostBlock` guard prevents
   the newly created canonical ops from being deleted in `removeOldScheduleOps`.
3. **`use_empty()` before erase.** Every op in `removeOldScheduleOps` and
   `removeExecuteRegionsFromMain` is checked for zero users before erasure, preventing
   use-after-free from dependent ops.
4. **Slice reconstruction is topologically ordered.** The two-pass algorithm (partition-direct
   first, then nested) ensures parents exist before children when slices are created.
