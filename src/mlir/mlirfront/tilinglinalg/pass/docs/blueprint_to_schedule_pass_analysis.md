<!-- Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->
# BlueprintToSchedule Pass Analysis

## Overview

After the `DmaphopTodfscheblueprintPass` completes, the IR contains dfscheblueprint ops that describe
the abstract schedule plan: tile groups, data slices, flow configurations, and flow transfers.
Two passes then consume this IR and produce dfschedule ops targeting two separate output files:

- **`BlueprintToSchedulePass`** → `host.cc` — emits host-side DMA setup, kernel load/launch, and DDR management
- **`BlueprintToScheduleKernelPass`** → `kernel.cc` — emits kernel-side driver module (buffers, locks, windows, main loop)

The module is **cloned** before these passes run so each pass sees the identical dfscheblueprint IR.

### Pass Orchestration (`pass/unitest/test.cpp`)

```
DmaphopTodfscheblueprintPass
        |
        +--- clone --> hostModule
        |                  BlueprintToSchedulePass
        |                  ScheduleCanonicalizePass
        |                  DfscheduleToApiPass
        |                  RoutingConstantFoldPass
        |                  CanonicalizerPass
        |                  translateToCpp --> host.cc
        |
        +--- clone --> kernelModule
                           BlueprintToScheduleKernelPass
                           DfscheduleToKernelApiPass
                           translateToCpp --> kernel.cc
```

---

## dfscheblueprint Input Ops (Stage 4 IR)

These are the ops present after `DmaphopTodfscheblueprintPass`. Source: `pass/unitest/ir/dfschedule/4_DmaphopTodfscheblueprintPass.mlir`.

### `dfscheblueprint.declare_data`

Wraps a constant tensor as a named data root. Acts as a logical origin for the data that will be DMA'd.

```mlir
%0 = dfscheblueprint.declare_data %cst : tensor<16x16xi8> -> tensor<16x16xi8>
```

### `dfscheblueprint.tile_group`

Names a group of (col, row) tile coordinates — either source (core) tiles or destination (shim) tiles.

```mlir
dfscheblueprint.tile_group @group_src_0 { tiles = [[0, 3], [1, 3]] }
dfscheblueprint.tile_group @group_dst_0 { tiles = [[2, 0]] }
```

### `dfscheblueprint.data_slice`

Associates a per-tile tensor slice with a symbol name so it can be referenced by `flowconfig.slice_symbols`.

```mlir
%4 = dfscheblueprint.data_slice @producer_slice_0_0 wrap %extracted_slice_0 : tensor<4x16xi8>
%5 = dfscheblueprint.data_slice @producer_slice_0_1 wrap %extracted_slice_1 : tensor<4x16xi8>
```

### `dfscheblueprint.flowconfig` (core type)

Describes the DMA configuration for the core (compute) tiles: which tile group, which tensor view, DMA direction (MM2S = core sends to shim), and the per-tile data slice symbols.

```mlir
dfscheblueprint.flowconfig @flow_src_0 {
  target = @group_src_0,
  view = %extracted_slice : tensor<8x16xi8>,
  distribution = "linear",
  dma = #dfscheblueprint.DMA<channels = [0], direction = MM2S>,
  slice_symbols = [@producer_slice_0_0, @producer_slice_0_1],
  type = "core"
}
```

### `dfscheblueprint.flowconfig` (shim type)

Describes the DMA configuration for the shim tile: which tile group, which tensor view, DMA direction (S2MM = shim receives from core), and a `data_id` for tensor grouping.

```mlir
dfscheblueprint.flowconfig @flow_dst_0 {
  target = @group_dst_0,
  view = %extracted_slice : tensor<8x16xi8>,
  distribution = "root",
  dma = #dfscheblueprint.DMA<channels = [0], direction = S2MM>,
  type = "shim",
  data_id = 0
}
```

### `dfscheblueprint.flow_transfer`

Links a core flowconfig to a shim flowconfig, specifying the transfer type, packet ID base, and flow index. This is the primary op that triggers both pass conversions.

```mlir
dfscheblueprint.flow_transfer @transfer_0 {
  type = "many_to_one",
  from = @flow_src_0,
  to = @flow_dst_0,
  ordering = "sequential",
  base_packet_id = 0,
  flow_index = 0
}
```

---

## dfschedule Output Ops (Stage 5 IR)

These ops are produced by the two passes. Source: `pass/unitest/ir/dfschedule/5_BlueprintToSchedulePass.mlir`.

| Op | Type | Description |
|---|---|---|
| `dfschedule.declaretile {col, row}` | `!dfschedule.tile` | Declares a tile handle by coordinate |
| `dfschedule.memref_mapping` | `memref<...>` | Logical SSA anchor; bridges tensor-sourced memref to DDR/L1 allocations |
| `dfschedule.alloc_device_mem` | `memref<N xi8, 1>` | Allocates DDR buffer (memspace=1) sized to tensor |
| `dfschedule.buffer_view` | `memref<N xi8, 1>` | Slice view into DDR buffer with offset and length |
| `dfschedule.bind_core_buffer` | `memref<N xi8>` | Binds a logical token to an L1 core buffer at a given byte offset (ping or pong) |
| `dfschedule.config.dma_bd` | `!dfschedule.bd_handle` | Configures a DMA Buffer Descriptor: buffer, tile, BD ID, packet, locks, next_bd |
| `dfschedule.config.create_io` | `!dfschedule.io_handle` | Creates an IO channel: BD handle, tile, channel, direction (MM2S/S2MM), operation (SEND/RECV) |
| `dfschedule.schedule.getbdid` | `i32` | Retrieves the runtime BD ID for a tile |
| `dfschedule.schedule.start_io` | `!dfschedule.event` | Starts a DMA IO channel, returns an event |
| `dfschedule.declare_kernel_config` | `!dfschedule.kernel_config` | Declares per-tile kernel config dict (tile_index, flow_index, packet_id, buffer_size, lock IDs, etc.) |
| `dfschedule.config.load_kernel_group` | `!dfschedule.kernelgroup` | Loads a kernel onto a set of tiles, with callee symbol and distributed_args |
| `dfschedule.schedule.launch_kernel_group` | `!dfschedule.event` | Launches the kernel group, returns an event |
| `dfschedule.schedule.wait` | void | Waits for multiple events (kernel + shim IO) |
| `dfschedule.free_device_mem` | void | Frees the DDR allocation |
| `dfschedule.dskernel_receiver` | symbol | Empty stub at module level (filled by kernel pass) |
| `dfschedule.module` | symbol | Kernel driver module (locks, buffers, windows, kernel_decl, main) |

---

## Host Pass: `BlueprintToSchedulePass`

**Source:** `pass/passblueprinttoschedule/passblueprinttoschedule.cpp`

**Conversion target:** all dfscheblueprint ops made illegal; dfschedule + arith + memref + scf + tensor remain legal.

### Pattern: `FlowTransferConversion`

This is the primary pattern. It fires on each `dfscheblueprint.flow_transfer` op and generates the complete
host-side DMA/kernel schedule.

#### Step 1 — Identify shim vs. core

Look up both `from` and `to` `FlowConfigOp`s. Check the `type` field (`"shim"` or `"core"`) to assign roles.
For the 2×2 example, `flow_dst_0` is the shim (S2MM receiver) and `flow_src_0` is the core (MM2S sender).

#### Step 2 — Shim DDR allocation and BD configuration

For the shim tile (e.g., col=2, row=0):

```
unrealized_conversion_cast(view_tensor → shaped_memref)     // type bridge
dfschedule.memref_mapping(shaped_memref)                    // logical SSA anchor
dfschedule.alloc_device_mem(token)                          // DDR buffer (memspace=1)
dfschedule.declaretile {col=2, row=0}                       // shim tile handle
dfschedule.buffer_view(ddr_buf, offset=0, len=128)          // full DDR slice
dfschedule.config.dma_bd(buf_view, shim_tile, bd_id=0) {   // shim BD
  len=128, enable_packet=true, packet_id=0,
  next_bd=4294967295, acquire_lock_id=0, release_lock_id=0,
  data_id=0                                                  // for ScheduleCanonicalizePass grouping
}
dfschedule.config.create_io(bd_handle, shim_tile) {
  channel=0, direction="S2MM", io_operation="RECV"
}
```

Key attributes:
- `data_id` is read from `shimFlowConfig.getDataId()` and propagated to the BD so `ScheduleCanonicalizePass` can group all shim BDs for the same root tensor.
- `next_bd=4294967295` (0xFFFFFFFF) means no next BD (shim only has one BD, no chaining).

#### Step 3 — Core tile ping-pong BD configuration

For each core tile in the `TileGroupOp` (e.g., tile (0,3) and tile (1,3)):

```
dfschedule.declaretile {col=0, row=3}
unrealized_conversion_cast(per_tile_slice_tensor → memref<64xi8>)
dfschedule.memref_mapping(memref)                          // logical channel token
dfschedule.bind_core_buffer(token, tile, offset=0)         // ping L1 buffer
dfschedule.bind_core_buffer(token, tile, offset=64)        // pong L1 buffer

// Pong BD first (no linked_bd, next_bd → ping):
dfschedule.config.dma_bd(pong_buf, tile, bd_id=1) {
  len=64, enable_packet=true, packet_id=tileIndex,
  next_bd=0,                                               // → ping
  acquire_lock_id=0, acquire_lock_val=-1,                  // wait for lock
  release_lock_id=1, release_lock_val=1,                   // signal lock
  data_id=-1
}
// Ping BD second (linked_bd=pong, next_bd → pong):
dfschedule.config.dma_bd(ping_buf, tile, bd_id=0, pong_bd_handle) {
  len=64, enable_packet=true, packet_id=tileIndex,
  next_bd=1,                                               // → pong
  acquire_lock_id=0, acquire_lock_val=-1,
  release_lock_id=1, release_lock_val=1,
  data_id=-1
}
dfschedule.config.create_io(ping_bd, tile) {
  channel=0, direction="MM2S", io_operation="SEND"
}
dfschedule.schedule.getbdid(tile)
dfschedule.schedule.start_io(io_handle, bd_id) { flow_index=0 }
```

BD ID and lock ID allocation is delegated to `ResourceMgr` (`hw/ResourceManager.h`):
- `resourceMgr->allocateTileLockPair(row, col, flowIndex, acquireLockId, releaseLockId)` — per-tile, per-flow lock pair
- `resourceMgr->allocateTileBd(row, col, flowIndex)` — per-tile BD IDs (ping=0, pong=1 for first allocation)

Per-tile data slices come from `flowconfig.slice_symbols` which index into the `DataSliceOp` table.

#### Step 4 — Kernel config and group launch

After all core tiles are processed:

```
dfschedule.declare_kernel_config @kernelconfig0 {
  tile_configs = [{
    tile_index=0, flow_index=0, packet_id=0, dma_channel=0,
    buffer_mode=1, num_buffers=2, buffer_size=64, buffer_offset=0,
    element_size=1, acquire_lock_id=0, release_lock_id=1
  }]
}
dfschedule.declare_kernel_config @kernelconfig1 {
  tile_configs = [{
    tile_index=1, flow_index=0, packet_id=1, dma_channel=0,
    buffer_mode=1, num_buffers=2, buffer_size=64, buffer_offset=64,
    element_size=1, acquire_lock_id=0, release_lock_id=1
  }]
}
dfschedule.config.load_kernel_group(tile0, tile1) {
  callee=[@dskernel_receiver],
  distributed_compute_kernel_args=[@compute0, @compute0],
  distributed_args=[@kernelconfig0, @kernelconfig1]
}
dfschedule.schedule.launch_kernel_group(kernel_group)     → kernel_event
dfschedule.schedule.getbdid(shim_tile)
dfschedule.schedule.start_io(shim_io, bd_id) { flow_index=0 } → shim_event
dfschedule.schedule.wait(kernel_event, shim_event)
dfschedule.free_device_mem(ddr_buffer)
```

`buffer_offset` encodes the logical byte offset within the partitioned tensor for each tile (0 for tile 0, 64 for tile 1 with a 64-byte partition).

#### Step 5 — `dskernel_receiver` stub

A module-level `dfschedule.dskernel_receiver @dskernel_receiver {}` is emitted once (guarded by `hasDSKernelReceiver`).
This stub is referenced by `load_kernel_group.callee`. The kernel pass fills in the body detail.

### Additional Patterns

| Pattern | Action |
|---|---|
| `EraseOpPattern<FlowConfigOp>` | Erase (attributes already consumed by `FlowTransferConversion`) |
| `EraseOpPattern<TileGroupOp>` | Erase (consumed by `FlowTransferConversion`) |
| `DeclareDataOpConversion` | Replace `declare_data %cst → %cst` (passthrough) |
| `DataSliceOpConversion` | Replace `data_slice wrap %t → %t` (passthrough) |

---

## Kernel Pass: `BlueprintToScheduleKernelPass`

**Source:** `pass/passblueprinttoschedulekernel/passblueprinttoschedulekernel.cpp`

**Conversion target:** same as host pass — dfscheblueprint ops illegal, dfschedule + standard dialects legal.

### Pattern: `FlowTransferConversion` (kernel variant)

Fires on each `flow_transfer`. Focuses on generating the kernel driver module; does not allocate DDR.

#### Step 1 — `generateDSKernelReceiver` → `generateKernelModule`

Emits `dfschedule.module @kernel_driver_dskernel_receiver { ... }` at module level:

```
dfschedule.module @kernel_driver_dskernel_receiver {
  // 1. Kernel metadata
  dfschedule.kernel_config_def @config {
    kernel_name = "compute_kernel",
    kernel_file = "compute_kernel.cc",
    buffer_size = 256,
    element_type = i32,
    vector_width = 4
  }

  // 2. Lock definitions (per input/output window)
  dfschedule.lock_def @LOCK_window_in_0_ACQ { id=48, init_value=2 }
  dfschedule.lock_def @LOCK_window_in_0_REL { id=49 }

  // 3. Buffer definitions (ping/pong pair per window)
  dfschedule.buffer_def @buf_in_ping_0 : memref<N x vector<4xi8>, "LOCAL">
  dfschedule.buffer_def @buf_in_pong_0 : memref<N x vector<4xi8>, "LOCAL">

  // 4. Window definition
  dfschedule.window_def @window_in_0 {
    direction="in", ping_buffer=@buf_in_ping_0, pong_buffer=@buf_in_pong_0,
    acquire_lock=@LOCK_window_in_0_ACQ, release_lock=@LOCK_window_in_0_REL,
    buffer_size=N, async=true
  }

  // 5. Kernel declaration
  dfschedule.kernel_decl @compute_kernel {
    inputs=[@window_in_0], outputs=[], iteration_style="internal"
  }

  // 6. Main entry point
  dfschedule.main @main {
    alloc_sync_buffer(8)
    sync_buffer_write(buf, 0, 0)
    log(1)
    win = window_init @window_in_0 : InputWindow<i8>
    kernel_invoke @compute_kernel(win)
    done
    kernel_return
  }
}
```

**Lock ID allocation via `KernelResourceManager`:**
- Input acquire lock: starts at 48, increments per input window
- Input release lock: starts at 49, increments per input window
- Output acquire lock: starts at 51, increments per output window
- Output release lock: starts at 50, increments per output window

**Direction analysis via `analyzeKernelParams`:**
Walk all `declare_data` ops → trace value chain through SSA users to find the `flowconfig` that consumes it → find the `flow_transfer` referencing that `flowconfig` → check `from/to` type:
- `shim → core`: input window (MM2S, data flows into kernel)
- `core → shim`: output window (S2MM, data flows out of kernel)
- `core → core`: skipped (inter-tile, not a kernel parameter)

#### Step 2 — Core tile DMA IO configuration

For each core tile, the kernel pass also emits lightweight DMA IO config ops (without ping-pong or DDR):

```
dfschedule.declaretile {col, row}
memref.alloc() : memref<perTileLen x elemType>
dfschedule.config.dma_bd(buf, tile, bd_id) { ... }
dfschedule.config.create_io(bd_handle, tile) { channel, direction, io_operation }
dfschedule.schedule.getbdid(tile)
dfschedule.schedule.start_io(io_handle, bd_id) { flow_index }
```

BD IDs for the kernel pass come from `KernelResourceManager.allocateBdId()` (simple incrementing counter starting at 0).

#### Step 3 — Post-pass cleanup

After conversion, the kernel pass erases all top-level ops that are not part of the kernel driver output:

```cpp
// Keep only:
DSKernelReceiverOp, KernelModuleOp,
DeclareTileOp, ConfigDmaBdOp, ConfigCreateIoOp, GetBdIdOp, StartIoOp,
arith::ConstantOp, memref::AllocOp
// Erase everything else (func.func @main, routing ops, etc.)
```

---

## Op Conversion Table

| dfscheblueprint op | dfschedule ops produced | Pass |
|---|---|---|
| `declare_data %cst` | replaced by `%cst` (passthrough) | Host |
| `tile_group @g` | erased | Both |
| `data_slice @s wrap %t` | replaced by `%t` (passthrough) | Both |
| `flowconfig @f {type="shim"}` | `declaretile` + `memref_mapping` + `alloc_device_mem` + `buffer_view` + `config.dma_bd` + `config.create_io` | Host |
| `flowconfig @f {type="core"}` | `declaretile` + `memref_mapping` + `bind_core_buffer(ping)` + `bind_core_buffer(pong)` + `config.dma_bd(pong)` + `config.dma_bd(ping, linked=pong)` + `config.create_io` + `schedule.getbdid` + `schedule.start_io` | Host |
| `flowconfig @f {type="shim"}` | erased (kernel pass ignores shim) | Kernel |
| `flowconfig @f {type="core"}` | `declaretile` + `memref.alloc` + `config.dma_bd` + `config.create_io` + `schedule.getbdid` + `schedule.start_io` | Kernel |
| `flow_transfer @t` | `declare_kernel_config(×N)` + `config.load_kernel_group` + `schedule.launch_kernel_group` + `schedule.getbdid(shim)` + `schedule.start_io(shim)` + `schedule.wait` + `free_device_mem` + `dskernel_receiver {}` (stub) | Host |
| `flow_transfer @t` | `dfschedule.module @kernel_driver_dskernel_receiver { ... }` | Kernel |

---

## IR Walkthrough: 2×2 GEMM Example

The unitest IR uses a 16×16 matrix partitioned into 2 row-partitions of 8×16, each assigned to 2 core tiles.

### Stage 4 IR Structure

Two `scf.execute_region` bodies (one per partition), each containing:
- `tile_group @group_src_N` → 2 core tiles at rows 3 or 4
- `tile_group @group_dst_N` → 1 shim tile at (2, 0)
- `data_slice @producer_slice_N_M` → 4×16 per-tile slices
- `flowconfig @flow_src_N` → core MM2S, channel 0, slices [@producer_slice_N_0, @producer_slice_N_1]
- `flowconfig @flow_dst_N` → shim S2MM, channel 0 or 1, data_id=0
- `flow_transfer @transfer_N` → base_packet_id=0, flow_index=N

### Stage 5 IR Structure (Host)

For each `flow_transfer`, in-place in the same `scf.execute_region`:

**Flow 0 (partition 0, tiles at row 3):**
1. Shim tile (2,0): `alloc_device_mem` → 128-byte DDR buf → `buffer_view(offset=0, len=128)` → `config.dma_bd(channel=0, S2MM, packet_id=0)` → `create_io(RECV)`
2. Core tile (0,3): `bind_core_buffer(ping, offset=0)` + `bind_core_buffer(pong, offset=64)` → pong BD (bd_id=1, next_bd=0, packet_id=0) + ping BD (bd_id=0, next_bd=1, packet_id=0, linked=pong) → `create_io(MM2S, SEND)` → `getbdid` → `start_io(flow_index=0)`
3. Core tile (1,3): same pattern, packet_id=1, bd_ids=0/1 (fresh per-tile allocation)
4. `declare_kernel_config @kernelconfig0` (tile_index=0, buffer_offset=0, packet_id=0)
5. `declare_kernel_config @kernelconfig1` (tile_index=1, buffer_offset=64, packet_id=1)
6. `load_kernel_group(tile(0,3), tile(1,3))` {callee=[@dskernel_receiver], distributed_args=[@kernelconfig0, @kernelconfig1]}
7. `launch_kernel_group` → kernel_event
8. `getbdid(shim_tile)` → `start_io(shim_io, flow_index=0)` → shim_event
9. `schedule.wait(kernel_event, shim_event)`
10. `free_device_mem(ddr_buf)`

**Flow 1 (partition 1, tiles at row 4):** identical structure, using `channel=1` for the shim, `flow_index=1`.

Module-level: `dfschedule.dskernel_receiver @dskernel_receiver {}` (emitted once after first flow_transfer).

---

## ResourceMgr Roles

### Host Pass: `ResourceMgr` (`hw/ResourceManager.h`)

The host pass creates a single shared `ResourceMgr` instance (Gen2 hardware model) and uses it for:

- **Lock allocation:** `allocateTileLockPair(row, col, flowIndex, &acquireId, &releaseId)` — tracks lock usage per tile to avoid conflicts between flows. Fallback formula: `lockBase = flowIndex * 8 + tileIndex * 2`.
- **BD ID allocation:** `allocateTileBd(row, col, flowIndex)` — tracks BD IDs per tile, returning sequential IDs (0 for ping, 1 for pong within a tile).

Both allocators are per-tile and per-flow, ensuring different flows or different tiles do not collide on lock or BD IDs.

### Kernel Pass: `KernelResourceManager` (local class)

A simpler local class in `passblueprinttoschedulekernel.cpp`:

```
nextBdId=0        — increments per allocateBdId() call
nextInputAcqLock=48  — increments per allocateInputAcquireLock()
nextInputRelLock=49  — increments per allocateInputReleaseLock()
nextOutputAcqLock=51 — increments per allocateOutputAcquireLock()
nextOutputRelLock=50 — increments per allocateOutputReleaseLock()
```

These IDs correspond to the AIEML lock numbering used inside core tiles for the ping-pong synchronization protocol between the DMA engine and the compute kernel.
