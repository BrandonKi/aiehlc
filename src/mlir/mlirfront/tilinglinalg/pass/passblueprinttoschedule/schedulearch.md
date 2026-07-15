<!-- Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->
# BlueprintToSchedulePass — Architecture Document

## Overview

`BlueprintToSchedulePass` (`lower-blueprint-to-schedule`) is a dialect-conversion MLIR pass that translates the **dfscheblueprint** dialect (abstract schedule manifests) into the **dfschedule** dialect (executable DMA/kernel schedule operations). It is step 5 of the host-path pipeline:

```
DmaphopTodfscheblueprintPass
        ↓
BlueprintToSchedulePass        ← this pass
        ↓
ScheduleCanonicalizePass
        ↓
DfscheduleToApiPass
```

Source file: `pass/passblueprinttoschedule/passblueprinttoschedule.cpp`
Header: `pass/passblueprinttoschedule/passblueprinttoschedule.h`

---

## Three-Phase Execution

`runOnOperation()` runs three sequential phases.

### Phase 1 — `preprocessConstantToMemref`

**Purpose:** Lower a `arith.constant` dense tensor (the root data tensor) into a writable `memref` *inside `@main`* without creating a module-level `memref.global`.

**Steps:**
1. Walk `@main` for the single `dfscheblueprint.declare_data` op.
2. Find the `arith.constant` that feeds it.
3. Create `bufferization.to_memref` → read-only view over the constant.
4. `memref.alloc` + `memref.copy` → writable buffer.
5. Insert `memref.dealloc` before the `func.return`.
6. Store `rootMemref`, `rootMemrefType`, `rootShape`, `elementType` in shared `BlueprintPassState`.

Result: every downstream pattern receives a live `Value` (`passState->rootMemref`) that points into the pre-allocated host-side DDR buffer.

---

### Phase 2 — Dialect Conversion

**Legal dialects (targets):**

| Dialect | Notes |
|---------|-------|
| `dfschedule` | Output dialect; all generated ops land here |
| `routing` | Retained; restructured in Phase 3 / later passes |
| `func`, `memref`, `arith`, `scf`, `tensor`, `bufferization`, `builtin` | Standard dialects |

**Illegal ops (must be converted):**

| Op | Action |
|----|--------|
| `dfscheblueprint.flow_config` | Erased by `EraseOpPattern` |
| `dfscheblueprint.tile_group` | Erased by `EraseOpPattern` |
| `dfscheblueprint.declare_data` | Replaced with its `init_tensor` operand (`DeclareDataOpConversion`) |
| `dfscheblueprint.flow_transfer` | Fully lowered by `FlowTransferConversion` |

**Registered patterns:**

| Pattern | Description |
|---------|-------------|
| `FlowTransferConversion` | Core lowering; see detailed breakdown below |
| `DataSliceOpConversion` | Replace `data_slice` with its `tensor_slice` operand |
| `DeclareDataOpConversion` | Replace `declare_data` with its `init_tensor` operand |
| `EraseOpPattern<FlowConfigOp>` | Erase after attributes have been consumed |
| `EraseOpPattern<TileGroupOp>` | Erase after tile coordinates have been consumed |

Conversion is driven by `applyPartialConversion`.

---

### Phase 3 — Dead-Op Elimination

Iteratively erases use-empty ops until a fixed point:
- `tensor.extract_slice` chains (partition → producer)
- `routing.partitiontensor`
- `arith.constant` with `RankedTensorType` result

This cleans up tensor IR that was only needed to carry offset/size metadata into `FlowTransferConversion`.

> **Note:** `scf.execute_region` restructuring is deferred to `ScheduleCanonicalizePass`. Generated `dfschedule` ops remain inside `routing.RoutingCreate` bodies (which carry the `SymbolTable` trait required by `DeclareKernelConfigOp`).

---

## FlowTransferConversion — Detailed Walk-Through

This is the primary conversion pattern. It matches each `dfscheblueprint.flow_transfer` op and emits a complete `dfschedule` sub-graph.

### Inputs Consumed

| Attribute / Operand | Source |
|---------------------|--------|
| `from` symbol ref | resolves to `FlowConfigOp` (shim or core) |
| `to` symbol ref | resolves to `FlowConfigOp` (shim or core) |
| `base_packet_id` | per-flow unique packet identifier base |
| `flow_index` | per-flow unique index (used for lock/BD allocation) |
| `shim.view` | tensor/memref slice representing the DDR partition |
| `shim.dma` / `core.dma` | `DmaAttr` carrying channel, direction |
| `core.slice_symbols[]` | per-tile `DataSliceOp` symbol refs |
| `passState->rootMemref` | pre-allocated DDR buffer from Phase 1 |
| `resourceMgr` | `ResourceMgr` for tile BD and lock allocation |

### Step 1 — Shim/Core Identification

The pass identifies which `FlowConfigOp` has `type="shim"`:

- **Shim is sender** (`type="shim"` on `from`): direction = MM2S / SEND
- **Shim is receiver** (`type="shim"` on `to`): direction = S2MM / RECV

### Step 2 — DDR Partition Subview

If `passState->rootMemref` exists and the shim view traces back to a `tensor.extract_slice`:

```
rootMemref
  └─ memref.subview [partOffsets, partSizes, partStrides]   ← partition subview
```

This subview is used as the DDR buffer for all subsequent ops.

### Step 3 — Shim Tile DMA Configuration

Emits exactly once per `flow_transfer`:

```
dfschedule.declaretile (shimCol, shimRow)
dfschedule.buffer_view  (ddrBuffer, offset=0, len=bufferLen)
dfschedule.config.dma_bd
    buffer        = shimBuffer
    tile          = shimTile
    bd_id         = 0
    enable_packet = true
    packet_id     = basePacketId
    next_bd       = 0xFFFFFFFF  (terminal)
    data_id       = shimFlowConfig.data_id   ← for ScheduleCanonicalizePass BD merging
dfschedule.config.create_io
    bd_config  = dma_bd result
    tile       = shimTile
    channel    = shimChannel
    direction  = "MM2S" | "S2MM"
    io_op      = "SEND" | "RECV"
```

### Step 4 — Per-Core-Tile Loop

For each tile in `coreTileGroup.tiles[]`:

#### 4a. Tile Declaration
```
dfschedule.declaretile (col, row)
```

#### 4b. Resource Allocation
- **Lock pair**: `ResourceMgr::allocateTileLockPair(row, col, flowIndex)` → `acquireLockId`, `releaseLockId`
  Fallback: `flowIndex*8 + tileIndex*2` if allocation fails.
- **BD IDs**: two calls to `ResourceMgr::allocateTileBd(row, col, flowIndex)` → `pingBdId`, `pongBdId`
  Fallback: `0` / `1`.

#### 4c. Per-Tile Memref Subview (from DataSliceOp)
```
partitionSubview
  └─ memref.subview [tileOffsets, tileSizes, tileStrides]   ← tile subview
       └─ dfschedule.memref_mapping                          ← strip strides, clean shape
            └─ dfschedule.bind_core_buffer (pingL1Offset=0)  ← ping buffer
            └─ dfschedule.bind_core_buffer (pongL1Offset=perTileTotalSize*elementSize) ← pong buffer
```

#### 4d. Ping-Pong DMA BD Configuration
Two `dfschedule.config.dma_bd` ops are chained in a ring:

```
pongBD:
    buffer    = pongL1
    bd_id     = pongBdId
    next_bd   = pingBdId       ← ring: pong → ping
    packet_id = basePacketId + tileIndex
    acquire   = (acquireLockId, -1)
    release   = (releaseLockId, +1)
    linked_bd = none

pingBD:
    buffer    = pingL1
    bd_id     = pingBdId
    next_bd   = pongBdId       ← ring: ping → pong
    packet_id = basePacketId + tileIndex
    acquire   = (acquireLockId, -1)
    release   = (releaseLockId, +1)
    linked_bd = pongBD         ← chaining: ping owns pong
```

#### 4e. Core IO Handle and Start
```
dfschedule.config.create_io (pingBD, coreTile, coreChannel, coreDmaDirection)
dfschedule.getbdid (coreTile)
dfschedule.schedule.start_io (coreIoHandle, bdId, flowIndex)
```

#### 4f. Per-Tile Config Dictionary (for KernelConfig)
Each tile produces an attribute dictionary with:
```
tile_index, flow_index, packet_id, dma_channel,
buffer_mode=1 (ping-pong), num_buffers=2,
buffer_size, buffer_offset, element_size,
acquire_lock_id, release_lock_id
```

### Step 5 — Kernel Group and Launch

```
# One DeclareKernelConfigOp per tile (named @kernelconfig0, @kernelconfig1, ...)
dfschedule.declare_kernel_config @kernelconfigN [singleTileConfigDict]

# Load kernel group (all core tiles share callee=@dskernel_receiver)
dfschedule.config.load_kernel_group
    tiles              = [coreTile0, coreTile1, ...]
    callees            = [@dskernel_receiver]
    compute_kernels    = [@compute0, @compute0, ...]
    distributed_args   = [@kernelconfig0, @kernelconfig1, ...]

# Launch
dfschedule.schedule.launch_kernel_group (kernelGroup)
```

### Step 6 — Shim IO Start and Synchronisation

```
dfschedule.getbdid (shimTile)
dfschedule.schedule.start_io (shimIoHandle, shimBdId, flowIndex)
dfschedule.schedule.wait [launchEvent, shimStartIoEvent]
dfschedule.free_device_mem (ddrBuffer)
```

### Step 7 — DSKernelReceiver Declaration (once per module)

```
dfschedule.dskernel_receiver @dskernel_receiver { /* empty body */ }
```

Generated only if a receiver with that name does not already exist at module level. Details (kernel module, buffers, locks) are filled in by a later pass.

---

## Helper Utilities

| Function | Purpose |
|----------|---------|
| `lookupSymbolOp<OpTy>` | Generic template: finds op by `sym_name` in same block then parent regions |
| `lookupTileGroup` | Wrapper → finds `dfscheblueprint.tile_group` |
| `lookupFlowConfig` | Wrapper → finds `dfscheblueprint.flow_config` |
| `lookupDataSlice` | Wrapper → finds `dfscheblueprint.data_slice` |
| `hasDSKernelReceiver` | Module-level scan: prevents duplicate `dskernel_receiver` |
| `hasKernelModule` | Module-level scan: prevents duplicate `kernel_module` |
| `getModuleOp` | Walks to root module op |
| `generateDSKernelReceiver` | Creates empty `dfschedule.dskernel_receiver` at module level |
| `toOpFoldResult` | Converts `ArrayRef<int64_t>` → `SmallVector<OpFoldResult>` for SubViewOp |
| `preprocessConstantToMemref` | Phase 1 implementation |
| `postprocessRestructure` | (declared, not called in main path; restructuring deferred to ScheduleCanonicalizePass) |

---

## Shared State (`BlueprintPassState`)

```cpp
struct BlueprintPassState {
    Value           rootMemref;      // alloc'd writable DDR buffer
    MemRefType      rootMemrefType;  // its type
    SmallVector<int64_t> rootShape;  // tensor shape
    Type            elementType;     // element type
};
```

Passed by `shared_ptr` between Phase 1 (writer) and `FlowTransferConversion` (reader).

---

## Resource Management

`ResourceMgr` (wraps `makeResource("Gen2")` hardware model) tracks:

- **Lock pairs per tile**: `allocateTileLockPair(row, col, ownerId)` returns `(acquireLockId, releaseLockId)`.
- **BD IDs per tile**: `allocateTileBd(row, col, ownerId)` returns one BD ID at a time; called twice for ping/pong.

Both have integer fallback formulas when the hardware model cannot satisfy an allocation, printing a `WARNING` to `llvm::errs()`.

---

## Generated IR Shape (per FlowTransfer)

```
// Inside routing.RoutingCreate body (later hoisted by ScheduleCanonicalizePass):

memref.subview %rootMemref[partOffsets][partSizes][partStrides]        // partition subview
  → dfschedule.declaretile (shimCol, shimRow)                          // shim tile
  → dfschedule.buffer_view ...                                          // DDR view
  → dfschedule.config.dma_bd  [shim]                                   // shim BD
  → dfschedule.config.create_io [shim]                                 // shim IO

  for each core tile:
    → dfschedule.declaretile (col, row)
    → memref.subview [tile offsets]
    → dfschedule.memref_mapping
    → dfschedule.bind_core_buffer (ping)
    → dfschedule.bind_core_buffer (pong)
    → arith.constant (pongBdId)
    → dfschedule.config.dma_bd  [pong, linked_bd=none]
    → arith.constant (pingBdId)
    → dfschedule.config.dma_bd  [ping, linked_bd=pongBD]
    → dfschedule.config.create_io [core]
    → dfschedule.getbdid
    → dfschedule.schedule.start_io [core]
    → dfschedule.declare_kernel_config @kernelconfigN

  → dfschedule.config.load_kernel_group
  → dfschedule.schedule.launch_kernel_group
  → dfschedule.getbdid [shim]
  → dfschedule.schedule.start_io [shim]
  → dfschedule.schedule.wait
  → dfschedule.free_device_mem

// At module level (emitted once):
dfschedule.dskernel_receiver @dskernel_receiver { }
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Phase 1 uses `bufferization.to_memref` + alloc/copy | Avoids `memref.global`; keeps everything inside `@main`; no module-level state needed |
| Shim BD `next_bd = 0xFFFFFFFF` | Hardware convention for terminal BD (no chaining on shim side) |
| Pong BD created *before* ping, then ping chains to it | Ensures `linked_bd` SSA value (pong BD handle) is available when ping BD is created |
| `data_id` propagated to shim `ConfigDmaBdOp` | Lets `ScheduleCanonicalizePass` identify and merge BDs for the same root tensor |
| `dskernel_receiver` body left empty | Separation of concerns; a later kernel pass fills in DMA BD config for the AIE core side |
| Restructuring deferred to `ScheduleCanonicalizePass` | `routing.RoutingCreate` provides the `SymbolTable` scope needed by `DeclareKernelConfigOp` at conversion time |

---

## Pass Registration

```cpp
class BlueprintToSchedulePass
    : public PassWrapper<BlueprintToSchedulePass, OperationPass<>> {
    StringRef getArgument()    const { return "lower-blueprint-to-schedule"; }
    StringRef getDescription() const { return "Lower dfscheblueprint dialect to dfschedule dialect"; }
};
```

Dependent dialects registered: `dfscheblueprint`, `dfschedule`, `func`, `memref`, `arith`, `scf`, `tensor`.
