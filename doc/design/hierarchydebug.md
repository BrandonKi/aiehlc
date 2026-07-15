# Hierarchical Debug System: Provenance Maps & aiediag

## Overview

The hierarchical debug system bridges the gap between compile-time IR transformations and runtime HW behavior. It consists of three components:

1. **DmaphopProvenanceMapPass** — extracts logical routing topology from dmaphop IR
2. **DfscheduleProvenanceMapPass** — extracts DMA/lock/BD configuration from dfschedule IR
3. **aiediag** — runtime diagnostic tool that reads live HW registers and cross-references both JSON maps

```
Compile-time                          Runtime
===========                          =======

dmaphop IR ──► DmaphopProvenanceMapPass ──► dmaphopprovenacemap.json ──┐
                                                                       ├──► aiediag ──► diagnosis
dfschedule IR ► DfscheduleProvenanceMapPass ► dfscheduleprovenancemap.json ┘     │
                                                                           aiedbg ──┘
                                                                        (live HW regs)
```

## Layer 1: DmaphopProvenanceMapPass

**Source:** `pass/passdmaphopprovenancemap/passdmaphopprovenancemap.cpp`
**Output:** `dmaphopprovenacemap.json`
**Runs on:** dmaphop IR (after `DmapToDmaphopPass`)

### What It Captures

The dmaphop IR represents physical routing topology: which tiles send to which tiles, through what hops, with what data shapes. This pass walks all `dmaphop.push` and `dmaphop.pull` operations and extracts:

| Field | Source | Description |
|-------|--------|-------------|
| `direction` | push/pull op | "push" (DDR→core) or "pull" (core→DDR) |
| `data.tensor_shape` | push/pull data operand type | e.g., `[16, 64]` for a 16x64 tensor |
| `data.element_type` | MLIR element type | e.g., "i8", "f32" |
| `data.total_bytes` | computed | shape * element_bytes |
| `data.partition_info` | `routing.partitiontensor` op | splitdim, splitnum, hw_axis_owner, replicate_on |
| `stages[].producer` | create_path producers | tile (col,row), port symbol, channel |
| `stages[].channel` | create_hop chain | hop-by-hop routing path |
| `stages[].consumer` | create_path consumers | tile (col,row), port symbol, dma_port, pkt_id |
| `invariants` | computed | data volume checks, hop chain length |

### Symbol Resolution Chain

Dmaphop uses symbolic references. The pass resolves them:

```
create_path.producers = [@sym]
  → findProducerByName(@sym) → producer.tp = @port_sym
    → findPortByName(@port_sym) → port.tile → tile(col, row)
```

Similarly for consumers:
```
create_path.consumers = [@sym]
  → findConsumerByName(@sym) → consumer.from = @port_sym
    → findPortByName(@port_sym) → port.tile → tile(col, row)
```

### Partition Info Tracing

For push ops, the data operand traces back through:
```
push.data → routingextract_data.operand(0) → partitiontensor
```
Extracting `splitdim`, `splitnum`, `hw_axis_owner`, and `replicate_on` attributes that describe how the tensor was partitioned across tiles.

### Invariants

- **Data volume**: For pull, verifies `sum(producer_buffer_bytes) == total_bytes`
- **DMA port consistency**: Checks if all consumer/producer dma_ports match
- **Packet IDs**: Lists packet IDs for pull paths (used for multiplexed streams)
- **Hop chain length**: Number of hops in the routing path

### JSON Structure

```json
{
  "version": 1,
  "module_attrs": {
    "tile_m": 8, "tile_n": 8,
    "tile_rows": 16, "tile_cols": 16,
    "effective_k": 64, "full_k": 64,
    "k_rounds": 1, "m_rounds": 2, "n_rounds": 2
  },
  "communication_paths": [
    {
      "id": "push_0",
      "direction": "push",
      "data": {
        "tensor_shape": [16, 64],
        "element_type": "i8",
        "element_bytes": 1,
        "total_bytes": 1024,
        "partition_info": {
          "splitdim": 0, "splitnum": 2,
          "hw_axis_owner": "row", "replicate_on": "col"
        }
      },
      "stages": [
        { "role": "producer", "tile": {"col": 0, "row": 0, "type": "shim"}, ... },
        { "role": "channel", "hops": [{"from": "...(0,0)", "to": "...(0,3)"}] },
        { "role": "consumer", "tiles": [{"col": 0, "row": 3, "type": "core"}, ...] }
      ],
      "invariants": [
        "producer total bytes (1024) broadcast to 2 consumers",
        "hop chain length: 1"
      ]
    }
  ]
}
```

## Layer 2: DfscheduleProvenanceMapPass

**Source:** `pass/passdfscheduleprovenancemap/passdfscheduleprovenancemap.cpp`
**Output:** `dfscheduleprovenancemap.json`
**Runs on:** dfschedule IR (after `BlueprintToSchedulePass`)

### What It Captures

The dfschedule IR is the final host-side representation before EmitC. It contains concrete DMA BD configurations, lock assignments, and start_io calls. This pass extracts the exact hardware configuration that will be programmed.

| Section | Source Ops | Description |
|---------|-----------|-------------|
| `tiles[]` | `DeclareTileOp` | All tiles with their DMA channel configs |
| `tiles[].dma_channels[]` | `ConfigCreateIoOp` | Per-channel: direction, enable_ooo, BD chain |
| `tiles[].dma_channels[].bd_chain[]` | `ConfigDmaBdOp` chain | BD id, len, lock, packet, next_bd, ooo, dim/iter |
| `tiles[].dma_channels[].start_io[]` | `StartIoOp` | repeat count, scf.for context |
| `kernel_configs[]` | `DeclareKernelConfigOp` | buffer size, lock IDs, iteration count per tile |
| `load_kernel_group` | `LoadKernelGroupOp` | kernel binary + target tiles |
| `flow_summary[]` | Grouped `StartIoOp` entries | All tiles participating in each flow |
| `invariant_checks[]` | Computed | Ping-pong chain cycle, lock symmetry |

### BD Chain Collection

The pass follows the linked list of `ConfigDmaBdOp` operations through their `linkedBd` SSA values:

```
ConfigDmaBdOp(bd_id=1, linkedBd=%bd0)
  └→ ConfigDmaBdOp(bd_id=0, linkedBd=null)  ← chain end
```

The chain is reversed to present BDs in programming order (BD0 first).

Each BD records:
- `bd_id`: Hardware BD slot number
- `buffer_offset`: Byte offset into the tile's memory
- `len`: Transfer length in bytes
- `enable_packet` / `packet_id`: Packet-switched mode
- `next_bd`: Next BD in ping-pong cycle
- `acquire_lock` / `release_lock`: Lock ID and value for synchronization
- `out_of_order_bd_id`: For OOO shim channels
- `dim_strides` / `dim_wraps`: Multi-dimensional addressing
- `iter_step_size` / `iter_wrap`: BD iteration (shim tiles)

### Start IO and scf.for Context

The pass distinguishes between top-level `start_io` ops and those nested inside `scf::ForOp`:

```cpp
// Top-level: start_io outside any loop
processStartIo(startOp, /*inLoop=*/false, 0, 0);

// Loop-nested: start_io inside scf.for
processStartIo(startOp, /*inLoop=*/true, lb, ub);
```

This is critical for diagnosis: a `start_io` inside `scf.for(0..2)` with `repeat=8` means 8 BD tasks per iteration, 16 total across 2 iterations.

### Flow Summary

Groups all `start_io` entries by `flow_index`, creating a unified view of which tiles participate in each data flow:

```json
{
  "flow_index": 2,
  "direction": "output",
  "entries": [
    {"tile_col": 0, "tile_row": 3, "channel": 0, "io_direction": "MM2S",
     "repeat_count": 1, "bd_len": 64},
    {"tile_col": 0, "tile_row": 0, "channel": 0, "io_direction": "S2MM",
     "repeat_count": 8, "inside_scf_for": true, "loop_range": "0..2", "bd_len": 64}
  ]
}
```

This connects senders (core MM2S) to receivers (shim S2MM), enabling aiediag to find connected tiles.

### Invariant Checks

1. **Ping-pong chain cycle**: Verifies BD0→BD1→BD0 forms a cycle via `next_bd` fields
2. **Lock symmetry**: Verifies all BDs in a chain use the same acquire/release lock IDs and values

### JSON Structure

```json
{
  "version": 1,
  "module_attrs": [{ "tile_m": 8, "tile_n": 8, ... }],
  "tiles": [
    {
      "col": 0, "row": 3, "type": "core",
      "dma_channels": [
        {
          "channel": 0, "direction": "S2MM",
          "enable_out_of_order": false,
          "flow_index": 0,
          "bd_chain": [
            { "bd_id": 0, "len": 512, "next_bd": 1,
              "acquire_lock": [{"id": 0, "val": -1}],
              "release_lock": [{"id": 0, "val": 1}] },
            { "bd_id": 1, "len": 512, "next_bd": 0, ... }
          ],
          "start_io": [{ "repeat_count": 1 }],
          "contract": "S2MM ch0: ping-pong receive, 512B each, lock 0/0"
        }
      ]
    }
  ],
  "kernel_configs": [...],
  "flow_summary": [...],
  "invariant_checks": [...]
}
```

## Layer 3: aiediag

**Source:** `src/tool/debug/aiediag.py`
**Inputs:** Both JSON provenance maps + live HW via `aiedbg`

### 7-Step Diagnostic Flow

```
aiediag dig COL ROW -mm2s0 [startcol N] [--json-dir PATH]
```

| Step | Action | Data Source |
|------|--------|------------|
| **[1]** | Read DMA status register of queried tile | `aiedbg reg read` |
| **[2]** | Show BD chain for queried channel | `dfscheduleprovenancemap.json` |
| **[3]** | Find connected tiles via flow_summary | `dfscheduleprovenancemap.json` |
| **[4]** | Read DMA status of connected tiles | `aiedbg reg read` |
| **[4b]** | Read shim event status registers | `aiedbg reg read` + `shimtile_events.json` |
| **[5]** | Show BD chains of connected tiles | `dfscheduleprovenancemap.json` |
| **[6]** | Show routing path (hops) | `dmaphopprovenacemap.json` |
| **[7]** | Automated root-cause diagnosis | Combines all above |

### Step 1: DMA Status Register Decode

Reads a 32-bit DMA status register at tile-type-specific offsets:

| Tile Type | S2MM Base | MM2S Base |
|-----------|-----------|-----------|
| Core | `0x1DF00` | `0x1DF10` |
| Shim (AIEML v5) | `0x1D220` | `0x1D228` |
| Shim (AIE 2ps) | `0x9320` | `0x9328` |

Channel stride: `+0x4` per channel.

Decoded fields:

| Bits | Field | Meaning |
|------|-------|---------|
| 1:0 | status | 0=Idle, 1=Running, 2=Paused |
| 2 | stall_lock_acq | Waiting for lock acquire |
| 3 | stall_lock_rel | Waiting for lock release |
| 4 | stall_stream | Stream backpressure (MM2S) or starvation (S2MM) |
| 5 | stall_tct | Task count stall |
| 10 | err_bd_unavail | No BD queued |
| 11 | err_bd_invalid | BD config error |
| 19 | channel_running | Channel is active |
| 22:20 | q_size | Number of BDs in queue |
| 27:24 | cur_bd | Current BD being executed |

### Step 3: Connected Tile Discovery

Uses `flow_summary` from the dfschedule JSON to find all tiles in the same data flow:

```python
flow_entry = find_flow(dfsche, flow_index)
connected = find_connected_tiles(flow_entry, col, row, direction)
```

For example, querying core tile(0,3) MM2S ch0 (output) finds:
- The shim tile(0,0) S2MM ch0 that receives this core's output
- Other core tiles in the same flow (if any)

### Step 4b: Shim Event Status

For shim tiles, reads the PL module event status registers (`0x34200`, `0x34204`) and decodes per-channel DMA events:

| Event | ID (S2MM ch0) | Meaning |
|-------|---------------|---------|
| START_TASK | 14 | DMA channel was started |
| FINISHED_TASK | 18 | DMA completed all tasks |
| STALLED_LOCK | 22 | Lock stall occurred |
| STREAM_STARVATION | 26 | No upstream data |
| MEMORY_BACKPRESSURE | 30 | DDR write pressure |

This disambiguates "Idle" status into "completed" vs "never started".

### Step 6: Routing Path Matching

Matches flow tiles against `communication_paths` from the dmaphop JSON:

```python
def find_routing_paths(dmaphop, flow_index, flow_entry):
    flow_tiles = {(e["tile_col"], e["tile_row"]) for e in flow_entry["entries"]}
    for path in dmaphop["communication_paths"]:
        path_endpoint_tiles = ...  # extract from stages[role=producer/consumer]
        if path_endpoint_tiles <= flow_tiles:
            matches.append(path)
```

This shows the physical hop chain from dmaphop IR, connecting the high-level routing to the low-level DMA configuration.

### Step 7: Diagnosis Engine

The `diagnose()` function combines all collected data to produce root-cause analysis:

**Stream Stall (MM2S)** = backpressure from downstream:
- Check if receiver (S2MM) is idle → "repeat/iter_wrap too low"
- Check if receiver is lock-stalled → "kernel not consuming input"
- Check if receiver is also stream-stalled → "routing congestion"

**Stream Stall (S2MM)** = starvation from upstream:
- Check if sender (MM2S) is idle → "never started or completed"
- Check if sender is lock-stalled → "kernel hasn't produced output"

**Lock Acquire Stall (MM2S)**:
- "Kernel hasn't released output buffer" → kernel compute is slow or stuck

**Lock Acquire Stall (S2MM)**:
- "DMA waiting for buffer to be available" → data not consumed fast enough

**Idle + connected tile active**:
- "Channel completed but connected tile still running" → repeat count mismatch

## Cross-Referencing Between Layers

The two JSON files serve complementary roles:

```
dmaphopprovenacemap.json          dfscheduleprovenancemap.json
========================          ============================
Logical routing topology          Physical DMA configuration
  - Which tiles talk to whom        - BD IDs, lengths, offsets
  - Through what hops               - Lock IDs and values
  - What data shapes                - Repeat counts
  - Partition strategy              - scf.for loop context
  - Packet IDs                      - Ping-pong chains
```

aiediag joins them through tile coordinates `(col, row)`:

1. User queries tile(0,3) MM2S ch0
2. `dfscheduleprovenancemap.json` → flow_index=2, connected to tile(0,0) S2MM ch0
3. Read both tiles' DMA status registers
4. `dmaphopprovenacemap.json` → this is `pull_0`, hop chain: tile(0,3)→tile(0,0)
5. Diagnose: core MM2S is lock-stalled → kernel hasn't released output → check kernel code

## Typical Debug Scenarios

### Scenario 1: Stream Deadlock (Repeat Mismatch)

```
[1] tile(0,3) MM2S ch0: Running, stall_stream=1
[4] tile(0,0) S2MM ch0: Idle, channel_running=0
[4b] Shim S2MM ch0: START_TASK=SET, FINISHED_TASK=SET
[7] MM2S ch0 has STREAM BACKPRESSURE
    >> Receiver tile(0,0) S2MM ch0 completed all transfers — repeat/iter_wrap too low?
```

Root cause: Shim S2MM consumed all its programmed BD tasks (repeat too low), went idle. Core MM2S still has data to send but has no receiver → stream backs up.

### Scenario 2: Kernel Hang (Lock Stall)

```
[1] tile(0,3) MM2S ch0: Running, stall_lock_acq=1
[4] tile(0,0) S2MM ch0: Running, stall_stream=1
[7] MM2S ch0 stalled on LOCK ACQUIRE — kernel hasn't released output buffer
```

Root cause: Kernel hasn't called `release_output_window()`. Could be stuck in computation, waiting for an input that never arrived, or hitting an infinite loop.

### Scenario 3: DMA Never Started

```
[1] tile(0,0) S2MM ch0: Idle
[4b] Shim S2MM ch0: START_TASK=not set
[7] Channel is IDLE — never started (START_TASK not set)
    Shim S2MM: START_TASK not set — start_io not issued
```

Root cause: The host code never reached the `__Runtime_startio()` call for this channel. Check if the host code is structured correctly or if an earlier operation blocked.

## Files

| File | Role |
|------|------|
| `pass/passdmaphopprovenancemap/passdmaphopprovenancemap.cpp` | Generates `dmaphopprovenacemap.json` |
| `pass/passdfscheduleprovenancemap/passdfscheduleprovenancemap.cpp` | Generates `dfscheduleprovenancemap.json` |
| `src/tool/debug/aiediag.py` | Runtime diagnostic tool |
| `~/aiejson/shimtile_events.json` | Shim event ID reference (optional) |
