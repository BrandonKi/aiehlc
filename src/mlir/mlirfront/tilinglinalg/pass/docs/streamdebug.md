<!-- Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->
# Stream Stall Debug: Packet ID Mismatch (DMA_MM2S_STREAM_BACKPRESSURE)

## Problem Summary

All 4 core tile DMA MM2S channels stall with `STALL_STREAM=1` after kernel launch.
The shim tile S2MM channel stalls with stream starvation.
The `device_teardown done` line is never reached — the host hangs at `__Runtime_wait`.

Observed in: `hwlog2` (AieRt debug snapshot after wait timeout)

---

## Root Cause: Packet ID Mismatch Between host.cc and routing.cc

The stream switch at each core tile uses packet-switched routing. It holds a table of
slave slot entries that match incoming packet headers by PktId. If no slot matches the
packet header sent by the DMA, the stream switch deasserts TREADY (flow control) back
toward the DMA, causing `DMA_MM2S_STREAM_BACKPRESSURE` (Event ID 148).

### Mask Semantics

`XAie_StrmPktSwSlaveSlotEnable(..., PktId, Mask, ...)` matches if:
```
(incoming_PktId & Mask) == (slot_PktId & Mask)
```
With `Mask=31` (0x1F, all 5 PktId bits set), this is an **exact equality** check.

### The Mismatch Table

| Tile   | BD pkt_id (host.cc, before fix) | Slot PktId (routing.cc) | Match? |
|--------|----------------------------------|--------------------------|--------|
| (0,3)  | 0                                | 1 (DMA slot, Mask=31)    | NO     |
| (1,3)  | 1                                | 2 (DMA slot, Mask=31)    | NO     |
| (0,4)  | 0                                | 1 (DMA slot, Mask=31)    | NO     |
| (1,4)  | 1                                | 2 (DMA slot, Mask=31)    | NO     |

The routing engine generates PktId values starting at 1 (1-based, one per column within
a row group). The BlueprintToSchedulePass was generating BD `packet_id` values starting
at 0 (0-based). This off-by-one mismatch caused all 4 DMA channels to stall.

---

## Event Chain (from AIE2PS Event Spec)

```
Core tile (0,3) DMA MM2S ch0
│  BD config: pkt_id=0 → DMA prepends packet header {PktId=0, PktType=0}
▼
Stream switch slave port DMA0
  Slot 0: PktId=1, Mask=31 → match check: (0 & 31) == (1 & 31) → 0 == 1 → NO MATCH
  Result: switch holds flit at slave port, deasserts TREADY to DMA
▼
Memory Module Event: DMA_MM2S_0_STREAM_BACKPRESSURE (ID 148) fires
Core Module Event:   STREAM_STALL (ID 43) fires (kernel waiting on DMA)
▼
Shim tile (2,0) S2MM ch0 — never receives data
  Shim Event: DMA_S2MM_0_STREAM_STARVATION (ID 26) fires
```

Same pattern for tiles (1,3), (0,4), (1,4).

---

## Debug Evidence (hwlog2)

```
[AieRt_Debug] tile(0,3) ch0 MM2S: STALLED: lock_acq=0 lock_rel=0 stream=1 tct=0
              BD0: pkt_id=0  next->BD1
[AieRt_Debug] tile(1,3) ch0 MM2S: STALLED: stream=1
              BD0: pkt_id=1  next->BD1
[AieRt_Debug] tile(0,4) ch0 MM2S: STALLED: stream=1
              BD0: pkt_id=0  next->BD1
[AieRt_Debug] tile(1,4) ch0 MM2S: STALLED: stream=1
              BD0: pkt_id=1  next->BD1
[AieRt_Debug] tile(2,0) ch0 S2MM: STALLED: stream=1 (stream starvation)
```

`lock_acq=0` confirms lock is NOT the stall cause. `stream=1` is the definitive indicator
of packet-switch slot mismatch.

---

## Fix Applied (host.cc)

The BD `packet_id` values for all 4 core tiles were incremented by 1 to match routing.cc:

| Tile   | BD pkt_id (before) | BD pkt_id (after fix) | Slot PktId (routing.cc) |
|--------|---------------------|------------------------|--------------------------|
| (0,3)  | 0                   | **1**                  | 1                        |
| (1,3)  | 1                   | **2**                  | 2                        |
| (0,4)  | 0                   | **1**                  | 1                        |
| (1,4)  | 1                   | **2**                  | 2                        |

Changed lines in `worklocal/host.cc`:

- Line 26/31 (tile 0,3 BD1/BD0): `packet_id` arg changed from `0` → `1`
- Line 40/45 (tile 1,3 BD1/BD0): `packet_id` arg changed from `1` → `2`
- Line 63/68 (tile 0,4 BD1/BD0): `packet_id` arg changed from `0` → `1`
- Line 77/82 (tile 1,4 BD1/BD0): `packet_id` arg changed from `1` → `2`

The `__Runtime_dma_bd_config` signature is:
```c
XAie_DmaDesc __Runtime_dma_bd_config(
    DevInst, tile, buf, bd_id, offset, len, next_bd,
    enable_packet, packet_id, packet_type,
    acquire_lock_val, release_lock_id, release_lock_val);
```
The 9th argument is `packet_id`. All 8 BD config calls for core tiles were updated.

---

## Root Cause in Compiler (BlueprintToSchedulePass)

The `BlueprintToSchedulePass` assigns `packet_id` to BDs using `base_packet_id` from the
`flow_transfer` op. The routing engine allocates PktId starting at 1 (slot 0 reserved or
PktId=0 is the "no packet" sentinel). The pass was directly using `flow_index` (0-based)
as `packet_id`, producing a 0-based sequence (0, 1, 2...) instead of matching the
routing-generated 1-based sequence (1, 2, 3...).

**Fix location in compiler**: `passblueprinttoschedule.cpp` — the `packet_id` assigned
to core tile BDs must use `base_packet_id + tile_index` from `flow_transfer`, which must
agree with the PktId values programmed in `routing.cc` by the routing passes.

---

## DMA_MM2S_STREAM_BACKPRESSURE vs DMA_MM2S_MEMORY_STARVATION

| Event | ID  | Condition |
|-------|-----|-----------|
| DMA_MM2S_STREAM_BACKPRESSURE | 148 | DMA has data ready, stream switch deasserts TREADY |
| DMA_MM2S_MEMORY_STARVATION   | 152 | Stream switch ready to accept, local memory too slow |

In our case the stall is unambiguously BACKPRESSURE: the lock was acquired (lock_acq=0
in stall status), the DMA read local memory successfully, but the stream switch rejected
the packet header.

---

## Additional Risk: SS_OVERFLOW (Event ID 60, Shim)

If a stalled packet keeps accumulating in stream switch internal buffers, `SS_OVERFLOW`
(shim event ID 60) may fire, potentially corrupting other streams sharing the switch.
Worth checking this bit in future hwlog captures when stream stalls are observed.

---

## Stream Switch Config Debug Tool

`AieRt_PrintStreamSwitchConfig` in `src/mlir/runtime/aie_runtime_debug.c` now outputs
JSON and classifies each port by switch type.

### Switch Type Classification

| Type | Ports | Meaning |
|------|-------|---------|
| `PKT` | DMA, CORE master with `pkt_en=1`; any slave with enabled slots | Packet-switched — carries packet headers |
| `CIRC` | Direction ports (`NORTH`/`SOUTH`/`EAST`/`WEST`) with `pkt_en=0` | Circuit-switched — raw data, no header |
| `CTRL` | `CTRL`, `FIFO0`, `TRACE`, `AIE_TRACE`, `MEM_TRACE` | Control/management ports |

### Master Port Fields

```json
{
  "index": 1,
  "port": "DMA0",
  "switch_type": "PKT",
  "enabled": true,
  "packet_switch": true,
  "drop_header": false,
  "drop_header_note": "DONOT_DROP_HEADER (preserve pkt hdr for downstream PKT-SW slave)",
  "config": "0x01",
  "raw": "0xC0000001"
}
```

`drop_header_note` is printed only for packet-switch masters. The rule:
- `DONOT_DROP_HEADER` — intermediate hop; packet header must pass through for downstream slave slot matching
- `DROP_HEADER` — final PKT-SW hop before a circuit-switch segment; strip header before forwarding

### Slave Slot Fields

```json
{
  "slot": 0,
  "enabled": true,
  "pkt_id": 1,
  "pkt_mask": "0x1F",
  "match_mode": "exact",
  "match_note": "exact: DMA BD pkt_id must equal 1 or DMA_MM2S_STREAM_BACKPRESSURE (event 148) fires",
  "msel": 0,
  "arb": 0,
  "raw": "0x01000100"
}
```

| `match_mode` | `pkt_mask` | Semantics |
|---|---|---|
| `exact` | `0x1F` | BD `pkt_id` must equal slot `pkt_id` — mismatch fires event 148 |
| `wildcard` | `0x00` | Any `pkt_id` accepted — used at merge-point NORTH/WEST slaves |
| `partial` | other | Partial bit match |

### Usage in host.cc

The debug snapshot is called via `AieRt_DebugSnapshotFromCoords` after wait timeout.
To also dump stream switch state, call `AieRt_PrintStreamSwitchConfigAll` with the
tile list before or after the snapshot:

```c
XAie_LocType ss_tiles[] = {
    XAie_TileLoc(0,3), XAie_TileLoc(1,3),
    XAie_TileLoc(0,4), XAie_TileLoc(1,4)
};
AieRt_PrintStreamSwitchConfigAll(g_DevInst, ss_tiles, 4, /*print_all=*/0);
```

---

## Status

| Fix | Location | Status |
|-----|----------|--------|
| Corrected BD packet_id values | `worklocal/host.cc` | **Applied** |
| Root cause in compiler pass | `passblueprinttoschedule.cpp` | Identified, fix needed in pass |
| Stream switch JSON debug tool | `src/mlir/runtime/aie_runtime_debug.c` | **Implemented** |
