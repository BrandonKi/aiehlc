<!-- Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->

# XAie API Validation Rules (AIEML)

Pre-extracted constraint tables from the XAie driver source so the agent can verify generated code without re-reading the driver each time. Target device: **AIE-ML** (Versal AI Core series).

Driver source root: `thirdparty/alib/aie-rt/driver/`

---

## 1. Tile Type Mapping

Determined by `XAie_TileLoc(col, row)` row value.

| Row   | Tile Type              | Constant                  |
|-------|------------------------|---------------------------|
| 0     | Shim (SHIMNOC/SHIMPL)  | `XAIEGBL_TILE_TYPE_SHIMNOC` (1) / `SHIMPL` (2) |
| 1-2   | MemTile                | `XAIEGBL_TILE_TYPE_MEMTILE` (3) |
| 3+    | AIE Tile (compute)     | `XAIEGBL_TILE_TYPE_AIETILE` (0) |

Source: `include/xaiengine/xaie_lite_hwcfg.h` (AIEML: `MEM_TILE_ROW_START=1`, `MEM_TILE_NUM_ROWS=2`, `AIE_TILE_ROW_START=3`, `AIE_TILE_NUM_ROWS=8`)

---

## 2. StrmSwPortType Enum

Source: `include/xaiengine/xaiegbl.h` lines 254-266

| Name             | Value |
|------------------|-------|
| CORE             | 0     |
| DMA              | 1     |
| CTRL             | 2     |
| FIFO             | 3     |
| SOUTH            | 4     |
| WEST             | 5     |
| NORTH            | 6     |
| EAST             | 7     |
| TRACE            | 8     |
| UCTRLR           | 9     |
| SS_PORT_TYPE_MAX | 10    |

Any port type value >= 10 is invalid.

---

## 3. Port Number Limits (AIEML)

Format: Master count / Slave count. A port number must be < the count for the corresponding direction (master or slave).

Source: `src/global/xaiemlgbl_reginit.c` lines 954-1210

### AIE Tile (row 3+)

| Port Type | Master | Slave |
|-----------|--------|-------|
| CORE      | 1      | 1     |
| DMA       | 2      | 2     |
| CTRL      | 1      | 1     |
| FIFO      | 1      | 1     |
| SOUTH     | 4      | 6     |
| WEST      | 4      | 4     |
| NORTH     | 6      | 4     |
| EAST      | 4      | 4     |
| TRACE     | 0      | 2     |

### MemTile (row 1-2)

| Port Type | Master | Slave |
|-----------|--------|-------|
| CORE      | 0      | 0     |
| DMA       | 6      | 6     |
| CTRL      | 1      | 1     |
| FIFO      | 0      | 0     |
| SOUTH     | 4      | 6     |
| WEST      | 0      | 0     |
| NORTH     | 6      | 4     |
| EAST      | 0      | 0     |
| TRACE     | 0      | 1     |

MemTile has NO CORE, FIFO, WEST, or EAST ports.

### Shim Tile (row 0)

| Port Type | Master | Slave |
|-----------|--------|-------|
| CORE      | 0      | 0     |
| DMA       | 0      | 0     |
| CTRL      | 1      | 1     |
| FIFO      | 1      | 1     |
| SOUTH     | 6      | 8     |
| WEST      | 4      | 4     |
| NORTH     | 6      | 4     |
| EAST      | 4      | 4     |
| TRACE     | 0      | 2     |

Shim tile has NO CORE or DMA stream ports (DMA is accessed via `XAie_EnableAieToShimDmaStrmPort` / `XAie_EnableShimDmaToAieStrmPort` instead).

---

## 4. PortVerify Rules (AIEML)

These rules define which slave-to-master port type combinations are valid within a single tile's stream switch. A connection that violates these rules will return `XAIE_ERR_STREAM_PORT` at runtime.

Source: `src/stream_switch/xaie_ss_aieml.c`

### AIE Tile (lines 50-96)

| Slave Type | Rule |
|------------|------|
| TRACE      | Master must be FIFO, SOUTH, or DMA with MstrPortNum == 0. All others rejected. |
| CORE       | Master must NOT be CORE. (CORE-to-CORE forbidden) |
| DMA        | If Master is DMA, then SlvPortNum must equal MstrPortNum. DMA to non-DMA is OK. |
| CTRL       | Master must NOT be DMA or CTRL. |
| FIFO       | No restrictions (any master OK). |
| SOUTH/WEST/NORTH/EAST | If Slave == Master (same direction), then SlvPortNum must equal MstrPortNum. Cross-direction is OK. |
| default    | Rejected. |

### MemTile (lines 117-154)

| Slave Type | Rule |
|------------|------|
| TRACE      | Master must be SOUTH or DMA with MstrPortNum == 5. All others rejected. |
| DMA        | If Master is DMA, then SlvPortNum must equal MstrPortNum. DMA to non-DMA is OK. |
| CTRL       | If Master is DMA, MstrPortNum must be 5. Master must NOT be CTRL. |
| SOUTH/NORTH | If Master is SOUTH or NORTH, SlvPortNum must equal MstrPortNum. |
| default    | Rejected (includes CORE, FIFO, WEST, EAST). |

### Shim Tile (lines 176-211)

| Slave Type | Rule |
|------------|------|
| TRACE      | Master must be FIFO, SOUTH, WEST(port 0), or EAST(port 0). All others rejected. |
| CTRL       | Master must NOT be CTRL. (CTRL-to-CTRL forbidden) |
| FIFO/SOUTH | No restrictions (any master OK). |
| WEST/NORTH/EAST | If Slave == Master, SlvPortNum must equal MstrPortNum. Cross-direction is OK. |
| default    | Rejected (includes CORE, DMA). |

---

## 5. Packet Switch Constants

Source: `src/stream_switch/xaie_ss.c` lines 50-53, `include/xaiengine/xaiegbl.h` line 50

| Constant              | Value | Used By |
|-----------------------|-------|---------|
| XAIE_PACKET_ID_MAX   | 31    | `XAie_StrmPktSwSlaveSlotEnable` Pkt.PktId |
| XAIE_SS_ARBITOR_MAX  | 7     | Both MstrPortEnable and SlaveSlotEnable Arbitor param |
| XAIE_SS_MSEL_MAX     | 3     | `XAie_StrmPktSwSlaveSlotEnable` MSel param |
| XAIE_SS_MASK         | 31    | `XAie_StrmPktSwSlaveSlotEnable` Mask param (0x1F) |
| XAIE_SS_MSELEN_MAX   | 15    | `XAie_StrmPktSwMstrPortEnable` MSelEn param (0xF) |
| NumSlaveSlots         | 4     | `XAie_StrmPktSwSlaveSlotEnable` SlotNum param |

### API Signatures

`XAie_StrmPktSwMstrPortEnable(DevInst, Loc, Master, MstrPortNum, DropHeader, Arbitor, MSelEn)`
- DropHeader: 0 (`XAIE_SS_PKT_DONOT_DROP_HEADER`) or 1 (`XAIE_SS_PKT_DROP_HEADER`)
- Arbitor: 0-7
- MSelEn: 0-15 (bitmask selecting which MSel values this master accepts)

`XAie_StrmPktSwSlaveSlotEnable(DevInst, Loc, Slave, SlvPortNum, SlotNum, Pkt, Mask, MSel, Arbitor)`
- SlotNum: 0-3
- Pkt: `{.PktId=N, .PktType=M}` where PktId 0-31
- Mask: 0-31 (packet ID match mask)
- MSel: 0-3
- Arbitor: 0-7

### Arbitor/MSel Consistency

For packet-switched routing on a given tile:
- A slave slot's `Arbitor` value selects which arbiter handles this packet
- A slave slot's `MSel` value is a 2-bit selector within that arbiter
- A master port's `Arbitor` value binds it to a specific arbiter
- A master port's `MSelEn` is a 4-bit bitmask; bit N means the master accepts packets with MSel == N
- Consistency check: for each slave slot with (Arbitor=A, MSel=M), there must be a master port on the same tile with Arbitor=A and bit M set in MSelEn

---

## 6. DMA Constraints (AIEML)

Source: `src/dma/xaie_dma.c`, `src/global/xaiemlgbl_reginit.c`

| Resource      | AIE Tile | MemTile | Shim |
|---------------|----------|---------|------|
| NumBds        | 16       | 16      | 48   |
| NumChannels   | 2        | 2       | 6    |
| NumLocks      | 16       | 64      | 16   |

BD-Channel mapping (AIEML, Shim only): BD 0-23 pair with even channels, BD 24-47 pair with odd channels.

### Key DMA API Constraints

| API | Parameter | Constraint |
|-----|-----------|------------|
| `XAie_DmaWriteBd` | BdNum | < NumBds for tile type |
| `XAie_DmaChannelPushBdToQueue` | BdNum | < NumBds |
| `XAie_DmaChannelPushBdToQueue` | ChNum | < NumChannels |
| `XAie_DmaChannelReset` | ChNum | < NumChannels |
| `XAie_DmaSetBdIteration` | StepSize | <= 131072 (tile), 8192 (shim), 1048576 (memtile) |
| `XAie_DmaSetBdIteration` | Wrap | 1-64 (must not be 0) |
| `XAie_DmaSetBdIteration` | IterCurr | <= 63 |

---

## 7. Lock Constraints (AIEML)

Source: `src/locks/xaie_locks.c`

| Resource          | AIE Tile | MemTile | Shim |
|-------------------|----------|---------|------|
| NumLocks          | 16       | 64      | 16   |
| LockValUpperBound | 63       | 63      | 63   |
| LockValLowerBound | -64      | -64     | -64  |

Special value: `XAIE_LOCK_WITH_NO_VALUE` = -1

### Lock API Constraints

| API | Parameter | Constraint |
|-----|-----------|------------|
| `XAie_LockAcquire` | Lock.LockId | < NumLocks for tile type |
| `XAie_LockAcquire` | Lock.LockVal | -64 to 63 |
| `XAie_LockRelease` | Lock.LockId | < NumLocks for tile type |
| `XAie_LockRelease` | Lock.LockVal | -64 to 63 |

---

## 8. Direction Adjacency Rules

When `XAie_StrmConnCctEnable` routes traffic out a master port in a given direction, the receiving tile must have a corresponding slave port in the opposite direction:

| Master Direction at tile(C,R) | Expected Slave at | Slave Direction |
|-------------------------------|-------------------|-----------------|
| NORTH                         | tile(C, R+1)      | SOUTH           |
| SOUTH                         | tile(C, R-1)      | NORTH           |
| EAST                          | tile(C+1, R)      | WEST            |
| WEST                          | tile(C-1, R)      | EAST            |

This applies to both circuit-switched (`XAie_StrmConnCctEnable`) and packet-switched routing where physical connections traverse tile boundaries.

---

## 9. Common Shim DMA Port APIs

These APIs configure the shim tile's DMA-to-stream and stream-to-DMA ports. They bypass the normal stream switch port numbering:

| API | Direction | Parameters |
|-----|-----------|------------|
| `XAie_EnableAieToShimDmaStrmPort` | AIE array → Shim DMA | (DevInst, Loc, PortNum) |
| `XAie_EnableShimDmaToAieStrmPort` | Shim DMA → AIE array | (DevInst, Loc, PortNum) |

These configure the PL interface and are required before shim DMA can receive/send data. The PortNum typically maps to a SOUTH port on the shim tile.
