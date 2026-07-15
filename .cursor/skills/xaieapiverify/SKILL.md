<!-- Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->
---
name: xaieapiverify
description: Verify correct XAie API usage in generated code (routing.cc, host.cc) by analyzing function calls against driver validation rules. Use when generated code needs verification before HW execution, or when debugging XAie errors from board runs.
---

# XAie API Verify

Statically verify that generated C/C++ code (routing.cc, host.cc) uses XAie driver APIs correctly, before deploying to hardware.

## Workflow

1. **Read** the target file(s). Default locations:
   - `src/mlir/mlirfront/tilinglinalg/pass/unitest/worklocal/routing.cc`
   - `src/mlir/mlirfront/tilinglinalg/pass/unitest/worklocal/host.cc`

2. **Extract** every `XAie_*` function call. For each call note: line number, function name, and all arguments (resolve simple variable assignments like `int32_t v2 = 0;` to their literal values).

3. **Determine tile types** from `XAie_TileLoc(col, row)` using the AIEML row mapping in [references/validation-rules.md](references/validation-rules.md):
   - Row 0 = Shim (SHIMNOC/SHIMPL)
   - Row 1-2 = MemTile
   - Row 3+ = AIE Tile

4. **Validate each call** against constraint tables in [references/validation-rules.md](references/validation-rules.md):

   **Stream switch APIs** (`XAie_StrmConnCctEnable`, `XAie_StrmPktSwMstrPortEnable`, `XAie_StrmPktSwSlaveSlotEnable`):
   - Port type must be a valid `StrmSwPortType` enum value (< 10)
   - Port number must be within the tile-type-specific master/slave limit
   - Apply PortVerify rules per tile type (see validation-rules.md section 4):
     - AIE tile: TRACE only to FIFO/SOUTH/DMA(port 0); CORE-to-CORE forbidden; DMA same-port only; CTRL cannot connect to DMA or CTRL; same-direction only if same port number
     - Mem tile: TRACE only to SOUTH/DMA(port 5); DMA same-port only; CTRL cannot connect to DMA(!=5) or CTRL; same-direction only if same port number; no WEST/EAST
     - Shim tile: TRACE to FIFO/SOUTH/WEST(0)/EAST(0); CTRL-to-CTRL forbidden; same-direction only if same port number
   - Packet switch: PktId <= 31, Arbitor <= 7, MSel <= 3, Mask <= 31, MSelEn <= 15, SlotNum < 4

   **DMA APIs** (`XAie_DmaWriteBd`, `XAie_DmaChannelPushBdToQueue`, etc.):
   - BD number within tile-type limit (Tile/MemTile: 16, Shim: 48)
   - Channel number within limit (Tile: 2, Shim: 6)
   - BD-channel mapping (AIEML): BD 0-23 for even channels, BD 24-47 for odd channels

   **Lock APIs** (`XAie_LockAcquire`, `XAie_LockRelease`):
   - Lock ID within tile-type limit (Tile/Shim: 16, MemTile: 64)
   - Lock value within range (-64 to 63 for AIEML)

5. **Trace routing paths** from `XAie_StrmConnCctEnable` and packet switch calls:
   - Build an adjacency graph: each call connects (tile, slave_port) to (tile, master_port)
   - Check direction consistency between adjacent tiles: a master EAST at tile(1,2) must correspond to a slave WEST at tile(2,2)
   - Check for dangling connections (master port with no matching slave at the neighboring tile)
   - Verify paths reach from source (DMA/compute tiles) to destination (shim tiles)

6. **Check packet switch consistency** (when both `XAie_StrmPktSwSlaveSlotEnable` and `XAie_StrmPktSwMstrPortEnable` are present):
   - Each slave slot's Arbitor/MSel must align with a master port's Arbitor/MSelEn
   - No arbiter conflicts: two slave slots on the same tile using the same arbiter but conflicting MSel

7. **Check DMA/Lock pairing** (when present):
   - Lock acquire/release calls should be balanced
   - BD configurations should reference valid lock IDs

8. **Report** findings using this format:

   ```
   === XAie API Verification: <filename> ===

   Line <N>: <function call>
     [ERROR] <description of violation, constraint, and valid range>
     [WARN]  <description of suspicious pattern>
     [PASS]  All checks passed

   --- Routing Path Analysis ---
   Path 1: tile(0,3) DMA:0 → tile(0,3) EAST:0 → tile(1,3) WEST:0 → ... → tile(2,0) SOUTH:1  [OK]
   Path 2: tile(0,4) DMA:0 → ...  [DANGLING at tile(X,Y)]

   --- Summary ---
   Total calls: N | PASS: N | WARN: N | ERROR: N
   ```

## When to cross-reference driver source

If `validation-rules.md` doesn't cover an API or a constraint seems ambiguous, read the actual driver validation code at:
- Stream switch: `thirdparty/alib/aie-rt/driver/src/stream_switch/xaie_ss.c`
- Port compatibility: `thirdparty/alib/aie-rt/driver/src/stream_switch/xaie_ss_aieml.c`
- DMA: `thirdparty/alib/aie-rt/driver/src/dma/xaie_dma.c`
- Locks: `thirdparty/alib/aie-rt/driver/src/locks/xaie_locks.c`
- Port limits: `thirdparty/alib/aie-rt/driver/src/global/xaiemlgbl_reginit.c`

## Resources

| When | File |
|------|------|
| All constraint tables (port limits, enums, constants) | [references/validation-rules.md](references/validation-rules.md) |
| Example generated routing code | `src/mlir/mlirfront/tilinglinalg/pass/unitest/worklocal/routing.cc` |
| Example generated host code | `src/mlir/mlirfront/tilinglinalg/pass/unitest/worklocal/host.cc` |
| Known-good manual XAie usage | `src/mlir/mlirfront/tilinglinalg/pass/routingimplement/codegenexample/aie_control.cpp` |
| XAie API guide | `doc/aieapi.md` |
