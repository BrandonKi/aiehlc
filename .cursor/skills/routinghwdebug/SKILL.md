<!-- Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->
---
name: routinghwdebug
description: Debug routing code generation issues by scanning generated routing.cc with xaieapiverify, tracing errors back through the MLIR dialect pipeline (EmitC -> routinghw -> dmaphop -> dmap -> routing), finding root causes in routingimplement (topology, BFS, resource manager), and providing fixes. Use when routing.cc fails XAie API verification or causes HW errors.
---

# RoutingHW Debug

End-to-end debugging skill for routing code generation. Scans generated C++ for XAie API errors, traces each error back through the MLIR lowering pipeline to its origin, identifies the root cause in the routing implementation, and proposes a fix.

## Workflow

### Step 1: Scan with xaieapiverify

Read and follow the xaieapiverify skill at `.cursor/skills/xaieapiverify/SKILL.md` to verify the target file (default: `src/mlir/mlirfront/tilinglinalg/pass/unitest/worklocal/routing.cc`). Collect all ERROR and WARN findings with line numbers.

If no errors are found, report PASS and stop.

### Step 2: Identify the pipeline path

Determine which pipeline produced the file:

| Output file | Pipeline | CLI arg | Function in test.cpp |
|-------------|----------|---------|----------------------|
| `routing.cc` | routing -> dmap -> dmaphop -> routinghw -> EmitC | `dmaphw` | `routingtodmap()` |
| `routing_hw.cc` | routing -> routinghw -> EmitC | `hw` | `routingtoroutinghw()` |
| `host.cc` | routing -> dmap -> dmaphop -> blueprint -> dfschedule -> EmitC | `dfschedule` | `routingtodfschedule()` |

### Step 3: Trace each error back through dialects

For each ERROR found in Step 1, trace it backward through the pipeline stages. Use the op mapping tables in [references/op-mapping.md](references/op-mapping.md).

**3a. C++ to EmitC:** Map the XAie function call to the `emitc.call_opaque` that generated it. The call arguments correspond to EmitC operands.

**3b. EmitC to routinghw:** Identify which routinghw op produced the EmitC call. Look in `pass/routinghwlower.cpp` for the lowering pattern. Key mappings:

| XAie API call | routinghw op |
|---------------|-------------|
| `XAie_StrmConnCctEnable(Dev, Loc, Slave, SlvPort, Master, MstrPort)` | `routinghw::ConnectStreamSingleSwitchPort` |
| `XAie_StrmPktSwSlaveSlotEnable(...)` | `routinghw::ConnectStreamPktSwitchPort` (slave part) |
| `XAie_StrmPktSwMstrPortEnable(...)` | `routinghw::ConnectStreamPktSwitchPort` (master part) |
| `XAie_EnableAieToShimDmaStrmPort(Dev, Loc, Port)` | `routinghw::EnableAieToExtShimPort` |
| `XAie_EnableShimDmaToAieStrmPort(Dev, Loc, Port)` | `routinghw::EnableExtToAieShimPort` |

**3c. routinghw to dmaphop (Path B: routing.cc):** Look in `pass/passdmaphoptoroutinghw/passdmaphoptoroutinghw.cpp` for which dmaphop ops lowered to the routinghw ops. The tile location, port direction, and port number are set during this lowering.

**3d. dmaphop to dmap:** Look in `pass/passdmaptodmaphop/dmaptodmaphop.cpp`. The `PushOpLowering` and `PullOpLowering` patterns create hops with tile locations and port info from the RoutingTopology BFS results.

**3e. dmap to routing:** Look in `pass/passroutingtodmap/routingtodmap.cpp`. The routing ops (mesh, data IO, broadcast) are lowered to dmap ports/streams using RoutingTopology path finding.

### Step 4: Examine the routing implementation

Once the trace identifies where the bad tile location or port assignment originated, examine the routing implementation code:

**4a. Port templates** — Check if the port configuration for the tile type matches the actual hardware:

File: `pass/routingimplement/hw/hwresource.cpp`
- `defaultPortTemplates()` defines per-tile-type port counts (Core, Mem, Shim, NocShim)
- Compare against the XAie driver's actual port limits in `.cursor/skills/xaieapiverify/references/validation-rules.md` section 3
- Common issue: Mem tiles given EAST/WEST ports (4 each) in the model but the AIEML hardware has 0

**4b. BFS path finding** — Check if the BFS routing respects tile-type constraints:

File: `pass/routingimplement/routing/routingpath.cpp`
- Row-0 rule (line 129): no horizontal movement on row 0 (shim)
- BFS priority (lines 101-106): Memory > Shim > Core
- Check: is there a rule preventing EAST/WEST movement on MemTile rows? If not, this is likely the root cause.
- Check: are `M_start_` / `M_end_` correct for the target device? (hardcoded as 1/1, but AIEML Gen-2 has Mem rows 1-2)

**4c. Resource manager** — Check link availability and occupation:

File: `pass/routingimplement/hw/ResourceManager.cpp`
- `linkAvailable()` (line 149): checks slave/master port vectors
- `occupyLink()` (line 172): allocates ports
- If Mem tiles have EAST/WEST ports in the model, `linkAvailable` will return true for horizontal moves on Mem rows even though the hardware doesn't support it

**4d. Topology construction** — Check how tiles are created:

File: `pass/routingimplement/routing/routingtopology.cpp`
- Creates `ResourceMgr` from `IHwResource` (which comes from hwresource.cpp)
- DataIO setup, RoutingPath creation

### Step 5: Determine root cause and provide fix

Common root causes and fixes:

**A. Port template mismatch (hwresource.cpp)**
- Problem: `defaultPortTemplates()` gives Mem tiles EAST/WEST ports, but AIEML MemTiles have 0.
- Fix: Create separate port templates for Mem tiles with EAST=0, WEST=0:

```cpp
// In hwresource.cpp defaultPortTemplates()
{ TileType::Mem, {
    { PortDirection::North, PortRole::Master, 4, {} },
    { PortDirection::South, PortRole::Master, 4, {} },
    { PortDirection::East, PortRole::Master, 0, {} },   // MemTile: no EAST
    { PortDirection::West, PortRole::Master, 0, {} },   // MemTile: no WEST
    { PortDirection::North, PortRole::Slave, 4, {} },
    { PortDirection::South, PortRole::Slave, 4, {} },
    { PortDirection::East, PortRole::Slave, 0, {} },    // MemTile: no EAST
    { PortDirection::West, PortRole::Slave, 0, {} },    // MemTile: no WEST
    { PortDirection::DMA, PortRole::Master, 6, {} },    // MemTile: 6 DMA ports
    { PortDirection::DMA, PortRole::Slave, 6, {} },
}}
```

**B. Missing BFS constraint for MemTile rows (routingpath.cpp)**
- Problem: BFS allows EAST/WEST movement on MemTile rows.
- Fix: Add a constraint similar to the row-0 rule:

```cpp
// In routingpath.cpp BFS loop, after row-0 rule
if (cur.r >= M_start_ && cur.r <= M_end_ && d.c != 0) continue; // MemTile: no horizontal
```

**C. M_end_ hardcoded incorrectly (routingpath.cpp)**
- Problem: `M_end_=1` but AIEML Gen-2 has Mem rows 1-2.
- Fix: Set `M_end_` from the device layout in hwresource.cpp.

**D. Lowering pass produces wrong port type for tile (routinghwlower.cpp or passdmaphoptoroutinghw.cpp)**
- Problem: The lowering pass emits a port direction that is valid for the routingimplement model but invalid on actual hardware.
- Fix: The lowering pass should consult the port limits or the lowering should reject invalid port assignments.

### Step 6: Verify the fix

After applying the fix:

1. Rebuild: `cd pass/unitest/build && cmake .. && make -j4`
2. Regenerate: `./test dmaphw` (or `./test hw` for Path A)
3. Re-run xaieapiverify on the regenerated `routing.cc`
4. If errors remain, repeat from Step 3

## Key Files

| Purpose | File |
|---------|------|
| Test driver (pipeline invocation) | `pass/unitest/test.cpp` |
| Port templates per tile type | `pass/routingimplement/hw/hwresource.cpp` |
| BFS path finding | `pass/routingimplement/routing/routingpath.cpp` |
| Tile topology model | `pass/routingimplement/routing/routingtopology.cpp` |
| Resource/port tracking | `pass/routingimplement/hw/ResourceManager.cpp` |
| routing -> routinghw lowering | `pass/routinglower.cpp` |
| routinghw -> EmitC lowering | `pass/routinghwlower.cpp` |
| routing -> dmap | `pass/passroutingtodmap/routingtodmap.cpp` |
| dmap -> dmaphop | `pass/passdmaptodmaphop/dmaptodmaphop.cpp` |
| dmaphop -> routinghw | `pass/passdmaphoptoroutinghw/passdmaphoptoroutinghw.cpp` |
| XAie API verification | `.cursor/skills/xaieapiverify/SKILL.md` |
| XAie constraint tables | `.cursor/skills/xaieapiverify/references/validation-rules.md` |
| Op mapping reference | [references/op-mapping.md](references/op-mapping.md) |

All file paths are relative to `src/mlir/mlirfront/tilinglinalg/` unless otherwise noted.
