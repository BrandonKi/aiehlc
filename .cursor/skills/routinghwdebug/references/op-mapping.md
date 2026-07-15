<!-- Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->

# Op Mapping: Routing Dialect Pipeline

Maps operations across each dialect stage so errors in generated C++ can be traced back to their origin.

All file paths relative to `src/mlir/mlirfront/tilinglinalg/`.

---

## 1. C++ (routing.cc) to EmitC

| C++ call | EmitC op |
|----------|----------|
| `XAie_StrmConnCctEnable(dev, loc, slv, slvPort, mstr, mstrPort)` | `emitc.call_opaque "XAie_StrmConnCctEnable"(dev, loc, slv, slvPort, mstr, mstrPort)` |
| `XAie_StrmPktSwSlaveSlotEnable(dev, loc, slv, slvPort, slot, pkt, mask, msel, arb)` | `emitc.call_opaque "XAie_StrmPktSwSlaveSlotEnable"(...)` |
| `XAie_StrmPktSwMstrPortEnable(dev, loc, mstr, mstrPort, drop, arb, mselEn)` | `emitc.call_opaque "XAie_StrmPktSwMstrPortEnable"(...)` |
| `XAie_EnableAieToShimDmaStrmPort(dev, loc, port)` | `emitc.call_opaque "XAie_EnableAieToShimDmaStrmPort"(...)` |
| `XAie_EnableShimDmaToAieStrmPort(dev, loc, port)` | `emitc.call_opaque "XAie_EnableShimDmaToAieStrmPort"(...)` |
| `XAie_TileLoc(col, row)` | `emitc.call_opaque "XAie_TileLoc"(col, row)` |
| `getOrCreateDeviceInstance()` | `emitc.call_opaque "getOrCreateDeviceInstance"()` |

Lowering pass: `pass/routinghwlower.cpp` (RoutingHWLowerPass)

---

## 2. EmitC to routinghw

| EmitC call | routinghw op | Key attributes |
|------------|-------------|----------------|
| `XAie_StrmConnCctEnable` | `routinghw.ConnectStreamSingleSwitchPort` | tile, slave_dir, slave_port, master_dir, master_port |
| `XAie_StrmPktSwSlaveSlotEnable` + `XAie_StrmPktSwMstrPortEnable` | `routinghw.ConnectStreamPktSwitchPort` | tile, slave_dir, slave_port, master_dir, master_port, pkt_id, arbitor, msel |
| `XAie_EnableAieToShimDmaStrmPort` | `routinghw.EnableAieToExtShimPort` | tile, port |
| `XAie_EnableShimDmaToAieStrmPort` | `routinghw.EnableExtToAieShimPort` | tile, port |

Lowering pass: `pass/routinghwlower.cpp`

Patterns (key classes in routinghwlower.cpp):
- `EnableExtToAieShimPortpattern` -> `XAie_EnableShimDmaToAieStrmPort`
- `EnableAieToExtShimPortpattern` -> `XAie_EnableAieToShimDmaStrmPort`
- `ConnectStreamSingleSwitchPortpattern` -> `XAie_StrmConnCctEnable`
- `ConnectStreamPktSwitchPortpattern` -> `XAie_StrmPktSwSlaveSlotEnable` + `XAie_StrmPktSwMstrPortEnable`

---

## 3. routinghw to dmaphop (Path B: routing.cc)

| routinghw op | dmaphop origin | How tile/port is determined |
|-------------|----------------|----------------------------|
| `ConnectStreamSingleSwitchPort` | `dmaphop::create_hop` chain | BFS path from RoutingPath; each hop segment produces one circuit-switched connection |
| `ConnectStreamPktSwitchPort` | `dmaphop::create_hop` (packet segment) | Packet switch at source/destination tiles; arbitor/msel from allocation |
| `EnableAieToExtShimPort` / `EnableExtToAieShimPort` | `dmaphop::create_path` (shim endpoint) | Shim DMA port from DataIO allocation |

Lowering pass: `pass/passdmaphoptoroutinghw/passdmaphoptoroutinghw.cpp`

The tile location and port direction come from the BFS path stored in the dmaphop IR. The lowering pass reads the hop chain and emits one routinghw op per hop segment.

---

## 4. dmaphop to dmap

| dmaphop op | dmap origin |
|-----------|-------------|
| `dmaphop::tile` | Created for each tile in the BFS path |
| `dmaphop::port` | Created for source/destination ports |
| `dmaphop::create_hop` | From `dmap::push` / `dmap::pull` lowering |
| `dmaphop::create_path` | Wraps the full source-to-destination path |
| `dmaphop::alloc_buffer` | From `dmap::create_data` |
| `dmaphop::sync` | From `dmap::sync` |

Lowering pass: `pass/passdmaptodmaphop/dmaptodmaphop.cpp`

Key patterns: `PushOpLowering`, `PullOpLowering`. These use `RoutingTopology` to find BFS paths and create the hop chain. The BFS result determines which tiles and port directions appear in the dmaphop IR.

---

## 5. dmap to routing

| dmap op | routing origin |
|---------|---------------|
| `dmap::port` | `routing::movedatabyio` / `routing::routinggatherout` |
| `dmap::stream` | `routing::broadcast` (multicast) |
| `dmap::push` / `dmap::pull` | `routing::movedatabyio` data movement direction |
| `dmap::create_data` | `routing::data` |
| `dmap::define_io_engine` | `routing::ioengine` |

Lowering pass: `pass/passroutingtodmap/routingtodmap.cpp`

---

## 6. routinghw to routing (Path A: routing_hw.cc)

| routinghw op | routing origin |
|-------------|---------------|
| `routinghw.ConnectStreamSingleSwitchPort` | `routing::movedatabyio` via RoutingLowerPass |
| `routinghw.ConnectStreamPktSwitchPort` | `routing::broadcast` via RoutingLowerPass |
| `routinghw.TileCreate` | `routing::tile` |

Lowering pass: `pass/routinglower.cpp` (RoutingLowerPass)

---

## 7. Where tile locations and port assignments originate

The critical chain for debugging port errors:

```
hwresource.cpp (port template: how many ports each tile type has)
    -> ResourceManager.cpp (linkAvailable: checks port counts)
        -> routingpath.cpp (BFS: finds path through tiles, uses linkAvailable)
            -> routingtopology.cpp (wraps BFS, exposes to passes)
                -> dmaptodmaphop.cpp / routingtodmap.cpp (calls topology.findPath)
                    -> passdmaphoptoroutinghw.cpp / routinglower.cpp (emits routinghw ops with tile/port)
                        -> routinghwlower.cpp (emits EmitC XAie calls)
                            -> translateToCpp (emits C++)
```

Port direction and port number are determined at the BFS step in `routingpath.cpp`. The BFS selects a direction based on tile adjacency, and the port number is assigned by `ResourceManager::occupyLink()` which picks the first available port slot.

If the port template in `hwresource.cpp` says a tile type has N ports for a direction, the BFS and ResourceManager will allow routing through that direction. If the actual hardware has 0 ports for that direction, the generated code will contain invalid API calls.

---

## 8. Tracing example: MemTile EAST/WEST error

Error: `XAie_StrmConnCctEnable(..., TileLoc(1,1), NORTH, 0, EAST, 0)` — MemTile has no EAST ports.

Trace:
1. **C++** line 18: `XAie_StrmConnCctEnable(..., NORTH, 0, EAST, 0)` on tile(1,1)
2. **EmitC**: `emitc.call_opaque "XAie_StrmConnCctEnable"` with EAST master direction
3. **routinghw**: `ConnectStreamSingleSwitchPort` with master_dir=EAST on tile(1,1)
4. **dmaphop**: A hop segment from tile(1,1) to tile(2,1) going EAST was created
5. **BFS**: `routingpath.cpp` found a path from column 1 to column 2 at row 1 (MemTile row)
6. **ResourceManager**: `linkAvailable(tile(1,1), tile(2,1))` returned true because Mem tile's EAST bank has 4 ports
7. **Root cause**: `hwresource.cpp` `defaultPortTemplates()` gives Mem tiles 4 EAST master + 4 EAST slave ports, but AIEML MemTiles have 0 EAST/WEST stream switch ports
