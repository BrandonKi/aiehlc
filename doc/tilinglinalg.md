# TilingLinalg — MLIR Progressive IR for Multi-Tile GEMM Offloading

TilingLinalg lowers a `linalg.matmul` through six custom MLIR dialects into
host C++ (`host.cc`), kernel C++ (`kernel.cc`), and routing C++ (`routing.cc`)
that run a multi-tile GEMM on AMD Versal AI Engine hardware.

---

## 1. Pipeline Overview

```
linalg.matmul  (tiled via transform dialect)
    │
    ▼
┌─────────┐   routing (abstract mesh, partition, broadcast, gather)
│ Path A  │──▶ RoutingUnrollingLower → RoutingLower → RoutingHWLower
│routing.cc│   → RoutingDeadArg → RoutingConstantFold → EmitC → routing.cc
└─────────┘
    │
    ▼  Path B (main host + kernel flow)
┌─────────────────────────────────────────────────────────────────────┐
│ RoutingUnrollingLower                                               │
│   → RoutingToDmap                                                   │
│     → DmapToDmaphop                                                 │
│       → DmaphopTodfscheblueprint                                    │
│         ├─ BlueprintToSchedule → ScheduleCanonicalize               │
│         │    → DfscheduleToApi → RoutingConstantFold → host.cc      │
│         └─ BlueprintToScheduleKernel                                │
│              → DfscheduleToKernelApi → kernel.cc                    │
└─────────────────────────────────────────────────────────────────────┘
```

**Entry point**: `tilinglinalg.cpp` creates a `linalg.matmul` (i8), tiles it
with `transform.tile_using_forall` (num_threads `{8,16}`, tile_sizes
`{64,64,32}`), then `tilingpass.cpp` (`TilingCodePass`) walks `scf.forall`
and `memref.subview` to build routing IR with `createroutingfuncByDim`.

---

## 2. Dialect Inventory

### 2.1 routing — Abstract Data Movement

Models data movement over an abstract mesh with partitioning and broadcast/gather.

| Op | Purpose |
|----|---------|
| `createhwmesh` | Create HW mesh (row, col) |
| `createscheduletensor` | Create schedule tensor with init data |
| `partitionmesh` | Partition mesh along axis |
| `partitiontensor` | Partition tensor (splitnum, splitdim, hw_axis_owner, replicate_on) |
| `extract_data` | Extract data slice from partitioned tensor |
| `extract_tiles` | Extract tile list from partitioned mesh |
| `RoutingCreate` | Structured routing region (scf_idx, Memo) with body |
| `createhwiowithtarget` | Create HW IO with target tile list |
| `movedatabyio` | Move data through IO to/from AIE |
| `routinggatherout` | Gather data from a tile list |
| `createtilearray` | Create tile array from row/col counts |
| `createdataio` | Create data IO (iotype, direction) |
| `createbroadcast` | Broadcast from IO to tile array |
| `yield` | Terminator for routing regions |

**Types**: `tilearray`, `dataio`
**Attrs**: `TileRangeAttr(rowNum, colNum)`, `TileRangeArrayAttr`
**Interface**: `TileInterface` — `getTileBase()` for base address

### 2.2 dmap — Logical Dataflow Streams

Models logical streams connecting IO engines, core groups, and port configurations.

| Op | Purpose |
|----|---------|
| `func` | Function with symbol table |
| `create_data` | Create data (shape, element_type) |
| `define_core_group` | Define core engine group |
| `define_io_engine` | Define IO engine (SHIM/MEM) |
| `define_port_configure` | Port config with access pattern |
| `create_core_group_with_config` | Core group with config map |
| `create_io_engin_with_config` | IO engine with access pattern |
| `createstream` | Create stream src→dst |
| `create_chain_stream` | Chain multiple streams (multi-hop) |
| `createport_group` | Group destination ports |
| `push` | Push data into stream |
| `pull` | Pull data from stream |
| `yield` | Terminator |

**Types**: `dmacoreenginegroup`, `dmapioenginetype`, `dmapport`, `dmapportgroup`,
`dmapportstream`, `dmapportchainstream`, `dmapdata`, `dmapioconfig`, `dmapportconfig`
**Enums**: `dmapdirection` (SEND/RECEIVE), `dmapio_enum` (SHIMIO/MEMTILEIO/CORETILEIO)
**Attrs**: `dataaccesspattern` (direction, sizepertran, pingpongbuffer, slidingaccess),
`dataconfigmap` (array of dataconfmapitem)

### 2.3 dmaphop — Physical Tile-to-Tile Hops

Models physical tiles, ports, and hop-by-hop paths.

| Op | Purpose |
|----|---------|
| `func` | Function with symbol table |
| `tile` | Physical tile (tiletype, col, row) |
| `port` | Port on tile (direction, channel) |
| `create_hop` | Hop from source port to destination port |
| `create_path` | Path from hops with producers, consumers, tee_points |
| `alloc_buffer` | Allocate buffer on tile |
| `push` | Push data into path to consumer buffers |
| `pull` | Pull data from path from producer buffers |
| `sync` | Sync path transfers |
| `dealloc_buffer` | Deallocate buffer |

**Types**: `tile`, `port`, `hop`, `path`, `dmacoreenginegroup`
**Attrs**: `DirectionAttr` (In/Out), `portaccesspattern` (direction, transize, totalsize, pingpongbuffer, slidingaccess)

### 2.4 dfscheblueprint — Schedule Blueprint

High-level description of flow configurations and transfers before host/kernel split.

| Op | Purpose |
|----|---------|
| `config` | Top-level schedule blueprint container |
| `transfer_manifest` | Transfer manifest (payload_slice, packet_id, source, destinations) |
| `tile_group` | Tile group (shim_gateway, compute_row) |
| `declare_data` | Declare logical data with init tensor |
| `partition` | Partition data into views |
| `extract` | Extract partition from view |
| `data_slice` | Wrap tensor slice with symbol |
| `flowconfig` | Bind resource to data view |
| `flow_transfer` | Collective transfer between bindings |

**Types**: `Blueprint`, `Config`, `Manifest`, `DataHandle`, `ViewHandle`
**Attrs**: `bp_direction` (MM2S/S2MM/BOTH), `DMAAttr` (channels, direction),
`SliceAttr` (data_type, offset, size, stride), `EndpointAttr` (row, col, direction, channel)

### 2.5 dfschedule — Host + Kernel Schedule

Final dialect before EmitC lowering. Contains both host schedule ops and kernel ops.

**Host ops:**

| Op | Purpose |
|----|---------|
| `host` | Host block (config, kernel launch, schedule) |
| `declaretensor` | Declare tensor and allocate device mem |
| `declaretile` | Declare tile (col, row) |
| `packet` | Packet (data, dma_channel) |
| `declare_kernel_config` | Kernel config for multiple tiles |
| `config.dma_bd` | Configure DMA buffer descriptor |
| `config.create_io` | Create IO handle |
| `config.load_kernel_group` | Load kernel group to tiles |
| `schedule.launch_kernel_group` | Launch kernel group |
| `schedule.getbdid` | Get BD ID for tile |
| `schedule.start_io` | Start IO |
| `schedule.wait` | Wait for events |

**Kernel ops:**

| Op | Purpose |
|----|---------|
| `dskernel.compute` | Compute kernel logic |
| `dskernel_receiver` | Receiver kernel with ping-pong buffering |
| `dskernel.lock_init` | Init lock |
| `dskernel.launch_dma_s2m_loop` | Launch DMA S2M loop |
| `dskernel.acquire_lock` | Acquire lock |
| `dskernel.release_lock` | Release lock |
| `core.compute` | Invoke compute logic |
| `kernel.read_tile_config` | Read tile config |
| `kernel.memalloc` | Allocate kernel memory |

**Config accessors**: `get_packet_id`, `get_dma_channel`, `get_buffer_addr`,
`get_buffer_mode`, `get_num_buffers`, `get_*_lock_id`

**Types**: `event`, `stream`, `packet`, `lock`, `tile`, `kernelgroup`,
`kernel_config`, `tile_config`, `compute`, `bd_handle`, `io_handle`,
`dma_channel`, `sync_buffer`, `input_window`, `output_window`

### 2.6 routinghw — Physical Routing (Path A only)

Physical stream switch and shim port configuration. Used only for `routing.cc`
(Path A), not in the main host/kernel flow.

| Op | Purpose |
|----|---------|
| `tilearrayhandlecreate` | Create tile array handle |
| `tilecreate` | Create tile (row, col, comments) |
| `ioshimtilecreate` | Create IO shim tile (row, col, IOID, dmadirection) |
| `connectiototile` | Connect IO shim → dest tile |
| `connectiototilearray` | Connect IO shim → tile array |
| `createshimstreamswitchport` | Create shim stream switch port |
| `connectstreamswitchport` | Connect stream switch ports between tiles |
| `connectsinglestreamswitchport` | Connect single tile stream switch ports |
| `connectpktstreamswitchport` | Connect packet stream switch port |
| `enableexttoaieshimport` | Enable external→AIE shim port |
| `enableaietoextshimport` | Enable AIE→external shim port |

**Types**: `tilearrayhandle`

---

## 3. Pass-Level Transformations

### 3.1 RoutingUnrollingLowerPass

**Input**: routing + `scf.for` → **Output**: unrolled routing IR

- Adds `codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"]` to module
- Fully unrolls all `scf.for` loops when bounds/step are constant
- Uses `mlir::loopUnrollByFactor((ub-lb)/step)`
- Runs canonicalization after unrolling

### 3.2 RoutingToDmapPass

**Input**: routing → **Output**: dmap

- `movedatabyio` → `define_io_engine` + `define_core_group` + `define_port_configure` + `createstream` + `push`/`pull`
- `routinggatherout` → `pull`
- `createhwiowithtarget` → `create_io_engin_with_config`
- Optional MEM tile path when `moplevel == 1` (creates `create_chain_stream`)
- Erases: `createdataio`, `createtilearray`, `createhwmesh`, `partitionmesh`, `extract_tiles`

### 3.3 DmapToDmaphopPass

**Input**: dmap → **Output**: dmaphop

- `push`/`pull` → `tile` (shim/core/mem) + `port` + `create_hop` chain + `create_path` + `push`/`pull` + `sync`
- Uses `RoutingTopology` for shim column/channel allocation
- Supports direct SHIM→CORE and 2-hop via MEM tile

### 3.4 DmaphopTodfscheblueprintPass

**Input**: dmaphop + routing tensors → **Output**: dfscheblueprint

- `push` → `ConfigOp` + `TileGroupOp` + `DeclareDataOp` + `FlowConfigOp` + `FlowTransferOp` (one_to_many)
- `pull` → same structure (many_to_one)
- `createscheduletensor` → `DeclareDataOp`
- Erases: `tile`, `port`, `create_hop`, `create_path`, `alloc_buffer`, `sync`

### 3.5 BlueprintToSchedulePass (host)

**Input**: dfscheblueprint → **Output**: dfschedule (host)

- `FlowTransferOp` → `DeclareTensorOp` + `DeclareTileOp` + `ConfigDmaBdOp` + `ConfigCreateIoOp` + `GetBdIdOp` + `StartIoOp`
- Core tiles → `DeclareKernelConfigOp` + `LoadKernelGroupOp` + `LaunchKernelGroupOp`
- `ScheduleWaitOp` on DMA + kernel events
- Uses `KernelResourceManager` for BD/lock ID allocation

### 3.6 BlueprintToScheduleKernelPass (kernel)

**Input**: dfscheblueprint → **Output**: dfschedule (kernel)

- Same FlowTransfer conversion as host, targeting kernel path
- Generates `KernelModuleOp` with `KernelConfigDefOp`, `LockDefOp`, `BufferDefOp`, `WindowDefOp`, `KernelDeclOp`, `KernelMainOp`

### 3.7 ScheduleCanonicalizePass

**Input/Output**: dfschedule → dfschedule (reorganized)

- Separates shim vs core tiles, deduplicates slice parameters
- Reorganizes ops for host vs kernel separation
- Consolidates schedule structure for API lowering

### 3.8 DfscheduleToApiPass → host.cc

**Input**: dfschedule → **Output**: EmitC → C++

- Converts `dfschedule` ops to `__Runtime_*` API calls
- Manages `PartitionTensor` and `ExtractSlice` for host data layout
- `arith.constant` (dense tensors) → `emitc.constant` or global arrays

### 3.9 DfscheduleToKernelApiPass → kernel.cc

**Input**: dfschedule kernel modules → **Output**: EmitC → C++

- `KernelConfigDefOp` → `#include`, `#define BUF_SZ`, `#define FOR_READ/FOR_WRITE`
- `LockDefOp` → `#define LOCK_*`
- `BufferDefOp` → `v4int32 buf[BUF_SZ];`
- `KernelDeclOp` → `#include "compute_kernel.cc"`
- `KernelMainOp` → `main()` with `window_init`, `kernel_invoke`, `done()`, `return`

### 3.10 Path A: RoutingLowerPass → RoutingHWLowerPass

Used only for `routing.cc` generation (stream switch configuration C code).

**RoutingLowerPass**: routing → routinghw
- `movedatabyio` → `TileCreate` + `IOShimTileCreate` + `ConnectStreamSingleSwitchPort` + `ConnectStreamPktSwitchPort`
- Uses `RoutingTopology` for shim allocation and `GetSeqPath`/`ParseTheCCTRoutingPath` for stream switch paths

**RoutingHWLowerPass**: routinghw → EmitC
- `EnableExtToAieShimPort` → `XAie_EnableShimDmaToAieStrmPort`
- `EnableAieToExtShimPort` → `XAie_EnableAieToShimDmaStrmPort`
- `ConnectStreamSingleSwitchPort` → `XAie_StrmConnCctEnable`
- `ConnectStreamPktSwitchPort` → `XAie_StrmPktSwSlaveSlotEnable` + `XAie_StrmPktSwMstrPortEnable`

### 3.11 Utility Passes

| Pass | Purpose |
|------|---------|
| `RoutingDeadArgPass` | Remove unused function arguments and update call sites |
| `RoutingConstantFoldPass` | Fold constant operands into `emitc.call` args, remove dead EmitC ops |

---

## 4. Routing Implementation Engine

Located in `pass/routingimplement/`.

### RoutingTopology

High-level routing API over `ResourceMgr`. Parameterized by generation (e.g. `"Gen2"`).

- `createDataIO(name, loc, direction)` — allocate shim column/channel
- `createPath(dioID, dsttiles)` — build routing path from shim to cores
- `ReserveTiles`, `occupyLink`, `occupyPointDirection`

### RoutingPath

2D BFS mesh tree router. Priority: Memory > SHIM > Core. Row-0 no
horizontal movement. Produces `MultiPath` tree of branches from source to sinks.

### ResourceManager

Tracks tile/link/port usage across the mesh.

- `RoutingTile` — master/slave port banks per direction
- `ShimTile` — MM2S/S2MM channel allocation
- `DataIO` — shim location, channel, direction
- `freeShimNoc()` — find free shim column/channel
- `occupyLink(a, b, ioId)` — reserve link between tiles

---

## 5. Test Flow

### 5.1 test.cpp CLI Modes

| Mode | Command | Action | Output |
|------|---------|--------|--------|
| dfschedule | `./test dfschedule` | Full host+kernel pipeline | `worklocal/host.cc`, `worklocal/kernel.cc` |
| hw | `./test hw` | Routing Path A | `worklocal/routing.cc` |
| test | `./test test` | Path contiguity check | Console only |
| default | `./test` | Both dfschedule + hw | All three files |

### 5.2 Build

```bash
cd src/mlir/mlirfront/tilinglinalg/pass/unitest
mkdir -p build && cd build
cmake ..    # uses LLVM from /usr/local or LLVM_INSTALL_DIR
make -j4
```

TableGen target `routingdilaect` runs `gen.sh` for all six custom dialects.

### 5.3 Generate host.cc + kernel.cc + routing.cc

```bash
cd build
./test dfschedule    # → worklocal/host.cc, worklocal/kernel.cc
./test hw            # → worklocal/routing.cc
# or just: ./test    # generates all three
```

### 5.4 Compile Kernel (xchesscc flow)

```bash
WORKLOCAL_DIR="$(pwd)/aout/worklocal" source script/compile_kernel.sh dskernel_receiver
```

Internally calls `script/kc.sh`:
1. `xchesscc` — `kernel.cc` → `kernel_orig.ll` (AIE2PS flags, **always-on `-g`** for DWARF)
2. `opt` — two `xlopt` passes on LLVM IR (preserves debug metadata)
3. `xchessmk` — `aie2ps.prx` + IR → `build/kernel` (AIE ELF, carries `.debug_line`)
4. `readelf --debug-dump=decodedline` → `build/kernel.decodedline.txt`, then
   `script/parse_linemap.py` → `build/kernel.linemap.json` (PC→source map)
5. `ld` — `aarch64-none-elf-ld -EL -r -b binary build/kernel -o kernel.o`
6. `objcopy` — rename symbols to `_binary_kernel_<func>_{start,end,size}`

Output: `build/kernel.o` (relocatable for host linking)

#### DWARF line info / PC→source debug

`-g` is always enabled in the three xchesscc flag strings (`kc.sh`, routed through
`+Wllvm,-O2,-g,...`), so the kernel ELF embeds a `.debug_line` table referencing
`kernel.cc`. The build emits two extra artifacts next to the ELF:

| Artifact | Producer | Use |
|----------|----------|-----|
| `build/kernel.decodedline.txt` | `readelf --debug-dump=decodedline` | human-readable addr→file:line |
| `build/kernel.linemap.json` | `script/parse_linemap.py` | sorted addr→{file,line} for tooling |

> **Artifact location (tiling pipeline):** the kernel is compiled by `kc.sh` via
> `aiehlc.sh → script/hostcompile.sh → script/compile_kernel.sh`, which build directly in
> `WORKLOCAL_DIR` (set to `aout/worklocal` by `aiehlc.sh`). The ELF and DWARF artifacts
> therefore land under **`aout/worklocal/build/`**. Verified end-to-end on
> `example/tileprogram/ccode/simplematmul2.cc` (kernel ELF + `.debug_line`,
> 412-entry `kernel.linemap.json`, offline `aiediag pc` lookup all OK).
> The standalone unitest flow (`piplinerun.sh`) sets `WORKLOCAL_DIR=unitest/worklocal`,
> so it builds under `unitest/worklocal/build/` instead.
> Note: `simpleconv2d.cc` currently aborts at `xchessmk` (no ELF produced) for a
> separate, non-`-g` reason — track that build issue independently.

Verify DWARF is present:

```bash
readelf -S build/kernel | grep debug_line                 # section exists
readelf --debug-dump=decodedline build/kernel | head      # non-empty, kernel.cc lines
grep -c DILocation build/kernel.ll                          # > 0: -g survived xchesscc+xlopt
python3 -c "import json;json.load(open('build/kernel.linemap.json'))"  # JSON parses
```

`.debug_line` parsing is architecture-agnostic, so the system (or aarch64) `readelf`
works on the AIE ELF. The artifacts are best-effort: a missing/empty table only warns,
the kernel build still succeeds. With `-O2` the line table is approximate (inlining/
optimization) — treat PC→line as coarse. If a future toolchain rejects bare `-g`, fall
back to `+Wllvm,-g`; if `xchessmk` strips DWARF at link, add a debug flag to the PRX
`llvm.xargs` in `kernelconfig.h`.

Map a stuck core's program counter back to source with `aiediag pc` (reads the AIE2PS
core PC register `0x30F00`, 20-bit value mask, and bisects the line map):

```bash
python3 src/tool/debug/aiediag.py pc <col> <row> startcol <s> -dev pal     # live HW read
python3 src/tool/debug/aiediag.py pc <col> <row> --pc 0x1234 --linemap build/kernel.linemap.json
```

`aiediag dig` also prints `Core PC=0x.... -> kernel.cc:<line>` for core tiles when the
line map is found.

### 5.5 Compile Host + Link

```bash
WORKLOCAL_DIR="$(pwd)/aout/worklocal" source script/hostcompile.sh
```

Steps:
1. Source `script/compile_kernel.sh` (kernel.o)
2. Fix `host.cc` → `host_fixed.cc` (sed: add forward decl, fix `void main→int main`)
3. `aarch64-none-elf-g++ -Os -std=c++17 -DAIE_GEN=5 -mcpu=cortex-a78 -c host_fixed.cc -o host.o`
4. `aarch64-none-elf-g++ ... -c aie_runtime.c -o aie_runtime.o`
5. Compile `routing.cc` if present → `routing.o`
6. Link all objects with `libxil`, `libaienginev2`, stdlib → `build/host` (baremetal ELF)

### 5.6 HW Execution via Remote Login (apppaltest.py)

Prerequisites: `PALIP`, `USERNAME`, `BOARDNAME` (or source `script/test/envlocal.sh`)

```bash
python3 script/test/apppaltest.py ./worklocal/build/host
```

Sequence:
1. **Copy ELF** to remote: `scp host → /home/{user}/aiehlc/`
2. **SSH Connection 1** (XSDB):
   - `systest` → `become "{boardname}"`
   - `power 0` → `power 1` (board power cycle)
   - Launch `xsdb`:
     - `conn` → `tar 1` → `device program {PDI}`
     - `tar 20` → `rst -proc`
     - `dow -force {elf}` → `con`
3. **SSH Connection 2** (Console):
   - `systest-client` → `connect com0`
   - Read console output for 10 seconds
4. **Cleanup**: `exit` xsdb, `power 0`

### 5.7 Verification Criteria

| Result | Console pattern |
|--------|----------------|
| **PASS** | Contains `device_teardown done` or `device_init OK` |
| **FAIL** | Contains `AIE ERROR`, `Invalid Tile Type`, or `Cannot find Tile Type` |

Automated via `verify_host.sh`:
```bash
bash .cursor/skills/hostcodegen/scripts/verify_host.sh ./worklocal/build/host
# Logs to script/test/.verify_host_console.log
# Exit 0 = pass, exit 1 = fail
```

### 5.8 piplinerun.sh (End-to-End Automation)

```bash
cd src/mlir/mlirfront/tilinglinalg/pass/unitest
source piplinerun.sh
```

| Step | Action |
|------|--------|
| 1 | `source script/setup.sh --path-set-only` + `source script/aiehlc.sh --aielib-only --aie-version 5` |
| 2 | `mkdir -p build` → `cmake ..` → `make -j4` |
| 3 | `./test dfschedule` → `host.cc`, `kernel.cc` |
| 3b | **Codegen validation**: grep generated files for `XAie_*` / `__Runtime_*` API calls. If none found, clean files, rebuild, and retry (up to 3 attempts). Exits with error if all retries fail. |
| 4 | `WORKLOCAL_DIR=$(pwd)/worklocal source ../../../../../../script/hostcompile.sh` → `worklocal/build/host` |
| 5 | `source envlocal.sh` → `apppaltest.py ./worklocal/build/host` |

---

## 6. Key File Paths

| Component | Path |
|-----------|------|
| IR entry point | `tilinglinalg.cpp`, `tilingpass.cpp` |
| Dialects | `routing/`, `routinghw/`, `dataflowmap/{dmap,dmaphop,dfscheblueprint,dfschedule}/` |
| Passes | `pass/routing*.cpp`, `pass/pass*/` |
| Routing engine | `pass/routingimplement/` |
| Test driver | `pass/unitest/test.cpp` |
| Build | `pass/unitest/CMakeLists.txt` |
| Automation | `pass/unitest/piplinerun.sh` |
| Host compile | `script/hostcompile.sh` |
| Kernel compile | `script/compile_kernel.sh`, `script/kc.sh` |
| HW run | `script/test/apppaltest.py` |
| Verification | `.cursor/skills/hostcodegen/scripts/verify_host.sh` |
| Example routing | `pass/routingimplement/codegenexample/aie_control.cpp` |

All paths relative to `src/mlir/mlirfront/tilinglinalg/` unless otherwise noted.
