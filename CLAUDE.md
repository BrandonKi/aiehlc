## Role
- MLIR and Spatial computing expert

# AIEHLC Project Architecture

AIEHLC (AIE High-Level Compiler) is a compilation/deployment solution for AMD Versal AI Engine. It has **two parts**:

1. **aiehlc** — Clang-based tool that compiles C++ (AIE driver C API) into host + kernel binaries for single-kernel AIE apps.
2. **tilinglinalg** — MLIR progressive lowering pipeline that offloads a GEMM operation across multiple AIE tiles, generating host, kernel, and routing C++ code through 6 custom dialects.

Both share the AIE runtime (`include/aie_runtime.h`) and target Versal AI Core Series with pre-built PDI.

## Directory Layout

```
aiehlc/
├── include/                   # aie_runtime.h, aie_device_map.h
├── src/
│   ├── llvm/aiehlc.cc         # Main aiehlc Clang tool
│   └── mlir/
│       ├── runtime/aie_runtime.c  # Runtime wrappers over XAie_* APIs
│       └── mlirfront/
│           ├── AieFrontEnd.cc     # Clang AST → MLIR
│           ├── AieDialect.cc      # AIE dialect (LoadKernel, etc.)
│           └── tilinglinalg/      # ★ Multi-tile GEMM pipeline
│               ├── routing/       # Abstract routing dialect
│               ├── routinghw/     # Physical routing dialect
│               ├── dataflowmap/   # dmap, dmaphop, dfscheblueprint, dfschedule dialects
│               └── pass/          # All lowering passes + unitest/
├── script/                    # setup.sh, aiehlc.sh, kc.sh, test/apppaltest.py
├── example/                   # AIE examples (perf, matmul, multi-kernel)
└── thirdparty/alib/           # XAie driver (excluded from this doc)
```

## Part 1: aiehlc (Single Kernel)

Compiles a user C++ file into a host ELF (ARM) + kernel ELF (AIE core):

```
User C++ → aiehlc (Clang AST) → AieFrontEnd (MLIR)
  → Kernel: xchesscc → xchessmk → kernel ELF (embedded in host via ld -r -b binary)
  → Host: aarch64-g++ (host.cc + aie_runtime.c + routing + kernel.o) → host ELF
```

Key runtime API: `__Runtime_device_init`, `__Runtime_load_kernel_group`, `__Runtime_launch_kernel_group`, `__Runtime_dma_bd_config`, `__Runtime_wait_event`, `__Runtime_device_teardown`.

Platforms: baremetal (`aarch64-none-elf-g++`) or Linux (`aarch64-linux-gnu-g++`).

## Part 2: tilinglinalg (Multi-Tile GEMM)

### Six Custom MLIR Dialects

| Dialect | Purpose |
|---------|---------|
| **routing** | Abstract tile arrays, data IO, broadcast, mesh partitioning |
| **routinghw** | Physical tiles, stream switch ports, packet flows |
| **dmap** | Logical dataflow: ports, streams, push/pull |
| **dmaphop** | Physical hops: tile-to-tile paths, DMA, buffers |
| **dfscheblueprint** | Schedule blueprint: transfer manifests, flow configs |
| **dfschedule** | Executable schedule: DMA BD, kernel launch, locks |

### Pass Pipeline

**Shared stages** (produces dfscheblueprint IR, then module is cloned):

1. `RoutingUnrollingLowerPass` — unroll abstract routing into per-tile ops
2. `RoutingToDmapPass` — routing → logical dataflow
3. `DmapToDmaphopPass` — logical → physical hops
4. `DmaphopTodfscheblueprintPass` — hops → schedule blueprints

**Host path** → `host.cc`:

5. `BlueprintToSchedulePass` → `ScheduleCanonicalizePass` → `DfscheduleToApiPass` → `RoutingConstantFoldPass` → `CanonicalizerPass` → EmitC → `host.cc`

**Kernel path** → `kernel.cc`:

5. `BlueprintToScheduleKernelPass` → `DfscheduleToKernelApiPass` → EmitC → `kernel.cc`

**Routing path** (alternative, Path A) → `routing.cc`:

1. `RoutingUnrollingLowerPass` → `RoutingLowerPass` → `RoutingHWLowerPass` → `RoutingDeadArgPass` → `RoutingConstantFoldPass` → `CanonicalizerPass` → EmitC → `routing.cc`

### Routing Implementation (`pass/routingimplement/`)

- **RoutingTopology**: Gen2 AIE tile topology model
- **RoutingPath**: BFS path finding (priority: Memory > SHIM > Core)
- **ResourceManager**: Tracks link/port usage to avoid conflicts

## Build

```bash
# Main aiehlc binary
mkdir build && cd build && cmake .. -DLLVM_INSTALL_DIR=/path/to/llvm/build && make -j$(nproc)

# TilingLinalg unitest (standalone)
 source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simplematmul2.cc
 python3 ./script/test/apppaltest.py -y  -nonreboot > ./applog 2>&1
```

Each dialect has `td/` (TableGen), `gen.sh` (runs mlir-tblgen), and `inc/` (generated .inc files).

## Test and Verification

### Unitest CLI 

```bash
source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simplematmul.cc
```

### End-to-End Flow

```
1. Generate    source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simplematmul.cc
2. HW run      python3 script/test/apppaltest.py aout/worklocal/build/host  → SSH+xsdb+console
3. Verify      script/test/verify_host.sh → pass: "device_teardown done", fail: "AIE ERROR"
```

### Per-Dialect Unit Tests

Each dialect has its own `unitest/` directory with independent CMake build:
- `routing/unitest/`, `routinghw/unitest/`
- `dataflowmap/{dmap,dmaphop,dfschedule,dfscheblueprint}/unitest/`
- `pass/routingimplement/{routing,hw}/unitest/`

## Additional Documentation

- **[doc/module_analysis.md](doc/module_analysis.md)** — 9-module breakdown of the AIEHLC project: M1 (aiehlc driver), M2 (MLIR frontend), M3 (pipeline orchestrator), M4 (spatial routing dialects), M5 (dataflow mapping dialects), M6 (routing engine), M7 (schedule generation), M8 (AIE runtime), M9 (build & test infra). Includes per-module key files, line counts, inputs/outputs, dependencies, and data flow diagram
- **[doc/aieapi.md](doc/aieapi.md)** — XAie driver API guide covering: simple single-tile flow (`aieml_perf.cc`), multi-tile manual routing/DMA/locks (`aie_control.cpp`), and production XAie call patterns from `common_layer/aeg_runtime_api.cpp` (transaction batching, broadcast core enable, BD recycling, ping-pong RTP, advanced DMA config)
- **[doc/tilinglinalg.md](doc/tilinglinalg.md)** — TilingLinalg deep dive: all 6 custom dialects (ops, types, attrs), 15 pass transformations, routing engine internals, and complete test/build/HW-run/verification flow including remote board login
- **[doc/lowering.md](doc/lowering.md)** — Concrete IR lowering trace: walks a 2x2 mesh / 16x16 tensor through every dialect stage (routing→dmap→dmaphop→blueprint→dfschedule→EmitC and routing→routinghw→EmitC) with actual IR snippets, op-to-API mapping tables, tile/lock/BD allocation, and data partitioning breakdown
- **[doc/design/tile_dim_structured_design.md](doc/design/tile_dim_structured_design.md)** — Structured `tile_dim` (size/stride/groups) design for `aie::SpatialPolicy`: replaces the flat scalar `tile_m/tile_n/tile_k` hints with a per-dimension descriptor and unifies the conv halo/overlap split (`TileMode::Overlap`) into the same field. Covers semantics (Partition vs Overlap, groups derivation, coverage validation), the implementation map across `aiehlc.cc` + `tilinglinalg_pipeline.cpp` (struct emitted as a string in THREE places), and migration notes
- **[doc/design/spatial_space_composition.md](doc/design/spatial_space_composition.md)** — Composition-based spatial op spaces: refactors the "fat" `aie::SpatialPolicy` into a lean generic policy plus per-op iteration spaces (`GemmSpace { policy; m,n,k }`, `Conv2dSpace { policy; oh,ow,oc,ic,kh,kw; stride,pad }`) that *compose* (not inherit) the policy. Covers the `auto` NTTP `port` change, per-port attachment, legacy-vs-composed AST detection (`field(0).isStruct()`), the lean SpatialPolicy field order, Conv2d input-geometry reconstruction, and the DmaTransform precedence rule (explicit wins over Conv2dSpace-derived im2col/spatial-halo)
- **Skill: xaieapiverify** (`.cursor/skills/xaieapiverify/SKILL.md`) — Agent-driven static verification of XAie API calls in generated code (routing.cc, host.cc). Checks port number limits, tile-type-aware PortVerify rules, packet switch consistency, DMA/Lock constraints, and routing path connectivity against driver validation rules
- **Skill: routinghwdebug** (`.cursor/skills/routinghwdebug/SKILL.md`) — End-to-end routing debug: scans routing.cc with xaieapiverify, traces errors backward through EmitC -> routinghw -> dmaphop -> dmap -> routing, examines routingimplement (hwresource port templates, BFS path finding, ResourceManager) to find root causes, and provides targeted fixes
- **Skill: dmabdverify** (`.cursor/skills/dmabdverify/SKILL.md`) — Static verification of DMA Buffer Descriptor configuration in generated host.cc. Checks BD IDs, lock IDs/values, ping-pong chaining, buffer lengths, packet IDs, and channel assignments against AIEML hardware limits. Catches DMA/lock issues before HW deployment
- **Skill: datacorrectness** (`.cursor/skills/datacorrectness/SKILL.md`) — Pre-HW-run checklist for data correctness: type width consistency, input/output direction, buffer bank assignment, address validity, tensor slice coverage, and lock protocol. Catches output-wrong/output-zero bugs before board runs
- **Skill: aiedriverkb** (`.cursor/skills/aiedriverkb/SKILL.md`) — AIE Driver Knowledge Base: structured reference for XAie driver APIs in `thirdparty/alib/aie-rt`. Covers ELF loading (KB-101: PT_LOAD-only segment loading, PM/DM bounds checking, file size vs loadable size distinction), and other driver internals. Use when analyzing HW logs or verifying assumptions about driver behavior
- **Skill: aiehwdmadebug** (`.cursor/skills/aiehwdmadebug/SKILL.md`) — Live AIE hardware DMA debug over JTAG/XSDB using aiediag (flow-aware), aiedbg (raw reg/mem ground truth), and aieshow (live grids). Connect via `export AIEDBG_TARGET=xsdb://xx.xx.xx.213:3121`. Decodes AIE2PS tile/shim DMA status registers from authoritative field maps, traces a stuck "started-but-never-finished" channel across the producer→hop→consumer chain. Documents the aiediag offset-as-value parsing bug (Pitfall 1): false BD_UNAVAIL/BD_INVALID from decoding the register *offset* `0x1DF10` instead of the *value* — always confirm with `aiedbg reg read`. Use when a board run hangs on DMA
- **Skill: aiesimloaddebug** (`.cursor/skills/aiesimloaddebug/SKILL.md`) — Debug AIE simulator (`aie2pssimmsm`) segfaults at PS.so load: the sim dies right after `AIE_WORK_DIR = ...` and before `AIEHLC PS IP started`, with no console error. Discriminator: grep the sim log for `PS IP started`/`__Runtime_init` — if absent, the crash is at PS.so load/elaboration. Most common root cause: a stale `script/sim/build/kernel_elf_init.cc` referencing the wrong embedded-kernel symbol `_binary_kernel_<name>_start` (its Makefile target was order-only, so it wasn't regenerated when the kernel changed; the tiling path didn't clean it) → undefined symbol → `dlopen(aiehlc_ps.so)` fails → segfault. Confirm with `nm -D -u aiehlc_ps.so | grep _binary_kernel_`. Also covers: missing untracked `example/aiesim_test/` stub source, stub not built under `--stub-tiles`, and that gdb is unusable here (crashes the working baseline identically via the vendor `libxv_isl_iostreams.so`). Minimal repro: `example/tileprogram/ccode/mini_tile.cc`
- **Command: data-mismatch-debug** (`.claude/commands/data-mismatch-debug.md`) — Systematic debug for DMA data mismatch: stream starvation, output-wrong, output-zero. Traces data volumes from shim tiles through core tiles to kernels (startio repeat, BD len/iter_wrap, kernel rounds), builds supply/demand tables, identifies root cause patterns (repeat mismatch, iter_wrap mismatch, lock credit exhaustion, BD chain length), and traces back to responsible pass
- **[doc/llvm_mlir_pitfalls.md](doc/llvm_mlir_pitfalls.md)** — Known LLVM/MLIR API pitfalls: ArrayRef dangling references, FileEntry/Rewrite crashes, StringRef lifetime, RTTI requirements, getBuffer null dereference, recursive TableGen, constant propagation pattern match, integer overflow, and MLIR SSA discipline
- **[script/verify_env.sh](script/verify_env.sh)** — Environment verification script: validates Vitis, LLVM, cross-compiler toolchain, aie-rt driver, BSP directories, Docker timezone, build directories, and PAL/board test variables. Run before build to catch setup issues early
- **[src/tool/debug/aiediag.py](src/tool/debug/aiediag.py)** — Flow-aware DMA diagnostic tool: reads DMA status registers via `aiedbg`, cross-references with provenance map JSONs (`dfscheduleprovenancemap.json`, `dmaphopprovenacemap.json`) and shim event status (`~/aiejson/shimtile_events.json`), identifies connected tiles and routing paths, and prints automated root-cause diagnosis for DMA stalls. See `README.md > Debug Tools` for full usage
- **[src/tool/debug/aiegdb.py](src/tool/debug/aiegdb.py)** — GDB-like scoped CLI frontend over `aiedbg` for live AIE debug. Stateful REPL keeps a *current scope* (partition → tile → channel); `target tile 0 0` / `tar tile(0,0)` and `target channel mm2s0` descend levels and the prompt/breadcrumb changes (`partition(startcol=3)/tile(0,0)/mm2s0>`). Commands default to the current scope: tile scope auto-injects `phys_col=col+startcol`, `row`; channel scope also auto-injects `dir/ch`. Imports `aiediag.py` as a library (offsets, `run_aiedbg_reg_read`, decoders, provenance) and shells out to `aiedbg -d <device>` for raw passthrough (unknown tokens pass through verbatim). Channel helpers: `dma status`, `bd` (JSON BD chain + live HW lengths), `event` (DMA start/finish/error), `dma counter` (AIE performance counters — reads MEM-module `0x11020/0x11024` on core tiles, PL-module `0x31020/24` on shim; decodes CONTROL0 start/stop event ids). Read-only by default; the only write is the explicit `dma counter setup [finished|started]`. `--dry-run` prints every `aiedbg` command without a board; `-c "cmd; cmd"` / `--script FILE` run non-interactively
- **[src/tool/debug/aiemcp.py](src/tool/debug/aiemcp.py)** — MCP (Model Context Protocol) server exposing `aiegdb` live AIE debug to Claude Code. A *third* front-end onto the same `AieGdb.run_line` dispatch (browser/daemon is the other; see `doc/design/aiegdb_live_debug_framework.md` §11): imports `aiegdb.AieGdb` **in-process** and keeps ONE long-lived instance so scope (partition → tile → channel) persists across tool calls. Tools: `aie_exec(cmd)` (primary — run any command line, e.g. `target tile 0 3`, `target channel mm2s0`, `dma status`, `bd`, `pc`), `aie_scope()` (`where`), `aie_commands()` (`?` — per-scope command discovery), `aie_help()` (`help`). Critical: FastMCP stdio uses fd 1 for JSON-RPC and `_passthrough` runs `aiedbg` uncaptured, so `_run()` does OS-fd-level `dup2` capture of fd 1/2 around `run_line()` (a lock serializes it) and traps `SystemExit`/`Exception` so a missing `aiedbg` never kills the server. Config via env (`.mcp.json`): `AIEDBG_TARGET`, `AIEMCP_DEVICE/STARTCOL/AIE_VERSION/JSON_DIR/DRY_RUN`; unset startcol/aie_version auto-resolve from provenance JSON. Registered as `aiegdb` in repo-root `.mcp.json`. Needs `pip install "mcp[cli]"`; smoke test `AIEMCP_DRY_RUN=1 mcp dev src/tool/debug/aiemcp.py`. **LLM-tab auto-connect:** the daemon (`schedule_debug_server.py`) no longer relies on cwd `.mcp.json` discovery — `DebugState._write_mcp_config()` writes a temp `.mcp.json` embedding the daemon-resolved HW config (`AIEDBG_TARGET/AIEMCP_DEVICE/STARTCOL/AIE_VERSION/JSON_DIR`, `command=sys.executable`) and `_llm_spawn` passes it via `--mcp-config <temp> --strict-mcp-config --allowedTools mcp__aiegdb__aie_{exec,scope,commands,help}`; a startup probe (`probe_mcp`, on by default, `--no-mcp-probe` to skip) verifies the connection and prints `LLM MCP: aiegdb connected/NOT connected` (warns without aborting on failure); the temp config is unlinked on Ctrl-C shutdown
- **[doc/design/live_debug_framework.md](doc/design/live_debug_framework.md)** — Design (no code yet) for a live AIE debug/test framework that unifies the static schedule viewer (`schedule_view.py` → `host_schedule.html`) with runtime tools. Adds a thin stdlib `http.server` daemon (`schedule_debug_server.py`, house-style per `host_cc_control.py:1847`) that imports `aiediag.py` (offsets, `run_aiedbg_reg_read`, decoders, `phys_col = col + startcol`), orchestrates `apppaltest.py -y -nonreboot` (`-u` unbuffered, output redirected to the repo-root `applog` file), and exposes polling JSON endpoints (`/run`, `/applog?offset=N` for realtime `applog` tail, `/grid?what=dma\|cores\|events`, `/cmd`) so the browser shows live DMA/core/event overlays and a per-tile debug console. Headline: static HTML stays for offline view but is insufficient for live features — evolution is additive, not a rewrite

## Key Terms

| Term | Definition |
|------|------------|
| **PDI** | Pre-built hardware design; decouples HW/SW development |
| **GMIO** | Global Memory I/O; DDR ↔ AIE via NoC |
| **Shim tile** | Row-0 tile bridging NoC/DDR and AIE array |
| **MemTile** | Large-memory tile for caching between DDR and compute tiles |
| **BD** | Buffer Descriptor; configures a DMA transfer |
| **DSKernel** | Data-streaming kernel (receives via DMA, computes, outputs via DMA) |
| **EmitC** | MLIR dialect for C/C++ emission; final stage before `translateToCpp` |
| **xchesscc** | Synopsys compiler for AIE cores (from Vitis) |
| **PAL** | Board environment for running ELFs on real AIE hardware |

## never do
- the api function > 200 lines
## Learn rule
- create skill when a issue fixed
- when do somthing wrong that is fix by user guide create related skill
## Document rule
- Maintain and update architecture doc and keep update after do some changes
## Process transperent rule
- after each task done, list all files that change or new created
## Memory and Lock
- ./doc/aiedifferentview.md

