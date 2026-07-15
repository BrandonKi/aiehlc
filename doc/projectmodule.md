

## Project module
### 9-Module Breakdown

```
                          User C++ Source
                               |
                    +----------+----------+
                    |                     |
              [M1] aiehlc           [M2] Frontend
           (single-kernel)        (Clang AST->MLIR)
                    |                     |
                    |              [M3] Pipeline Orchestrator
                    |                     |
                    v         +-----------+-----------+
              host+kernel     |                       |
              ELF             v                       v
                    [M4] Spatial Routing       [M5] Dataflow Mapping
                    (routing + routinghw)      (dmap + dmaphop)
                              |                       |
                              v                       v
                    [M6] Routing Engine        [M7] Schedule Generation
                    (BFS, ResourceMgr)         (blueprint + dfschedule)
                              |                       |
                              +-------+-------+-------+
                                      |
                              [M8] AIE Runtime
                              (XAie driver wrapper)
                                      |
                              [M9] Build & Test Infra
                              (scripts, unitest, HW run)
```

---

#### M1 - aiehlc Compiler Driver (Single-Kernel Path)
**Responsibility:** Clang-based tool that compiles user C++ into host ELF + kernel ELF for single-tile AIE apps
**Key Files:**
| File | Lines | Role |
|------|-------|------|
| `src/llvm/aiehlc.cc` | 3,091 | Main driver: parses C++, invokes xchesscc for kernel, aarch64-g++ for host |

**Inputs:** User C++ with AIE driver API calls
**Outputs:** `host` ELF (ARM) + `kernel` ELF (AIE core)
**Dependencies:** M2 (Frontend), M8 (Runtime)

---

#### M2 - MLIR Frontend (Clang AST to MLIR)
**Responsibility:** Parse user C++ into MLIR; define the base AIE dialect (LoadKernel, etc.)
**Key Files:**
| File | Lines | Role |
|------|-------|------|
| `src/mlir/mlirfront/AieFrontEnd.cc` | 527 | Clang AST visitor -> MLIR IR generation |
| `src/mlir/mlirfront/AieFrontEnd.h` | - | Frontend interface |
| `src/mlir/mlirfront/AieDialect.cc` | 671 | AIE dialect ops, types, attrs definition |
| `src/mlir/mlirfront/AieDialect.h` | - | Dialect header |
| `src/mlir/mlirfront/AieLinkDialect.h` | - | Link-time dialect |
| `include/aie_spatial_types.h` | 70 | Spatial type tags for mesh partitioning |

**Inputs:** Clang AST from user C++
**Outputs:** MLIR ModuleOp with AIE dialect ops
**Dependencies:** LLVM/Clang, MLIR core

---

#### M3 - Pipeline Orchestrator
**Responsibility:** Wire all passes together, build initial routing IR, manage host/kernel/routing code emission
**Key Files:**
| File | Lines | Role |
|------|-------|------|
| `tilinglinalg/pass/tilinglinalg_pipeline.cpp` | 1,167 | `TilingLinalgPipeline::runPipeline()` - orchestrates all 15+ passes |
| `tilinglinalg/pass/tilinglinalg_pipeline.h` | 163 | Pipeline API + TensorParam/SplitModel/MeshKernelDesc structs |
| `tilinglinalg/pass/kernelconfig/kernelconfig.h` | 141 | Kernel configuration parameters |

**Inputs:** MLIR ModuleOp from M2, mesh/tensor/split parameters
**Outputs:** `host.cc`, `kernel.cc`, `routing.cc`, `aieml.bcf`, `aieml.prx`
**Dependencies:** M4, M5, M6, M7 (all dialects & passes)

---

#### M4 - Spatial Routing Dialects (routing + routinghw)
**Responsibility:** Model abstract tile arrays & physical stream-switch routing
**Key Files:**
| File | Lines | Role |
|------|-------|------|
| `routing/routingmanager.cpp` | 874 | **routing dialect** - abstract mesh, data IO, broadcast, partitioning |
| `routing/routingmanager.h` | - | Routing dialect interface |
| `routing/td/*.td` | 4 files | TableGen: ops, types, attrs, interfaces |
| `routinghw/routinghwmanager.cpp` | 94 | **routinghw dialect** - physical tiles, stream switch ports, packet flows |
| `routinghw/routinghwmanager.h` | - | RoutingHW dialect interface |
| `routinghw/td/*.td` | 3 files | TableGen: ops, types, attrs |

**Lowering passes in this module's domain:**
| Pass | Lines | Direction |
|------|-------|-----------|
| `passroutingtodmap/routingtodmap.cpp` | 1,218 | routing -> dmap |
| `passdmaphoptoroutinghw/passdmaphoptoroutinghw.cpp` | 1,245 | dmaphop -> routinghw (Path A) |

---

#### M5 - Dataflow Mapping Dialects (dmap + dmaphop)
**Responsibility:** Model logical dataflow (ports, streams, push/pull) and physical hop-by-hop paths (tile-to-tile DMA, buffers)
**Key Files:**
| File | Lines | Role |
|------|-------|------|
| `dataflowmap/dmap/dmapmanager.cpp` | 311 | **dmap dialect** - logical ports, streams, push/pull |
| `dataflowmap/dmap/td/*.td` | 3 files | TableGen: ops, types, attrs |
| `dataflowmap/dmaphop/dmaphopmanager.cpp` | 392 | **dmaphop dialect** - physical hops, DMA configs, buffers |
| `dataflowmap/dmaphop/td/*.td` | 3 files | TableGen: ops, types, attrs |

**Lowering passes in this module's domain:**
| Pass | Lines | Direction |
|------|-------|-----------|
| `passdmaptodmaphop/dmaptodmaphop.cpp` | 653 | dmap -> dmaphop |
| `passdmaphoptodfscheblueprint/passdmaphoptodfscheblueprint.cpp` | 1,456 | dmaphop -> dfscheblueprint |
| `passdmaphopprovenancemap/passdmaphopprovenancemap.cpp` | 970 | dmaphop provenance tracking (debug) |

---

#### M6 - Routing Engine (BFS + Resource Management)
**Responsibility:** Concrete routing algorithm - BFS path finding across AIE topology, port/link resource tracking
**Key Files:**
| File | Lines | Role |
|------|-------|------|
| `pass/routingimplement/routing/routingpath.cpp` | 263 | BFS path finding (priority: Memory > SHIM > Core) |
| `pass/routingimplement/routing/routingtopology.cpp` | 103 | Gen2 AIE tile topology model |
| `pass/routingimplement/routing/routingcontruct.cpp` | 4 | Routing graph construction |
| `pass/routingimplement/routing/routingtile.cpp` | 4 | Tile type definitions |
| `pass/routingimplement/hw/hwresource.cpp` | 421 | Port templates, HW resource definitions |
| `pass/routingimplement/hw/ResourceManager.cpp` | 814 | Resource allocation, conflict avoidance |
| `pass/routingimplement/include/` | - | Headers for routing & hw modules |

**Dependencies:** M4 (uses routing/routinghw dialect types)

---

#### M7 - Schedule Generation (dfscheblueprint + dfschedule)
**Responsibility:** Blueprint -> executable DMA/lock/kernel-launch schedule; final EmitC code generation
**Key Files:**
| File | Lines | Role |
|------|-------|------|
| `dataflowmap/dfscheblueprint/dfscheblueprintmanager.cpp` | 671 | **dfscheblueprint dialect** - transfer manifests, flow configs |
| `dataflowmap/dfscheblueprint/td/*.td` | 3 files | TableGen: ops, types, attrs |
| `dataflowmap/dfschedule/dfschedulemanager.cpp` | 1,286 | **dfschedule dialect** - DMA BD, kernel launch, locks, waits |
| `dataflowmap/dfschedule/td/*.td` | 3 files | TableGen: ops, types, attrs |

**Lowering passes in this module's domain:**
| Pass | Lines | Direction |
|------|-------|-----------|
| `passblueprinttoschedule/passblueprinttoschedule.cpp` | 2,471 | blueprint -> dfschedule (host path) |
| `passblueprinttoschedulekernel/passblueprinttoschedulekernel.cpp` | 1,280 | blueprint -> dfschedule (kernel path) |
| `passschedulecanonicalize/passschedulecanonicalize.cpp` | 644 | Schedule optimization |
| `passschedulesequentialop/passschedulesequentialop.cpp` | 283 | Sequential ordering |
| `passwaitmerge/passwaitmerge.cpp` | 185 | Merge wait operations |
| `passdfscheduletoapi/passdfscheduletoapi.cpp` | 3,595 | dfschedule -> EmitC (host.cc) **LARGEST PASS** |
| `passdfscheduletokernelapi/passdfscheduletokernelapi.cpp` | 369 | dfschedule -> EmitC (kernel.cc) |
| `passdfscheduleprovenancemap/passdfscheduleprovenancemap.cpp` | 800 | Provenance tracking (debug) |

---

#### M8 - AIE Runtime
**Responsibility:** C wrapper layer over XAie driver APIs; device init/teardown, DMA, locks, kernel load
**Key Files:**
| File | Lines | Role |
|------|-------|------|
| `src/mlir/runtime/aie_runtime.c` | 1,363 | Core runtime: device_init, load_kernel, dma_bd_config, wait_event |
| `src/mlir/runtime/aie_runtime.h` | 380 | Runtime API declarations |
| `src/mlir/runtime/aie_runtime_common.c` | 585 | Common utilities |
| `src/mlir/runtime/aie_runtime_debug.c` | 2,143 | Debug/diagnostic wrappers |
| `src/mlir/runtime/aie_runtime_stream_debug.c` | 42 | Stream debug |
| `src/mlir/runtime/kernel_log.h` | 80 | Kernel logging |
| `include/aie_device_map.h` | 40 | Device tile-type mapping |
| `thirdparty/alib/` | - | XAie driver library (external) |

---

#### M9 - Build, Test & Deployment Infrastructure
**Responsibility:** Build system, test harness, HW board deployment, diagnostic tools
**Key Files:**
| File | Lines | Role |
|------|-------|------|
| `script/aiehlc.sh` | 492 | Main build script (source to run unitest) |
| `script/kc.sh` | 278 | Kernel compilation script |
| `script/setup.sh` | 210 | Environment setup |
| `script/verify_env.sh` | - | Environment validation |
| `script/test/apppaltest.py` | - | HW board test (SSH+xsdb+console) |
| `src/tool/debug/aiediag.py` | - | DMA diagnostic tool |
| `script/verify_generated.py` | - | Generated code verification |
| `pass/unitest/test.cpp` | 1,810 | Integration test driver |
| `pass/unitest/piplinerun.sh` | - | End-to-end pipeline automation |
| `example/tileprogram/ccode/` | - | Example programs (simplematmul, simpleconv2d) |
| `CMakeLists.txt` (various) | - | Build system |
| `doc/` | - | Architecture & API documentation |

---

### Summary Table

| # | Module | Responsibility | LOC (approx) | Key Artifact |
|---|--------|---------------|--------------|--------------|
| M1 | aiehlc Driver | Single-kernel C++ -> ELF compiler | 3,100 | `aiehlc` binary |
| M2 | MLIR Frontend | Clang AST -> MLIR + AIE dialect | 1,200 | MLIR ModuleOp |
| M3 | Pipeline Orchestrator | Wires 15+ passes, manages emission | 1,500 | host/kernel/routing.cc |
| M4 | Spatial Routing Dialects | Abstract & physical routing IR | 2,500 | routing/routinghw ops |
| M5 | Dataflow Mapping Dialects | Logical & physical dataflow IR | 3,800 | dmap/dmaphop ops |
| M6 | Routing Engine | BFS path finding + resource mgmt | 1,600 | Concrete routes |
| M7 | Schedule Generation | Blueprint->schedule->EmitC | 10,600 | DMA/lock C code |
| M8 | AIE Runtime | XAie driver C wrappers | 5,700 | libruntime |
| M9 | Build & Test Infra | Scripts, tests, HW deployment | 3,000+ | test harness |

### Data Flow Through Modules

```
User C++ --(M1/M2)--> MLIR IR --(M3)--> routing IR
  --(M4: routing->routinghw)--> physical routing --(M6: BFS)--> resolved routes
  --(M4: routing->dmap)------> logical dataflow
  --(M5: dmap->dmaphop)------> physical hops
  --(M5: dmaphop->blueprint)--> schedule blueprint
  --(M7: blueprint->schedule)--> DMA/lock/kernel schedule
  --(M7: schedule->EmitC)-----> host.cc + kernel.cc + routing.cc
  --(M8: runtime)--------------> linked into final ELF
  --(M9: build+HW run)--------> deployed on AIE hardware
```

For the full module analysis with detailed file listings, see [doc/module_analysis.md](doc/module_analysis.md).