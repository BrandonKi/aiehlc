# aietriton Internal Architecture — Lowering `triton_matmul.py` to `matrixmul.cc`

## 1. Overview

This document describes the **compiler-internal** lowering path from a Triton-style Python kernel (`triton_matmul.py`) to the C output (`matrixmul.cc`) that runs on AMD Versal AI Engine. Where `tritton.md` is the user-facing design document explaining API and concept mapping, this document traces each compiler stage, the MLIR passes involved, and how every construct in the Python source becomes routing IR and ultimately C code.

Reference files:

| File | Role |
|------|------|
| `example/tileprogram/design/triton/triton_matmul.py` | Triton-style Python input |
| `example/tileprogram/design/triton/tritton.md` | User-facing design doc (structure template) |
| `example/tileprogram/design/ccode/matrixmul.cc` | C output target |
| `src/mlir/mlirfront/tilinglinalg/pass/tilinglinalg_pipeline.h` | `TensorParam` + `buildRoutingIR` / `runPipeline` API |
| `src/mlir/mlirfront/tilinglinalg/pass/tilinglinalg_pipeline.cpp` | Pipeline implementation |
| `src/mlir/mlirfront/tilinglinalg/routing/routingmanager.cpp` | `createroutingfuncGEMM` / `createroutingfuncByDim` |
| `src/mlir/mlirfront/aietriton/__init__.py` | Python package: `jit` decorator, `_JitKernel`, `_KernelLauncher`, `mesh()` |
| `src/mlir/mlirfront/aietriton/language.py` | `tl.` namespace stubs: types, ops (parsed via AST, not executed) |
| `src/mlir/mlirfront/aietriton/_compiler.py` | AST parser + pipeline invocation: `compile_and_run()` |
| `src/mlir/mlirfront/aietriton/aietriton_pybind.cpp` | pybind11 wrapper: `_aietriton_core.run_aie_pipeline()` |
| `src/mlir/mlirfront/aietriton/CMakeLists.txt` | Build `_aietriton_core.so`, links `mlirtestlib` + MLIR libs |

---

## 2. End-to-End Compilation Flow

```
triton_matmul.py
  │  import aietriton (__init__.py)
  │  @aie_triton.jit → _JitKernel wraps function
  │  kernel[grid]    → _KernelLauncher.__getitem__
  │  kernel[grid]()  → _KernelLauncher.__call__
  ▼
_compiler.py: compile_and_run()
  │  inspect.getsource(fn) + ast.parse()
  │  _classify_params() → tensor/scalar/constexpr
  │  _is_input_tensor() → AST walk for tl.store targets
  │  numpy args → shape, dtype → tensor_specs
  ▼
_aietriton_core.run_aie_pipeline() [pybind11]
  │  tensor_specs → TensorParam[]
  │  registers MLIR dialects, calls buildRoutingIR + runPipeline
  ▼
TensorParam[] + meshRows + meshCols + userKernelBody
  │  buildRoutingIR(ctx, 2, 2, tensors)
  │  (tilinglinalg_pipeline.cpp:128)
  ▼
routing dialect IR (ModuleOp)
  │  runPipeline(ctx, module, outputDir, userKernelBody, userKernelFuncName)
  │  (tilinglinalg_pipeline.cpp:181)
  │
  │  ┌─── Shared passes ─────────────────────────────────────────┐
  │  │  RoutingUnrollingLowerPass                                 │
  │  │  RoutingToDmapPass(Gen2)                                   │
  │  │  DmapToDmaphopPass(Gen2)                                   │
  │  │  DmaphopTodfscheblueprintPass                              │
  │  └────────────────────────────────────────────────────────────┘
  │           │                              │
  │      clone → hostModule             clone → kernelModule
  │           │                              │
  │  ┌─── Host path ──────────┐    ┌─── Kernel path ────────────┐
  │  │  BlueprintToSchedulePass│    │  BlueprintToScheduleKernel │
  │  │  ScheduleCanonicalizeP. │    │  DfscheduleToKernelApiPass │
  │  │  DfscheduleToApiPass    │    │  translateToCpp → kernel.cc│
  │  │  CanonicalizerPass      │    └────────────────────────────┘
  │  │  RoutingConstantFoldP.  │
  │  │  translateToCpp → host.cc│
  │  └─────────────────────────┘
  │
  │  ┌─── Routing path (separate module rebuild) ──────────────┐
  │  │  RoutingUnrollingLowerPass                               │
  │  │  RoutingLowerPass(Gen2)                                  │
  │  │  RoutingHWLowerPass(Gen2)                                │
  │  │  RoutingDeadArgPass                                      │
  │  │  RoutingConstantFoldPass                                 │
  │  │  CanonicalizerPass                                       │
  │  │  translateToCpp → routing.cc                             │
  │  └──────────────────────────────────────────────────────────┘
  │
  ▼
Output file set:
  host.cc           XAie driver API calls (DMA BDs, locks, core load/enable)
  kernel.cc         Window acquire/release wrapper
  <kernelName>.cc   User compute kernel (matmul body) or auto-generated
  routing.cc        Stream switch configuration
  aieml.bcf         Linker control file (tile-local memory layout)
  aieml.prx         Project file for xchesscc
  │
  │  Cross-compile
  ▼
kernel.cc → xchesscc → kernel ELF (AIE core binary)
host.cc   → aarch64-g++ → host ELF (links kernel.o + aie_runtime)
  │
  │  Deploy on AIE HW
  ▼
Load ELFs, configure DMA BDs, run cores
```

---

## 3. Stage 1: Python AST Parse

**Input:** `triton_matmul.py` with `@aie_triton.jit` decorated functions.

### 3.1 Python Package Architecture

The Python frontend is a 3-layer design:

| Layer | File | Responsibility |
|-------|------|----------------|
| **API surface** | `aietriton/__init__.py` | Public API: `jit` decorator, `mesh()`, `set_device()`, `synchronize()` |
| **AST engine** | `aietriton/_compiler.py` | AST parsing, parameter classification, pipeline invocation |
| **C++ bridge** | `aietriton/_aietriton_core` (`.so`) | pybind11 module wrapping `TilingLinalgPipeline` C++ API |

Supporting module:

| File | Role |
|------|------|
| `aietriton/language.py` | `tl.` namespace stubs — types (`int8`, `float32`) and ops (`load`, `store`, `dot`, `zeros`, etc.) that exist so kernel functions can be parsed by Python's AST module. These functions return `None`/`0` and are **never executed** at runtime. |

Class hierarchy:

```
_Mesh
  └── rows, cols (tile mesh dimensions)

_JitKernel
  ├── _fn    (original Python function reference)
  ├── _name  (function.__name__)
  └── __getitem__(grid) → _KernelLauncher

_KernelLauncher
  ├── _fn, _name, _grid
  └── __call__(*args, **kwargs) → compile_and_run()
```

### 3.2 Decorator and Launch Protocol

The launch protocol follows Triton's `kernel[grid](*args)` pattern:

1. **`@aie_triton.jit`** → `_JitKernel(fn)` — stores the function reference, does **not** parse or compile anything yet.

2. **`kernel[grid]`** → `_JitKernel.__getitem__(grid)` returns `_KernelLauncher(fn, name, grid)`. The `grid` is a tuple `(mesh.rows, mesh.cols)` from `aie_triton.mesh()`.

3. **`kernel[grid](A, B, C, ...)`** → `_KernelLauncher.__call__(*args, **kwargs)` triggers `compile_and_run(fn, name, grid, args, kwargs)`.

4. **Lazy compilation:** The pybind11 module `_aietriton_core` is imported inside `compile_and_run()` (not at package load time). This allows `import aietriton` to succeed even without a built `.so` — useful for IDE autocomplete and standalone AST parser testing.

### 3.3 AST Analysis Engine (`_compiler.py`)

`compile_and_run()` implements a 3-phase pipeline:

**Phase 1 — AST parse + tensor spec extraction:**

```python
source = textwrap.dedent(inspect.getsource(fn))
tree   = ast.parse(source)
kernel_def = _find_kernel_def(tree, name)       # find FunctionDef node
tensor_params, scalar_params, constexpr_params = _classify_params(kernel_def, param_names)
```

For each tensor parameter, runtime numpy array metadata is extracted:
- `shape = list(arr.shape)` → e.g. `[16, 16]`
- `bits = arr.dtype.itemsize * 8` → e.g. `8` for `np.int8`
- `is_input = _is_input_tensor(kernel_def, pname)` → AST-based input/output classification

**Phase 2 — Kernel body extraction:**

```python
kernel_body = _extract_kernel_body(func_def, param_names, tensor_params, constexpr_params, kwargs)
```

Currently a stub returning `""` (empty string), which triggers auto-generated GEMM compute kernel. Future: translate Python AST → C code (`tl.load` → `acquire_input_window`, etc.).

**Phase 3 — C++ pipeline invocation:**

```python
from . import _aietriton_core
success = _aietriton_core.run_aie_pipeline(
    mesh_rows, mesh_cols, tensor_specs, output_dir, kernel_body, name
)
```

**Key analysis functions:**

| Function | Algorithm |
|----------|-----------|
| `_classify_params(func_def, param_names)` | For each parameter: if annotated with `tl.constexpr` → constexpr; if name ends with `_ptr` or is in `{A, B, C, a_ptr, b_ptr, c_ptr}` → tensor; otherwise → scalar |
| `_is_input_tensor(func_def, param_name)` | Walks the entire function AST. If the parameter appears as the first argument to a `tl.store(param, ...)` call → output (`False`). Otherwise → input (`True`). |
| `_references_name(node, name)` | Recursive AST walk checking if any `ast.Name` node in the subtree has `id == name` |
| `_find_kernel_def(tree, name)` | Walks AST to find `ast.FunctionDef` with matching function name |
| `_numpy_dtype_to_bits(dtype)` | `dtype.itemsize * 8` — converts numpy dtype to bit width |

### 3.4 pybind11 Binding Layer (`aietriton_pybind.cpp`)

The C++ bridge exposes a single entry point:

```cpp
static bool run_aie_pipeline(
    int meshRows, int meshCols,
    const std::vector<std::tuple<std::vector<int64_t>, int, bool>> &tensorSpecs,
    const std::string &outputDir,
    const std::string &userKernelBody,
    const std::string &userKernelFuncName);
```

**Type mapping (Python → C++):**

| Python type | C++ type | Description |
|-------------|----------|-------------|
| `int` | `int` | `meshRows`, `meshCols` |
| `list[tuple[list[int], int, bool]]` | `vector<tuple<vector<int64_t>, int, bool>>` | `tensor_specs`: shape, bit width, isInput |
| `str` | `std::string` | `outputDir`, `userKernelBody`, `userKernelFuncName` |

**Internal flow:**

1. Constructs `std::vector<TensorParam>` from the Python-provided `tensorSpecs` tuples
2. Calls `TilingLinalgPipeline::registerDialects(ctx)` to register all 6 custom dialects
3. Calls `TilingLinalgPipeline::buildRoutingIR(ctx, meshRows, meshCols, tensors)` → routing IR module
4. Calls `TilingLinalgPipeline::runPipeline(ctx, module, outputDir, userKernelBody, userKernelFuncName)` → generates output files

The pybind11 module is named `_aietriton_core` (underscore prefix = private implementation detail, not part of public API).

### 3.5 Output

A `TensorParam[]` vector, mesh dimensions, and kernel body text:

```cpp
// Constructed in _compiler.py as tensor_specs, passed to _aietriton_core.run_aie_pipeline()
std::vector<TensorParam> tensors = {
    {{16, 16}, 8, true },   // A — tl.load target
    {{16, 16}, 8, true },   // B — tl.load target
    {{16, 16}, 8, false},   // C — tl.store target
};
int meshRows = 2, meshCols = 2;
std::string userKernelBody = "...";       // matmul function body
std::string userKernelFuncName = "matmul"; // from @aie_triton.jit def name
```

---

## 4. Stage 2: Build Routing IR (`buildRoutingIR`)

**Implementation:** `tilinglinalg_pipeline.cpp:128` — `TilingLinalgPipeline::buildRoutingIR()`

**Input:**
- `TensorParam[] = [{shape={16,16}, bitWidth=8, isInput=true}, {shape={16,16}, bitWidth=8, isInput=true}, {shape={16,16}, bitWidth=8, isInput=false}]`
- `meshRows=2`, `meshCols=2`

**Steps:**

### 4.1 Create mesh

```cpp
auto mesh = builder.create<createhwmesh>(loc, meshRows, meshCols);
// → routing.routingcreatehwmesh row=2, col=2
```

### 4.2 Create schedule tensors

For each `TensorParam`, a dense-initialized tensor is created:

```cpp
auto tensor = builder.create<createscheduletensor>(loc, tensorType,
    initConstant.getResult(), shapeAttr, dimAttr);
```

This produces:
- `tensorA = routingcreatescheduletensor [16,16]` (init values 1..256)
- `tensorB = routingcreatescheduletensor [16,16]` (init values 1..256)
- `tensorC = routingcreatescheduletensor [16,16]` (init values 1..256)

### 4.3 Create routing per tensor (`createroutingfuncByDim`)

For each tensor, `routingmanager::createroutingfuncByDim()` (`routingmanager.cpp:543`) is called:

```cpp
rm.createroutingfuncByDim(builder, &ctx, tp.isInput, mesh, tensor, meshRows, "row");
```

This generates (per tensor):

```
scf.execute_region {routing_memo = "row"} {
    partitionmesh  mesh, splitnum=2, splitaxis="row"
    partitiontensor tensor, splitnum=2, splitdim=0,
                    hw_axis_owner="row", replicate_on="col", single_tile_owner=""

    scf.for %i = 0 to 2 step 1 {
        routing.RoutingCreate<Memo = "row"> (scf_idx = %idx) {
            extract_tiles(partmesh, %idx)       → tile list for group i
            extract_data(partTensor, %idx)       → tensor slice

            // For inputs (A, B):
            createhwiowithtarget(tiles, "input", "mem2")  → hwio
            movedatabyio(slice, hwio)                      → schedule DMA

            // For output (C):
            routinggatherout(tiles, slice)                 → gathered output
            createhwiowithtarget(tiles, "output", "mem2")  → hwio
            movedatabyio(gathered, hwio)                   → schedule DMA

            routing.yield
        }
    }
    scf.yield
}
```

### 4.4 Concrete routing IR for the GEMM case

When called via `buildRoutingIR` with the three tensors, the generated routing IR contains:

**Tensor A (input):**
```
partitionmesh  mesh, splitnum=2, splitaxis="row"
partitiontensor A: splitnum=2, splitdim=0, hw_axis_owner="row", replicate_on="col"
  → A split along rows: A[0:7,:] to group 0, A[8:15,:] to group 1
  → Each half replicated to both columns in that tile row
```

**Tensor B (input):**
```
partitionmesh  mesh, splitnum=2, splitaxis="row"
partitiontensor B: splitnum=2, splitdim=0, hw_axis_owner="row", replicate_on="col"
  → Note: createroutingfuncByDim always uses splitnum=meshRows for all tensors
  → B is split the same way as A in the per-tensor approach
```

**Tensor C (output):**
```
partitionmesh  mesh, splitnum=2, splitaxis="row"
partitiontensor C: splitnum=2, splitdim=0, hw_axis_owner="row", replicate_on="col"
  → C split along rows + routinggatherout per tile group
  → C[0:7,:] gathered from group 0, C[8:15,:] gathered from group 1
```

> **Note:** `buildRoutingIR` uses `createroutingfuncByDim` (per-tensor), not `createroutingfuncGEMM`. The GEMM-specific function (`createroutingfuncGEMM`) is used in `ops_testNew` and handles B's broadcast explicitly with `splitnum=1`. The per-tensor approach creates independent routing regions for each tensor. See [Section 13](#13-design-decisions) for the trade-off.

These routing IR ops correspond to **Steps 1-3** in `matrixmul.cc`'s header comments.

---

## 5. Stage 3: MLIR Progressive Lowering Pipeline

**Implementation:** `tilinglinalg_pipeline.cpp:181` — `TilingLinalgPipeline::runPipeline()`

### 5.1 Shared Passes (routing → dfscheblueprint)

These passes run on the single module before cloning:

| # | Pass | Input | Output | `matrixmul.cc` reference |
|---|------|-------|--------|--------------------------|
| 1 | `RoutingUnrollingLowerPass` | routing IR with `scf.for` loops | Unrolled per-tile-group ops (no more `scf.for`) | Step 3: "For each tile group i" |
| 2 | `RoutingToDmapPass(Gen2)` | routing ops | dmap dialect: `define_io_engine` (SHIM), `define_core_group` (2 cores per row), `create_stream` (SHIM→cores, cores→SHIM) | Host comment: "Logical dataflow: IO engines, ports, streams" |
| 3 | `DmapToDmaphopPass(Gen2)` | dmap | dmaphop: concrete tiles `core(0,3)`, `core(1,3)`, `core(0,4)`, `core(1,4)`, `shim(2,0)` with hop chains | Step 4: physical tile mapping |
| 4 | `DmaphopTodfscheblueprintPass` | dmaphop | dfscheblueprint: transfer manifests, flow configs | Step 4→5 transition |

After these 4 passes, the module is cloned into `hostModule` and `kernelModule`.

### 5.2 Host Path (dfscheblueprint → host.cc)

| # | Pass | What it does | `matrixmul.cc` reference |
|---|------|-------------|--------------------------|
| 5 | `BlueprintToSchedulePass(0.5)` | Allocate BD chains, locks, ping-pong buffers per tile. 2 BDs per data stream, lock acquire/release pairs. | Step 5: "2 ping-pong buffer pairs per input" |
| 6 | `ScheduleCanonicalizePass` | Merge redundant schedule ops | — |
| 7 | `DfscheduleToApiPass(debug=true)` | Lower dfschedule ops → EmitC ops representing XAie API calls (`XAie_DmaDescInit`, `XAie_DmaBdSetLock`, `XAie_LoadElf`, etc.) | Host comment: "EmitC C code emission" |
| 8 | `CanonicalizerPass` | Standard MLIR canonicalization | — |
| 9 | `RoutingConstantFoldPass` | Fold remaining routing constants | — |

Final step: `mlir::emitc::translateToCpp(hostModule, stream)` → `host.cc`

### 5.3 Kernel Path (dfscheblueprint → kernel.cc)

| # | Pass | What it does |
|---|------|-------------|
| 5 | `BlueprintToScheduleKernelPass(0.5)` | Generate kernel-side schedule: window declarations, acquire/release sequences |
| 6 | `DfscheduleToKernelApiPass` | Lower to EmitC: window acquire/release API calls |

Before `DfscheduleToKernelApiPass`, the pipeline walks `KernelDeclOp` to extract:
- `numInputWindows` (2 for GEMM: A and B)
- `numOutputWindows` (1 for GEMM: C)
- `kernelElementType` ("int8" for this example)
- `computeKernelName` (overridden to `userKernelFuncName` if provided)

Final step: `mlir::emitc::translateToCpp(kernelModule, stream)` → `kernel.cc`

### 5.4 Compute Kernel Emission

If `userKernelBody` is non-empty (extracted from `@aie_triton.jit` function), it is written verbatim to `<kernelName>.cc`. Otherwise, an auto-generated GEMM compute kernel is emitted with:
- `acquire_input_window` / `release_input_window` calls
- `acquire_output_window` / `release_output_window` calls
- A `v4intN` MAC loop body

### 5.5 Routing Path (separate module, routing → routing.cc)

A fresh routing module is rebuilt via `ops_testNew()` and lowered through a different pass sequence:

| # | Pass | What it does |
|---|------|-------------|
| 1 | `RoutingUnrollingLowerPass` | Unroll routing loops |
| 2 | `RoutingLowerPass(Gen2)` | Lower routing to routinghw |
| 3 | `RoutingHWLowerPass(Gen2)` | Lower routinghw to stream switch configs |
| 4 | `RoutingDeadArgPass` | Remove dead arguments |
| 5 | `RoutingConstantFoldPass` | Fold constants |
| 6 | `CanonicalizerPass` | Standard canonicalization |

Final step: `translateToCpp` → `routing.cc`

### 5.6 BCF/PRX Generation

After the host path completes, `ResourceMgr::instance()->coreMemAllocator()` provides tile-local memory allocations. These are emitted as:
- `aieml.bcf` — linker control file with symbol addresses (stack, reserved DMB regions, buffer symbols)
- `aieml.prx` — project file for xchesscc

---

## 6. Kernel Body Conversion: Python `@aie_triton.jit` → C `__global__`

This section traces the concrete transformation from the Python kernel function body to the C kernel code that runs on each AIE tile. The conversion has two separate concerns:

1. **Function signature** — transformed by the compiler (routing IR → window API)
2. **Compute body** — either auto-generated or extracted from user-provided C

### 6.1 Function Signature Transformation

The Python kernel takes global pointers and dimensions. The C kernel takes window handles (DMA-managed local buffers):

```
Python:                                       C:
─────────────────────────────────────────     ──────────────────────────────────────────
@aie_triton.jit                               __global__ void matmul(
def matmul_simple(                                input_window_int8 *window_in_0,
    a_ptr,                                        input_window_int8 *window_in_1,
    b_ptr,                                        output_window_int8 *window_out_0) {
    c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr = 8,
    BLOCK_N: tl.constexpr = 8,
    BLOCK_K: tl.constexpr = 8,
):
```

**How the compiler transforms the signature:**

| Python parameter | Classification | C equivalent |
|------------------|---------------|--------------|
| `a_ptr` | tensor (name in `{A, a_ptr, ...}`) + `_is_input_tensor()` → `True` | `input_window_int8 *window_in_0` |
| `b_ptr` | tensor + `_is_input_tensor()` → `True` | `input_window_int8 *window_in_1` |
| `c_ptr` | tensor + `_is_input_tensor()` → `False` (target of `tl.store`) | `output_window_int8 *window_out_0` |
| `M, N, K` | scalar (not tensor, not constexpr) | Folded into DMA config by `buildRoutingIR` (not in C kernel signature) |
| `BLOCK_M, BLOCK_N, BLOCK_K` | constexpr (annotated `tl.constexpr`) | `#define M_TILE (M/2)` — compile-time constants in generated code |

The `_classify_params()` function in `_compiler.py` performs this classification. The `_is_input_tensor()` function walks the AST to determine direction: parameters that appear as the first argument to `tl.store(param, ...)` are outputs; all others are inputs.

The window type (`input_window_int8`) is derived from the numpy array's dtype via `_numpy_dtype_to_bits()` → `elementBitWidth=8` → `int8`. The window numbering (`window_in_0`, `window_in_1`, `window_out_0`) follows the order tensors appear in the parameter list.

### 6.2 Tile Identity

```
Python:                                       C:
─────────────────────────────────────────     ──────────────────────────────────────────
tile_row = tl.program_id(axis=0)              unsigned coreid = get_coreid();
tile_col = tl.program_id(axis=1)              int row = coreid & 0x1F;
                                              int col = coreid >> 16;
```

On GPU Triton, `program_id` returns the block index in the launch grid. On AIE, each tile has a hardware core ID. The compiler maps `axis=0` to the row component and `axis=1` to the column component of the core ID. The Python `tile_row`/`tile_col` values are **not** passed to the C kernel — on AIE, `get_coreid()` is a hardware intrinsic.

> **Note:** In the current pipeline, `tl.program_id()` is parsed by the AST but does **not** generate routing IR ops. Tile identity is implicit — the physical tile coordinates are determined by `DmapToDmaphopPass` during mesh mapping, and each tile's data partition is configured by the DMA buffer descriptors, not by runtime program-id indexing.

### 6.3 Data Access: `tl.load` / `tl.store` → Window Acquire / Release

This is the core semantic transformation. In Python, the kernel reads/writes global memory with pointer arithmetic. In C, the DMA subsystem delivers data to tile-local windows:

```
Python (per K-iteration):                     C (per ping-pong iteration):
─────────────────────────────────────────     ──────────────────────────────────────────
for k_start in range(0, K, BLOCK_K):          for (int iter = 0; iter < 2; iter++) {

    a_block = tl.load(                            int8_t *A_tile =
        a_ptr + offsets_A                             (int8_t *)acquire_input_window(
    )                                                     window_in_0);

    b_block = tl.load(                            int8_t *B =
        b_ptr + offsets_B                             (int8_t *)acquire_input_window(
    )                                                     window_in_1);

    acc += tl.dot(a_block, b_block)               // GEMM triple loop:
                                                  for (i) for (j) for (kk)
                                                      sum += A_tile[i*K+kk]
                                                           * B[kk*N+j];

                                                  release_input_window(window_in_0);
                                                  release_input_window(window_in_1);
```

**What the compiler does:**

| Python construct | Routing IR op | C code |
|------------------|---------------|--------|
| `tl.load(a_ptr + ...)` | `movedatabyio(A_slice, hwio_input)` | `acquire_input_window(window_in_0)` — DMA has filled local buffer, lock acquired |
| `tl.load(b_ptr + ...)` | `movedatabyio(B_slice, hwio_input)` | `acquire_input_window(window_in_1)` |
| `tl.dot(a, b)` | (not in routing IR — kernel body) | GEMM triple loop with `int16_t` accumulation, or `v4int8` vector MAC on real HW |
| `tl.store(c_ptr + ..., result)` | `routinggatherout` + `movedatabyio(C, hwio_output)` | `acquire_output_window(window_out_0)` + write + `release_output_window(window_out_0)` |
| `range(0, K, BLOCK_K)` | (implicit in BD chain length) | `for (int iter = 0; iter < 2; iter++)` — ping-pong loop count matches K/BLOCK_K |
| end of loop body | (implicit) | `release_input_window(window_in_0/1)` — signal DMA for next buffer |

The key insight: Python pointer arithmetic (`a_ptr + row_offsets * K + k_offsets`) disappears entirely in the C kernel. The DMA subsystem partitions the tensors and delivers the correct slices to each tile's local memory via buffer descriptors configured by the host. The kernel just calls `acquire_input_window` to get a pointer to the pre-filled local buffer.

### 6.4 Accumulator and Type Conversion

```
Python:                                       C:
─────────────────────────────────────────     ──────────────────────────────────────────
acc = tl.zeros(                               // Local buffer at BCF-assigned address
    (BLOCK_M, BLOCK_N),                       // (32 KB tile data memory)
    dtype=tl.int32                            int16_t sum = 0;
)

acc += tl.dot(a_block, b_block)               sum += (int16_t)A_tile[i*K+kk]
                                                   * (int16_t)B[kk*N+j];

result = acc.to(tl.int8)                      if (sum > 127)       sum = 127;
                                              else if (sum < -128) sum = -128;
                                              C_tile[i*N+j] = (int8_t)sum;
```

- `tl.zeros(shape, dtype=tl.int32)` → a local accumulator buffer in tile data memory (address assigned by BCF linker control file)
- `tl.dot(a, b)` → on real HW this maps to AIE vector intrinsics (`aie::mmul`); the scalar triple loop in `matrixmul.cc` is the reference implementation
- `.to(tl.int8)` → saturating cast from wider accumulator to output type (`int16_t` → `int8` with clamp to [-128, 127])

### 6.5 Output Store

```
Python:                                       C:
─────────────────────────────────────────     ──────────────────────────────────────────
c_offsets = (                                 int8_t *C_tile =
    (row_start + arange(BLOCK_M)[:,None]) * N     acquire_output_window(window_out_0);
    + (col_start + arange(BLOCK_N)[None,:])
)
result = acc.to(tl.int8)                      C_tile[i * N + j] = (int8_t)sum;
tl.store(c_ptr + c_offsets, result)           release_output_window(window_out_0);
```

Python computes 2D offset expressions into global memory. The C kernel writes directly into a DMA-managed output window. The compiler:

1. Detects `c_ptr` as output tensor (`_is_input_tensor()` finds it as first arg to `tl.store` → `False`)
2. Creates `routinggatherout` + `movedatabyio(C, hwio_output)` in routing IR
3. `DfscheduleToApiPass` generates `acquire_output_window` / `release_output_window` calls
4. Host DMA gathers tile-local C slices back to DDR

### 6.6 Host Code Conversion

The host launch section maps directly:

```
Python:                                       C (matrixmul.cc):
─────────────────────────────────────────     ──────────────────────────────────────────
M, N, K = 16, 16, 16                         #define M 16
                                              #define K 16
                                              #define N 16
                                              #define M_TILE (M / 2)

aie_triton.set_device(0)                      aieSetDevice(0);

mesh = aie_triton.mesh(rows=2, cols=2)        aieDim mesh(2, 2);

A = np.arange(1, M*K+1,                      int8_t *A = (int8_t *)malloc(M*K);
    dtype=np.int8).reshape(M, K)              for (int i = 0; i < M*K; i++)
                                                  A[i] = (int8_t)((i % 7) - 3);

B = np.arange(1, K*N+1,                      int8_t *B = (int8_t *)malloc(K*N);
    dtype=np.int8).reshape(K, N)              for (int i = 0; i < K*N; i++)
                                                  B[i] = (int8_t)((i % 5) - 2);

C = np.zeros((M, N), dtype=np.int8)           int8_t *C = (int8_t *)malloc(M*N);
                                              for (int i = 0; i < M*N; i++)
                                                  C[i] = 0;

grid = (mesh.rows, mesh.cols)
matmul_simple[grid](A, B, C,                  matmul<<<mesh>>>(A, B, C);
    M, N, K, BLOCK_M=8, ...)

aie_triton.synchronize()                      aieDeviceSynchronize();

# numpy verification                          // CPU reference verification loop
C_ref = (A.astype(np.int16) @                 for (int i = 0; i < M; i++)
    B.astype(np.int16))                         for (int j = 0; j < N; j++) {
    .clip(-128, 127).astype(np.int8)              int16_t sum = 0; ...
                                                }
# garbage collected                            free(A); free(B); free(C);
```

### 6.7 Kernel Variant: Strided Triton Style (`matmul_strided`)

`triton_matmul.py` includes a second kernel variant using Triton 2.0+ `make_block_ptr` / `advance` API. Both variants produce identical routing IR because the frontend only extracts tensor metadata, not the kernel body:

```
matmul_simple:                                matmul_strided:
─────────────────────────────────────────     ──────────────────────────────────────────
a_block = tl.load(                            a_block_ptr = tl.make_block_ptr(
    a_ptr + explicit_offsets                      base=a_ptr, shape=(M,K),
)                                                 strides=(...), offsets=(...),
                                                  block_shape=(BLOCK_M, BLOCK_K))
                                              a_block = tl.load(a_block_ptr)

for k_start in range(0, K, BLOCK_K):         for k in range(0, K, BLOCK_K):
    ...                                           ...
                                                  a_block_ptr = tl.advance(
                                                      a_block_ptr, (0, BLOCK_K))
```

Both normalize to the same `tensor_specs` list:

```python
tensor_specs = [
    ([16, 16], 8, True),    # A — input
    ([16, 16], 8, True),    # B — input
    ([16, 16], 8, False),   # C — output
]
```

The strided variant has additional scalar parameters (`stride_am`, `stride_ak`, ...) that are classified as scalars by `_classify_params()` and do not affect routing IR generation.

### 6.8 Kernel Body Status

`_extract_kernel_body()` implements the AST -> KernelOps -> MLIR EmitC -> C pipeline (see Section 6.9). If the pipeline produces an empty result (e.g., unrecognized AST patterns), it falls back to auto-generated GEMM compute kernel.

### 6.9 AST -> C Translation Pass (`aie_pass/`)

The kernel body translation uses a 3-layer pipeline that converts the Python `@aie_triton.jit` function body into C code via structured intermediate representation:

```
Python AST (@aie_triton.jit body)
  -> aie_pass/ast_to_kernelops.py: AST walk -> list of KernelOp dicts
  -> pybind11: build_kernel_body() passes KernelOp list to C++
  -> C++ KernelBodyEmitter: builds MLIR EmitC VerbatimOps from KernelOps
  -> mlir::emitc::translateToCpp() -> C kernel body string
  -> returned to Python -> passed as userKernelBody -> written to <kernelName>.cc
```

#### Layer 1: Python AST -> KernelOp List (`aie_pass/ast_to_kernelops.py`)

The `_ASTWalker` class walks the function AST and produces a flat list of typed dicts:

| Python AST pattern | KernelOp |
|---|---|
| `tl.program_id(axis=N)` | `{"op": "get_coreid"}` |
| `tl.zeros(shape, dtype)` | (tracked internally, no op emitted) |
| `tl.load(a_ptr + ...)` | `{"op": "acquire_input", "window_idx": N}` |
| `tl.dot(a, b)` | `{"op": "gemm_body", "m": M, "n": N, "k": K}` |
| `acc.to(tl.int8)` | (folded into gemm_body) |
| `tl.store(c_ptr + ...)` | `{"op": "acquire_output"}` + `{"op": "release_output"}` |
| `for k in range(0, K, BLOCK_K)` | `{"op": "for_range", "trip_count": K/BLOCK_K}` / `{"op": "end_for"}` |

Key features:
- `_get_ptr_base()` walks `ast.BinOp` trees to find the tensor parameter name
- `_compute_trip_count()` evaluates `range(start, stop, step)` using constexpr values
- Input/output releases are automatically emitted at end of for-loop bodies

#### Layer 2: pybind11 Bridge (`aietriton_pybind.cpp`)

`build_kernel_body()` converts Python `list[dict]` to `vector<KernelOp>` structs:

```cpp
static std::string build_kernel_body(
    const std::string &kernelName,
    const std::string &elementType,
    int numInputWindows, int numOutputWindows,
    const py::list &kernelOpsList);
```

#### Layer 3: C++ MLIR EmitC Builder (`aie_pass/kernel_body_emitter.h/.cpp`)

`KernelBodyEmitter` iterates over `KernelOp` structs and creates `emitc::VerbatimOp` for each construct -- the same pattern used by `DfscheduleToKernelApiPass`. After all ops are built into a `ModuleOp`, `mlir::emitc::translateToCpp()` emits the final C string.

**Generated C structure** (for `matmul_simple` with M=8, N=16, K=16):

```c
void matmul_simple(input_window_int8 *window_in_0,
                   input_window_int8 *window_in_1,
                   output_window_int8 *window_out_0) {
    unsigned coreid = get_coreid();
    int col = coreid >> 16;
    int row = coreid & 0x1F;
    for (int iter = 0; iter < 1; iter++) {
        klog("CENk", iter);
        int8_t *in0 = (int8_t *)acquire_input_window(window_in_0);
        int8_t *in1 = (int8_t *)acquire_input_window(window_in_1);
        int8_t *out0 = acquire_output_window(window_out_0);
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 16; j++) {
                int16_t sum = 0;
                for (int kk = 0; kk < 16; kk++)
                    sum += (int16_t)in0[i * 16 + kk] * (int16_t)in1[kk * 16 + j];
                if (sum > 127) sum = 127;
                else if (sum < -128) sum = -128;
                out0[i * 16 + j] = (int8_t)sum;
            }
        }
        klog("CLOP", 128);
        release_input_window(window_in_0);
        release_input_window(window_in_1);
        release_output_window(window_out_0);
        klog("CEXT", 1);
    }
}
```

#### Files

| File | Purpose |
|------|---------|
| `aie_pass/__init__.py` | Package marker |
| `aie_pass/ast_to_kernelops.py` | Python AST -> KernelOp list |
| `aie_pass/kernel_body_emitter.h` | C++ header: `KernelBodyEmitter` class + `KernelOp` struct |
| `aie_pass/kernel_body_emitter.cpp` | C++ impl: EmitC VerbatimOp construction + translateToCpp |
| `aie_pass/test_ast_to_c.py` | Unit test: AST -> KernelOps -> C verification |

#### Testing

```bash
# AST -> KernelOps only (no .so needed):
cd src/mlir/mlirfront && python -m aietriton.aie_pass.test_ast_to_c

# Full pipeline including C generation (requires built .so):
cd src/mlir/mlirfront && python -m aietriton.aie_pass.test_ast_to_c --full
```

---

## 7. Triton Python ↔ Routing IR ↔ C Summary Table

Quick-reference mapping showing how each Triton construct becomes routing IR, then C:

| `triton_matmul.py` | Routing IR Op | `matrixmul.cc` |
|---------------------|---------------|-----------------|
| `@aie_triton.jit` | (AST parse entry point) | `__global__ void matmul(...)` |
| `tl.program_id(0)` | (implicit in tile group `scf.for` loop) | `get_coreid()` → `row = coreid & 0x1F` |
| `tl.program_id(1)` | (implicit in tile group `scf.for` loop) | `get_coreid()` → `col = coreid >> 16` |
| `tl.load(a_ptr + ...)` | `movedatabyio(A_slice, hwio_input)` | `acquire_input_window(window_in_0)` |
| `tl.load(b_ptr + ...)` | `movedatabyio(B_slice, hwio_input)` | `acquire_input_window(window_in_1)` |
| `tl.dot(a, b)` | (kernel body — not in routing IR) | GEMM triple loop / `v4int8` MAC |
| `acc = tl.zeros(...)` | (kernel body — not in routing IR) | Local buffer at BCF-assigned address |
| `acc.to(tl.int8)` | (kernel body — not in routing IR) | Saturate `int16` sum to `int8` range |
| `tl.store(c_ptr + ...)` | `routinggatherout` + `movedatabyio(C, hwio_output)` | `acquire_output_window` + write + `release_output_window` |
| `aie_triton.mesh(2,2)` | `createhwmesh(2, 2)` | `aieDim mesh(2, 2)` |
| `BLOCK_M=8` / `tl.constexpr` | `partitiontensor splitnum=2` (M/BLOCK_M = 16/8 = 2) | `#define M_TILE (M / 2)` |
| `range(0, K, BLOCK_K)` | (K-loop in kernel body) | `for (int iter = 0; iter < 2; iter++)` |
| `matmul_simple[grid](A, B, C, ...)` | `buildRoutingIR(ctx, 2, 2, tensors)` → full routing module | `matmul<<<mesh>>>(A, B, C)` |
| `aie_triton.set_device(0)` | (host-side, not in routing IR) | `aieSetDevice(0)` |
| `aie_triton.synchronize()` | (host-side, not in routing IR) | `aieDeviceSynchronize()` |

See [Section 6](#6-kernel-body-conversion-python-aie_tritonjit--c-__global__) for the detailed code-level walkthrough of each conversion.

---

## 8. Data Partitioning Deep Dive

### 8.1 Partition Strategy

The `partitiontensor` ops map to the physical data flow described in `matrixmul.cc`'s header:

**Tensor A (input) — row-partitioned, replicated on col:**
```
partitiontensor A: splitnum=2, splitdim=0, hw_axis_owner="row", replicate_on="col"

  A[16x16] split along dim 0 (rows):
    Group 0 → A[0:7, :]   (8 rows, all 16 K columns) = 128 bytes
    Group 1 → A[8:15, :]  (8 rows, all 16 K columns) = 128 bytes

  Each group's slice is replicated to both tiles in the column dimension:
    Group 0: core(0,3) gets A[0:7,:], core(1,3) also gets A[0:7,:]
    Group 1: core(0,4) gets A[8:15,:], core(1,4) also gets A[8:15,:]
```

**Tensor B (input) — broadcast:**

In `createroutingfuncGEMM` (used by `ops_testNew`):
```
partitiontensor B: splitnum=1, splitdim=0, hw_axis_owner="", replicate_on="row"
  → B is NOT split (splitnum=1)
  → Full B[16x16] = 256 bytes sent to every tile group
```

In `createroutingfuncByDim` (used by `buildRoutingIR`):
```
partitiontensor B: splitnum=2, splitdim=0, hw_axis_owner="row", replicate_on="col"
  → B is split like A (per-tensor approach treats all tensors uniformly)
```

**Tensor C (output) — row-partitioned + gather:**
```
partitiontensor C: splitnum=2, splitdim=0, hw_axis_owner="row", replicate_on="col"
routinggatherout per tile group

  C[16x16] split along dim 0:
    Group 0 → C[0:7, :]   gathered from row 0 tiles = 128 bytes
    Group 1 → C[8:15, :]  gathered from row 1 tiles = 128 bytes
```

### 8.2 Physical Tile Map

After `DmapToDmaphopPass`, the abstract mesh maps to concrete AIE tiles:

```
        DDR (host memory)
        ┌──────────────────┐
        │ A[16x16] B[16x16]│
        └────────┬─────────┘
                 │ GMIO via NoC
        ┌────────┴─────────┐
        │   SHIM tile(2,0) │
        │   DMA BD chains  │
        └──┬────────────┬──┘
    stream │            │ stream
 ┌─────────┴──┐   ┌────┴────────┐
 │ core(0,3)  │   │ core(1,3)   │  Tile group 0 (row 0)
 │ A[0:7,:]   │   │ A[0:7,:]    │  (A replicated on col)
 │ B[0:15,:]  │   │ B[0:15,:]   │  (B broadcast)
 │ → C[0:7,:] │   │ → C[0:7,:]  │
 └────────────┘   └─────────────┘
 ┌────────────┐   ┌─────────────┐
 │ core(0,4)  │   │ core(1,4)   │  Tile group 1 (row 1)
 │ A[8:15,:]  │   │ A[8:15,:]   │
 │ B[0:15,:]  │   │ B[0:15,:]   │
 │ → C[8:15,:]│   │ → C[8:15,:] │
 └──┬─────────┘   └────┬────────┘
    │ stream            │ stream
 ┌──┴───────────────────┴──┐
 │   SHIM tile(2,0)        │
 └────────┬────────────────┘
          │ GMIO via NoC
        ┌─┴────────────────┐
        │ C[16x16] (result)│
        └──────────────────┘
```

### 8.3 Data paths (dmaphop)

Each path uses packet-switched routing with unique packet IDs:
- **SHIM port_out → core port_in** — DDR → tile (input data: A slices, B)
- **core port_out → core port_in** — tile-to-tile hop for broadcast
- **core port_out → SHIM port_in** — tile → DDR (output data: C slices)

### 8.4 Memory per tile

From `matrixmul.cc` header:
```
Input A:  M_TILE * K = 8 * 16 = 128 bytes
Input B:  K * N      = 16 * 16 = 256 bytes (broadcast, full)
Output C: M_TILE * N = 8 * 16 = 128 bytes
Total:    512 bytes per tile (fits in AIE local data memory, ~32 KB)
```

With ping-pong buffering, double this to ~1024 bytes — still well within the 32 KB limit.

---

## 9. Code Emission: EmitC → C Output

### 9.1 Host code (`DfscheduleToApiPass` → `host.cc`)

`DfscheduleToApiPass` converts dfschedule ops into EmitC ops that represent XAie driver API calls:

| dfschedule Op | XAie API Call | Purpose |
|---------------|---------------|---------|
| DMA BD config | `XAie_DmaDescInit`, `XAie_DmaBdSetLock`, `XAie_DmaBdSetBuffer` | Configure buffer descriptors |
| Lock init | `XAie_LockInit`, `XAie_LockAcquire`, `XAie_LockRelease` | Producer-consumer sync |
| Kernel load | `XAie_LoadElf` | Load kernel ELF to AIE core |
| Core enable | `XAie_CoreEnable` | Start core execution |
| Core wait | `XAie_CoreWaitForDone` | Wait for core completion |

The EmitC module is then translated to C++ via `mlir::emitc::translateToCpp()`.

### 9.2 Kernel code (`DfscheduleToKernelApiPass` → `kernel.cc`)

`DfscheduleToKernelApiPass` generates the kernel wrapper with window acquire/release calls:

```c
void kernel_wrapper(input_window_int8 *window_in_0,
                    input_window_int8 *window_in_1,
                    output_window_int8 *window_out_0) {
    // acquire, call user compute, release
}
```

### 9.3 User kernel body injection

When `userKernelBody` is non-empty (extracted from `@aie_triton.jit`), the pipeline:

1. Renames `KernelDeclOp` in both `hostModule` and `kernelModule` IR to match `userKernelFuncName`
2. Updates `KernelInvokeOp` symbol references
3. Updates `KernelConfigDefOp` attributes (`kernel_name`, `kernel_file`)
4. Writes the user body verbatim to `<kernelName>.cc`

This is how `matmul` from `matrixmul.cc` gets its function body — the user writes the GEMM compute logic in Python (or C), and the compiler wraps it with the DMA/window infrastructure.

### 9.4 Output file set

| File | Generated by | Content |
|------|-------------|---------|
| `host.cc` | `translateToCpp(hostModule)` | XAie driver API calls: BD config, lock init, ELF load, core enable |
| `kernel.cc` | `translateToCpp(kernelModule)` | Window acquire/release wrapper |
| `<kernelName>.cc` | Direct write (user body or auto-gen) | Compute kernel (GEMM loops) |
| `routing.cc` | `translateToCpp(routingModule)` | Stream switch port/packet configuration |
| `aieml.bcf` | `TilingBcf::exportToFile()` | Tile-local memory layout (stack, reserved DMB, buffer symbols) |
| `aieml.prx` | `TilingPrx::exportToFile()` | xchesscc project file |

---

## 10. Cross-compilation and Deployment

### 10.1 Kernel compilation

```
kernel.cc + <kernelName>.cc → xchesscc (Synopsys AIE compiler)
  → kernel ELF (AIE core binary)

Configuration:
  - aieml.bcf provides symbol addresses and memory layout
  - aieml.prx provides project settings
```

### 10.2 Host compilation

```
host.cc → aarch64-g++ (cross-compiler)
  → host ELF (ARM binary)

Links with:
  - kernel.o (kernel ELF embedded via ld -r -b binary)
  - aie_runtime.c (XAie driver wrapper from include/aie_runtime.h)
  - routing.cc (stream switch configuration)
```

### 10.3 Deployment

The host ELF runs on the ARM processor and uses XAie driver APIs to:
1. Initialize the AIE device
2. Load kernel ELFs to AIE cores
3. Configure DMA buffer descriptors and locks
4. Configure stream switch routing
5. Enable cores and wait for completion
6. Read results from DDR

---

## 11. Build System Integration

### 11.1 `_aietriton_core.so` Build (`aietriton/CMakeLists.txt`)

The pybind11 shared object is built as part of the main `mlirfront` CMake project:

```cmake
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

pybind11_add_module(_aietriton_core aietriton_pybind.cpp)

target_link_libraries(_aietriton_core PRIVATE
    mlirtestlib          # TilingLinalgPipeline + all dialect managers
    clangTooling clangBasic clangASTMatchers
    MLIREmitCDialect MLIRTargetCpp
    MLIRIR MLIRParser MLIRPass MLIRTransforms MLIRSupport
    MLIRFuncDialect MLIRLinalgDialect MLIRMemRefDialect
    ... (same MLIR libs as mlirtest executable)
)

install(TARGETS _aietriton_core DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
```

Key points:
- Links against `mlirtestlib` (the static library containing all C++ pipeline code) — same library the `mlirtest` executable uses
- Include paths mirror the `mlirfront/CMakeLists.txt` include structure (all dialect `inc/` directories)
- `install` target copies the built `.so` next to `__init__.py` so `import aietriton` can find it

### 11.2 Integration with Parent Build (`mlirfront/CMakeLists.txt`)

```cmake
# aie_triton Python bindings (pybind11)
add_subdirectory(aietriton)
```

Added at the end of `mlirfront/CMakeLists.txt`, after `mlirtest` and `mlirtestlib` targets are defined. The `_aietriton_core` target depends on `mlirtestlib` which depends on all tablegen custom targets.

### 11.3 Build and Run

```bash
# Build (from project root)
cd build && cmake .. -DLLVM_INSTALL_DIR=/path/to/llvm/build && make -j$(nproc)
# Produces: build/src/mlir/mlirfront/aietriton/_aietriton_core.cpython-3X-*.so

# Install .so next to Python package
make install
# Copies .so to: src/mlir/mlirfront/aietriton/_aietriton_core.cpython-3X-*.so

# Set PYTHONPATH so 'import aietriton' resolves
export PYTHONPATH=src/mlir/mlirfront

# Run
python3 example/tileprogram/design/triton/triton_matmul.py
```

---

## 12. Key Implementation Files

| File | Role |
|------|------|
| `aietriton/__init__.py` | Python package API: `jit` decorator, `_JitKernel`, `_KernelLauncher`, `mesh()`, `set_device()`, `synchronize()` |
| `aietriton/language.py` | `tl.` stub namespace: type classes (`_DType`, `constexpr`), op stubs (`load`, `store`, `dot`, `zeros`, `arange`, etc.) |
| `aietriton/_compiler.py` | AST analysis engine: `compile_and_run()`, `_classify_params()`, `_is_input_tensor()`, `_extract_kernel_body()` |
| `aietriton/aietriton_pybind.cpp` | pybind11 C++ bridge: `_aietriton_core.run_aie_pipeline()` wrapping `TilingLinalgPipeline` |
| `aietriton/CMakeLists.txt` | Build `_aietriton_core.so`: `pybind11_add_module`, links `mlirtestlib` + MLIR libs |
| `tilinglinalg/pass/tilinglinalg_pipeline.h` | `TensorParam` struct, `buildRoutingIR()` and `runPipeline()` public API |
| `tilinglinalg/pass/tilinglinalg_pipeline.cpp` | Pipeline orchestration: pass ordering, module cloning, EmitC output, BCF/PRX generation |
| `tilinglinalg/routing/routingmanager.cpp` | `createroutingfuncGEMM()` (GEMM-specific routing with B broadcast), `createroutingfuncByDim()` (per-tensor generic routing) |
| `tilinglinalg/routing/` | Routing dialect: abstract tile arrays, mesh partitioning, data IO ops |
| `tilinglinalg/dataflowmap/dmap/` | dmap dialect: logical dataflow — IO engines, core groups, streams |
| `tilinglinalg/dataflowmap/dmaphop/` | dmaphop dialect: physical tile-to-tile hops, concrete port assignments |
| `tilinglinalg/dataflowmap/dfscheblueprint/` | dfscheblueprint dialect: schedule blueprints, transfer manifests |
| `tilinglinalg/dataflowmap/dfschedule/` | dfschedule dialect: executable schedules — BD chains, locks, kernel load/enable |
| `tilinglinalg/pass/routingimplement/` | BFS path finding (`RoutingPath`), topology model (`RoutingTopology`), resource tracking (`ResourceManager`) |
| `tilinglinalg/pass/kernelconfig.h` | `TilingBcf` and `TilingPrx` for BCF/PRX file generation |

---

## 13. Design Decisions

### `createroutingfuncByDim` (per-tensor) vs `createroutingfuncGEMM` (GEMM-specific)

`buildRoutingIR` uses `createroutingfuncByDim` — it iterates over the `TensorParam[]` vector and creates an independent routing region for each tensor. This is **generic**: it works for any tensor count and doesn't embed GEMM-specific knowledge.

`createroutingfuncGEMM` (used by `ops_testNew`) creates all three tensors' routing in a single `scf.execute_region`, with GEMM-specific handling: B uses `splitnum=1` for broadcast, C uses `routinggatherout`. This is **optimized for GEMM** but less reusable.

The per-tensor approach (`buildRoutingIR`) treats each tensor uniformly with `splitnum=meshRows`. The downstream passes then handle the actual data distribution. This trade-off favors generality over GEMM-specific optimization at the routing IR level.

### Why M-dimension split (not K or N)

Row-partitioning (split along M) is natural for output-stationary dataflow:
- Each tile group owns a contiguous row slice of C
- No reduction across tiles is needed (each tile computes its full dot product across K)
- The `gatherout` at the end simply concatenates row slices back to DDR
- Splitting along K would require cross-tile accumulation; splitting along N would produce column-interleaved output

### Why broadcast B (`splitnum=1` in GEMM-specific path)

Each tile needs the full K-dimension of B to compute its row slice of C:
```
C_tile[M_TILE x N] = A_tile[M_TILE x K] * B[K x N]
```
Broadcasting the full B avoids partial products and cross-tile reduction.

### Why 2 ping-pong iterations in `matrixmul.cc`

The `for (int iter = 0; iter < 2; iter++)` loop in `matrixmul.cc` matches `BlueprintToSchedulePass`'s BD pair allocation:
- 2 BDs per data stream enable double buffering
- While the kernel processes buffer 0, DMA fills buffer 1
- On next iteration, roles swap
- For this 16x16 GEMM with K=16, BLOCK_K=8, the 2 iterations also correspond to the 2 K-chunks

The `BlueprintToSchedulePass(0.5)` parameter (0.5) controls the buffer allocation ratio for ping-pong scheduling.

### Package naming: `aietriton` (directory) vs `aie_triton` (import)

The Python package directory is named `aietriton` (no underscore), matching the C++ naming convention used elsewhere in the project. The example script `triton_matmul.py` uses `import aie_triton`, which requires either renaming the directory to `aie_triton` or using an import alias. Current state: the directory is `aietriton` and `import aietriton` is the working import path. The `aie_triton` name in the example is aspirational — a mismatch that needs resolution (either rename the directory or update the example import).

### Lazy pybind11 import

`_aietriton_core` is imported inside `compile_and_run()`, not at package load time (`__init__.py`):

```python
def compile_and_run(fn, name, grid, args, kwargs):
    ...
    from . import _aietriton_core   # ← imported here, not at top-level
```

This allows `import aietriton` to succeed even without a built `_aietriton_core.so`. Benefits:
- IDE autocomplete works without building the C++ extension
- The AST parser (`_classify_params`, `_is_input_tensor`) can be tested standalone
- Clean error message at call time ("module not found") rather than import-time failure

### Stub-based language module

`language.py` functions (`load`, `store`, `dot`, `zeros`, etc.) return `None` or `0` — they exist so `@aie_triton.jit` kernel functions can be **parsed** by Python's `ast` module without execution errors.

The kernel function body is never actually executed in Python. Instead:
1. `inspect.getsource(fn)` retrieves the source text
2. `ast.parse()` builds an AST tree
3. `_classify_params()` and `_is_input_tensor()` analyze the AST statically

This mirrors Triton's approach where kernel code is compiled to GPU IR, not interpreted as Python. The stubs allow kernel functions to be valid Python (importable, type-checkable) while their semantics are defined by the compiler, not by Python execution.
