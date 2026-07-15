# aietriton — Triton-style Python Front End for AIE

`aietriton` is a **Triton-style Python front end** that compiles an
`@aie_triton.jit`-decorated GEMM kernel into the AIE output file set
(`host.cc`, `kernel.cc`, `<kernel>.cc`, `routing.cc`, `aieml.bcf`, `aieml.prx`)
by calling the `tilinglinalg` C++ pipeline **directly** through a pybind11
module (`_aietriton_core`).

```
triton_matmul.py ──(import aietriton)──► _compiler.py (AST parse)
   │  tensor specs + kernel body
   ▼
_aietriton_core.run_aie_pipeline()  (pybind11 → C++)
   │  buildRoutingIR + runPipeline
   ▼
./worklocal/{host,kernel,<kernel>,routing}.cc + aieml.bcf/.prx
```

Unlike `rcom` (a pure-Python, pipeline-*external* tool that emits `.cc` source
and shells out to `aiehlc.sh`), `aietriton` is **pipeline-internal**: the Python
layer only parses the kernel AST to recover tensor shapes/dtypes and a C kernel
body, then hands them to the same C++ `TilingLinalgPipeline` used by the
`mlirtest` executable.

It is a *recognize-and-map* front end: the kernel body is not executed as
Python. `tl.*` calls are AST-matched to AIE primitives (window
acquire/release + a GEMM MAC body). Unrecognized bodies fall back to an
auto-generated GEMM compute kernel.

---

## 1. Layout

Package root: `src/mlir/mlirfront/frontend/aietriton/`

| File | Role |
|------|------|
| `__init__.py` | Public API: `jit` decorator, `mesh()`, `set_device()`, `synchronize()`, `_JitKernel`, `_KernelLauncher` |
| `language.py` | `tl.` stub namespace: types (`int8/16/32`, `float32`, `constexpr`) + op stubs (`load`, `store`, `dot`, `zeros`, `arange`, `program_id`, `make_block_ptr`, `advance`). Parsed via AST, **never executed** |
| `_compiler.py` | AST engine: `compile_and_run()`, `_classify_params()`, `_is_input_tensor()`, `_extract_kernel_body()` |
| `aietriton_pybind.cpp` | pybind11 bridge: `_aietriton_core.run_aie_pipeline()` and `build_kernel_body()` |
| `aie_pass/ast_to_kernelops.py` | Python AST → flat `KernelOp` dict list |
| `aie_pass/kernel_body_emitter.{h,cpp}` | C++ `KernelBodyEmitter`: `KernelOp[]` → MLIR EmitC `VerbatimOp` → `translateToCpp` → C string |
| `aie_pass/test_ast_to_c.py` | Unit test: AST → KernelOps → C |
| `CMakeLists.txt` | Builds `_aietriton_core.so`, links `mlirtestlib` + MLIR libs |
| `architecture.md` | Deep-dive: every compiler stage, pass ordering, and Triton↔C mapping |

Examples: `example/tileprogram/design/triton/triton_matmul.py`,
`example/tileprogram/design/triton/resnet18_triton.py`.

---

## 2. How to use

### 2.1 Build the pybind module

`_aietriton_core.so` is built as part of the main `mlirfront` CMake project and
installed next to the Python package:

```bash
cd build
cmake .. -DLLVM_INSTALL_DIR=/path/to/llvm/build
make -j$(nproc) _aietriton_core
make install                     # copies the .so next to __init__.py
```

> Without the `.so`, `import aietriton` still succeeds (the pybind module is
> imported lazily inside `compile_and_run()`), so the AST parser can be tested
> standalone. The `.so` is only required when you actually launch a kernel.

### 2.2 Set the import path

The package **directory is `aietriton`**, so the working import is
`import aietriton`:

```bash
export PYTHONPATH=src/mlir/mlirfront/frontend:$PYTHONPATH
python3 -c "import aietriton; print('ok')"
```

> **Naming note:** the example scripts use `import aie_triton` (with an
> underscore), which does **not** match the `aietriton` directory. To run those
> examples unchanged, either edit the import to `import aietriton as aie_triton`,
> or expose the package under the `aie_triton` name on `PYTHONPATH`. This
> mismatch is tracked in `architecture.md` §13.

### 2.3 Write a kernel

```python
import aietriton as aie_triton
import aietriton.language as tl
import numpy as np

@aie_triton.jit
def matmul(a_ptr, b_ptr, c_ptr, M, N, K,
           BLOCK_M: tl.constexpr = 8,
           BLOCK_N: tl.constexpr = 8,
           BLOCK_K: tl.constexpr = 8):
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + ...)     # → acquire_input_window(window_in_0)
        b = tl.load(b_ptr + ...)     # → acquire_input_window(window_in_1)
        acc += tl.dot(a, b)          # → GEMM MAC body
    tl.store(c_ptr + ..., acc.to(tl.int8))   # → acquire/write/release output
```

### 2.4 Launch (triggers compilation)

```python
M, N, K = 16, 16, 16
A = np.arange(1, M*K+1, dtype=np.int8).reshape(M, K)
B = np.arange(1, K*N+1, dtype=np.int8).reshape(K, N)
C = np.zeros((M, N), dtype=np.int8)

aie_triton.set_device(0)
mesh = aie_triton.mesh(rows=2, cols=2)
grid = (mesh.rows, mesh.cols)

matmul[grid](A, B, C, M, N, K, BLOCK_M=8, BLOCK_N=8, BLOCK_K=8)
aie_triton.synchronize()
```

Compilation happens at the `matmul[grid](...)` call. Output files are written to
`./worklocal/`. From there, cross-compile with the normal scripts:

```bash
WORKLOCAL_DIR=worklocal source script/compile_kernel.sh   # kernel ELF
WORKLOCAL_DIR=worklocal source script/hostcompile.sh      # host ELF
python3 script/test/apppaltest.py worklocal/build/host    # run on board
```

### 2.5 Run the example

```bash
export PYTHONPATH=src/mlir/mlirfront/frontend
python3 example/tileprogram/design/triton/triton_matmul.py   # edit import first (see 2.2)
```

---

## 3. Launch protocol (API surface)

Follows Triton's `kernel[grid](*args)` shape (`__init__.py`):

| Call | Result |
|------|--------|
| `@aie_triton.jit` | `_JitKernel(fn)` — stores the function; parses nothing yet |
| `kernel[grid]` | `_JitKernel.__getitem__` → `_KernelLauncher(fn, name, grid)`; `grid = (rows, cols)` |
| `kernel[grid](A, B, C, ...)` | `_KernelLauncher.__call__` → `compile_and_run(fn, name, grid, args, kwargs)` |
| `aie_triton.mesh(rows, cols)` | `_Mesh` holding mesh dims → `buildRoutingIR(ctx, rows, cols, ...)` |
| `aie_triton.set_device(id)` / `synchronize()` | Host-side no-ops in this front end (present for API parity) |

---

## 4. Architecture

`compile_and_run()` (`_compiler.py`) runs three phases, then calls C++.

```
             _compiler.py: compile_and_run()
 ┌───────────────────────────────────────────────────────────────┐
 │  Phase 1  AST parse + tensor specs                            │
 │    inspect.getsource(fn) → ast.parse → _find_kernel_def       │
 │    _classify_params()   → tensor / scalar / constexpr         │
 │    _is_input_tensor()   → target of tl.store ⇒ output         │
 │    numpy args           → shape, bits, is_input               │
 │        → tensor_specs = [(shape, bits, is_input), ...]        │
 │                                                               │
 │  Phase 2  Kernel body extraction (aie_pass/)                  │
 │    ast_to_kernel_ops()  → KernelOp dict list                 │
 │    _aietriton_core.build_kernel_body() → C string            │
 │    (empty ⇒ fall back to auto-generated GEMM kernel)          │
 │                                                               │
 │  Phase 3  C++ pipeline invocation (pybind11)                  │
 │    _aietriton_core.run_aie_pipeline(                          │
 │        rows, cols, tensor_specs, "./worklocal",              │
 │        kernel_body, name)                                     │
 └───────────────────────────────────────────────────────────────┘
                              │
                              ▼
   TilingLinalgPipeline (C++):
     buildRoutingIR(ctx, rows, cols, TensorParam[])   → routing IR
     runPipeline(ctx, module, outDir, body, name):
       routing → dmap → dmaphop → dfscheblueprint     (shared)
         ├─ host   path → host.cc
         ├─ kernel path → kernel.cc
         └─ routing path → routing.cc  (+ aieml.bcf / aieml.prx)
```

### 4.1 Parameter classification (`_classify_params`)

For each function parameter:

- annotated `tl.constexpr` → **constexpr** (folded into codegen, e.g. `BLOCK_*`);
- name ends with `_ptr` or in `{A, B, C, a_ptr, b_ptr, c_ptr}` → **tensor**;
- otherwise → **scalar** (e.g. `M, N, K`, strides — not in the C kernel signature).

### 4.2 Direction inference (`_is_input_tensor`)

A tensor that appears as the **first argument of `tl.store(param, ...)`** is an
**output** (`isInput=false`); every other tensor is an **input**. This drives
`window_in_N` / `window_out_N` numbering and the `numInputWindows` /
`numOutputWindows` counts.

### 4.3 Dtype → window element type

`_numpy_dtype_to_bits(dtype) = dtype.itemsize * 8` → `int8/16/32`
(`_bits_to_element_type`), which selects the window type (`input_window_int8`,
etc.). The element type is taken from the first tensor argument.

### 4.4 Kernel body: AST → KernelOps → EmitC → C

`aie_pass/` translates the recognized kernel body into C via a structured IR
(schema in `ast_to_kernelops.py`):

| Python AST pattern | KernelOp |
|--------------------|----------|
| `tl.program_id(axis=N)` | `{"op": "get_coreid"}` (emitted once) |
| `tl.zeros(shape, dtype)` | tracked as accumulator, no op |
| `tl.load(a_ptr + ...)` | `{"op": "acquire_input", "window_idx": N, "var": "inN"}` |
| `tl.dot(a, b)` (or `acc += tl.dot(...)`) | `{"op": "gemm_body", "m", "n", "k"}` (from constexpr) |
| `acc.to(tl.int8)` | folded into `gemm_body` (saturating cast) |
| `tl.store(c_ptr + ..., v)` | `acquire_output` + `release_output` |
| `for k in range(0, K, BLOCK_K)` | `for_range` (trip count from constexpr) … `end_for` |
| `tl.make_block_ptr` / `tl.advance` | tracked, no op (strided variant normalizes to the same ops) |

`_ASTWalker` auto-emits input/output **releases** at the end of each `for` body.
The list crosses the pybind11 boundary to `KernelBodyEmitter`, which builds one
`emitc::VerbatimOp` per KernelOp and calls `mlir::emitc::translateToCpp()`. The
resulting C string is passed back as `userKernelBody` and written verbatim to
`<kernel>.cc`.

### 4.5 pybind11 bridge (`aietriton_pybind.cpp`)

| Entry point | Signature (Python → C++) |
|-------------|--------------------------|
| `run_aie_pipeline` | `(int mesh_rows, int mesh_cols, list[(shape, bits, isInput)] tensor_specs, str output_dir, str user_kernel_body="", str user_kernel_func_name="", list[(splitDim, hwAxisOwner, replicateOn)] split_specs=[]) → bool` |
| `build_kernel_body` | `(str kernel_name, str element_type, int num_input_windows, int num_output_windows, list[KernelOp] kernel_ops) → str` |

`run_aie_pipeline` builds `std::vector<TensorParam>`, registers the six custom
dialects, builds a `SplitModel` (from `split_specs`, or `SplitModel::gemm()` when
omitted — as `_compiler.py` does today), then calls `buildRoutingIR` +
`runPipeline`. `build_kernel_body` converts the KernelOp dicts to
`std::vector<KernelOp>` and runs `KernelBodyEmitter::emit`.

---

## 5. Output file set

Written to `./worklocal/` (the `output_dir` in `_compiler.py`):

| File | Content |
|------|---------|
| `host.cc` | XAie driver calls: DMA BD config, locks, ELF load, core enable |
| `kernel.cc` | Window acquire/release wrapper |
| `<kernel>.cc` | Compute kernel body (from `@aie_triton.jit`, or auto-generated GEMM) |
| `routing.cc` | Stream-switch port/packet configuration |
| `aieml.bcf` | Tile-local memory layout for the linker |
| `aieml.prx` | xchesscc project file |

---

## 6. Testing

```bash
# AST → KernelOps only (no .so needed):
cd src/mlir/mlirfront/frontend && python3 -m aietriton.aie_pass.test_ast_to_c

# Full pipeline incl. C generation (requires built _aietriton_core.so):
cd src/mlir/mlirfront/frontend && python3 -m aietriton.aie_pass.test_ast_to_c --full
```

---

## 7. Limitations

- **GEMM-oriented:** the AST→C body path recognizes the matmul pattern
  (`tl.load` / `tl.dot` / `tl.store` in a K-loop). Other bodies fall back to the
  auto-generated GEMM kernel — they are *recognized*, not faithfully translated.
- **Split strategy defaults to GEMM:** when `_compiler.py` calls
  `run_aie_pipeline` it omits `split_specs`, so the C++ side uses
  `SplitModel::gemm()`. A custom per-tensor split requires passing `split_specs`
  (list of `(splitDim, hwAxisOwner, replicateOn)` tuples), which the current
  Python layer does not expose.
- **Import name mismatch:** examples import `aie_triton`; the package is
  `aietriton` (see §2.2).
- `set_device` / `synchronize` are API-parity no-ops.

---

## 8. See also

- `architecture.md` — full stage-by-stage lowering trace and Triton↔C tables.
- `../rcom/README.md` — the sibling ROCm/HIP front end (pipeline-external).
- `doc/tilinglinalg.md`, `doc/lowering.md` — the C++ pipeline this front end drives.
