# rcom — ROCm/HIP → AIE Matmul Front End

`rcom` is a standalone Python front end that turns a **ROCm/HIP** int8 GEMM
source into a **CUDA-style `.cc` + `.h`** (the same dialect as
`example/tileprogram/ccode/simplematmul2.cc`) and then drives the **unchanged**
`script/aiehlc.sh` pipeline to build host + kernel **ELFs**.

```
matmul_hip.cpp ──(rcom.py)──► <name>.cc + <name>.h ──(aiehlc.sh)──► aout/main.elf
```

It is a *recognize-and-template* front end, not a source-to-source translator:
rcom confirms the HIP kernel is a GEMM, then emits the proven AIE
"cache-A / stream-B" matmul template rather than translating the thread-indexed
HIP body line-for-line.

---

## Quick start (how to compile)

Always initialise the toolchain first:

```bash
source script/setup.sh
```

### One-shot: HIP source → ELF

```bash
source script/rcom.sh --rocm-source-file example/tileprogram/rocm/matmul_hip.cpp --aie-version 5
# Produces aout/main.elf
```

### Two-step: generate, inspect, then build

```bash
# 1. Emit the generated AIE source (no build)
python3 src/tool/frontend/rcom.py example/tileprogram/rocm/matmul_hip.cpp \
    --emit-only --out ./gen
#    → ./gen/matmul.cc  and  ./gen/matmul.h

# 2. Build it with the normal pipeline
source script/aiehlc.sh --aie-version 5 --runtime-source-file ./gen/matmul.cc
```

### Run on hardware (data check)

```bash
python3 ./script/test/apppaltest.py -y -nonreboot > ./applog 2>&1
# PASS: "PASS: all ... match" + "device_teardown done", no "AIE ERROR"
```

> Note: `aiehlc.sh` runs `rm -rf $(pwd)/aout/` at start, so `rcom.sh` writes the
> generated files **outside** `aout/` (default `example/tileprogram/rocm/gen/`).
> If you pass `--out` yourself, keep it outside `aout/`.

---

## CLI reference

`rcom.py <hip_src> [options]`

| Option | Default | Meaning |
|---|---|---|
| `--name NAME` | `matmul` | Generated kernel / function name (and `<NAME>_policy`). |
| `--out DIR` | `./gen` | Output directory for `<name>.cc` / `<name>.h`. |
| `--mesh RxC` | `4x4` | AIE tile mesh (rows × cols) → `HW_ROWS`/`HW_COLS`. |
| `--M / --N / --K` | from source | Override GEMM dims (CLI wins over `#define`). |
| `--aie-version V` | `5` | Passed through to `aiehlc.sh` when `--build`. |
| `--emit-only` | (default) | Only generate the `.cc`/`.h`. |
| `--build` | — | Generate, then invoke `script/aiehlc.sh` to build ELFs. |

`script/rcom.sh` mirrors `aiehlc.sh` arg style:
`--rocm-source-file <hip> [--aie-version 5] [--mesh RxC] [--name NAME] [--out DIR]`.

---

## Architecture

rcom is a thin, dependency-free (`stdlib`-only) tool with four stages, all in
`rcom.py`.

```
                        rcom.py
 ┌───────────────────────────────────────────────────────────────┐
 │  hip_src                                                        │
 │     │                                                          │
 │     ▼                                                          │
 │  ┌───────────────┐   dict{name,dtype,                          │
 │  │  parse_hip()  │──► inputs,output,M,N,K}                      │
 │  └───────────────┘                                             │
 │     │  (validated GEMM)                                        │
 │     ▼                                                          │
 │  ┌────────────────────┐   (m_tile, n_tile, k_chunk)           │
 │  │  compute_tiling()  │──►                                     │
 │  └────────────────────┘                                       │
 │     │                                                          │
 │     ▼                                                          │
 │  ┌───────────────────────┐   <name>.cc  (kernel + host main)  │
 │  │  emit_cc() / emit_h()  │──► <name>.h  (M/K/N, HW dims,      │
 │  └───────────────────────┘       scalar_matmul, verify)       │
 │     │                                                          │
 │     ▼  (--build)                                               │
 │  subprocess ► script/aiehlc.sh ► aout/main.elf                │
 └───────────────────────────────────────────────────────────────┘
```

### 1. HIP parser — `parse_hip(src)`

Lightweight regex/text extraction (comments stripped first) from the canonical
HIP GEMM:

- **Kernel signature:** finds `__global__ void NAME(<params>)` and extracts the
  balanced-brace body.
- **Pointer params & dtype:** each `*` param's element type is classified;
  `int8_t` / `signed char` / `char` → `int8`. Anything else → hard error
  (v1 is int8-only).
- **Direction:** the pointer written via `NAME[...] = ...` is the **output**;
  the other two pointer params are **inputs A, B** (fallback: 3rd param is
  output).
- **Dims M/N/K:** from `#define` or `const int` (CLI overrides win).
- **GEMM validation:** requires an output store, a MAC (`sum += a*b` or
  `a[..] * b[..]`), and a `k` loop; otherwise raises `ParseError`
  ("not a recognized GEMM") so bad input fails fast with no silent wrong output.

### 2. Tiling heuristics — `compute_tiling(M, N, K)`

Returns `(m_tile, n_tile, k_chunk)`. For the HW-validated `M=N=K=256` config it
reproduces the proven values exactly (`m_tile=n_tile=16`, `k_chunk=64`). For
other dims it uses documented heuristics: `tile_size≈16` (falls back to the full
dim when not divisible by 16) and a K-chunk chosen so `K % chunk == 0`
(`K/4` when divisible). Deviating from the proven config prints a warning
because only that path is HW-validated.

### 3. Code generator — `emit_cc(cfg)` / `emit_h(cfg)`

Python string templates (`_CC_TEMPLATE`, `_H_TEMPLATE`) derived verbatim from
`simplematmul2.cc` + `simplematmul.h`, with `@@TOKEN@@` substitution points:
`M/K/N`, `HW_ROWS/HW_COLS`, kernel name, `<name>_policy`, the GemmSpace
`d1/d2 {tile_size,stride}` + K-chunk, and the partition end-column
(`{0, cols-1, 0, 6}`). The emitted `.cc` contains the three `GemmSpace`
policies (`RowBA`, `ColBB`, `LtoR_Merge`), the `__global__(<name>_policy)`
kernel, and the host `main()` (device partition, alloc, init, `<name><<<mesh>>>`,
`verify_matmul`).

### 4. Build delegation

`--build` (or `script/rcom.sh`) shells out to `script/aiehlc.sh
--aie-version <v> --runtime-source-file <out>/<name>.cc`, which runs the aiehlc
Clang→MLIR→EmitC pipeline and links the ELFs. rcom makes **no** C++/MLIR
changes — it sits strictly in front of the existing pipeline.

### HIP → AIE `.cc` mapping

| HIP construct | Generated AIE `.cc` |
|---|---|
| `__global__ void gemm(A,B,C,M,N,K)` | `__global__(matmul_policy) void matmul(port<...,RowBA> win_a, ...ColBB win_b, ...LtoR_Merge win_c)` |
| int8 pointer params | `input_window_int8*` / `output_window_int8*` |
| thread-indexed `sum+=A*B; C=sat(sum)` | proven cache-A / stream-B kernel body |
| host dims `M,N,K` | `#define M/K/N` in `<name>.h` |
| launch grid/block | `#define HW_ROWS/HW_COLS` + `device.partition({0,cols-1,0,6}, HW_ROWS, HW_COLS)` |
| `hipMalloc`/`hipMemcpy`/launch/verify | host `main()`: `device.alloc`, init loops, `matmul<<<mesh>>>`, `verify_matmul` |

---

## Files

| Path | Role |
|---|---|
| `src/tool/frontend/rcom.py` | The tool (parser + tiling + generator + CLI). |
| `script/rcom.sh` | Wrapper: `rcom.py --emit-only` then `source aiehlc.sh`. |
| `example/tileprogram/rocm/matmul_hip.cpp` | Sample canonical HIP int8 GEMM input. |

## Limitations (v1)

- **int8 only**; other dtypes error out in the parser.
- Only the **`M=N=K=256`, 4×4 mesh** config is HW-validated. Other dims/mesh use
  heuristic tile sizing and emit a warning (untested on hardware).
- The kernel body is **templated, not faithfully translated** — a custom HIP
  GEMM body (fused epilogue, different accumulation) is recognized as a GEMM but
  emitted as the standard matmul template.

## Relationship to `aietriton`

`rcom` is independent from the `aietriton` pybind front end
(`src/mlir/mlirfront/aietriton/`). It reuses only the *ideas* (input/output
tensor classification, dtype detection) but is a pure-Python, pipeline-external
tool that emits C++ source, whereas `aietriton` calls the C++ pipeline directly
via pybind11.
