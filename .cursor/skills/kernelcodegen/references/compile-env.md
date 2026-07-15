<!-- Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->

# Kernel compile environment and compile_kernel.sh

## Setup

Source the repo env before compiling:

```bash
# From repo root
source script/setup.sh --path-set-only
```

Requires `XILINX_VITIS` (Vitis with aietools). compile_kernel.sh sources setup.sh from the repo when run from worklocal.

## compile_kernel.sh

- **Location**: `src/mlir/mlirfront/tilinglinalg/pass/unitest/worklocal/compile_kernel.sh`
- **Usage**: Run from worklocal. Requires `kernel.cc` in worklocal (generate first via unitest). If kernel.cc includes it, the pre-existing `compute_kernel.cc` must also be present.

**Steps** (AIE2PS):

1. **xchesscc**: Compile kernel.cc to LLVM IR (`kernel_orig.ll`). Flags: `-aiearch aie2ps -s +f -p me`, include paths for aietools and local alib.
2. **opt (pass 1)**: Run xlopt on kernel_orig.ll → kernel.ll.
3. **opt (pass 2)**: Run xlopt again on kernel.ll.
4. **xchessmk**: Link with aie2ps.prx to produce kernel ELF.

**Output**: `worklocal/build/kernel` (kernel ELF).

**Includes**: Script uses `XILINX_VITIS/aietools` and repo `thirdparty/alib/include`; if local alib has xaiengine, that is used as AIETOOLS include base.

## Common errors and fixes

| Symptom | Cause | Fix |
|---------|--------|-----|
| kernel.cc not found | Unitest not run or wrong worklocal | Run unitest (`./test`) from unitest build dir first. |
| XILINX_VITIS not set | Env not sourced | source script/setup.sh from repo root; compile_kernel.sh also sources it if AIEHLC_DIR/script/setup.sh exists. |
| xchesscc/opt/xchessmk not found | Vitis/aietools not in PATH | Ensure Vitis settings64.sh or script/setup.sh sets PATH to include aietools. |
| Undefined symbol / API mismatch | kernel.cc and aie_api or driver headers mismatch | Align kernel code with AIE version (aie2ps); check include paths in compile_kernel.sh. |
| `unknown type name 'window_internal'` or `undeclared identifier 'window_out_0_ping'/'window_out_0_pong'` | Codegen does not emit window types/helpers or uses wrong buffer names | In kernel.cc: (1) Add typedef for `window_internal`, `output_window_int8`, and inline `window_init`, `get_output_async_window_int8`, `acquire_output_window`, `release_output_window` (see kernel.bak.cc or example/work/kernel.cc). (2) Use `buf_out_ping_0`/`buf_out_pong_0` in `window_init`, not `window_out_0_ping`/`window_out_0_pong`. (3) Call `compute_kernel(get_output_async_window_int8(window_window_out_0))`. (4) Add `#include "compute_kernel.cc"` before main. |

## Quick check

From repo root after generating kernel.cc:

```bash
source script/setup.sh --path-set-only
cd src/mlir/mlirfront/tilinglinalg/pass/unitest/worklocal
./compile_kernel.sh
# Kernel ELF: build/kernel
```
