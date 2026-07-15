<!-- Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->
---
name: hostcodegen
description: Host code generation, compilation, HW run, verification, and bug fixing for the aiehlc AIE flow. Use when you need to (1) Generate host/kernel C++ from the MLIR unitest pipeline, (2) Compile host + aie_runtime for baremetal/linux, (3) Run the host ELF on PAL/HW and capture console, (4) Verify output or debug failures, (5) Fix bugs in codegen, generated host.cc, or runtime.
---

# Hostcodegen

This skill drives the generate → compile → run on HW → verify → fix loop for aiehlc host code.

## Workflow

1. **Generate**: Build the unitest, run `./test` (e.g. dfschedule). Outputs: `worklocal/host.cc`, `worklocal/kernel.cc`. For key paths see [references/project-layout.md](references/project-layout.md).

2. **Compile**: From worklocal (or with correct paths), run `hostcompile.sh`. ELF output: `worklocal/build/host`. For env and options see [references/compile-env.md](references/compile-env.md).

3. **Run on HW** (optional): Run `apppaltest.py <path-to-ELF>` (e.g. `./worklocal/build/host`). Set `USERNAME`, `PALIP`, `BOARDNAME` or use `script/test/envlocal.sh`. Details: [references/hw-test.md](references/hw-test.md).

4. **Verify**: Run [src/mlir/mlirfront/tilinglinalg/pass/unitest/piplinerun.sh](src/mlir/mlirfront/tilinglinalg/pass/unitest/piplinerun.sh) (from repo root) to run generate + compile + HW test and check console for success/errors. Or run apppaltest.py manually and check console for expected prints and XAie errors.

5. **Bug fix** (decision flow):
   - **Pass pipeline / codegen error** → Fix in MLIR passes or unitest (e.g. `passdfscheduletoapi`, `src/mlir/mlirfront/tilinglinalg/pass/unitest/test.cpp`).
   - **Compile error in host.cc** → Fix generation (emitc/DfscheduleToApi) or apply local fix in hostcompile.sh sed for known C++ quirks (e.g. `void main` → `int main`).
   - **Link/runtime error** → Check `src/mlir/runtime/aie_runtime.c`, includes, and hostcompile.sh libs/flags.
   - **HW/board failure** → Check env, xsdb, and [references/hw-test.md](references/hw-test.md).

## Full pipeline

`piplinerun.sh` (from unitest dir) runs generate + compile + HW test in one go. Use it for automation when all steps are needed.

## Resources

| When | File |
|------|------|
| Paths for unitest, worklocal, host.cc, scripts | [references/project-layout.md](references/project-layout.md) |
| setup.sh, hostcompile.sh options, compile errors | [references/compile-env.md](references/compile-env.md) |
| apppaltest.py, USERNAME/PALIP/BOARDNAME, console output | [references/hw-test.md](references/hw-test.md) |
| Verify host on HW (generate + compile + run, check console) | [src/mlir/mlirfront/tilinglinalg/pass/unitest/piplinerun.sh](src/mlir/mlirfront/tilinglinalg/pass/unitest/piplinerun.sh) |