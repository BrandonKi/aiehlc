<!-- Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->

# Project layout (paths from repo root)

Use these paths when generating, compiling, or running host code. Resolve from repo root so they work from any cwd.

## Repo root

Workspace root (e.g. `/scratch/staff/huaj/aiehlc/aiehlc` or your clone). All paths below are relative to this.

## Unitest (generate host/kernel)

| Item | Path |
|------|------|
| Unitest directory | `src/mlir/mlirfront/tilinglinalg/pass/unitest` |
| Unitest build dir | `src/mlir/mlirfront/tilinglinalg/pass/unitest/build` |
| Unitest binary | `src/mlir/mlirfront/tilinglinalg/pass/unitest/build/test` |
| Full pipeline script | `src/mlir/mlirfront/tilinglinalg/pass/unitest/piplinerun.sh` |

Build: from unitest dir, `cd build && make`. Run: `./test` (or `./test dfschedule`). Generates host.cc and kernel.cc into worklocal.

## Worklocal (generated outputs and host compile)

| Item | Path |
|------|------|
| Worklocal directory | `src/mlir/mlirfront/tilinglinalg/pass/unitest/worklocal` |
| Generated host | `src/mlir/mlirfront/tilinglinalg/pass/unitest/worklocal/host.cc` |
| Generated kernel | `src/mlir/mlirfront/tilinglinalg/pass/unitest/worklocal/kernel.cc` |
| Host compile script | `src/mlir/mlirfront/tilinglinalg/pass/unitest/worklocal/hostcompile.sh` |
| Host ELF output | `src/mlir/mlirfront/tilinglinalg/pass/unitest/worklocal/build/host` |

Run hostcompile.sh from worklocal or pass correct paths; it sources setup.sh and compiles host.cc + aie_runtime.c.

## Runtime and env

| Item | Path |
|------|------|
| AIE runtime C | `src/mlir/runtime/aie_runtime.c` |
| Runtime header | `include/aie_runtime.h` |
| Device map header | `include/aie_device_map.h` |
| Env setup script | `script/setup.sh` |

## HW test

| Item | Path |
|------|------|
| Host verify script (run on HW + check console) | `script/test/verify_host.sh` |
| PAL board test script | `script/test/apppaltest.py` |
| Local env (USERNAME, PALIP, BOARDNAME) | `script/test/envlocal.sh` |

Run: `python3 script/test/apppaltest.py <path-to-ELF>` (e.g. `script/test/apppaltest.py src/mlir/mlirfront/tilinglinalg/pass/unitest/worklocal/build/host`). ELF is copied to remote and run via xsdb; console is captured on com0.

## Codegen (for bug fixes)

| Item | Path |
|------|------|
| Dfschedule to API pass | `src/mlir/mlirfront/tilinglinalg/pass/passdfscheduletoapi/` |
| Unitest driver (writes host.cc) | `src/mlir/mlirfront/tilinglinalg/pass/unitest/test.cpp` |

Host C++ is emitted via emitc from the host module after DfscheduleToApiPass; test.cpp writes it to worklocal/host.cc.
