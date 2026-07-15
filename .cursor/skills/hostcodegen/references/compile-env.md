<!-- Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->

# Compile environment and hostcompile.sh

## Setup

Source the repo env before compiling:

```bash
# From repo root
source script/setup.sh --path-set-only
```

This sets `XILINX_VITIS` and paths. For hostcompile.sh, `PATH_SET_ONLY=1` is used so setup only sets paths (no full BSP/install).

## hostcompile.sh

- **Location**: `src/mlir/mlirfront/tilinglinalg/pass/unitest/worklocal/hostcompile.sh`
- **Usage**: Run from worklocal, or invoke with script path. Requires `host.cc` in worklocal (generate first via unitest).

**Options** (match aiehlc.sh):

| Option / env | Effect |
|--------------|--------|
| `--aie-version 2` or `--aie-version 5` | AIE generation; 2 = AIEML (cortex-a72), 5 = AIE2PS (cortex-a78). Default: 2; auto-switches to 5 if only libxaienginea78.a exists. |
| `--platform baremetal` or `--platform linux` | baremetal → `aarch64-none-elf-`, linux → `aarch64-linux-gnu-`. Default: baremetal. |
| `CROSS_COMPILE` | Override toolchain prefix (e.g. `CROSS_COMPILE=aarch64-none-elf-`). |
| `AIE_VERSION` / `PLATFORM` | Same as options if set in env. |

**Defines**: `-DAIE_GEN=<aie_version>` (1, 2, or 5). Must match runtime and XAie lib.

**Output**: `worklocal/build/host` (ELF).

## Toolchain

- **Baremetal**: `aarch64-none-elf-g++` / `gcc`, `-mcpu=cortex-a72` (gen 1/2) or `-mcpu=cortex-a78` (gen 5).
- **Linux**: `aarch64-linux-gnu-g++` / `gcc`.

Linking uses BSP libs (e.g. `thirdparty/arch/...`), `libxaiengine*`, and `aie_runtime.c`. Runtime source: `src/mlir/runtime/aie_runtime.c`.

## Common errors and fixes

| Symptom | Cause | Fix |
|---------|--------|-----|
| `host.cc not found` | Unitest not run or wrong worklocal | Run unitest (`./test`) from unitest build dir first; confirm worklocal path in hostcompile.sh. |
| `XILINX_VITIS not set` / undefined `XAie_*` | Env not sourced | `source script/setup.sh --path-set-only` from repo root; re-run hostcompile.sh. |
| Wrong AIE_GEN / AieGen mismatch | Build uses different gen than runtime or lib | Use same `--aie-version` (and AIE_GEN define) as the XAie lib and aie_runtime.c (e.g. 5 for AIE2PS). |
| Link errors (missing BSP / xil / timer) | Missing arch or lib path | Ensure thirdparty/arch and BSP layout exist; hostcompile.sh expects arch dirs under `thirdparty/arch` (e.g. psv_cortexa72_0, cortexa78_0). |
| Compile error in host.cc (e.g. `void main`, missing `return 0`) | Generated C++ quirks | hostcompile.sh applies sed to add `host_canonicalized` forward decl, `int main()`, `return 0`, `__global__`. If new quirks appear, fix in DfscheduleToApi/emitc or add sed in hostcompile.sh. |

## Quick check

From repo root after generating host.cc:

```bash
source script/setup.sh --path-set-only
cd src/mlir/mlirfront/tilinglinalg/pass/unitest/worklocal
./hostcompile.sh
# ELF: src/mlir/mlirfront/tilinglinalg/pass/unitest/worklocal/build/host
```
