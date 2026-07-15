<!-- Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->

# Kernel code generation flow

Code flow that produces `worklocal/kernel.cc`. Entry: unitest `./test dfschedule` (function `routingtodfschedule()` in `src/mlir/mlirfront/tilinglinalg/pass/unitest/test.cpp`).

## Files involved in the generate flow

| File | Role |
|------|------|
| `src/mlir/mlirfront/tilinglinalg/pass/unitest/test.cpp` | Driver: builds shared pipeline to dfscheblueprint, clones module, runs kernel passes, calls emitc and writes kernel.cc. |
| `src/mlir/mlirfront/tilinglinalg/pass/passblueprinttoschedulekernel/passblueprinttoschedulekernel.cpp` | **In charge of kernel dfschedule generation**: BlueprintToScheduleKernelPass turns dfscheblueprint IR into kernel schedule IR. |
| `src/mlir/mlirfront/tilinglinalg/pass/passdfscheduletokernelapi/passdfscheduletokernelapi.cpp` | **In charge of kernel.cc generation**: DfscheduleToKernelApiPass lowers kernel schedule IR to EmitC (kernel API); emitc::translateToCpp then produces the C++ file kernel.cc. |
| (shared pipeline) Routing/Dmap/Dmaphop/DmaphopTodfscheblueprint pass .cpp files | Produce the blueprint IR that is cloned for the kernel path; see §1 below. |

So for the **kernel** path specifically: **passblueprinttoschedulekernel.cpp** is responsible for kernel dfschedule; **passdfscheduletokernelapi.cpp** is responsible for the lowering that leads to kernel.cc (plus test.cpp for orchestration and emitc write).

## 1. Shared pipeline (up to dfscheblueprint)

Input: routing module from `mtest.ops_testNew(&ctx, 1)`.

| Stage | Pass | Result |
|-------|------|--------|
| 1 | RoutingUnrollingLowerPass | Unrolled routing |
| 2 | RoutingToDmapPass(rtopology) | Dmap IR |
| 3 | DmapToDmaphopPass(rtopology) | Dmaphop IR |
| 4 | DmaphopTodfscheblueprintPass | **Dfscheblueprint IR** |

After `pm.run(module1)`, `module1` holds the dfscheblueprint (blueprint) IR.

## 2. Clone for host and kernel

- `kernelModule = cast<ModuleOp>(module1->clone())`
- `hostModule = cast<ModuleOp>(module1->clone())`

Kernel and host codegen start from the same blueprint; each has its own pass pipeline.

## 3. Kernel-only pipeline

Applied to `kernelModule`:

| Step | Pass | Result |
|------|------|--------|
| 1 | BlueprintToScheduleKernelPass | Kernel schedule IR from blueprint |
| 2 | DfscheduleToKernelApiPass | Kernel API (EmitC) IR |
| 3 | emitc::translateToCpp(kernelModule, kernelStream) | C++ text → **kernel.cc** |

Pass manager: `pmkernel`; output path: `worklocalDir + "/kernel.cc"` (worklocal is unitest/worklocal).

## 4. Host path (for reference)

Applied to `hostModule`: BlueprintToSchedulePass → ScheduleCanonicalizePass → DfscheduleToApiPass → Canonicalizer → RoutingConstantFoldPass, then emitc → host.cc. Kernel path does not run these.

## Key files (summary)

| Role | Path |
|------|------|
| Unitest driver (orchestrates passes, writes kernel.cc) | `src/mlir/mlirfront/tilinglinalg/pass/unitest/test.cpp` |
| Kernel dfschedule generation | `src/mlir/mlirfront/tilinglinalg/pass/passblueprinttoschedulekernel/passblueprinttoschedulekernel.cpp` |
| kernel.cc generation (schedule → EmitC → C++) | `src/mlir/mlirfront/tilinglinalg/pass/passdfscheduletokernelapi/passdfscheduletokernelapi.cpp` |
| EmitC → C++ text | MLIR emitc dialect `translateToCpp` (called from test.cpp) |

## Bug fixes

- **Wrong or missing kernel C++** → Inspect BlueprintToScheduleKernelPass and DfscheduleToKernelApiPass; check kernel module after each pass (e.g. add PrintIRPass).
- **EmitC translate failure** → Check kernel MLIR is valid EmitC; fix lowering in DfscheduleToKernelApiPass.
