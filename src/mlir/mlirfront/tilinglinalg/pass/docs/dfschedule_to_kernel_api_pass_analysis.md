<!-- Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->
# DfscheduleToKernelApiPass Analysis

## Overview

`DfscheduleToKernelApiPass` is the final lowering step on the kernel path. It runs on the
`kernelModule` (cloned from the shared dfscheblueprint IR), immediately after
`BlueprintToScheduleKernelPass`.

Its input (`10_BlueprintToScheduleKernelPass.mlir`) contains a single `dfschedule.module`
op that encodes the AIE core kernel driver in structured dfschedule ops.
Its output (`11_DfscheduleToKernelApiPass.mlir`) is pure EmitC dialect — `emitc.verbatim`
lines and an `emitc.func @main` — which `translateToCpp` turns into `kernel.cc`.

**Source:** `pass/passdfscheduletokernelapi/passdfscheduletokernelapi.cpp`

---

## Pass Pipeline Position

```
BlueprintToScheduleKernelPass  →  DfscheduleToKernelApiPass  →  translateToCpp
(dfschedule.module with kernel     (emitc verbatim + func)        kernel.cc
 driver ops)
```

---

## Conversion Target

```cpp
target.addLegalDialect<emitc::EmitCDialect, func::FuncDialect,
                        arith::ArithDialect, memref::MemRefDialect>();
target.addIllegalOp<KernelModuleOp>();
```

`KernelModuleOp` (`dfschedule.module`) is the only illegal op. The single conversion pattern
`KernelModuleToEmitCPattern` matches it, converts all nested ops to EmitC, then erases the module.

---

## Pattern: `KernelModuleToEmitCPattern`

This pattern does **all work in one shot** — it walks the module body directly rather than relying
on nested patterns to fire first. This avoids MLIR's conversion ordering issues where the parent
op may be visited before its nested regions are lowered.

### Phase 1 — Pre-scan: Build `windowInfoMap`

Before emitting any code, walk the `dfschedule.module` body once and collect all
`dfschedule.window_def` ops into a `StringMap<WindowInfo>`:

```
WindowInfo {
  pingBuffer  : symbol name of the ping buffer (e.g. "buf_out_ping_0")
  pongBuffer  : symbol name of the pong buffer (e.g. "buf_out_pong_0")
  acquireLock : symbol name of the acquire lock (e.g. "LOCK_window_out_0_ACQ")
  releaseLock : symbol name of the release lock (e.g. "LOCK_window_out_0_REL")
}
```

keyed by `window_def` symbol name (e.g. `"window_out_0"`). This map is used later when
`window_init` is emitted in `convertMainToEmitC`, so that the correct buffer and lock names
are referenced by name without needing SSA values.

### Phase 2 — Config state defaults

```cpp
int32_t bufferSize = 256;
int32_t vectorWidth = 4;
std::string elementType = "int32";
std::string kernelFileName = "compute_kernel.cc";
```

These are overwritten when a `KernelConfigDefOp` is encountered in the body.

### Phase 3 — Sequential body emission

The module body is walked **in order** (not with a walker — direct iteration over `body.getOps()`).
Each dfschedule op is handled:

#### `dfschedule.kernel_config_def`

Reads config attributes and emits the file-level boilerplate:

| Emitted line | Purpose |
|---|---|
| `#include <stdint.h>` | Standard integer types |
| `#include <adf.h>` | ADF (AI Data Flow) types: `window_internal`, `output_window_int8`, `get_output_async_window_int8` |
| `#include <aie_api/aie.hpp>` | AIE intrinsics |
| `#include <aie_api/aie_adf.hpp>` | AIE ADF helpers |
| `#define FOR_READ  1` | Direction constant |
| `#define FOR_WRITE 0` | Direction constant |
| `#define BUF_SZ <bufferSize>` | Ping/pong buffer size in elements |
| `inline int8_t* acquire_output_window(...)` | Helper to cast `output_window_int8*` to `int8_t*` |
| `inline void release_output_window(...)` | Helper that calls `chess_memory_fence()` on release |

`BUF_SZ` is set from `kernel_config_def.buffer_size` attribute (default 256).

The element type (`i8`, `i16`, `i32`, `f32`) and vector width are read here for use by
`buffer_def` emission below.

#### `dfschedule.lock_def`

Emits a `#define` mapping the lock symbol to its numeric ID:

```c
#define LOCK_window_out_0_ACQ 51
#define LOCK_window_out_0_REL 50
```

The lock `id` comes directly from `LockDefOp.getId()`. The symbol name is used as the macro name.

#### `dfschedule.buffer_def`

Reads the buffer type (`memref<N x vector<W x T>, "LOCAL">`), maps the element/vector type to an
AIE C type, and emits a global array declaration:

```c
v4int8 buf_out_ping_0[BUF_SZ];
v4int8 buf_out_pong_0[BUF_SZ];
```

Type mapping:

| MLIR type | C type |
|---|---|
| `vector<W x i32>` | `v{W}int32` |
| `vector<W x i16>` | `v{W}int16` |
| `vector<W x i8>` | `v{W}int8` |
| `vector<W x f32>` | `v{W}float` |

Default (fallback if type parsing fails): `v4int32`.

Array size uses `BUF_SZ` (the macro defined from `kernel_config_def`), not a literal integer.

#### `dfschedule.window_def`

Emits a comment only (the window's actual buffers and locks are already emitted by `buffer_def`
and `lock_def` above):

```c
// window_def window_out_0
```

The real work for `window_def` information was done in Phase 1 (`windowInfoMap`).

#### `dfschedule.kernel_decl`

Includes the compute kernel source file and emits a comment:

```c
#include "compute_kernel.cc"
// kernel_decl compute_kernel
```

The filename comes from `kernelFileName` (read from `kernel_config_def.kernel_file`).
This is the file compiled by `xchesscc` that contains the user's computation function.

#### `dfschedule.main`

Delegates to `convertMainToEmitC` (described below).

---

## `convertMainToEmitC`: Generating the `main` Function

Creates an `emitc.func @main() -> i32` in the **parent block** (before the `KernelModuleOp`,
not inside it). A `DenseMap<Value, std::string>` (`valueToCName`) tracks SSA→C name mapping
for values that need to be referenced later (e.g. window handles).

### Preamble (before body ops)

```c
volatile static int sync_buffer[8] = {0, -1};
sync_buffer[0] = 0;
```

These lines are emitted unconditionally at the start of `main`, regardless of whether
`alloc_sync_buffer` or `sync_buffer_write` appear in the body.

### Op-by-op body emission

#### `dfschedule.alloc_sync_buffer`

```c
// alloc_sync_buffer
```

Comment only. The actual `sync_buffer` declaration is in the preamble.

#### `dfschedule.sync_buffer_write`

```c
// sync_buffer_write
```

Comment only.

#### `dfschedule.log`

```c
// log(...)
```

Comment only (logging is suppressed in the generated code).

#### `dfschedule.window_init`

Emits a `window_internal` struct array declaration and the `window_init` call.
Buffer and lock names are looked up from `windowInfoMap` by the window symbol name.

```c
window_internal window_window_out_0[1];
window_init(window_window_out_0, 1,
            buf_out_ping_0, LOCK_window_out_0_ACQ,
            buf_out_pong_0, LOCK_window_out_0_REL,
            BUF_SZ, BUF_SZ);
```

The SSA result of `window_init` is mapped in `valueToCName`:
`result → "window_window_out_0"` (prefix `"window_"` + symbol name).

`window_init` signature (from `<adf.h>`):

```c
void window_init(window_internal* win, int count,
                 void* ping,    int ping_acq_lock,
                 void* pong,    int pong_rel_lock,
                 int   ping_sz, int pong_sz);
```

#### `dfschedule.kernel_invoke`

Emits a comment and the actual function call. Each argument is an SSA value — its C name is
looked up from `valueToCName`. Window arguments are wrapped in `get_output_async_window_int8()`:

```c
// kernel_invoke compute_kernel
compute_kernel(get_output_async_window_int8(window_window_out_0));
```

`get_output_async_window_int8` (from `<adf.h>`) converts the `window_internal*` to the
`output_window_int8` type expected by the kernel function signature.

#### `dfschedule.done`

```c
done();
```

ADF runtime call that signals to the AIE shim that this tile's computation iteration is complete.

#### `dfschedule.kernel_return`

```c
return 0;   // emitted as: emitc.constant(0), emitc.return
```

Terminates the `emitc.func @main`. After this, the loop breaks (no further ops processed).

---

## IR Walkthrough (2×2 GEMM — Core-to-Shim Output Example)

### Input: `dfschedule.module` (`10_BlueprintToScheduleKernelPass.mlir`)

```mlir
dfschedule.module @kernel_driver_dskernel_receiver {
  dfschedule.kernel_config_def @config {
    buffer_size = 256, element_type = i32,
    kernel_file = "compute_kernel.cc", kernel_name = "compute_kernel", vector_width = 4
  }
  dfschedule.lock_def @LOCK_window_out_0_ACQ { id = 51, init_value = 2 }
  dfschedule.lock_def @LOCK_window_out_0_REL { id = 50 }
  dfschedule.buffer_def @buf_out_ping_0 : memref<256xvector<4xi8>, "LOCAL">
  dfschedule.buffer_def @buf_out_pong_0 : memref<256xvector<4xi8>, "LOCAL">
  dfschedule.window_def @window_out_0 {
    direction = "out",
    ping_buffer = @buf_out_ping_0, pong_buffer = @buf_out_pong_0,
    acquire_lock = @LOCK_window_out_0_ACQ, release_lock = @LOCK_window_out_0_REL,
    buffer_size = 256, async = true
  }
  dfschedule.kernel_decl @compute_kernel {
    inputs = [], outputs = [@window_out_0], iteration_style = "internal"
  }
  dfschedule.main @main {
    %0 = dfschedule.alloc_sync_buffer { size = 8 }
    %c0  = arith.constant 0 : i32
    dfschedule.sync_buffer_write(%0, %c0) { index = 0 }
    %c1  = arith.constant 1 : i32
    dfschedule.log(%c1)
    %1   = dfschedule.window_init(@window_out_0)
    dfschedule.kernel_invoke @compute_kernel(%1)
    dfschedule.done
    dfschedule.kernel_return
  }
}
```

### Output: EmitC (`11_DfscheduleToKernelApiPass.mlir`)

```mlir
module {
  emitc.verbatim "#include <stdint.h>"
  emitc.verbatim "#include <adf.h>"
  emitc.verbatim "#include <aie_api/aie.hpp>"
  emitc.verbatim "#include <aie_api/aie_adf.hpp>"
  emitc.verbatim "#define FOR_READ  1"
  emitc.verbatim "#define FOR_WRITE 0"
  emitc.verbatim "#define BUF_SZ 256"
  emitc.verbatim "inline int8_t* acquire_output_window(...) { return (int8_t*)win; }"
  emitc.verbatim "inline void release_output_window(...) { chess_memory_fence(); }"
  emitc.verbatim "#define LOCK_window_out_0_ACQ 51"
  emitc.verbatim "#define LOCK_window_out_0_REL 50"
  emitc.verbatim "v4int8 buf_out_ping_0[BUF_SZ];"
  emitc.verbatim "v4int8 buf_out_pong_0[BUF_SZ];"
  emitc.verbatim "// window_def window_out_0"
  emitc.verbatim "#include \"compute_kernel.cc\""
  emitc.verbatim "// kernel_decl compute_kernel"
  emitc.func @main() -> i32 {
    emitc.verbatim "volatile static int sync_buffer[8] = {0, -1};"
    emitc.verbatim "sync_buffer[0] = 0;"
    emitc.verbatim "// alloc_sync_buffer"
    emitc.verbatim "// sync_buffer_write"
    emitc.verbatim "// log(...)"
    emitc.verbatim "window_internal window_window_out_0[1];"
    emitc.verbatim "window_init(window_window_out_0, 1, buf_out_ping_0, LOCK_window_out_0_ACQ, buf_out_pong_0, LOCK_window_out_0_REL, BUF_SZ, BUF_SZ);"
    emitc.verbatim "// kernel_invoke compute_kernel"
    emitc.verbatim "compute_kernel(get_output_async_window_int8(window_window_out_0));"
    emitc.verbatim "done();"
    %0 = emitc.constant 0 : i32
    emitc.return %0 : i32
  }
}
```

### Final `kernel.cc` (after `translateToCpp`)

```c
#include <stdint.h>
#include <adf.h>
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>
#define FOR_READ  1
#define FOR_WRITE 0
#define BUF_SZ 256
inline int8_t* acquire_output_window(output_window_int8* win) { return (int8_t*)win; }
inline void release_output_window(output_window_int8* win) { chess_memory_fence(); }
#define LOCK_window_out_0_ACQ 51
#define LOCK_window_out_0_REL 50
v4int8 buf_out_ping_0[BUF_SZ];
v4int8 buf_out_pong_0[BUF_SZ];
// window_def window_out_0
#include "compute_kernel.cc"
// kernel_decl compute_kernel
int32_t main() {
  volatile static int sync_buffer[8] = {0, -1};
  sync_buffer[0] = 0;
  // alloc_sync_buffer
  // sync_buffer_write
  // log(...)
  window_internal window_window_out_0[1];
  window_init(window_window_out_0, 1, buf_out_ping_0, LOCK_window_out_0_ACQ,
              buf_out_pong_0, LOCK_window_out_0_REL, BUF_SZ, BUF_SZ);
  // kernel_invoke compute_kernel
  compute_kernel(get_output_async_window_int8(window_window_out_0));
  done();
  int32_t v1 = 0;
  return v1;
}
```

---

## Op Conversion Table

| dfschedule op (inside `dfschedule.module`) | Emitted C / EmitC | Scope |
|---|---|---|
| `dfschedule.kernel_config_def` | `#include` headers + `#define BUF_SZ` + helper inlines | file-level verbatim |
| `dfschedule.lock_def @NAME { id = N }` | `#define NAME N` | file-level verbatim |
| `dfschedule.buffer_def @NAME : memref<N x vector<W x T>>` | `v{W}{T} NAME[BUF_SZ];` | file-level verbatim |
| `dfschedule.window_def @NAME { ... }` | `// window_def NAME` (comment only) | file-level verbatim |
| `dfschedule.kernel_decl @NAME { kernel_file = "F" }` | `#include "F"` + `// kernel_decl NAME` | file-level verbatim |
| `dfschedule.main @main { ... }` | `emitc.func @main() -> i32 { ... }` | function |
| `dfschedule.alloc_sync_buffer` | `// alloc_sync_buffer` (comment) | inside main |
| `dfschedule.sync_buffer_write` | `// sync_buffer_write` (comment) | inside main |
| `dfschedule.log` | `// log(...)` (comment) | inside main |
| `dfschedule.window_init(@SYM)` | `window_internal window_SYM[1];` + `window_init(...)` | inside main |
| `dfschedule.kernel_invoke @KERNEL(%win)` | `KERNEL(get_output_async_window_int8(window_SYM));` | inside main |
| `dfschedule.done` | `done();` | inside main |
| `dfschedule.kernel_return` | `emitc.constant 0; emitc.return` → `return 0;` | terminates main |

The `dfschedule.module` op itself is **erased** after all nested ops are processed.

---

## Key Design Points

### Sequential emission, not pattern matching on nested ops

Unlike a typical dialect conversion where nested ops have their own patterns, this pass walks
the `dfschedule.module` body directly in order. This is intentional: MLIR's conversion framework
may visit parent ops before descending into nested regions, so relying on nested patterns would
require explicit legalization ordering. Direct iteration is simpler and guarantees emission order.

### Two-phase window handling

Window information is collected in a pre-scan (Phase 1) before any emission starts. This avoids
forward-reference issues: `window_init` in `dfschedule.main` needs buffer/lock names that are
declared earlier in the module body by `buffer_def` and `lock_def`.

### Preamble is unconditional

The `volatile static int sync_buffer[8]` declaration and `sync_buffer[0] = 0` initialization
are emitted directly in `convertMainToEmitC` before the body loop, not driven by
`alloc_sync_buffer` / `sync_buffer_write` ops. Those ops just emit comments.

### `window_init` naming convention

The `window_internal` variable name is `"window_" + windowSymName`.
For `@window_out_0`, this produces `window_window_out_0`. This prefix-plus-sym naming prevents
collision if there are multiple windows.

### Argument passing to compute kernel

`dfschedule.kernel_invoke` receives SSA values (window handles). These are looked up in
`valueToCName`. Window handle SSA values always map to `"window_" + symName`.
The C call wraps each window argument in `get_output_async_window_int8()` to convert the
`window_internal*` to the `output_window_int8` type expected by the kernel's ADF signature.

### `element_type` read but `vectorWidth` ignored for buffer type

The `kernel_config_def.element_type` is read into `elementType` (string like `"int8"`) but
the actual vector type for `buffer_def` emission is read directly from the `BufferDefOp`'s
`MemRefType` element type — not from the config. The config `element_type` and `vector_width`
fields serve as documentation/metadata but the `buffer_def` type is authoritative.
