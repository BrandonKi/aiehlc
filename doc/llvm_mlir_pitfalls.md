<!-- Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->
# LLVM/MLIR API Pitfalls

Known LLVM and MLIR API pitfalls encountered in the AIEHLC project. Each entry documents the bug, root cause, fix, and a rule to avoid reoccurrence.

## Table of Contents

1. [ArrayRef Dangling Reference](#1-arrayref-dangling-reference)
2. [FileEntry / Rewrite Crash](#2-fileentry--rewrite-crash)
3. [StringRef Lifetime](#3-stringref-lifetime)
4. [RTTI Requirement](#4-rtti-requirement)
5. [getBuffer Null Dereference](#5-getbuffer-null-dereference)
6. [Recursive TableGen Op Definitions](#6-recursive-tablegen-op-definitions)
7. [Constant Propagation Pattern Match](#7-constant-propagation-pattern-match)
8. [Integer Overflow in Position Tracking](#8-integer-overflow-in-position-tracking)
9. [MLIR SSA Use-Def Chain Discipline](#9-mlir-ssa-use-def-chain-discipline)

---

## 1. ArrayRef Dangling Reference

**Commit**: `1142285` (2025-09-08)

**Symptom**: `argNames[0]` returns a string with garbage size and points to invalid memory. Intermittent crash, hard to reproduce.

**Bad code**:
```cpp
llvm::ArrayRef<llvm::StringRef> argNames = {"mesh", "tensor", "splitnum",
                                             "axisidx", "partensor_dim"};
// argNames[0].size() returns a very big number
// argNames[0].data() points to invalid address
```

**Root cause**: The initializer list `{"mesh", ...}` creates temporary `llvm::StringRef` objects. These temporaries are stored in a compiler-generated temporary array. `ArrayRef` takes a pointer to this array, but the temporaries are destroyed at the end of the statement. After construction, `argNames` holds a dangling pointer.

This is LLVM-version-specific behavior. In some LLVM versions, the initializer list lifetime is extended; in others (like the one used in this project), it is not.

**Fix**:
```cpp
llvm::SmallVector<llvm::StringRef, 8> argNamesStorage = {"mesh", "tensor",
    "splitnum", "axisidx", "partensor_dim",
    "axisowner", "partensor_replicateon", "partensor_singleowner"};
llvm::ArrayRef<llvm::StringRef> argNames(argNamesStorage);
```

**Rule**: Never initialize an `ArrayRef` directly from an initializer list. Always use a named `SmallVector` or `std::vector` as the owning storage, then create the `ArrayRef` from it.

**Broader pattern**: Any function that takes `ArrayRef` as a parameter is fine (the temporary lives until the function returns). The problem is only when storing `ArrayRef` in a variable that outlives the temporary.

---

## 2. FileEntry / Rewrite Crash

**Commit**: `0c5def3` (2024-02-19)

**Symptom**: Crash when mixing keyword replacement in `BeginSourceFileAction` with AST rewriting in the visitor.

**Root cause**: Clang's `FileEntry` has a fixed size. When `SourceManager::overrideFileContents` is used to replace the main file's buffer with a modified version, the `FileEntry` metadata (especially size) becomes inconsistent if the new content has a different length. The Rewriter then operates on stale SourceManager state.

Additionally, if the replacement buffer is a local variable, it gets freed when the function returns. Some LLVM APIs copy the buffer, but `overrideFileContents` may only take a reference, leading to use-after-free.

**Fix**: Create a virtual file with `FileManager::getVirtualFileRef` and a new `FileID`, rather than overriding the existing entry:

```cpp
const FileEntryRef SourceFile = FileMgr->getVirtualFileRef(
    "newfile.cpp", ModifiedBuffer->getBufferSize(), 0);
SourceMgr.overrideFileContents(SourceFile, std::move(ModifiedBuffer));
FileID MainFileID = SourceMgr.getOrCreateFileID(SourceFile, SrcMgr::C_User);
SourceMgr.setMainFileID(MainFileID);
```

**Rule**: Never override the contents of an existing `FileEntry` when the new content has a different size. Always create a new virtual file. Use `std::move` for the buffer to ensure ownership transfer.

---

## 3. StringRef Lifetime

**Location**: `src/llvm/aiehlc.cc`

**Pattern**: `StringRef` is a non-owning view into a character buffer. It becomes invalid when the underlying buffer is modified or freed.

**Dangerous patterns**:
```cpp
// BAD: StringRef into a buffer that will be modified by Rewriter
StringRef text = SourceMgr.getBufferData(fileID);
Rewrite.ReplaceText(...);  // invalidates text

// BAD: StringRef into a local string
StringRef name = std::string("hello");  // temporary destroyed immediately

// BAD: returning StringRef to local
StringRef getName() {
    std::string s = computeName();
    return s;  // dangling after return
}
```

**Safe patterns**:
```cpp
// OK: use StringRef immediately, no modification in between
StringRef text = SourceMgr.getBufferData(fileID);
const char* begin = text.begin() + offset;  // use immediately

// OK: StringRef into a long-lived string
std::string name = computeName();
StringRef nameRef = name;  // safe as long as name is alive

// OK: StringRef from function that returns view into stable storage
StringRef name = FileEntry->getName();  // FileEntry outlives this scope
```

**Rule**: Never store a `StringRef` that outlives its source buffer. If in doubt, copy to `std::string`.

---

## 4. RTTI Requirement

**Location**: `doc/build.md`, CMake configuration

**Symptom**: `undefined reference to typeinfo for ...` linker errors. `mlir::dyn_cast` and `mlir::isa` fail at runtime.

**Root cause**: LLVM is built without RTTI by default (`-fno-rtti`). When project code uses `dynamic_cast` or MLIR's type system (which requires RTTI), the linker cannot find typeinfo symbols.

**Fix**: Build LLVM with RTTI enabled:
```cmake
-DLLVM_ENABLE_RTTI=ON
```

**Rule**: Always build LLVM/MLIR with `-DLLVM_ENABLE_RTTI=ON` for this project. Document this in the build instructions.

---

## 5. getBuffer Null Dereference

**Location**: `src/llvm/aiehlc.cc`

**Symptom**: Crash when calling `getBuffer()` on a `FileID` that has no associated buffer.

**Bad code**:
```cpp
// BAD: getFileEntryForID can return nullptr
SourceMgr.getFileEntryForID(SourceMgr.getMainFileID())->getName();
```

**Fix**: Use the `Optional`-returning variants and check before use:
```cpp
// GOOD: check optional
const auto FileEntryRef = SourceMgr.getFileEntryRefForID(SourceMgr.getMainFileID());
if (FileEntryRef) {
    llvm::StringRef FileName = FileEntryRef->getName();
}

// GOOD: getBufferOrNone instead of getBuffer
const auto MainFileBufferOp = SourceMgr.getBufferOrNone(MainFileID);
if (MainFileBufferOp) {
    const llvm::MemoryBufferRef *MainFileBuffer = &MainFileBufferOp.value();
}
```

**Rule**: Always use `getFileEntryRefForID` (returns `Optional`) instead of `getFileEntryForID` (returns raw pointer). Always use `getBufferOrNone` instead of `getBuffer`.

---

## 6. Recursive TableGen Op Definitions

**Commit**: `2a64a5f` (2024-07-22)

**Symptom**: Infinite recursion crash when processing window/kernel ops.

**Root cause**: TableGen op definitions had circular references where op A referenced op B's type, and op B referenced op A's type, creating an infinite loop during type resolution.

**Rule**: Audit TableGen `.td` files for circular type dependencies. Use forward declarations or break cycles with intermediate types.

---

## 7. Constant Propagation Pattern Match

**Commit**: `a5b4e51` (2025-12-11)

**Symptom**: `RoutingConstantFoldPass` does not fold constants as expected. IR still contains un-folded operations.

**Root cause**: The pattern matcher was not correctly matching the op's operand structure. MLIR's `matchAndRewrite` requires exact pattern matching; if the op's region structure or operand types don't match the pattern, the fold silently does nothing.

**Fix**: 96-line rewrite of `routingconstantfold.cpp` to properly match the op structure.

**Rule**: When writing MLIR rewrite patterns, add debug output to verify the pattern actually fires. A silent no-match is hard to detect.

---

## 8. Integer Overflow in Position Tracking

**Commit**: `4bb2426` (2024-04-18)

**Symptom**: Position value becomes unexpectedly large (appears as a huge number).

**Root cause**: Likely unsigned integer underflow (subtracting from 0) or signed/unsigned mismatch causing a small negative number to become a large positive number when interpreted as unsigned.

**Rule**: Be explicit about signed vs unsigned types for positions and indices. Use `int64_t` for positions that can be negative. When converting between signed and unsigned, add bounds checks.

---

## 9. MLIR SSA Use-Def Chain Discipline

**Location**: `doc/schedule_canonicalize_redesign.md`, `passschedulecanonicalize.cpp`

**Pattern**: MLIR's SSA form requires that every value definition has a proper use chain. Breaking this chain (e.g., creating a BD but not connecting it to a `create_io`) causes verifier errors.

**Anti-patterns**:
- Discarding SSA relationships and rebuilding from positional indices
- Deduplicating values without updating all users
- Creating ops in one phase and referencing them by index in another (indices may shift)

**Safe patterns**:
- Always maintain SSA use-def chains through `mlir::Value` references
- When deduplicating, use `replaceAllUsesWith` to redirect users
- When creating ops that reference other ops, use the result `Value` directly, not a positional index

**Rule**: Treat MLIR IR as a graph, not a linear sequence. Always use `Value` references to connect ops, never positional indices.

---

## Quick Reference: Safe vs Dangerous Patterns

| Pattern | Safe | Dangerous |
|---------|------|-----------|
| `ArrayRef` from initializer list | Store in `SmallVector` first | Direct assignment to `ArrayRef` |
| `StringRef` | From stable storage, use immediately | From temporary, store for later |
| `FileEntry` modification | Create new virtual file | Override existing entry with different size |
| `getBuffer` | Use `getBufferOrNone`, check optional | Use `getBuffer`, dereference without check |
| MLIR op references | Use `Value` result directly | Use positional index |
| Constant fold | Add debug logging to verify pattern fires | Assume silent success |
| TableGen types | Check for circular dependencies | Assume acyclic |
| Integer positions | Use `int64_t`, check bounds | Use `unsigned`, assume non-negative |
