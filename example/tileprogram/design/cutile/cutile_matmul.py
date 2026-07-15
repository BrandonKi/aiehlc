###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# AIE Programming Model — CuTile-style Python Frontend for Matrix Multiply
#
# This file is the definitive CuTile-style Python input example for matmul
# targeting AMD Versal AI Engine via aiehlc. It contains two kernel variants
# and a host section that matches proposal3.cc exactly.
#
# CuTile (NVIDIA CUDA Tile API, CUDA 13.1+) uses index-based tile loads
# rather than Triton's pointer arithmetic. This maps more naturally to AIE's
# DMA buffer descriptor model, where transfers are specified by tile index
# and shape rather than base pointer + offset computation.
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │                        Compilation Flow                                │
# │                                                                        │
# │  cutile_matmul.py                                                      │
# │    │  Python AST parse (@ct.kernel functions)                          │
# │    ▼                                                                   │
# │  routing dialect IR                                                    │
# │    │  buildRoutingIR(ctx, meshRows=2, meshCols=2, tensors=[...])       │
# │    │  (tilinglinalg_pipeline.h: TensorParam for A, B, C)              │
# │    ▼                                                                   │
# │  MLIR lowering pipeline                                                │
# │    │  routing → dmap → dmaphop → dfscheblueprint → dfschedule         │
# │    ▼                                                                   │
# │  host.cc + kernel.cc + routing.cc + aieml.bcf + aieml.prx             │
# │    │  cross-compile: xchesscc (kernel ELF), aarch64-g++ (host ELF)    │
# │    ▼                                                                   │
# │  Deploy on AIE HW: load ELFs, configure DMA BDs, run cores            │
# └─────────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │          CuTile ←→ Triton ←→ AIE / proposal3.cc Concept Map           │
# │                                                                        │
# │  CuTile Python        │ Triton Python            │ AIE C / proposal3  │
# │  ─────────────────────┼──────────────────────────┼─────────────────── │
# │  @ct.kernel           │ @aie_triton.jit          │ __global__          │
# │  ct.program_id(0)     │ tl.program_id(0)         │ get_coreid()&0x1F  │
# │  ct.program_id(1)     │ tl.program_id(1)         │ get_coreid()>>16   │
# │  ct.load(A, idx, shp) │ tl.load(a_ptr+offsets)   │ acquire_input_win  │
# │  ct.store(C, idx, tl) │ tl.store(c_ptr+off, v)   │ acquire/write/rel  │
# │  ct.mma(a, b, acc)    │ tl.dot(a, b)             │ v4int8 MAC loop    │
# │  ct.full(shp, 0, dt)  │ tl.zeros(shape, dtype)   │ BCF local buffer   │
# │  ct.astype(acc, int8) │ acc.to(tl.int8)          │ saturate i32→i8    │
# │  ct.Constant[int]     │ tl.constexpr             │ C template param   │
# │  ct.num_tiles(A,ax,s) │ range(0, K, BLOCK_K)     │ for(k=0;k<2;k++)  │
# │  ct.set_device(0)     │ aie_triton.set_device(0) │ aieSetDevice(0)    │
# │  ct.mesh(rows, cols)  │ aie_triton.mesh(r, c)    │ aieDim mesh(r, c)  │
# │  ct.synchronize()     │ aie_triton.synchronize() │ aieDeviceSync()    │
# │  np.array / np.zeros  │ np.array / np.zeros      │ malloc() for DDR   │
# └─────────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │              buildRoutingIR Mapping (tilinglinalg_pipeline.h)          │
# │                                                                        │
# │  Python frontend extracts from @ct.kernel + kernel[grid]():            │
# │    meshRows=2, meshCols=2  (from ct.mesh / grid)                      │
# │    tensors = [                                                         │
# │      TensorParam{shape={16,16}, elementBitWidth=8, isInput=true },     │
# │      TensorParam{shape={16,16}, elementBitWidth=8, isInput=true },     │
# │      TensorParam{shape={16,16}, elementBitWidth=8, isInput=false},     │
# │    ]                                                                   │
# │  Then calls: buildRoutingIR(ctx, 2, 2, tensors) → routing IR          │
# └─────────────────────────────────────────────────────────────────────────┘
#
# Data partitioning on a 2x2 mesh for 16x16 @ int8:
#   tile(0,0) → C[0:8,  0:8 ]   tile(0,1) → C[0:8,  8:16]
#   tile(1,0) → C[8:16, 0:8 ]   tile(1,1) → C[8:16, 8:16]
#
# Each tile: tm=8, tn=8, tk=8
#   A partition: 8x16 (full K, partial M) → streamed in K-dim as 8x8 chunks
#   B partition: 16x8 (full K, partial N) → streamed in K-dim as 8x8 chunks
#   C partition: 8x8  (written back to DDR via DMA)
#
###############################################################################

import cuda.tile as ct
import numpy as np


###############################################################################
#  KERNEL VARIANT 1 — Simple / Index-based CuTile style
#
#  Closest to proposal3.cc's window_in / window_out pattern.
#  Uses ct.load / ct.mma / ct.store with index-based tile addressing,
#  which maps naturally to AIE DMA buffer descriptor configuration
#  (tile index + shape, no pointer arithmetic).
#
#  proposal3.cc equivalent (per K-iteration):
#    int8_t *in0 = (int8_t *)acquire_input_window(window_in_0);
#    int8_t *in1 = (int8_t *)acquire_input_window(window_in_1);
#    int8_t *out = acquire_output_window(window_out_0);
#    for (int i = 0; i < BUF_SZ; i++) {
#        v4int8 data0 = *((v4int8 *)&in0[i * 4]);
#        v4int8 data1 = *((v4int8 *)&in1[i * 4]);
#        *((v4int8 *)&out[i * 4]) = data0;   // placeholder — real: MAC
#    }
#    release_input_window(window_in_0);
#    release_input_window(window_in_1);
#    release_output_window(window_out_0);
#
#  Triton equivalent:
#    a_block = tl.load(a_ptr + offsets)   →  ct.load(A, index=(...), shape=(...))
#    b_block = tl.load(b_ptr + offsets)   →  ct.load(B, index=(...), shape=(...))
#    acc += tl.dot(a_block, b_block)      →  acc = ct.mma(a, b, acc)
#    tl.store(c_ptr + offsets, result)    →  ct.store(C, index=(...), tile=result)
###############################################################################

@ct.kernel
def matmul_simple(
    # Tensors in global memory (DDR) — runtime partitions them to tiles via DMA
    A: ct.Tensor,       # → proposal3.cc: window_in_0  (input_window_int8 *)
                        # → Triton:      a_ptr
    B: ct.Tensor,       # → proposal3.cc: window_in_1  (input_window_int8 *)
                        # → Triton:      b_ptr
    C: ct.Tensor,       # → proposal3.cc: window_out_0 (output_window_int8 *)
                        # → Triton:      c_ptr
    # Tile dimensions — constrained by AIE tile local memory (~32 KB)
    tm: ct.Constant[int],   # rows per tile    → Triton: BLOCK_M: tl.constexpr = 8
    tn: ct.Constant[int],   # cols per tile    → Triton: BLOCK_N: tl.constexpr = 8
    tk: ct.Constant[int],   # K-chunk per DMA  → Triton: BLOCK_K: tl.constexpr = 8
):
    """Simple index-based CuTile matmul kernel for one AIE tile.

    Mapping to proposal3.cc:
        @ct.kernel              → __global__
        ct.program_id(0)        → get_coreid() & 0x1F  (row)
        ct.program_id(1)        → get_coreid() >> 16    (col)
        ct.load(A, idx, shp)    → acquire_input_window(window_in_0)
        ct.load(B, idx, shp)    → acquire_input_window(window_in_1)
        ct.mma(a, b, acc)       → v4int8 MAC loop (BUF_SZ iterations)
        ct.store(C, idx, tile)  → acquire_output_window + write + release
        for k in range(num_k)   → for (int k = 0; k < 2; k++) in proposal3

    Mapping to Triton:
        @ct.kernel              → @aie_triton.jit
        ct.program_id(axis)     → tl.program_id(axis)
        ct.load(A, idx, shp)    → tl.load(a_ptr + offsets)
        ct.mma(a, b, acc)       → acc += tl.dot(a_block, b_block)
        ct.store(C, idx, tile)  → tl.store(c_ptr + offsets, result)
        ct.num_tiles(A, ax, s)  → range(0, K, BLOCK_K)
        ct.Constant[int]        → tl.constexpr
    """
    # ── Which tile am I? ──
    # proposal3.cc:  unsigned coreid = get_coreid();
    #                int row = coreid & 0x1F;
    #                int col = coreid >> 16;
    # Triton:        tile_row = tl.program_id(axis=0)
    #                tile_col = tl.program_id(axis=1)
    bidx = ct.program_id(0)  # → row = coreid & 0x1F     (Triton: tl.program_id(0))
    bidy = ct.program_id(1)  # → col = coreid >> 16      (Triton: tl.program_id(1))

    # ── K-loop iteration count ──
    # proposal3.cc:  for (int k = 0; k < 2; k++)
    # Triton:        for k_start in range(0, K, BLOCK_K)
    # CuTile makes this explicit via ct.num_tiles:
    #   K=16, tk=8 → num_k=2, matching proposal3.cc's loop count
    num_k = ct.num_tiles(A, axis=1, shape=(tm, tk))   # → 2 (Triton: K // BLOCK_K)

    # ── Accumulator in tile-local memory ──
    # proposal3.cc:  local buffer at BCF-assigned address
    # Triton:        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    acc = ct.full((tm, tn), 0, dtype=ct.int32)        # → BCF local buffer

    # ── K-dimension loop ──
    # Each iteration maps to one acquire/compute/release cycle:
    #   ct.load   → acquire_input_window  (Triton: tl.load)
    #   ct.mma    → v4int8 MAC            (Triton: tl.dot)
    #   (release) → release_input_window  (Triton: implicit)
    for k in range(num_k):

        # ── Load A tile [tm × tk] ──
        # proposal3.cc:  int8_t *in0 = (int8_t *)acquire_input_window(window_in_0);
        # Triton:        a_block = tl.load(a_ptr + offsets)   [shape: BLOCK_M × BLOCK_K]
        # CuTile index-based: no pointer arithmetic — just tile index + shape
        #   index=(bidx, k) tells the DMA which (row_block, k_block) to fetch
        a = ct.load(A, index=(bidx, k), shape=(tm, tk))      # → acquire_input_window

        # ── Load B tile [tk × tn] ──
        # proposal3.cc:  int8_t *in1 = (int8_t *)acquire_input_window(window_in_1);
        # Triton:        b_block = tl.load(b_ptr + offsets)   [shape: BLOCK_K × BLOCK_N]
        b = ct.load(B, index=(k, bidy), shape=(tk, tn))      # → acquire_input_window

        # ── Multiply-accumulate ──
        # proposal3.cc:  v4int8 data0 = *((v4int8 *)&in0[i*4]);
        #                v4int8 data1 = *((v4int8 *)&in1[i*4]);
        #                // real: acc += data0 * data1 (vector MAC)
        # Triton:        acc += tl.dot(a_block, b_block)
        acc = ct.mma(a, b, acc)                               # → v4int8/v16int8 MAC

        # ── End of iteration ──
        # proposal3.cc:  release_input_window(window_in_0);
        #                release_input_window(window_in_1);
        # Triton:        (implicit — compiler inserts lock release at DMA boundary)
        # CuTile:        (implicit — same as Triton)

    # ── Store result C tile [tm × tn] ──
    # proposal3.cc:  int8_t *out = acquire_output_window(window_out_0);
    #                *((v4int8 *)&out[i*4]) = result;
    #                release_output_window(window_out_0);
    # Triton:        result = acc.to(tl.int8)
    #                tl.store(c_ptr + c_offsets, result)
    result = ct.astype(acc, ct.int8)                          # saturate int32 → int8
    ct.store(C, index=(bidx, bidy), tile=result)              # → acquire/write/release


###############################################################################
#  KERNEL VARIANT 2 — With block swizzling (standard CuTile pattern)
#
#  Standard CuTile pattern for GPU cache locality via block index swizzling.
#  On AIE this maps to the same static tile allocation as matmul_simple —
#  the swizzle is a no-op since tiles are physically fixed. Included for
#  API completeness and to show the full CuTile idiom.
#
#  Same lowering path — the Python frontend normalizes both variants into
#  identical routing IR via buildRoutingIR().
#
#  Note: swizzle_2d is a standard CuTile optimization for L2 cache reuse
#  on GPU. On AIE, tile-to-memory affinity is fixed by the NoC topology,
#  so the swizzle permutation has no effect on performance.
###############################################################################

def swizzle_2d(bidx, bidy, num_rows, GROUP_SIZE_M):
    """Block index swizzling for improved cache locality (GPU optimization).

    On GPU: reorders block (bidx, bidy) within groups of GROUP_SIZE_M rows
    to improve L2 cache hit rate.

    On AIE: no-op — tile (row, col) mapping is physically fixed.
    Included to show CuTile API completeness.

    Triton equivalent: manual pid_m/pid_n decomposition with group_id.
    """
    group_id = bidx // GROUP_SIZE_M
    first_row = group_id * GROUP_SIZE_M
    group_size = min(num_rows - first_row, GROUP_SIZE_M)
    row_in_group = bidx % group_size
    col = bidy
    return first_row + row_in_group, col


@ct.kernel
def matmul_swizzle(
    A: ct.Tensor, B: ct.Tensor, C: ct.Tensor,
    tm: ct.Constant[int], tn: ct.Constant[int], tk: ct.Constant[int],
    GROUP_SIZE_M: ct.Constant[int],   # swizzle group size (GPU: ~8; AIE: mesh.rows)
):
    """CuTile matmul with block swizzling — standard GPU optimization pattern.

    On GPU, swizzle_2d reorders (bidx, bidy) within row-groups for L2 reuse.
    On AIE, tiles are physically fixed — the swizzle is a no-op but shows
    the full CuTile API surface.

    Triton equivalent (from matmul tutorial):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        ...
        pid_m = first_pid_m + (pid % num_pid_in_group) % GROUP_SIZE_M
        pid_n = (pid % num_pid_in_group) // GROUP_SIZE_M
    """
    # ── Swizzled tile identity ──
    # On AIE: bidx_sw == bidx, bidy_sw == bidy (no permutation)
    num_rows = ct.num_tiles(A, axis=0, shape=(tm, tk))   # mesh rows
    bidx_sw, bidy_sw = swizzle_2d(
        ct.program_id(0), ct.program_id(1), num_rows, GROUP_SIZE_M
    )

    # ── Same compute as matmul_simple ──
    num_k = ct.num_tiles(A, axis=1, shape=(tm, tk))
    acc = ct.full((tm, tn), 0, dtype=ct.int32)

    for k in range(num_k):
        a = ct.load(A, index=(bidx_sw, k), shape=(tm, tk))    # → acquire_input_window
        b = ct.load(B, index=(k, bidy_sw), shape=(tk, tn))    # → acquire_input_window
        acc = ct.mma(a, b, acc)                                 # → v4int8 MAC

    result = ct.astype(acc, ct.int8)
    ct.store(C, index=(bidx_sw, bidy_sw), tile=result)         # → acquire/write/release


###############################################################################
#  HOST — matches proposal3.cc main() exactly
#
#  proposal3.cc:                          CuTile Python:
#    const int M=16, N=16, K=16;            M, N, K = 16, 16, 16
#    aieSetDevice(0);                       ct.set_device(0)
#    aieDim mesh(2, 2);                     mesh = ct.mesh(rows=2, cols=2)
#    int32_t *A = malloc(M*K*sizeof);       A = np.arange(1, M*K+1, ...)
#    for(i) A[i] = i+1;                    (numpy arange does this in one line)
#    matmul<<<mesh>>>(A,B,C,M,N,K);        matmul_simple[grid](A, B, C, ...)
#    aieDeviceSynchronize();                ct.synchronize()
#    verify loop                            numpy comparison
#    free(A); free(B); free(C);             (garbage collected)
#
#  Triton equivalent:
#    aie_triton.set_device(0)               ct.set_device(0)
#    mesh = aie_triton.mesh(rows=2, cols=2) mesh = ct.mesh(rows=2, cols=2)
#    matmul_simple[grid](...)               matmul_simple[grid](...)
#    aie_triton.synchronize()               ct.synchronize()
###############################################################################

def main():
    print("=== CuTile-style Matrix Multiply on AIE Tile Mesh ===")

    M, N, K = 16, 16, 16           # → proposal3.cc: const int M=16, N=16, K=16;

    # ── Device setup ──
    # proposal3.cc: aieSetDevice(0);
    # Triton:       aie_triton.set_device(0)
    ct.set_device(0)

    # ── Host memory allocation ──
    # proposal3.cc: int32_t *A = (int32_t *)malloc(M * K * sizeof(int32_t));
    #               for (int i = 0; i < M * K; i++) A[i] = i + 1;
    # Triton:       A = np.arange(1, M*K+1, dtype=np.int8).reshape(M, K)
    # Note: proposal3.cc uses int32_t for host buffers but int8 on tile;
    # here we use int8 end-to-end for simplicity.
    A = np.arange(1, M * K + 1, dtype=np.int8).reshape(M, K)   # A[i] = i+1
    B = np.arange(1, K * N + 1, dtype=np.int8).reshape(K, N)   # B[i] = i+1
    C = np.zeros((M, N), dtype=np.int8)

    # ── Define tile mesh ──
    # proposal3.cc: aieDim mesh(2, 2);
    # Triton:       mesh = aie_triton.mesh(rows=2, cols=2)
    # → buildRoutingIR(ctx, meshRows=2, meshCols=2, tensors)
    mesh = ct.mesh(rows=2, cols=2)

    # ── Launch kernel ──
    # proposal3.cc: matmul<<<mesh>>>(A, B, C, M, N, K);
    # Triton:       matmul_simple[grid](A, B, C, M, N, K, BLOCK_M=8, ...)
    #
    # Under the hood, the Python frontend:
    #   1. Parses @ct.kernel AST → extracts tensor shapes + dtypes
    #   2. Constructs TensorParam list:
    #        TensorParam{shape={16,16}, elementBitWidth=8, isInput=true }  // A
    #        TensorParam{shape={16,16}, elementBitWidth=8, isInput=true }  // B
    #        TensorParam{shape={16,16}, elementBitWidth=8, isInput=false}  // C
    #   3. Calls buildRoutingIR(ctx, 2, 2, tensors) → routing dialect IR
    #   4. Runs MLIR pipeline: routing→dmap→dmaphop→blueprint→dfschedule
    #   5. Emits host.cc, kernel.cc, routing.cc, aieml.bcf, aieml.prx
    #   6. Cross-compiles: xchesscc → kernel ELF, aarch64-g++ → host ELF
    #   7. Deploys to board: loads ELFs, configures DMA BDs, runs cores
    grid = (mesh.rows, mesh.cols)

    # --- Option A: simple index-based CuTile kernel ---
    matmul_simple[grid](
        A, B, C,
        tm=8, tn=8, tk=8,
    )

    # --- Option B: swizzled CuTile kernel (uncomment to use) ---
    # matmul_swizzle[grid](
    #     A, B, C,
    #     tm=8, tn=8, tk=8,
    #     GROUP_SIZE_M=2,       # = mesh.rows (no-op on AIE)
    # )

    # ── Wait for AIE completion ──
    # proposal3.cc: aieDeviceSynchronize();
    # Triton:       aie_triton.synchronize()
    ct.synchronize()

    # ── Verify against CPU reference ──
    # proposal3.cc:
    #   for (int i = 0; i < M; i++)
    #     for (int j = 0; j < N; j++) {
    #       int32_t expected = 0;
    #       for (int k = 0; k < K; k++)
    #         expected += A[i*K+k] * B[k*N+j];
    #       if (C[i*N+j] != expected) { printf("MISMATCH..."); mismatches++; }
    #     }
    # Triton:       C_ref = (A.astype(np.int16) @ B.astype(np.int16)).clip(...)
    C_ref = (A.astype(np.int16) @ B.astype(np.int16)).clip(-128, 127).astype(np.int8)

    mismatches = 0
    for i in range(M):
        for j in range(N):
            if C[i, j] != C_ref[i, j]:
                # proposal3.cc: printf("MISMATCH C[%d][%d]: got %d, expected %d\n", ...)
                print(f"MISMATCH C[{i}][{j}]: got {C[i,j]}, expected {C_ref[i,j]}")
                mismatches += 1

    if mismatches == 0:
        # proposal3.cc: printf("PASS: all %d elements match.\n", M * N);
        print(f"PASS: all {M * N} elements match.")
    else:
        # proposal3.cc: printf("FAIL: %d mismatches.\n", mismatches);
        print(f"FAIL: {mismatches} mismatches.")

    print(f"\nResult C (first 4x4 block):")
    print(C[:4, :4])


if __name__ == "__main__":
    main()
