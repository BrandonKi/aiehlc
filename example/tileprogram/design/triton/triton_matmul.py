###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# AIE Programming Model — Triton-style Python Frontend for Matrix Multiply
#
# This file is the definitive Triton-style Python input example for matmul
# targeting AMD Versal AI Engine via aiehlc. It contains two kernel variants
# and a host section that matches proposal3.cc exactly.
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │                        Compilation Flow                                │
# │                                                                        │
# │  triton_matmul.py                                                      │
# │    │  Python AST parse (@aie_triton.jit functions)                     │
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
# │               Triton ←→ AIE / proposal3.cc Concept Map                │
# │                                                                        │
# │  Triton Python                  │ AIE C / proposal3.cc                 │
# │  ──────────────────────────────┼────────────────────────────────────── │
# │  @aie_triton.jit               │ __global__ void matmul(...)           │
# │  tl.program_id(0), (1)         │ get_coreid(): row = id & 0x1F,       │
# │                                │               col = id >> 16          │
# │  tl.load(a_ptr + offsets)      │ acquire_input_window(window_in_0)     │
# │  tl.load(b_ptr + offsets)      │ acquire_input_window(window_in_1)     │
# │  tl.store(c_ptr + offsets, v)  │ acquire_output_window(window_out_0)   │
# │                                │ + write + release_output_window       │
# │  tl.dot(a, b)                  │ v4int8 MAC loop over BUF_SZ           │
# │  tl.zeros(shape, dtype)        │ local buffer via BCF (known address)  │
# │  acc.to(tl.int8)               │ saturate int32 → int8 in kernel       │
# │  aie_triton.mesh(rows, cols)   │ aieDim mesh(2, 2)                     │
# │  kernel[grid](A, B, C, ...)    │ matmul<<<mesh>>>(A, B, C, M, N, K)   │
# │  aie_triton.set_device(0)      │ aieSetDevice(0)                       │
# │  aie_triton.synchronize()      │ aieDeviceSynchronize()                │
# │  np.array / np.zeros           │ malloc() for DDR buffers              │
# │  A = np.arange(1, M*K+1, ...)  │ for(i=0;i<M*K;i++) A[i]=i+1;        │
# └─────────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │              buildRoutingIR Mapping (tilinglinalg_pipeline.h)          │
# │                                                                        │
# │  Python frontend extracts from @aie_triton.jit + kernel[grid]():       │
# │    meshRows=2, meshCols=2  (from aie_triton.mesh / grid)              │
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
# Each tile: BLOCK_M=8, BLOCK_N=8, BLOCK_K=8
#   A partition: 8x16 (full K, partial M) → streamed in K-dim as 8x8 chunks
#   B partition: 16x8 (full K, partial N) → streamed in K-dim as 8x8 chunks
#   C partition: 8x8  (written back to DDR via DMA)
#
###############################################################################

import aie_triton
import aie_triton.language as tl
import numpy as np


###############################################################################
#  KERNEL VARIANT 1 — Simple / Window API style
#
#  Closest to proposal3.cc's window_in / window_out pattern.
#  Uses tl.load / tl.store with fixed 8×8 blocks, directly mapping to
#  acquire_input_window / release_output_window.
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
###############################################################################

@aie_triton.jit
def matmul_simple(
    # Pointers to DDR matrices (runtime partitions them to each tile via DMA)
    a_ptr,      # → proposal3.cc: window_in_0 (input_window_int8 *)
    b_ptr,      # → proposal3.cc: window_in_1 (input_window_int8 *)
    c_ptr,      # → proposal3.cc: window_out_0 (output_window_int8 *)
    # Matrix dimensions
    M, N, K,
    # Block sizes — constrained by AIE tile local memory (~32 KB)
    BLOCK_M: tl.constexpr = 8,   # rows per tile
    BLOCK_N: tl.constexpr = 8,   # cols per tile
    BLOCK_K: tl.constexpr = 8,   # K-chunk streamed per DMA iteration
):
    """Simple window-style matmul kernel for one AIE tile.

    Mapping to proposal3.cc:
        @aie_triton.jit         → __global__
        tl.program_id(0)        → get_coreid() & 0x1F  (row)
        tl.program_id(1)        → get_coreid() >> 16    (col)
        tl.load(a_ptr + ...)    → acquire_input_window(window_in_0)
        tl.load(b_ptr + ...)    → acquire_input_window(window_in_1)
        tl.dot(a, b)            → v4int8 MAC loop (BUF_SZ iterations)
        tl.store(c_ptr + ...)   → acquire_output_window + write + release
        for k_start in range    → for (int k = 0; k < 2; k++) in proposal3
    """
    # ── Which tile am I? ──
    # proposal3.cc:  unsigned coreid = get_coreid();
    #                int row = coreid & 0x1F;
    #                int col = coreid >> 16;
    tile_row = tl.program_id(axis=0)  # → row = coreid & 0x1F
    tile_col = tl.program_id(axis=1)  # → col = coreid >> 16

    # ── Block offsets (implicit in proposal3.cc — handled by DMA partitioning) ──
    row_start = tile_row * BLOCK_M
    col_start = tile_col * BLOCK_N

    # ── Accumulator in tile-local memory ──
    # proposal3.cc: local buffer at BCF-assigned address
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    # ── K-dimension loop ──
    # proposal3.cc:  for (int k = 0; k < 2; k++) {
    #   K=16, BLOCK_K=8 → 2 iterations, matching proposal3.cc's loop count
    #
    # Each iteration maps to one acquire/compute/release cycle:
    #   acquire_input_window → tl.load
    #   MAC computation      → tl.dot
    #   release_input_window → (implicit at end of tl.load lifetime)
    for k_start in range(0, K, BLOCK_K):

        # ── Load A block [BLOCK_M × BLOCK_K] ──
        # proposal3.cc:  int8_t *in0 = (int8_t *)acquire_input_window(window_in_0);
        a_block = tl.load(                                    # → acquire_input_window(window_in_0)
            a_ptr + (row_start + tl.arange(0, BLOCK_M)[:, None]) * K
                  + (k_start  + tl.arange(0, BLOCK_K)[None, :])
        )  # shape: [8, 8]

        # ── Load B block [BLOCK_K × BLOCK_N] ──
        # proposal3.cc:  int8_t *in1 = (int8_t *)acquire_input_window(window_in_1);
        b_block = tl.load(                                    # → acquire_input_window(window_in_1)
            b_ptr + (k_start  + tl.arange(0, BLOCK_K)[:, None]) * N
                  + (col_start + tl.arange(0, BLOCK_N)[None, :])
        )  # shape: [8, 8]

        # ── Multiply-accumulate ──
        # proposal3.cc:  v4int8 data0 = *((v4int8 *)&in0[i*4]);
        #                v4int8 data1 = *((v4int8 *)&in1[i*4]);
        #                // real impl: acc += data0 * data1 (vector MAC)
        acc += tl.dot(a_block, b_block)                       # → AIE v4int8/v16int8 MAC

        # ── End of iteration ──
        # proposal3.cc:  release_input_window(window_in_0);
        #                release_input_window(window_in_1);
        # (implicit — aiehlc inserts lock release at DMA boundary)

    # ── Store result C block [BLOCK_M × BLOCK_N] ──
    # proposal3.cc:  int8_t *out = acquire_output_window(window_out_0);
    #                *((v4int8 *)&out[i*4]) = result;
    #                release_output_window(window_out_0);
    c_offsets = (
        (row_start + tl.arange(0, BLOCK_M)[:, None]) * N
        + (col_start + tl.arange(0, BLOCK_N)[None, :])
    )
    result = acc.to(tl.int8)                                  # saturate int32 → int8
    tl.store(c_ptr + c_offsets, result)                       # → acquire_output + write + release


###############################################################################
#  KERNEL VARIANT 2 — Strided Triton style
#
#  Standard Triton matmul pattern with explicit strides, for users familiar
#  with GPU Triton. Uses make_block_ptr and masking (masks are no-ops on AIE
#  when dimensions are exact multiples of block size).
#
#  Same lowering path — the Python frontend normalizes both variants into
#  identical routing IR via buildRoutingIR().
###############################################################################

@aie_triton.jit
def matmul_strided(
    # Pointers + strides (standard Triton convention)
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,   # A strides: row-major → stride_am=K, stride_ak=1
    stride_bk, stride_bn,   # B strides: row-major → stride_bk=N, stride_bn=1
    stride_cm, stride_cn,   # C strides: row-major → stride_cm=N, stride_cn=1
    BLOCK_M: tl.constexpr = 8,
    BLOCK_N: tl.constexpr = 8,
    BLOCK_K: tl.constexpr = 8,
):
    """Strided Triton-style matmul kernel for one AIE tile.

    This follows the standard OpenAI Triton matmul tutorial structure but
    adapted for AIE constraints:
      - int8 data type (AIE AIEML native vector width)
      - Small blocks (8×8) due to 32 KB tile-local memory
      - No masking needed when M, N, K are exact multiples of block sizes

    The Python frontend maps this to the same proposal3.cc primitives:
      make_block_ptr + advance → acquire_input_window per K-iteration
      tl.dot                   → v4int8 MAC loop
      tl.store via block_ptr   → acquire_output_window + write + release
    """
    # ── Tile identity ──
    # In GPU Triton: pid = tl.program_id(0), then split into pid_m, pid_n
    # On AIE: directly mapped to tile row/col in the mesh
    pid_m = tl.program_id(axis=0)   # → get_coreid() & 0x1F
    pid_n = tl.program_id(axis=1)   # → get_coreid() >> 16

    # ── Block pointers (Triton 2.0+ style) ──
    # make_block_ptr creates a structured pointer with shape, strides, and offsets.
    # On AIE: this tells the frontend which DMA region to fetch (window_in_0, _1).
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),         # row slice, K starts at 0
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),                         # row-major
    )

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),         # K starts at 0, col slice
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    # ── Accumulator ──
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    # ── K-dimension reduction loop ──
    # Each iteration: advance block pointers along K, load, MAC.
    # On AIE this maps to the same acquire/compute/release cycle as matmul_simple.
    for k in range(0, K, BLOCK_K):
        # Load blocks via structured pointers
        # → acquire_input_window(window_in_0) / acquire_input_window(window_in_1)
        a_block = tl.load(a_block_ptr)      # [BLOCK_M, BLOCK_K]
        b_block = tl.load(b_block_ptr)      # [BLOCK_K, BLOCK_N]

        # MAC → v4int8 vector multiply-accumulate on AIE
        acc += tl.dot(a_block, b_block)

        # Advance pointers along K for next iteration
        # On AIE: next DMA BD fires, ping-pong buffer swap
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    # ── Store result ──
    # → acquire_output_window(window_out_0) + write + release_output_window
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    result = acc.to(tl.int8)
    tl.store(c_block_ptr, result)


###############################################################################
#  HOST — matches proposal3.cc main() exactly
#
#  proposal3.cc:                          Python equivalent:
#    const int M=16, N=16, K=16;            M, N, K = 16, 16, 16
#    aieSetDevice(0);                       aie_triton.set_device(0)
#    aieDim mesh(2, 2);                     mesh = aie_triton.mesh(rows=2, cols=2)
#    int32_t *A = malloc(M*K*sizeof);       A = np.arange(1, M*K+1, ...)
#    for(i) A[i] = i+1;                    (numpy arange does this in one line)
#    matmul<<<mesh>>>(A,B,C,M,N,K);        matmul_simple[grid](A, B, C, ...)
#    aieDeviceSynchronize();                aie_triton.synchronize()
#    verify loop                            numpy comparison
#    free(A); free(B); free(C);             (garbage collected)
###############################################################################

def main():
    print("=== Triton-style Matrix Multiply on AIE Tile Mesh ===")

    M, N, K = 16, 16, 16           # → proposal3.cc: const int M=16, N=16, K=16;

    # ── Device setup ──
    aie_triton.set_device(0)        # → proposal3.cc: aieSetDevice(0);

    # ── Host memory allocation ──
    # proposal3.cc: int32_t *A = (int32_t *)malloc(M * K * sizeof(int32_t));
    #               for (int i = 0; i < M * K; i++) A[i] = i + 1;
    # Note: proposal3.cc uses int32_t for host buffers but int8 on tile;
    # here we use int8 end-to-end for simplicity.
    A = np.arange(1, M * K + 1, dtype=np.int8).reshape(M, K)   # A[i] = i+1
    B = np.arange(1, K * N + 1, dtype=np.int8).reshape(K, N)   # B[i] = i+1
    C = np.zeros((M, N), dtype=np.int8)

    # ── Define tile mesh ──
    # proposal3.cc: aieDim mesh(2, 2);
    # → buildRoutingIR(ctx, meshRows=2, meshCols=2, tensors)
    mesh = aie_triton.mesh(rows=2, cols=2)

    # ── Launch kernel ──
    # proposal3.cc: matmul<<<mesh>>>(A, B, C, M, N, K);
    #
    # Under the hood, the Python frontend:
    #   1. Parses @aie_triton.jit AST → extracts tensor shapes + dtypes
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

    # --- Option A: simple window-style kernel ---
    matmul_simple[grid](
        A, B, C,
        M, N, K,
        BLOCK_M=8, BLOCK_N=8, BLOCK_K=8,
    )

    # --- Option B: strided Triton-style kernel (uncomment to use) ---
    # matmul_strided[grid](
    #     A, B, C,
    #     M, N, K,
    #     stride_am=K, stride_ak=1,    # A is row-major [M, K]
    #     stride_bk=N, stride_bn=1,    # B is row-major [K, N]
    #     stride_cm=N, stride_cn=1,    # C is row-major [M, N]
    #     BLOCK_M=8, BLOCK_N=8, BLOCK_K=8,
    # )

    # ── Wait for AIE completion ──
    # proposal3.cc: aieDeviceSynchronize();
    aie_triton.synchronize()

    # ── Verify against CPU reference ──
    # proposal3.cc:
    #   for (int i = 0; i < M; i++)
    #     for (int j = 0; j < N; j++) {
    #       int32_t expected = 0;
    #       for (int k = 0; k < K; k++)
    #         expected += A[i*K+k] * B[k*N+j];
    #       if (C[i*N+j] != expected) { printf("MISMATCH..."); mismatches++; }
    #     }
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
