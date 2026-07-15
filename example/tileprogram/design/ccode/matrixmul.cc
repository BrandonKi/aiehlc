/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * AIE Programming Model — Matrix Multiplication (GEMM)
 *
 * Demonstrates how the MLIR tilinglinalg pipeline maps a GEMM operation
 * (C = A * B) onto a 2x2 AIE tile mesh.
 *
 * Data partitioning strategy (matches tilinglinalg routing dialect):
 *
 *   A [16x16] — row-partitioned across tile rows, replicated on cols
 *     tile row 0: A[0:7,  0:15]  (rows 0-7,   all 16 columns)
 *     tile row 1: A[8:15, 0:15]  (rows 8-15,  all 16 columns)
 *
 *   B [16x16] — broadcast (full tensor) to all tile groups
 *     every tile receives B[0:15, 0:15]
 *
 *   C [16x16] — row-partitioned across tile rows, gathered from tiles
 *     tile row 0: computes C[0:7,  0:15]
 *     tile row 1: computes C[8:15, 0:15]
 *
 * This matches `routingmanager::createroutingfuncGEMM()` in the MLIR pipeline:
 *   - partitiontensor A: splitnum=2, splitdim=0, hw_axis_owner="row", replicate_on="col"
 *   - partitiontensor B: splitnum=1 (no split, broadcast along split axis)
 *   - partitiontensor C: splitnum=2, splitdim=0, hw_axis_owner="row", replicate_on="col"
 *
 * Each tile receives:
 *   window_in_0  = A partition [M_TILE x K]    (8x16 = 128 bytes)
 *   window_in_1  = B full      [K x N]         (16x16 = 256 bytes)
 *   window_out_0 = C partition [M_TILE x N]    (8x16 = 128 bytes)
 *
 * The kernel computes: C_tile = A_tile * B
 *   C_tile[i][j] = sum_k( A_tile[i][k] * B[k][j] )
 *
 * CUDA concepts kept (honest mapping):
 *   __global__             - kernel runs on AIE tiles
 *   kernel<<<mesh>>>()     - launch kernel across tile mesh
 *   aieDeviceSynchronize() - wait for all tiles to finish
 *   malloc/free            - plain C host memory allocation
 *
 * What the compiler handles automatically:
 *   DDR <-> tile DMA transfers, tensor partitioning, stream switch routing,
 *   buffer descriptors, lock synchronization, core load/run/wait
 *
 ******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// ═══════════════════════════════════════════════════════════════════════════
// Dimensions
//
// Full GEMM: C[M x N] = A[M x K] * B[K x N]
// Tile partition: M is split across tile rows (M_TILE = M / 2)
// K and N are not split — each tile does a full dot product across K,
// producing a full row of N output columns.
// ═══════════════════════════════════════════════════════════════════════════
#define M       16
#define K       16
#define N       16
#define M_TILE  (M / 2)   // 8 rows per tile row in a 2x2 mesh
#define BUF_SZ  (M_TILE * K / 4)  // buffer size in v4int8 units (for vectorized copy)

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL: matmul — C_tile = A_tile * B
//
// Each tile receives its row-partition of A and the full B matrix.
// Computes its corresponding rows of C.
//
// Inputs:
//   window_in_0: A_tile [M_TILE x K] = [8 x 16] int8 — row slice of A
//   window_in_1: B      [K x N]      = [16 x 16] int8 — full B (broadcast)
// Output:
//   window_out_0: C_tile [M_TILE x N] = [8 x 16] int8 — row slice of C
//
// The outer loop (k=0..1) implements double-buffering: the MLIR pipeline
// generates ping-pong buffer descriptors so DMA can fill the next buffer
// while the kernel processes the current one. In this example with a
// single GEMM pass, both iterations compute the same result.
// ═══════════════════════════════════════════════════════════════════════════
__global__ void matmul(input_window_int8 *window_in_0,
                       input_window_int8 *window_in_1,
                       output_window_int8 *window_out_0) {
    unsigned coreid = get_coreid();
    int col = coreid >> 16;
    int row = coreid & 0x1F;

    for (int iter = 0; iter < 2; iter++) {
        klog("CENk", iter);
        int8_t *A_tile = (int8_t *)acquire_input_window(window_in_0);
        int8_t *B      = (int8_t *)acquire_input_window(window_in_1);
        int8_t *C_tile = acquire_output_window(window_out_0);

        klog("IN0", (int8_t)(uintptr_t)A_tile);
        klog("IN1", (int8_t)(uintptr_t)B);
        klog("OUT", (int8_t)(uintptr_t)C_tile);

        // GEMM: C_tile[i][j] = sum_k( A_tile[i][k] * B[k][j] )
        for (int i = 0; i < M_TILE; i++) {
            for (int j = 0; j < N; j++) {
                int16_t sum = 0;
                for (int kk = 0; kk < K; kk++) {
                    sum += (int16_t)A_tile[i * K + kk] *
                           (int16_t)B[kk * N + j];
                }
                // Saturate to int8 range
                if (sum > 127)       sum = 127;
                else if (sum < -128) sum = -128;
                C_tile[i * N + j] = (int8_t)sum;
            }
        }
        klog("CLOP", M_TILE * N);

        release_input_window(window_in_0);
        release_input_window(window_in_1);
        release_output_window(window_out_0);
        klog("CEXT", 1);
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// HOST
//
// MLIR lowering flow (what the compiler does behind the scenes):
//
//   1. routing dialect:
//      - routingcreatehwmesh row=2, col=2
//      - partitiontensor A: splitnum=2, splitdim=0  → A[0:7,:] and A[8:15,:]
//      - partitiontensor B: splitnum=1 (broadcast)  → full B to each tile group
//      - partitiontensor C: splitnum=2, splitdim=0  → C[0:7,:] and C[8:15,:]
//
//   2. routing → dmap → dmaphop → blueprint → dfschedule:
//      For each tensor partition, the pipeline generates:
//        - DMA buffer descriptors (BDs) with ping-pong double buffering
//        - Lock acquire/release sequences for producer-consumer sync
//        - Stream switch routing from SHIM tile to compute tile
//        - Kernel load and launch per tile
//
//   3. dfschedule → EmitC → host.cc + kernel.cc:
//      Final C code generation with XAie driver API calls
//
// ═══════════════════════════════════════════════════════════════════════════
int main() {
    printf("=== Matrix Multiply (GEMM) on AIE 2x2 Tile Mesh ===\n");
    printf("    C[%dx%d] = A[%dx%d] * B[%dx%d], int8\n", M, N, M, K, K, N);

    // --- Device + mesh ---
    aieSetDevice(0);
    aieDim mesh(2, 2);

    // --- Allocate host memory ---
    int8_t *A = (int8_t *)malloc(M * K * sizeof(int8_t));
    int8_t *B = (int8_t *)malloc(K * N * sizeof(int8_t));
    int8_t *C = (int8_t *)malloc(M * N * sizeof(int8_t));

    // --- Initialize input matrices ---
    // A: sequential pattern starting from 1
    for (int i = 0; i < M * K; i++)
        A[i] = (int8_t)((i % 7) - 3);  // values in [-3, 3] to avoid overflow

    // B: sequential pattern starting from 1
    for (int i = 0; i < K * N; i++)
        B[i] = (int8_t)((i % 5) - 2);  // values in [-2, 2] to avoid overflow

    // Zero output
    for (int i = 0; i < M * N; i++)
        C[i] = 0;

    // --- Launch kernel on tile mesh ---
    // The compiler partitions A by rows and broadcasts B to all tiles.
    // Each tile computes M_TILE rows of C using its A partition and full B.
    //
    // Tile row 0 (tiles [0,0] and [0,1]):
    //   receives A[0:7, 0:15], B[0:15, 0:15]
    //   computes C[0:7, 0:15]
    //
    // Tile row 1 (tiles [1,0] and [1,1]):
    //   receives A[8:15, 0:15], B[0:15, 0:15]
    //   computes C[8:15, 0:15]
    matmul<<<mesh>>>(A, B, C);

    // --- Wait for completion ---
    aieDeviceSynchronize();

    // --- CPU reference computation ---
    int8_t *C_ref = (int8_t *)malloc(M * N * sizeof(int8_t));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int16_t sum = 0;
            for (int kk = 0; kk < K; kk++) {
                sum += (int16_t)A[i * K + kk] * (int16_t)B[kk * N + j];
            }
            if (sum > 127)       sum = 127;
            else if (sum < -128) sum = -128;
            C_ref[i * N + j] = (int8_t)sum;
        }
    }

    // --- Verify ---
    int mismatches = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            if (C[idx] != C_ref[idx]) {
                printf("MISMATCH C[%d][%d]: got %d, expected %d\n",
                       i, j, C[idx], C_ref[idx]);
                mismatches++;
            }
        }
    }

    if (mismatches == 0)
        printf("PASS: all %d elements match.\n", M * N);
    else
        printf("FAIL: %d mismatches out of %d.\n", mismatches, M * N);

    // --- Print a few elements to show correctness ---
    printf("\nSample output C[0:3][0:3]:\n");
    for (int i = 0; i < 4; i++) {
        printf("  [");
        for (int j = 0; j < 4; j++) {
            printf("%4d", C_ref[i * N + j]);
            if (j < 3) printf(",");
        }
        printf("]\n");
    }

    // --- Print data partitioning summary ---
    printf("\n--- MLIR Data Partitioning Summary ---\n");
    printf("  Mesh: 2 rows x 2 cols = 4 tiles\n");
    printf("  A[%dx%d]: row-partitioned (splitnum=2, splitdim=0)\n", M, K);
    printf("    tile row 0 → A[0:%d, 0:%d]  (%d bytes)\n",
           M_TILE - 1, K - 1, M_TILE * K);
    printf("    tile row 1 → A[%d:%d, 0:%d]  (%d bytes)\n",
           M_TILE, M - 1, K - 1, M_TILE * K);
    printf("  B[%dx%d]: broadcast (splitnum=1, full tensor to each group)\n", K, N);
    printf("    all tiles  → B[0:%d, 0:%d]  (%d bytes)\n",
           K - 1, N - 1, K * N);
    printf("  C[%dx%d]: row-partitioned (splitnum=2, splitdim=0)\n", M, N);
    printf("    tile row 0 → C[0:%d, 0:%d]  (%d bytes)\n",
           M_TILE - 1, N - 1, M_TILE * N);
    printf("    tile row 1 → C[%d:%d, 0:%d]  (%d bytes)\n",
           M_TILE, M - 1, N - 1, M_TILE * N);

    // --- Cleanup ---
    free(A);
    free(B);
    free(C);
    free(C_ref);
    return 0;
}
