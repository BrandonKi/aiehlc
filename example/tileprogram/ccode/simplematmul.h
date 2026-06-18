/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 **/
// ═══════════════════════════════════════════════════════════════════════════
// Verification
//
// scalar_matmul: pure C[M][N] = A[M][K] * B^T[N][K], completely independent
//   of pipeline topology, tile partitioning, or DMA layout.
//
// verify_matmul: computes full matmul, then compares flat C[] against
//   reference. Host assembles tile outputs into the full C matrix.
// ═══════════════════════════════════════════════════════════════════════════
/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * AIE Programming Model — Matrix Multiplication (Parameterized Kernel API)
 *
 * GEMM: C[MxN] = A[MxK] * B^T[NxK], int8
 * Deployed on a HW_ROWS x HW_COLS AIE tile mesh.
 *
 * Data distribution:
 *   A[MxK]: broadcast per row (each tile row gets TILE_ROWS x K)
 *   B[NxK]: broadcast per col (each tile col gets TILE_COLS x K)
 *
 * Kernel strategy: "cache all A, stream B"
 *   Phase 1: Receive all A chunks into local memory
 *   Phase 2: Stream B chunks one at a time, computing against all cached A
 *   Phase 3: Output results
 *
 * All tiling parameters (rounds, buffer sizes, tile dimensions) are resolved
 * at compile time via aie::get_*() built-in query functions. The compiler
 * replaces these calls with integer literals derived from M/K/N, mesh dims,
 * and per-port SpatialPolicy (pp_depth, max_buffer_bytes).
 *
 ******************************************************************************/
// ═══════════════════════════════════════════════════════════════════════════
// KERNEL: matmul
//
// "Cache all A, stream B" design with compiler-resolved parameters:
//   - aie::get_num_rounds(win)  -> number of DMA input/output rounds
//   - aie::get_buffer_size(win) -> elements per round
//   - aie::get_tile_rows()      -> output rows per tile (M / HW_ROWS)
//   - aie::get_tile_cols()      -> output cols per tile (N / HW_COLS)
//   - aie::get_k_dim()          -> inner product dimension K
//
// Per-tile computation:
//   C_tile[tile_rows x tile_cols] = A_tile[tile_rows x K] * B_tile^T[tile_cols x K]
// ═══════════════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════════════
// Spatial Policy Definitions
//
// Each policy describes how data is distributed across the AIE tile mesh:
//   pattern:          Broadcast | Scatter | Gather
//   distribution:     Row | Col (which mesh axis owns the partition)
//   merge_order:      LeftToRight | RightToLeft (for output gathering)
//   pp_depth:         Ping-pong buffer depth (2, 4, 8)
//   max_buffer_bytes: Maximum per-buffer size in bytes
// ═══════════════════════════════════════════════════════════════════════════

#ifndef __AIESIM__
#include "xil_cache.h"
#endif
#include "aie_timer.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
// GEMM dimensions (user-specified, overridable via -D)
#ifndef M
#define M 256 // 4096
#endif
#ifndef K
#define K 256 // 4096
#endif
#ifndef N
#define N 256 // 4096
#endif

#ifndef HW_ROWS
#define HW_ROWS 4
#endif
#ifndef HW_COLS
#define HW_COLS 4
#endif

// Derived constants for host-side verification (not used in kernel)
// #define TILE_ROWS (M / HW_ROWS)
// #define TILE_COLS (N / HW_COLS)

static int verify_matmul(const int8_t *A, const int8_t *B, const int8_t *C);
static int verify_mat_transpose(const int8_t *A, const int8_t *B, const int8_t *C);

// Pure scalar matmul: C_ref[M][N] = A[M][K] * B^T[N][K]
static void scalar_matmul(int8_t *C_ref, const int8_t *A, const int8_t *B) {
    // Xil_DCacheEnable();
    // Xil_ICacheEnable();
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int16_t sum = 0;
            for (int k = 0; k < K; k++)
                sum += (int16_t)A[i * K + k] * (int16_t)B[j * K + k];
            if (sum > 127)
                sum = 127;
            else if (sum < -128)
                sum = -128;
            C_ref[i * N + j] = (int8_t)sum;
        }
    }
    // Xil_DCacheDisable();
    // Xil_ICacheDisable();
}

// Verify AIE output C against CPU reference
static int verify_matmul(const int8_t *A, const int8_t *B, const int8_t *C) {
    const int dataprintsize = 64;
    int mismatches = 0;
    int8_t C_ref[M * N];
    printf("before scalar_matmul\n");
    XTime t_start, t_end;
    XTime_GetTime(&t_start);
    scalar_matmul(C_ref, A, B);
    XTime_GetTime(&t_end);
    double elapsed_ms = 1.0 * (t_end - t_start) / COUNTS_PER_SECOND * 1000.0;
    printf("after scalar_matmul\n");
    printf("scalar_matmul time: %.3f ms\n", elapsed_ms);
    for (int i = 0; i < M * N; i++) {
        if (C[i] != C_ref[i]) {
            printf("MISMATCH C[%d]: got %d, expected %d\n", i, C[i], C_ref[i]);
            mismatches++;
            if (mismatches > 128)
                break;
        }
    }
    // Print A
    printf("\nA [%dx%d]:\n", M, K);
    for (int i = 0; i < (M > dataprintsize ? dataprintsize : M); i++) {
        printf("  [");
        for (int j = 0; j < (K > dataprintsize ? dataprintsize : K); j++) {
            printf("%4d", A[i * K + j]);
            if (j < K - 1)
                printf(",");
        }
        printf("]\n");
    }

    // Print B
    printf("\nB [%dx%d]:\n", K, N);
    for (int i = 0; i < (K > dataprintsize ? dataprintsize : K); i++) {
        printf("  [");
        for (int j = 0; j < (N > dataprintsize ? dataprintsize : N); j++) {
            printf("%4d", B[i * N + j]);
            if (j < N - 1)
                printf(",");
        }
        printf("]\n");
    }

    // Print C
    printf("\nC [%dx%d]:\n", M, N);
    for (int i = 0; i < (M > dataprintsize ? dataprintsize : M); i++) {
        printf("  [");
        for (int j = 0; j < (N > dataprintsize ? dataprintsize : N); j++) {
            printf("%4d", C[i * N + j]);
            if (j < N - 1)
                printf(",");
        }
        printf("]\n");
    }

    // Print C_ref
    printf("\nC_ref [%dx%d]:\n", M, N);
    for (int i = 0; i < (M > dataprintsize ? dataprintsize : M); i++) {
        printf("  [");
        for (int j = 0; j < (N > dataprintsize ? dataprintsize : N); j++) {
            printf("%4d", C_ref[i * N + j]);
            if (j < N - 1)
                printf(",");
        }
        printf("]\n");
    }

    int total_elements = M * N;
    if (mismatches == 0)
        printf("PASS: all %d elements match.\n", total_elements);
    else
        printf("FAIL: %d mismatches out of %d.\n", mismatches, total_elements);

    return mismatches;
}

// Verify raw tile-order output against CPU reference
// Assumes tiles output in row-major tile order: (0,0),(0,1),...
// Within each tile: sequential row-major TILE_ROWS x TILE_COLS
/*
static int verify_mat_transpose(const int8_t *A, const int8_t *B, const int8_t *C) {
    int8_t C_ref[M * N];
    scalar_matmul(C_ref, A, B);

    const int TILE_OUT_SZ = TILE_ROWS * TILE_COLS;
    int total = HW_ROWS * HW_COLS * TILE_OUT_SZ;
    int mismatches = 0;

    for (int flat = 0; flat < total; flat++) {
        int tile_idx = flat / TILE_OUT_SZ;
        int local = flat % TILE_OUT_SZ;
        int hw_row = tile_idx / HW_COLS;
        int hw_col = tile_idx % HW_COLS;
        int local_r = local / TILE_COLS;
        int local_c = local % TILE_COLS;
        int global_r = hw_row * TILE_ROWS + local_r;
        int global_c = hw_col * TILE_COLS + local_c;
        int8_t expected = C_ref[global_r * N + global_c];

        if (C[flat] != expected) {
            printf("MISMATCH C[%d] (tile(%d,%d) local[%d,%d] -> global[%d,%d]): "
                   "got %d, expected %d\n",
                   flat, hw_row, hw_col, local_r, local_c, global_r, global_c, C[flat], expected);
            mismatches++;
        }
    }

    if (mismatches == 0)
        printf("verify_mat_transpose PASS: all %d elements match.\n", total);
    else
        printf("verify_mat_transpose FAIL: %d mismatches out of %d.\n", mismatches, total);

    // Print A
    printf("\nA [%dx%d]:\n", M, K);
    for (int i = 0; i < M; i++) {
        printf("  [");
        for (int j = 0; j < K; j++) {
            printf("%4d", A[i * K + j]);
            if (j < K - 1)
                printf(",");
        }
        printf("]\n");
    }

    // Print B
    printf("\nB [%dx%d]:\n", K, N);
    for (int i = 0; i < K; i++) {
        printf("  [");
        for (int j = 0; j < N; j++) {
            printf("%4d", B[i * N + j]);
            if (j < N - 1)
                printf(",");
        }
        printf("]\n");
    }

    // Print C
    printf("\nC [%dx%d]:\n", M, N);
    for (int i = 0; i < M; i++) {
        printf("  [");
        for (int j = 0; j < N; j++) {
            printf("%4d", C[i * N + j]);
            if (j < N - 1)
                printf(",");
        }
        printf("]\n");
    }

    // Print C_ref
    printf("\nC_ref [%dx%d]:\n", M, N);
    for (int i = 0; i < M; i++) {
        printf("  [");
        for (int j = 0; j < N; j++) {
            printf("%4d", C_ref[i * N + j]);
            if (j < N - 1)
                printf(",");
        }
        printf("]\n");
    }
    return mismatches;
}
*/