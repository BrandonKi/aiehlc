/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
 
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define DEBUG_OUTPUT_ORDER 1
#define M 16
#define K 16
#define N 16

// HW mesh dimensions (number of AIE tile rows and columns)
#define HW_ROWS 4
#define HW_COLS 4

// --- Tile dimensions derived from M, K, N, HW_ROWS, HW_COLS ---
#define ROWS_PER_ROUND ((M / HW_ROWS) / 2)       // 4: A/B rows per DMA round
#define COLS_PER_ROUND ((N / HW_COLS) / 2)       // 4: B rows per DMA round (= output cols of partial block)
#define OUT_STRIDE (N / HW_COLS)                 // 8: output row width (full tile cols)
#define K_DIM K                                  // 16: inner product dimension
#define BUF_SZ_OUT (ROWS_PER_ROUND * OUT_STRIDE) // 32: output bytes per round
// #define DEBUG_OUTPUT_ORDER 1

__global__ void matmul(input_window_int8 *window_in_0, input_window_int8 *window_in_1,
                       output_window_int8 *window_out_0) {
    // Local cache for round-0 data
    int8_t cache_A[ROWS_PER_ROUND * K_DIM];
    int8_t cache_B[ROWS_PER_ROUND * K_DIM];

    // ===== Iter 0: read A[0:1,:], B[0:1,:], cache, write partial output =====
    int8_t *A0 = (int8_t *)acquire_input_window(window_in_0);
    int8_t *B0 = (int8_t *)acquire_input_window(window_in_1);

    // Cache for use in iter 1
    for (int i = 0; i < ROWS_PER_ROUND * K_DIM; i++) {
        cache_A[i] = A0[i];
        cache_B[i] = B0[i];
#if DEBUG_OUTPUT_ORDER
        if (i < 8) {
            klog("A   ", (int32_t)A0[i]);
            klog("B   ", (int32_t)B0[i]);
        }
#endif
    }

    // Write partial output: C[0:RPR-1, 0:CPR-1] = A0 * B0^T, cols CPR..OUT_STRIDE-1 = 0
    {
        int8_t *out = acquire_output_window(window_out_0);
        for (int i = 0; i < ROWS_PER_ROUND; i++) {
            for (int j = 0; j < COLS_PER_ROUND; j++) {
                int16_t sum = 0;
                for (int k = 0; k < K_DIM; k++) {
                    sum += (int16_t)A0[i * K_DIM + k] * (int16_t)B0[j * K_DIM + k];
                }
                if (sum > 127)
                    sum = 127;
                else if (sum < -128)
                    sum = -128;
                out[i * OUT_STRIDE + j] = (int8_t)sum;
            }
            for (int j = COLS_PER_ROUND; j < OUT_STRIDE; j++) {
                out[i * OUT_STRIDE + j] = 0;
            }
        }
        release_output_window(window_out_0);
    }

    release_input_window(window_in_0);
    release_input_window(window_in_1);

    // ===== Iter 1: read A[2:3,:], B[2:3,:], compute remaining output =====
    int8_t *A1 = (int8_t *)acquire_input_window(window_in_0);
    int8_t *B1 = (int8_t *)acquire_input_window(window_in_1);

    {
        int8_t *out = acquire_output_window(window_out_0);
        // C[RPR:2*RPR-1, 0:CPR-1] = A1 * cached_B0^T
        for (int i = 0; i < ROWS_PER_ROUND; i++) {
            for (int j = 0; j < COLS_PER_ROUND; j++) {
                int16_t sum = 0;
                for (int k = 0; k < K_DIM; k++) {
                    sum += (int16_t)A1[i * K_DIM + k] * (int16_t)cache_B[j * K_DIM + k];
                }
                if (sum > 127)
                    sum = 127;
                else if (sum < -128)
                    sum = -128;
                out[i * OUT_STRIDE + j] = (int8_t)sum;
            }
#if DEBUG_OUTPUT_ORDER
            if (i < 8) {
                klog("C   ", (int32_t)out[i]);
            }
#endif
        }
        // C[RPR:2*RPR-1, CPR:2*CPR-1] = A1 * B1^T
        for (int i = 0; i < ROWS_PER_ROUND; i++) {
            for (int j = 0; j < COLS_PER_ROUND; j++) {
                int16_t sum = 0;
                for (int k = 0; k < K_DIM; k++) {
                    sum += (int16_t)A1[i * K_DIM + k] * (int16_t)B1[j * K_DIM + k];
                }
                if (sum > 127)
                    sum = 127;
                else if (sum < -128)
                    sum = -128;
                out[i * OUT_STRIDE + j + COLS_PER_ROUND] = (int8_t)sum;
            }
        }

        release_output_window(window_out_0);
    }

    release_input_window(window_in_0);
    release_input_window(window_in_1);
}
int main() {
    // --- Device + mesh ---
    aieArray device;
    aieMesh mesh = device.partition(HW_ROWS, HW_COLS); // startCol=0; sets _dev before alloc()
    // --- Allocate host memory ---
    int8_t *A = (int8_t *)device.alloc(M * K * sizeof(int8_t));
    int8_t *B = (int8_t *)device.alloc(K * N * sizeof(int8_t));
    int8_t *C = (int8_t *)device.alloc(M * N * sizeof(int8_t));

    // --- Initialize input matrices ---
    for (int i = 0; i < M * K; i++)
        A[i] = (int8_t)((i % 7) - 3);
    for (int i = 0; i < K * N; i++)
        B[i] = (int8_t)((i % 5) - 2);
    for (int i = 0; i < M * N; i++)
        C[i] = 0;

    // --- Launch kernel on tile mesh ---
    matmul<<<mesh>>>(A, B, C);

    printf("Output C:\n");
    for (int i = 0; i < M; i++) {
        printf("[");
        for (int j = 0; j < N; j++) {
            printf("%d", C[i * N + j]);
            if (j < N - 1)
                printf(", ");
        }
        printf("]\n");
    }
    // --- Cleanup ---
    device.free(A);
    device.free(B);
    device.free(C);
    return 1;
}
