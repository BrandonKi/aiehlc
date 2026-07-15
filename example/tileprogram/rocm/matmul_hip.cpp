/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Canonical ROCm/HIP int8 GEMM used as the reference input for the `rcom`
 * front end (src/tool/frontend/rcom.py).
 *
 * Semantics: C[M x N] = A[M x K] * B^T[N x K]  (int8, saturating to int8).
 *   B is stored row-major as B[N][K] (i.e. already transposed relative to the
 *   textbook C = A * B), matching the AIE `simplematmul2.cc` data layout.
 *
 * rcom recognizes this as a GEMM (triple loop with `sum += A[..] * B[..]`,
 * output written via `C[...] = ...`) and emits the proven AIE matmul template
 * rather than translating the thread-indexed body line-for-line.
 ******************************************************************************/
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

#define M 256
#define N 256
#define K 256

// Device GEMM kernel: each thread computes one C[row][col].
// A is [M][K], B is [N][K] (B^T layout), C is [M][N].
__global__ void gemm(const int8_t *A, const int8_t *B, int8_t *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        int16_t sum = 0;
        for (int k = 0; k < K; k++) {
            sum += (int16_t)A[row * K + k] * (int16_t)B[col * K + k];
        }
        if (sum > 127)
            sum = 127;
        else if (sum < -128)
            sum = -128;
        C[row * N + col] = (int8_t)sum;
    }
}

int main() {
    const int sizeA = M * K * sizeof(int8_t);
    const int sizeB = K * N * sizeof(int8_t);
    const int sizeC = M * N * sizeof(int8_t);

    int8_t *hA = (int8_t *)malloc(sizeA);
    int8_t *hB = (int8_t *)malloc(sizeB);
    int8_t *hC = (int8_t *)malloc(sizeC);

    for (int i = 0; i < M * K; i++)
        hA[i] = (int8_t)((i % 7) - 3);
    for (int i = 0; i < K * N; i++)
        hB[i] = (int8_t)((i % 5) - 2);

    int8_t *dA, *dB, *dC;
    hipMalloc(&dA, sizeA);
    hipMalloc(&dB, sizeB);
    hipMalloc(&dC, sizeC);

    hipMemcpy(dA, hA, sizeA, hipMemcpyHostToDevice);
    hipMemcpy(dB, hB, sizeB, hipMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    gemm<<<grid, block>>>(dA, dB, dC, M, N, K);

    hipMemcpy(hC, dC, sizeC, hipMemcpyDeviceToHost);

    hipFree(dA);
    hipFree(dB);
    hipFree(dC);
    free(hA);
    free(hB);
    free(hC);
    return 0;
}
