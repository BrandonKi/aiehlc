/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <stdio.h>
typedef int input_window_int32;
typedef int output_window_int32;
__global__ void square(input_window_int32 * win, output_window_int32 *out) {
    int a = 0;
    /*
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
    */
}

int main() {
    int n = 256;  // Matrix size, for simplicity assuming square matrices.
    size_t bytes = n * n * sizeof(float);

    // Allocate memory on the host
    float *h_a = (float *)malloc(bytes * 2);
    float *h_b = (float *)((uint8_t*)h_a + bytes);
    float *h_c = (float *)malloc(bytes);
    // Initialize matrices
    for (int i = 0; i < n * n; i++) {
        h_a[i] = 1.0;  // Example values
        h_b[i] = 2.0;
    }
    square<12, 0>(h_a, hc);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}

