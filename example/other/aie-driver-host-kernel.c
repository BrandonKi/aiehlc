/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <stdio.h>
typedef int input_window_int32;
typedef int output_window_int32;
__global__ void square(input_window_int32 * win, output_window_int32 *out) {
    int a = 0;
    // some kernel logic.
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
    XAie_PartitionInitalize();
    XAie_Kernel_Inst kinst = XAie_LoadKernel(square, {12, 3}/*aie tile location*/);
    XAie_RoutingHostToAie(kinst,  {2/*shim column*/, 0/*mm2s channel 0*/, 3/*port 3*/});
    XAie_RoutingAieToHost(kinst,  {35/*shim column*/, 0/*mm2s channel 0*/, 2/*port 2*/});
    XAie_MoveDataToAie(kinst, h_a);
    XAie_MoveDataToAie(kinst, h_b);
    XAie_MoveAieToData(kinst, h_c);
    XAie_RunCore(kinst);
    XAie_WaitOutData(kinst);
    XAie_Finish();
    for (int i = 0; i < n * n; i++) {
	    printf(" %x \n", h_c[i]);
    }

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}

