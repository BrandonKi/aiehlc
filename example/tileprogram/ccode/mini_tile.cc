/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define M 16
#define K 64
#define N 16
#define HW_ROWS 1
#define HW_COLS 1

constexpr aie::GemmSpace InSpace = {.policy = {.map = {.act = aie::Pattern::Broadcast, .layout = aie::Layout::Row},
                                               .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
                                               .sched = {.pp_depth = 1, .l1_budget = aie::Bytes{4096}}},
                                    .d1 = {.fullsize = M, .tile_size = 16, .stride = 16},
                                    .d2 = {.fullsize = K, .tile_size = 64, .stride = 64}};

constexpr aie::GemmSpace OutSpace = {
    .policy = {.map = {.layout = aie::Layout::Row, .merge_order = aie::Flow::LeftToRight},
               .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
               .sched = {.pp_depth = 1, .l1_budget = aie::Bytes{4096}}},
    .d1 = {.fullsize = M, .tile_size = 16, .stride = 16},
    .d2 = {.fullsize = N, .tile_size = 16, .stride = 16}};

__global__ void copyk(aie::port<input_window_int8 *, InSpace> win_in,
                      aie::port<output_window_int8 *, OutSpace> win_out) {
    int8_t *in = (int8_t *)acquire_input_window(win_in);
    int8_t *out = (int8_t *)acquire_output_window(win_out);
    for (int i = 0; i < M * N; i++)
        out[i] = in[i];
    release_input_window(win_in);
    release_output_window(win_out);
}

int main() {
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({0, 3, 0, 6}, HW_ROWS, HW_COLS);
    int8_t *A = (int8_t *)device.alloc(M * K * sizeof(int8_t) * 4);
    int8_t *C = (int8_t *)device.alloc(M * N * sizeof(int8_t) * 4);
    for (int i = 0; i < M * K; i++)
        A[i] = (int8_t)(i % 7);
    for (int i = 0; i < M * N; i++)
        C[i] = 0;
    copyk<<<mesh>>>(A, C);
    device.free(A);
    device.free(C);
    return 0;
}
