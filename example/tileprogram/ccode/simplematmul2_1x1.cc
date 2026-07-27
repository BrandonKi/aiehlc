/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * 1x1 debug variant of simplematmul2.cc
 * Single tile, reduced M=N=16 K=64 — isolates DMA/lock deadlock on one tile.
 */
#undef HW_ROWS
#undef HW_COLS
#undef M
#undef N
#undef K
#define HW_ROWS 1
#define HW_COLS 1
#define M 16
#define N 16
#define K 64
#include "simplematmul.h"
#pragma aie_debug_level(2 | AIE_DEBUG_FLAG_DISABLE_PARTITIONTEARDOWN)

// 1x1 mesh: each tile gets the full M-tile × K-chunk for A,
//           the full N-tile × K-chunk for B, and outputs full C.
// d1.tile_size == d1.fullsize (stride == size) → no spatial partitioning across tiles.
constexpr aie::GemmSpace RowBA = {
    .policy = {.map = {.act = aie::Pattern::Broadcast, .layout = aie::Layout::Row},
               .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
               .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{4096}}},
    .d1 = {.fullsize = M, .tile_size = 16, .stride = 16},  // A: M-tile = all M rows
    .d2 = {.fullsize = K, .tile_size = 64, .stride = 64}}; // A: K chunk (1 k-round for K=64)
constexpr aie::GemmSpace ColBB = {.policy = {.map = {.wgt = aie::Pattern::Broadcast, .layout = aie::Layout::Col},
                                             .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
                                             .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{4096}}},
                                  .d1 = {.fullsize = N, .tile_size = 16, .stride = 16},  // B: N-tile = all N cols
                                  .d2 = {.fullsize = K, .tile_size = 64, .stride = 64}}; // B: K chunk
constexpr aie::GemmSpace LtoR_Merge = {
    .policy = {.map = {.layout = aie::Layout::Row, .merge_order = aie::Flow::LeftToRight},
               .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
               .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{4096}}},
    .d1 = {.fullsize = M, .tile_size = 16, .stride = 16},  // C: M-tile
    .d2 = {.fullsize = N, .tile_size = 16, .stride = 16}}; // C: N-tile

#define DEBUG_OUTPUT_ORDER 1
constexpr aie::GlobalPolicy matmul_policy = {.fullconnect_auto = 1};
__global__(matmul_policy) void matmul(aie::port<input_window_int8 *, RowBA> win_a,
                                      aie::port<input_window_int8 *, ColBB> win_b,
                                      aie::port<output_window_int8 *, LtoR_Merge> win_c) {

    const int tile_rows = aie::get_tile_rows();
    const int tile_cols = aie::get_tile_cols();
    const int eff_k = aie::get_effective_k();
    const int k_rounds = aie::get_k_rounds();
    const int num_a_rounds = aie::get_num_rounds(win_a);
    const int num_b_rounds = aie::get_num_rounds(win_b);
    const int num_c_rounds = aie::get_num_rounds(win_c);
    const int buf_sz_a = aie::get_buffer_size(win_a);
    const int buf_sz_b = aie::get_buffer_size(win_b);
    const int buf_sz_c = aie::get_buffer_size(win_c);
    const int m_rounds = aie::get_spatial_multiple_rounds(win_a);
    const int n_rounds = aie::get_spatial_multiple_rounds(win_b);

    const int rows_per_round = buf_sz_a / eff_k;
    const int cols_per_round = buf_sz_b / eff_k;

#if DEBUG_OUTPUT_ORDER
    unsigned coreid = get_coreid();
    int col = coreid >> 16;
    int row = coreid & 0x1F;
    klog("DEBUG", 3);
#endif

    int8_t all_A[tile_rows * eff_k];
    int16_t accum[tile_rows * tile_cols];
    int8_t local_out[tile_rows * tile_cols];

    for (int mr = 0; mr < m_rounds * n_rounds; mr++) {
        klog("MR  ", (int32_t)mr);
        for (int i = 0; i < tile_rows * tile_cols; i++)
            accum[i] = 0;

        for (int kr = 0; kr < k_rounds; kr++) {
            klog("KRA ", (int32_t)kr);
            for (int ra = 0; ra < num_a_rounds; ra++) {
                int8_t *A_ptr = (int8_t *)acquire_input_window(win_a);
                for (int i = 0; i < buf_sz_a; i++)
                    all_A[ra * buf_sz_a + i] = A_ptr[i];
#if DEBUG_OUTPUT_ORDER
                for (int l = 0; l < (buf_sz_a < 8 ? buf_sz_a : 8); l++)
                    klog("A   ", (int32_t)A_ptr[l]);
#endif
                release_input_window(win_a);
            }

            for (int rb = 0; rb < num_b_rounds; rb++) {
                int8_t *B_ptr = (int8_t *)acquire_input_window(win_b);
                for (int i = 0; i < tile_rows; i++) {
                    for (int j = 0; j < cols_per_round; j++) {
                        int16_t sum = 0;
                        for (int k = 0; k < eff_k; k++)
                            sum += (int16_t)all_A[i * eff_k + k] * (int16_t)B_ptr[j * eff_k + k];
                        accum[i * tile_cols + rb * cols_per_round + j] += sum;
                    }
                }
#if DEBUG_OUTPUT_ORDER
                klog("B0  ", (int32_t)B_ptr[0]);
#endif
                release_input_window(win_b);
            }
        }

        for (int i = 0; i < tile_rows * tile_cols; i++) {
            int16_t val = accum[i];
            if (val > 127)
                val = 127;
            else if (val < -128)
                val = -128;
            local_out[i] = (int8_t)val;
        }
        for (int rc = 0; rc < num_c_rounds; rc++) {
            int8_t *out = (int8_t *)acquire_output_window(win_c);
            const int rows_per_c_round = buf_sz_c / tile_cols;
            for (int i = 0; i < rows_per_c_round; i++)
                for (int j = 0; j < tile_cols; j++)
                    out[i * tile_cols + j] = local_out[rc * buf_sz_c + i * tile_cols + j];
#if DEBUG_OUTPUT_ORDER
            klog("C0 ", (int32_t)out[0]);
#endif
            release_output_window(win_c);
        }
    }
}

// HOST
int main() {
    printf("=== 1x1 Debug GEMM: C[%dx%d] = A[%dx%d] * B^T[%dx%d], int8 ===\n", M, N, M, K, K, N);
    aieSetDevice(0);
    aieArray device;
    // 1x1 mesh: AIE2PS NMU switch init requires cols 0+1 even for 1-tile partitions.
    // Use endCol=1 (numCols=2) to satisfy driver; only col 0 row 3 has a compute tile.
    aieMesh mesh = device.partition({0, 1, 0, 3}, HW_ROWS, HW_COLS);
    int8_t *A = (int8_t *)device.alloc(M * K * sizeof(int8_t) * 4);
    int8_t *B = (int8_t *)device.alloc(K * N * sizeof(int8_t) * 4);
    int8_t *C = (int8_t *)device.alloc(M * N * sizeof(int8_t) * 4);
    for (int i = 0; i < M * K; i++)
        A[i] = (int8_t)((i % 7) - 3);
    for (int i = 0; i < K * N; i++)
        B[i] = (int8_t)((i % 5) - 2);
    for (int i = 0; i < M * N; i++)
        C[i] = 0;
    matmul<<<mesh>>>(A, B, C, M, N, K);
    int result = verify_matmul(A, B, C);
    device.free(A);
    device.free(B);
    device.free(C);
    return result;
}
