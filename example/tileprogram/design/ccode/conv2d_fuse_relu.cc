/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * AIE Programming Model — Conv2d + BatchNorm + LeakyReLU Fusion
 *
 * ResNet-style fused layer example using the tile programming model.
 * Two sequential fused kernels: conv3x3 + BN + LeakyReLU each.
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

// --- Dimensions ---
// Input feature map: 8x8, 1 channel, int8
// Each tile in 2x2 mesh gets a 4x4 spatial partition
#define FEAT_H  8
#define FEAT_W  8
#define TILE_H  4
#define TILE_W  4
#define TILE_SZ (TILE_H * TILE_W)   // 16 elements per tile

// Conv filter: 3x3, 1 input channel, 1 output channel
#define FILTER_SZ  9                // 3x3
#define PARAM_SZ   11               // 9 (filter) + 1 (bn scale) + 1 (bn bias)

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL 1: Conv2d 3x3 + BatchNorm + LeakyReLU (fused)
//
// Each tile processes its 4x4 spatial partition of the 8x8 feature map.
// BN is pre-fused as scale+bias (common inference optimization).
// LeakyReLU: positive pass-through, negative scaled by ~0.125 (>>3).
// ═══════════════════════════════════════════════════════════════════════════
__global__ void conv_bn_relu_1(input_window_int8 *window_in_feat,
                                input_window_int8 *window_in_params,
                                output_window_int8 *window_out_feat) {
    unsigned coreid = get_coreid();
    int col = coreid >> 16;
    int row = coreid & 0x1F;

    for (int k = 0; k < 2; k++) {
        klog("CENk", k);
        int8_t *feat_in  = (int8_t *)acquire_input_window(window_in_feat);
        int8_t *params   = (int8_t *)acquire_input_window(window_in_params);
        int8_t *feat_out = acquire_output_window(window_out_feat);

        klog("IN0", (int8_t)(uintptr_t)feat_in);
        klog("IN1", (int8_t)(uintptr_t)params);
        klog("OUT", (int8_t)(uintptr_t)feat_out);

        // Pre-load BN parameters from end of params buffer
        int8_t bn_scale = params[FILTER_SZ];      // index 9
        int8_t bn_bias  = params[FILTER_SZ + 1];  // index 10

        // --- Conv2d 3x3 + BN + LeakyReLU (fused) ---
        for (int i = 0; i < TILE_H; i++) {
            for (int j = 0; j < TILE_W; j++) {
                int16_t sum = 0;

                // 3x3 convolution with zero-padding at tile boundary
                for (int kh = -1; kh <= 1; kh++) {
                    for (int kw = -1; kw <= 1; kw++) {
                        int ih = i + kh;
                        int iw = j + kw;
                        if (ih >= 0 && ih < TILE_H && iw >= 0 && iw < TILE_W) {
                            sum += (int16_t)feat_in[ih * TILE_W + iw] *
                                   (int16_t)params[(kh + 1) * 3 + (kw + 1)];
                        }
                    }
                }
                int8_t conv_out = (int8_t)sum;

                // BatchNorm (pre-fused): out = scale * x >> 7 + bias
                int16_t bn_out = ((int16_t)conv_out * (int16_t)bn_scale) >> 7;
                bn_out += bn_bias;

                // LeakyReLU: positive pass-through, negative >> 3 (~alpha=0.125)
                int8_t result;
                if (bn_out >= 0)
                    result = (int8_t)bn_out;
                else
                    result = (int8_t)(bn_out >> 3);

                feat_out[i * TILE_W + j] = result;
            }
        }
        klog("CLOP", TILE_SZ);

        release_input_window(window_in_feat);
        release_input_window(window_in_params);
        release_output_window(window_out_feat);
        klog("CEXT", 1);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL 2: Conv2d 3x3 + BatchNorm + LeakyReLU (fused, second layer)
//
// Same structure as kernel 1, operates on the intermediate feature map
// produced by kernel 1. Uses a separate set of weights and BN params.
// ═══════════════════════════════════════════════════════════════════════════
__global__ void conv_bn_relu_2(input_window_int8 *window_in_feat,
                                input_window_int8 *window_in_params,
                                output_window_int8 *window_out_feat) {
    unsigned coreid = get_coreid();
    int col = coreid >> 16;
    int row = coreid & 0x1F;

    for (int k = 0; k < 2; k++) {
        klog("CENk", k);
        int8_t *feat_in  = (int8_t *)acquire_input_window(window_in_feat);
        int8_t *params   = (int8_t *)acquire_input_window(window_in_params);
        int8_t *feat_out = acquire_output_window(window_out_feat);

        klog("IN0", (int8_t)(uintptr_t)feat_in);
        klog("IN1", (int8_t)(uintptr_t)params);
        klog("OUT", (int8_t)(uintptr_t)feat_out);

        int8_t bn_scale = params[FILTER_SZ];
        int8_t bn_bias  = params[FILTER_SZ + 1];

        // --- Conv2d 3x3 + BN + LeakyReLU (fused) ---
        for (int i = 0; i < TILE_H; i++) {
            for (int j = 0; j < TILE_W; j++) {
                int16_t sum = 0;

                for (int kh = -1; kh <= 1; kh++) {
                    for (int kw = -1; kw <= 1; kw++) {
                        int ih = i + kh;
                        int iw = j + kw;
                        if (ih >= 0 && ih < TILE_H && iw >= 0 && iw < TILE_W) {
                            sum += (int16_t)feat_in[ih * TILE_W + iw] *
                                   (int16_t)params[(kh + 1) * 3 + (kw + 1)];
                        }
                    }
                }
                int8_t conv_out = (int8_t)sum;

                int16_t bn_out = ((int16_t)conv_out * (int16_t)bn_scale) >> 7;
                bn_out += bn_bias;

                int8_t result;
                if (bn_out >= 0)
                    result = (int8_t)bn_out;
                else
                    result = (int8_t)(bn_out >> 3);

                feat_out[i * TILE_W + j] = result;
            }
        }
        klog("CLOP", TILE_SZ);

        release_input_window(window_in_feat);
        release_input_window(window_in_params);
        release_output_window(window_out_feat);
        klog("CEXT", 1);
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// HOST
// ═══════════════════════════════════════════════════════════════════════════

// CPU reference: conv3x3 + BN(scale,bias) + LeakyReLU over full 8x8 map
static void cpu_conv_bn_relu(const int8_t *input, const int8_t *params,
                              int8_t *output, int H, int W) {
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            int16_t sum = 0;
            for (int kh = -1; kh <= 1; kh++) {
                for (int kw = -1; kw <= 1; kw++) {
                    int ih = i + kh;
                    int iw = j + kw;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        sum += (int16_t)input[ih * W + iw] *
                               (int16_t)params[(kh + 1) * 3 + (kw + 1)];
                    }
                }
            }
            int8_t conv_out = (int8_t)sum;

            int8_t bn_scale = params[FILTER_SZ];
            int8_t bn_bias  = params[FILTER_SZ + 1];
            int16_t bn_out = ((int16_t)conv_out * (int16_t)bn_scale) >> 7;
            bn_out += bn_bias;

            if (bn_out >= 0)
                output[i * W + j] = (int8_t)bn_out;
            else
                output[i * W + j] = (int8_t)(bn_out >> 3);
        }
    }
}

int main() {
    // --- Device + mesh ---
    aieSetDevice(0);
    aieDim mesh(2, 2);

    // --- Allocate host memory ---
    int8_t *input        = (int8_t *)malloc(FEAT_H * FEAT_W * sizeof(int8_t));
    int8_t *params1      = (int8_t *)malloc(PARAM_SZ * sizeof(int8_t));  // conv1 weights + bn1
    int8_t *params2      = (int8_t *)malloc(PARAM_SZ * sizeof(int8_t));  // conv2 weights + bn2
    int8_t *intermediate = (int8_t *)malloc(FEAT_H * FEAT_W * sizeof(int8_t));
    int8_t *output       = (int8_t *)malloc(FEAT_H * FEAT_W * sizeof(int8_t));

    // --- Initialize input with test pattern ---
    for (int i = 0; i < FEAT_H * FEAT_W; i++)
        input[i] = (int8_t)(i + 1);

    // --- Conv1 weights: simple 3x3 edge-detect style filter ---
    //  0 -1  0
    // -1  4 -1
    //  0 -1  0
    int8_t filter1[FILTER_SZ] = { 0, -1, 0,  -1, 4, -1,  0, -1, 0 };
    for (int i = 0; i < FILTER_SZ; i++) params1[i] = filter1[i];
    params1[9]  = 64;   // bn1 scale (0.5 in Q7 fixed-point: 64/128)
    params1[10] = 1;    // bn1 bias

    // --- Conv2 weights: simple smoothing filter ---
    //  1  1  1
    //  1  1  1
    //  1  1  1
    for (int i = 0; i < FILTER_SZ; i++) params2[i] = 1;
    params2[9]  = 32;   // bn2 scale (0.25 in Q7: 32/128)
    params2[10] = 0;    // bn2 bias

    // --- Launch kernel 1: conv + bn + relu on input ---
    conv_bn_relu_1<<<mesh>>>(input, params1, intermediate);

    // --- Launch kernel 2: conv + bn + relu on intermediate ---
    conv_bn_relu_2<<<mesh>>>(intermediate, params2, output);

    // --- Wait for completion ---
    aieDeviceSynchronize();

    // --- CPU reference computation ---
    int8_t *ref_inter = (int8_t *)malloc(FEAT_H * FEAT_W * sizeof(int8_t));
    int8_t *ref_out   = (int8_t *)malloc(FEAT_H * FEAT_W * sizeof(int8_t));

    cpu_conv_bn_relu(input, params1, ref_inter, FEAT_H, FEAT_W);
    cpu_conv_bn_relu(ref_inter, params2, ref_out, FEAT_H, FEAT_W);

    // --- Verify ---
    int mismatches = 0;
    for (int i = 0; i < FEAT_H; i++) {
        for (int j = 0; j < FEAT_W; j++) {
            int idx = i * FEAT_W + j;
            if (output[idx] != ref_out[idx]) {
                printf("MISMATCH [%d][%d]: got %d, expected %d\n",
                       i, j, output[idx], ref_out[idx]);
                mismatches++;
            }
        }
    }
    if (mismatches == 0)
        printf("PASS: all %d elements match.\n", FEAT_H * FEAT_W);
    else
        printf("FAIL: %d mismatches out of %d.\n", mismatches, FEAT_H * FEAT_W);

    // --- Cleanup ---
    free(input);
    free(params1);
    free(params2);
    free(intermediate);
    free(output);
    free(ref_inter);
    free(ref_out);
    return 0;
}
