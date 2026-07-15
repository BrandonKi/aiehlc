/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * AIE Programming Model — Full ResNet18 Inference
 *
 * Complete ResNet18 with 8 BasicBlocks, residual skip connections,
 * 1x1 downsample projections, global average pooling, and FC classifier.
 * Scaled-down dimensions: 8x8 input, channels 4→8→16→32, 4 classes.
 * Data type: int8 with Q7 fixed-point for BN scale/bias.
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
#include <string.h>

// ═══════════════════════════════════════════════════════════════════════════
// Dimensions (scaled-down ResNet18)
// ═══════════════════════════════════════════════════════════════════════════
#define INPUT_H       8
#define INPUT_W       8
#define INPUT_C       1
#define NUM_CLASSES   4

// Channel progression: 4 → 8 → 16 → 32
#define CH0           4     // conv1 output, layer1
#define CH1           8     // layer2
#define CH2           16    // layer3
#define CH3           32    // layer4

// Spatial sizes after each downsampling (stride=2)
#define S0            8     // after conv1, layer1:  8x8
#define S1            4     // after layer2:         4x4
#define S2            2     // after layer3:         2x2
#define S3            1     // after layer4:         1x1

// Q7 fixed-point scale for BN (0.5 = 64/128)
#define BN_SCALE_DEFAULT  64
#define BN_BIAS_DEFAULT   0

// ═══════════════════════════════════════════════════════════════════════════
// Helper: compute param buffer size for a conv layer
//   conv weights: Cin * Cout * K * K
//   bn params:    Cout (scale) + Cout (bias)
// We pack: [config: 6 bytes] [weights] [bn_scale] [bn_bias]
//   config = { H, W, Cin, Cout, K, stride }
// ═══════════════════════════════════════════════════════════════════════════
#define CONFIG_SZ     6
#define CONV_PARAM_SZ(Cin, Cout, K)  (CONFIG_SZ + (Cin)*(Cout)*(K)*(K) + (Cout)*2)
#define FC_PARAM_SZ(Cin, Cout)       ((Cin)*(Cout) + (Cout))  // weights + bias

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL 1: conv_bn_relu — Conv + BatchNorm + ReLU (fused)
//
// Parameterized by config packed in params buffer header:
//   params[0..5] = { H, W, Cin, Cout, K, stride }
//   params[6 .. 6+Cin*Cout*K*K-1] = conv weights
//   params[6+Cin*Cout*K*K .. ]    = bn_scale[Cout], bn_bias[Cout]
//
// Handles both 3x3 (K=3) and 1x1 (K=1) convolutions.
// ═══════════════════════════════════════════════════════════════════════════
__global__ void conv_bn_relu(input_window_int8 *window_in_feat,
                              input_window_int8 *window_in_params,
                              output_window_int8 *window_out_feat) {
    int8_t *feat_in  = (int8_t *)acquire_input_window(window_in_feat);
    int8_t *params   = (int8_t *)acquire_input_window(window_in_params);
    int8_t *feat_out = acquire_output_window(window_out_feat);

    // Unpack config
    int H      = (uint8_t)params[0];
    int W      = (uint8_t)params[1];
    int Cin    = (uint8_t)params[2];
    int Cout   = (uint8_t)params[3];
    int K      = (uint8_t)params[4];
    int stride = (uint8_t)params[5];
    int pad    = K / 2;  // 1 for 3x3, 0 for 1x1

    int8_t *weights  = &params[CONFIG_SZ];
    int wt_sz        = Cin * Cout * K * K;
    int8_t *bn_scale = &params[CONFIG_SZ + wt_sz];
    int8_t *bn_bias  = &params[CONFIG_SZ + wt_sz + Cout];

    int outH = H / stride;
    int outW = W / stride;

    for (int oc = 0; oc < Cout; oc++) {
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                int16_t sum = 0;
                for (int ic = 0; ic < Cin; ic++) {
                    for (int kh = 0; kh < K; kh++) {
                        for (int kw = 0; kw < K; kw++) {
                            int ih = oh * stride + kh - pad;
                            int iw = ow * stride + kw - pad;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                int in_idx = ic * H * W + ih * W + iw;
                                int wt_idx = oc * Cin * K * K + ic * K * K + kh * K + kw;
                                sum += (int16_t)feat_in[in_idx] * (int16_t)weights[wt_idx];
                            }
                        }
                    }
                }
                // BatchNorm: out = (conv * scale) >> 7 + bias
                int16_t bn_out = (sum * (int16_t)bn_scale[oc]) >> 7;
                bn_out += bn_bias[oc];
                // ReLU
                int8_t result = (bn_out > 127) ? 127 : (bn_out < 0) ? 0 : (int8_t)bn_out;
                feat_out[oc * outH * outW + oh * outW + ow] = result;
            }
        }
    }

    release_input_window(window_in_feat);
    release_input_window(window_in_params);
    release_output_window(window_out_feat);
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL 2: conv_bn — Conv + BatchNorm (no activation)
//
// Same param layout as conv_bn_relu, but no ReLU at the end.
// Used for the second conv in each BasicBlock (before residual add).
// ═══════════════════════════════════════════════════════════════════════════
__global__ void conv_bn(input_window_int8 *window_in_feat,
                         input_window_int8 *window_in_params,
                         output_window_int8 *window_out_feat) {
    int8_t *feat_in  = (int8_t *)acquire_input_window(window_in_feat);
    int8_t *params   = (int8_t *)acquire_input_window(window_in_params);
    int8_t *feat_out = acquire_output_window(window_out_feat);

    int H      = (uint8_t)params[0];
    int W      = (uint8_t)params[1];
    int Cin    = (uint8_t)params[2];
    int Cout   = (uint8_t)params[3];
    int K      = (uint8_t)params[4];
    int stride = (uint8_t)params[5];
    int pad    = K / 2;

    int8_t *weights  = &params[CONFIG_SZ];
    int wt_sz        = Cin * Cout * K * K;
    int8_t *bn_scale = &params[CONFIG_SZ + wt_sz];
    int8_t *bn_bias  = &params[CONFIG_SZ + wt_sz + Cout];

    int outH = H / stride;
    int outW = W / stride;

    for (int oc = 0; oc < Cout; oc++) {
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                int16_t sum = 0;
                for (int ic = 0; ic < Cin; ic++) {
                    for (int kh = 0; kh < K; kh++) {
                        for (int kw = 0; kw < K; kw++) {
                            int ih = oh * stride + kh - pad;
                            int iw = ow * stride + kw - pad;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                int in_idx = ic * H * W + ih * W + iw;
                                int wt_idx = oc * Cin * K * K + ic * K * K + kh * K + kw;
                                sum += (int16_t)feat_in[in_idx] * (int16_t)weights[wt_idx];
                            }
                        }
                    }
                }
                // BatchNorm only (no ReLU)
                int16_t bn_out = (sum * (int16_t)bn_scale[oc]) >> 7;
                bn_out += bn_bias[oc];
                int8_t result = (bn_out > 127) ? 127 : (bn_out < -128) ? -128 : (int8_t)bn_out;
                feat_out[oc * outH * outW + oh * outW + ow] = result;
            }
        }
    }

    release_input_window(window_in_feat);
    release_input_window(window_in_params);
    release_output_window(window_out_feat);
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL 3: residual_add_relu — Element-wise add + ReLU
//
// Adds main path and skip path tensors, then applies ReLU.
// Both inputs must have the same size (len elements).
// ═══════════════════════════════════════════════════════════════════════════
__global__ void residual_add_relu(input_window_int8 *window_in_main,
                                   input_window_int8 *window_in_skip,
                                   output_window_int8 *window_out,
                                   int len) {
    int8_t *main_in = (int8_t *)acquire_input_window(window_in_main);
    int8_t *skip_in = (int8_t *)acquire_input_window(window_in_skip);
    int8_t *out     = acquire_output_window(window_out);

    for (int i = 0; i < len; i++) {
        int16_t sum = (int16_t)main_in[i] + (int16_t)skip_in[i];
        // Saturate + ReLU
        if (sum > 127)  sum = 127;
        if (sum < 0)    sum = 0;
        out[i] = (int8_t)sum;
    }

    release_input_window(window_in_main);
    release_input_window(window_in_skip);
    release_output_window(window_out);
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL 4: avgpool_fc — Global Average Pooling + Fully-Connected
//
// Params buffer: [fc_weights: Cin*Cout] [fc_bias: Cout]
// Input:  spatial_h x spatial_w x channels (CHW layout)
// Output: num_classes logits
// ═══════════════════════════════════════════════════════════════════════════
__global__ void avgpool_fc(input_window_int8 *window_in_feat,
                            input_window_int8 *window_in_params,
                            output_window_int8 *window_out_logits,
                            int spatial_h, int spatial_w, int channels,
                            int num_classes) {
    int8_t *feat_in   = (int8_t *)acquire_input_window(window_in_feat);
    int8_t *fc_params = (int8_t *)acquire_input_window(window_in_params);
    int8_t *logits    = acquire_output_window(window_out_logits);

    int8_t *fc_weights = fc_params;
    int8_t *fc_bias    = &fc_params[channels * num_classes];

    // Global average pooling: average over spatial dims per channel
    int spatial_sz = spatial_h * spatial_w;
    int8_t pooled[32];  // max channels = CH3 = 32
    for (int c = 0; c < channels; c++) {
        int16_t sum = 0;
        for (int s = 0; s < spatial_sz; s++) {
            sum += (int16_t)feat_in[c * spatial_sz + s];
        }
        pooled[c] = (int8_t)(sum / spatial_sz);
    }

    // Fully-connected: logits[j] = sum_i(pooled[i] * weight[i * num_classes + j]) + bias[j]
    for (int j = 0; j < num_classes; j++) {
        int16_t acc = 0;
        for (int i = 0; i < channels; i++) {
            acc += (int16_t)pooled[i] * (int16_t)fc_weights[i * num_classes + j];
        }
        acc += fc_bias[j];
        logits[j] = (acc > 127) ? 127 : (acc < -128) ? -128 : (int8_t)acc;
    }

    release_input_window(window_in_feat);
    release_input_window(window_in_params);
    release_output_window(window_out_logits);
}


// ═══════════════════════════════════════════════════════════════════════════
// CPU REFERENCE FUNCTIONS (for verification)
// ═══════════════════════════════════════════════════════════════════════════

// Pack conv config into param buffer header
static void pack_config(int8_t *params, int H, int W, int Cin, int Cout,
                         int K, int stride) {
    params[0] = (int8_t)H;
    params[1] = (int8_t)W;
    params[2] = (int8_t)Cin;
    params[3] = (int8_t)Cout;
    params[4] = (int8_t)K;
    params[5] = (int8_t)stride;
}

// Initialize conv weights with alternating 1/-1 pattern
static void init_conv_weights(int8_t *dst, int count) {
    for (int i = 0; i < count; i++)
        dst[i] = (i % 2 == 0) ? 1 : -1;
}

// Initialize BN params: scale=BN_SCALE_DEFAULT, bias=BN_BIAS_DEFAULT
static void init_bn_params(int8_t *scale, int8_t *bias, int channels) {
    for (int i = 0; i < channels; i++) {
        scale[i] = BN_SCALE_DEFAULT;
        bias[i]  = BN_BIAS_DEFAULT;
    }
}

// CPU: Conv + BN + ReLU
static void cpu_conv_bn_relu(const int8_t *input, const int8_t *params,
                              int8_t *output) {
    int H      = (uint8_t)params[0];
    int W      = (uint8_t)params[1];
    int Cin    = (uint8_t)params[2];
    int Cout   = (uint8_t)params[3];
    int K      = (uint8_t)params[4];
    int stride = (uint8_t)params[5];
    int pad    = K / 2;

    const int8_t *weights  = &params[CONFIG_SZ];
    int wt_sz              = Cin * Cout * K * K;
    const int8_t *bn_scale = &params[CONFIG_SZ + wt_sz];
    const int8_t *bn_bias  = &params[CONFIG_SZ + wt_sz + Cout];

    int outH = H / stride;
    int outW = W / stride;

    for (int oc = 0; oc < Cout; oc++) {
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                int16_t sum = 0;
                for (int ic = 0; ic < Cin; ic++) {
                    for (int kh = 0; kh < K; kh++) {
                        for (int kw = 0; kw < K; kw++) {
                            int ih = oh * stride + kh - pad;
                            int iw = ow * stride + kw - pad;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                int in_idx = ic * H * W + ih * W + iw;
                                int wt_idx = oc * Cin * K * K + ic * K * K + kh * K + kw;
                                sum += (int16_t)input[in_idx] * (int16_t)weights[wt_idx];
                            }
                        }
                    }
                }
                int16_t bn_out = (sum * (int16_t)bn_scale[oc]) >> 7;
                bn_out += bn_bias[oc];
                int8_t result = (bn_out > 127) ? 127 : (bn_out < 0) ? 0 : (int8_t)bn_out;
                output[oc * outH * outW + oh * outW + ow] = result;
            }
        }
    }
}

// CPU: Conv + BN (no activation)
static void cpu_conv_bn(const int8_t *input, const int8_t *params,
                          int8_t *output) {
    int H      = (uint8_t)params[0];
    int W      = (uint8_t)params[1];
    int Cin    = (uint8_t)params[2];
    int Cout   = (uint8_t)params[3];
    int K      = (uint8_t)params[4];
    int stride = (uint8_t)params[5];
    int pad    = K / 2;

    const int8_t *weights  = &params[CONFIG_SZ];
    int wt_sz              = Cin * Cout * K * K;
    const int8_t *bn_scale = &params[CONFIG_SZ + wt_sz];
    const int8_t *bn_bias  = &params[CONFIG_SZ + wt_sz + Cout];

    int outH = H / stride;
    int outW = W / stride;

    for (int oc = 0; oc < Cout; oc++) {
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                int16_t sum = 0;
                for (int ic = 0; ic < Cin; ic++) {
                    for (int kh = 0; kh < K; kh++) {
                        for (int kw = 0; kw < K; kw++) {
                            int ih = oh * stride + kh - pad;
                            int iw = ow * stride + kw - pad;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                int in_idx = ic * H * W + ih * W + iw;
                                int wt_idx = oc * Cin * K * K + ic * K * K + kh * K + kw;
                                sum += (int16_t)input[in_idx] * (int16_t)weights[wt_idx];
                            }
                        }
                    }
                }
                int16_t bn_out = (sum * (int16_t)bn_scale[oc]) >> 7;
                bn_out += bn_bias[oc];
                int8_t result = (bn_out > 127) ? 127 : (bn_out < -128) ? -128 : (int8_t)bn_out;
                output[oc * outH * outW + oh * outW + ow] = result;
            }
        }
    }
}

// CPU: Element-wise add + ReLU
static void cpu_add_relu(const int8_t *main_path, const int8_t *skip_path,
                           int8_t *output, int len) {
    for (int i = 0; i < len; i++) {
        int16_t sum = (int16_t)main_path[i] + (int16_t)skip_path[i];
        if (sum > 127)  sum = 127;
        if (sum < 0)    sum = 0;
        output[i] = (int8_t)sum;
    }
}

// CPU: Global Average Pooling + FC
static void cpu_avgpool_fc(const int8_t *feat, const int8_t *fc_params,
                             int8_t *logits, int spatial_h, int spatial_w,
                             int channels, int num_classes) {
    int spatial_sz = spatial_h * spatial_w;
    const int8_t *fc_weights = fc_params;
    const int8_t *fc_bias    = &fc_params[channels * num_classes];

    int8_t pooled[32];
    for (int c = 0; c < channels; c++) {
        int16_t sum = 0;
        for (int s = 0; s < spatial_sz; s++)
            sum += (int16_t)feat[c * spatial_sz + s];
        pooled[c] = (int8_t)(sum / spatial_sz);
    }

    for (int j = 0; j < num_classes; j++) {
        int16_t acc = 0;
        for (int i = 0; i < channels; i++)
            acc += (int16_t)pooled[i] * (int16_t)fc_weights[i * num_classes + j];
        acc += fc_bias[j];
        logits[j] = (acc > 127) ? 127 : (acc < -128) ? -128 : (int8_t)acc;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Allocate + initialize a conv param buffer
// Returns malloc'd buffer with config + weights + BN packed
// ═══════════════════════════════════════════════════════════════════════════
static int8_t *make_conv_params(int H, int W, int Cin, int Cout,
                                 int K, int stride) {
    int param_sz = CONV_PARAM_SZ(Cin, Cout, K);
    int8_t *buf = (int8_t *)malloc(param_sz);
    memset(buf, 0, param_sz);

    pack_config(buf, H, W, Cin, Cout, K, stride);

    int wt_count = Cin * Cout * K * K;
    init_conv_weights(&buf[CONFIG_SZ], wt_count);
    init_bn_params(&buf[CONFIG_SZ + wt_count],
                   &buf[CONFIG_SZ + wt_count + Cout], Cout);
    return buf;
}

// Allocate + initialize FC param buffer (weights + bias)
static int8_t *make_fc_params(int Cin, int Cout) {
    int param_sz = FC_PARAM_SZ(Cin, Cout);
    int8_t *buf = (int8_t *)malloc(param_sz);
    // Uniform weights = 1, bias = 0
    for (int i = 0; i < Cin * Cout; i++)
        buf[i] = 1;
    for (int i = 0; i < Cout; i++)
        buf[Cin * Cout + i] = 0;
    return buf;
}


// ═══════════════════════════════════════════════════════════════════════════
// HOST: main
//
// ResNet18 architecture (scaled-down):
//   conv1 → layer1 (2 blocks) → layer2 (2 blocks, downsample)
//   → layer3 (2 blocks, downsample) → layer4 (2 blocks, downsample)
//   → avgpool → fc
// ═══════════════════════════════════════════════════════════════════════════
int main() {
    printf("=== ResNet18 (scaled-down) on AIE Tile Mesh ===\n");

    // --- Device + mesh ---
    aieSetDevice(0);
    aieDim mesh(2, 2);

    // ─── Input ───
    int input_sz = INPUT_H * INPUT_W * INPUT_C;
    int8_t *input = (int8_t *)malloc(input_sz);
    for (int i = 0; i < input_sz; i++)
        input[i] = (int8_t)((i % 7) + 1);  // test pattern: 1..7 repeating

    // ─── Feature map buffers (reusable scratch) ───
    // Max feature map size across the network: 8*8*CH0 = 256
    int max_feat_sz = S0 * S0 * CH0;  // 256
    int8_t *feat1   = (int8_t *)malloc(max_feat_sz);  // persistent between blocks
    int8_t *feat2   = (int8_t *)malloc(max_feat_sz);
    int8_t *feat3   = (int8_t *)malloc(max_feat_sz);
    int8_t *tmp1    = (int8_t *)malloc(max_feat_sz);  // scratch
    int8_t *tmp2    = (int8_t *)malloc(max_feat_sz);  // scratch
    int8_t *skip_ds = (int8_t *)malloc(max_feat_sz);  // downsample skip path

    // Output logits
    int8_t *logits = (int8_t *)malloc(NUM_CLASSES);

    // ─── Weight parameter buffers ───
    // conv1: 1→4, 3x3, stride=1, input 8x8
    int8_t *params_conv1 = make_conv_params(S0, S0, INPUT_C, CH0, 3, 1);

    // layer1: BasicBlock 0 & 1 (4→4, 3x3, stride=1, 8x8)
    int8_t *params_l1b0_c1 = make_conv_params(S0, S0, CH0, CH0, 3, 1);
    int8_t *params_l1b0_c2 = make_conv_params(S0, S0, CH0, CH0, 3, 1);
    int8_t *params_l1b1_c1 = make_conv_params(S0, S0, CH0, CH0, 3, 1);
    int8_t *params_l1b1_c2 = make_conv_params(S0, S0, CH0, CH0, 3, 1);

    // layer2: BasicBlock 0 (4→8, stride=2) + downsample, BasicBlock 1 (8→8)
    int8_t *params_l2b0_c1 = make_conv_params(S0, S0, CH0, CH1, 3, 2);  // 8x8→4x4
    int8_t *params_l2b0_c2 = make_conv_params(S1, S1, CH1, CH1, 3, 1);
    int8_t *params_l2_ds   = make_conv_params(S0, S0, CH0, CH1, 1, 2);  // 1x1 downsample
    int8_t *params_l2b1_c1 = make_conv_params(S1, S1, CH1, CH1, 3, 1);
    int8_t *params_l2b1_c2 = make_conv_params(S1, S1, CH1, CH1, 3, 1);

    // layer3: BasicBlock 0 (8→16, stride=2) + downsample, BasicBlock 1 (16→16)
    int8_t *params_l3b0_c1 = make_conv_params(S1, S1, CH1, CH2, 3, 2);  // 4x4→2x2
    int8_t *params_l3b0_c2 = make_conv_params(S2, S2, CH2, CH2, 3, 1);
    int8_t *params_l3_ds   = make_conv_params(S1, S1, CH1, CH2, 1, 2);
    int8_t *params_l3b1_c1 = make_conv_params(S2, S2, CH2, CH2, 3, 1);
    int8_t *params_l3b1_c2 = make_conv_params(S2, S2, CH2, CH2, 3, 1);

    // layer4: BasicBlock 0 (16→32, stride=2) + downsample, BasicBlock 1 (32→32)
    int8_t *params_l4b0_c1 = make_conv_params(S2, S2, CH2, CH3, 3, 2);  // 2x2→1x1
    int8_t *params_l4b0_c2 = make_conv_params(S3, S3, CH3, CH3, 3, 1);
    int8_t *params_l4_ds   = make_conv_params(S2, S2, CH2, CH3, 1, 2);
    int8_t *params_l4b1_c1 = make_conv_params(S3, S3, CH3, CH3, 3, 1);
    int8_t *params_l4b1_c2 = make_conv_params(S3, S3, CH3, CH3, 3, 1);

    // FC: 32→4
    int8_t *params_fc = make_fc_params(CH3, NUM_CLASSES);

    // ═══════════════════════════════════════════════════════════════════════
    // Forward pass: sequential kernel launches
    // ═══════════════════════════════════════════════════════════════════════

    // --- conv1: 8x8x1 → 8x8x4 ---
    int feat_sz_s0 = S0 * S0 * CH0;   // 256
    conv_bn_relu<<<mesh>>>(input, params_conv1, feat1);

    // ─── layer1: BasicBlock 0 ───
    // conv3x3(4→4) + BN + ReLU
    conv_bn_relu<<<mesh>>>(feat1, params_l1b0_c1, tmp1);
    // conv3x3(4→4) + BN (no activation)
    conv_bn<<<mesh>>>(tmp1, params_l1b0_c2, tmp2);
    // residual add + ReLU: tmp2 + feat1 → feat2
    residual_add_relu<<<mesh>>>(tmp2, feat1, feat2, feat_sz_s0);

    // ─── layer1: BasicBlock 1 ───
    conv_bn_relu<<<mesh>>>(feat2, params_l1b1_c1, tmp1);
    conv_bn<<<mesh>>>(tmp1, params_l1b1_c2, tmp2);
    residual_add_relu<<<mesh>>>(tmp2, feat2, feat3, feat_sz_s0);
    // feat3 now holds layer1 output: 8x8x4

    // ─── layer2: BasicBlock 0 (downsample 4→8, stride=2) ───
    int feat_sz_s1 = S1 * S1 * CH1;   // 128
    // Main path: 8x8x4 → 4x4x8
    conv_bn_relu<<<mesh>>>(feat3, params_l2b0_c1, tmp1);
    conv_bn<<<mesh>>>(tmp1, params_l2b0_c2, tmp2);
    // Skip path: 1x1 downsample 8x8x4 → 4x4x8
    conv_bn<<<mesh>>>(feat3, params_l2_ds, skip_ds);
    // Residual add
    residual_add_relu<<<mesh>>>(tmp2, skip_ds, feat1, feat_sz_s1);
    // feat1 now holds layer2 block0 output: 4x4x8

    // ─── layer2: BasicBlock 1 ───
    conv_bn_relu<<<mesh>>>(feat1, params_l2b1_c1, tmp1);
    conv_bn<<<mesh>>>(tmp1, params_l2b1_c2, tmp2);
    residual_add_relu<<<mesh>>>(tmp2, feat1, feat2, feat_sz_s1);
    // feat2 now holds layer2 output: 4x4x8

    // ─── layer3: BasicBlock 0 (downsample 8→16, stride=2) ───
    int feat_sz_s2 = S2 * S2 * CH2;   // 64
    conv_bn_relu<<<mesh>>>(feat2, params_l3b0_c1, tmp1);
    conv_bn<<<mesh>>>(tmp1, params_l3b0_c2, tmp2);
    conv_bn<<<mesh>>>(feat2, params_l3_ds, skip_ds);
    residual_add_relu<<<mesh>>>(tmp2, skip_ds, feat3, feat_sz_s2);
    // feat3: 2x2x16

    // ─── layer3: BasicBlock 1 ───
    conv_bn_relu<<<mesh>>>(feat3, params_l3b1_c1, tmp1);
    conv_bn<<<mesh>>>(tmp1, params_l3b1_c2, tmp2);
    residual_add_relu<<<mesh>>>(tmp2, feat3, feat1, feat_sz_s2);
    // feat1: 2x2x16

    // ─── layer4: BasicBlock 0 (downsample 16→32, stride=2) ───
    int feat_sz_s3 = S3 * S3 * CH3;   // 32
    conv_bn_relu<<<mesh>>>(feat1, params_l4b0_c1, tmp1);
    conv_bn<<<mesh>>>(tmp1, params_l4b0_c2, tmp2);
    conv_bn<<<mesh>>>(feat1, params_l4_ds, skip_ds);
    residual_add_relu<<<mesh>>>(tmp2, skip_ds, feat2, feat_sz_s3);
    // feat2: 1x1x32

    // ─── layer4: BasicBlock 1 ───
    conv_bn_relu<<<mesh>>>(feat2, params_l4b1_c1, tmp1);
    conv_bn<<<mesh>>>(tmp1, params_l4b1_c2, tmp2);
    residual_add_relu<<<mesh>>>(tmp2, feat2, feat3, feat_sz_s3);
    // feat3: 1x1x32

    // ─── Classifier: avgpool + FC ───
    avgpool_fc<<<mesh>>>(feat3, params_fc, logits, S3, S3, CH3, NUM_CLASSES);

    // --- Wait for all tiles to finish ---
    aieDeviceSynchronize();

    // ═══════════════════════════════════════════════════════════════════════
    // CPU reference computation (mirror exact same ops)
    // ═══════════════════════════════════════════════════════════════════════
    int8_t *ref_f1   = (int8_t *)malloc(max_feat_sz);
    int8_t *ref_f2   = (int8_t *)malloc(max_feat_sz);
    int8_t *ref_f3   = (int8_t *)malloc(max_feat_sz);
    int8_t *ref_t1   = (int8_t *)malloc(max_feat_sz);
    int8_t *ref_t2   = (int8_t *)malloc(max_feat_sz);
    int8_t *ref_skip = (int8_t *)malloc(max_feat_sz);
    int8_t *ref_logits = (int8_t *)malloc(NUM_CLASSES);

    // conv1
    cpu_conv_bn_relu(input, params_conv1, ref_f1);

    // layer1 block0
    cpu_conv_bn_relu(ref_f1, params_l1b0_c1, ref_t1);
    cpu_conv_bn(ref_t1, params_l1b0_c2, ref_t2);
    cpu_add_relu(ref_t2, ref_f1, ref_f2, feat_sz_s0);

    // layer1 block1
    cpu_conv_bn_relu(ref_f2, params_l1b1_c1, ref_t1);
    cpu_conv_bn(ref_t1, params_l1b1_c2, ref_t2);
    cpu_add_relu(ref_t2, ref_f2, ref_f3, feat_sz_s0);

    // layer2 block0 (downsample)
    cpu_conv_bn_relu(ref_f3, params_l2b0_c1, ref_t1);
    cpu_conv_bn(ref_t1, params_l2b0_c2, ref_t2);
    cpu_conv_bn(ref_f3, params_l2_ds, ref_skip);
    cpu_add_relu(ref_t2, ref_skip, ref_f1, feat_sz_s1);

    // layer2 block1
    cpu_conv_bn_relu(ref_f1, params_l2b1_c1, ref_t1);
    cpu_conv_bn(ref_t1, params_l2b1_c2, ref_t2);
    cpu_add_relu(ref_t2, ref_f1, ref_f2, feat_sz_s1);

    // layer3 block0 (downsample)
    cpu_conv_bn_relu(ref_f2, params_l3b0_c1, ref_t1);
    cpu_conv_bn(ref_t1, params_l3b0_c2, ref_t2);
    cpu_conv_bn(ref_f2, params_l3_ds, ref_skip);
    cpu_add_relu(ref_t2, ref_skip, ref_f3, feat_sz_s2);

    // layer3 block1
    cpu_conv_bn_relu(ref_f3, params_l3b1_c1, ref_t1);
    cpu_conv_bn(ref_t1, params_l3b1_c2, ref_t2);
    cpu_add_relu(ref_t2, ref_f3, ref_f1, feat_sz_s2);

    // layer4 block0 (downsample)
    cpu_conv_bn_relu(ref_f1, params_l4b0_c1, ref_t1);
    cpu_conv_bn(ref_t1, params_l4b0_c2, ref_t2);
    cpu_conv_bn(ref_f1, params_l4_ds, ref_skip);
    cpu_add_relu(ref_t2, ref_skip, ref_f2, feat_sz_s3);

    // layer4 block1
    cpu_conv_bn_relu(ref_f2, params_l4b1_c1, ref_t1);
    cpu_conv_bn(ref_t1, params_l4b1_c2, ref_t2);
    cpu_add_relu(ref_t2, ref_f2, ref_f3, feat_sz_s3);

    // classifier
    cpu_avgpool_fc(ref_f3, params_fc, ref_logits, S3, S3, CH3, NUM_CLASSES);

    // ═══════════════════════════════════════════════════════════════════════
    // Verification
    // ═══════════════════════════════════════════════════════════════════════
    int mismatches = 0;
    for (int i = 0; i < NUM_CLASSES; i++) {
        if (logits[i] != ref_logits[i]) {
            printf("MISMATCH logit[%d]: got %d, expected %d\n",
                   i, logits[i], ref_logits[i]);
            mismatches++;
        }
    }

    if (mismatches == 0)
        printf("PASS: all %d logits match.\n", NUM_CLASSES);
    else
        printf("FAIL: %d logit mismatches out of %d.\n", mismatches, NUM_CLASSES);

    // Print classification result
    int best_class = 0;
    int8_t best_val = ref_logits[0];
    printf("Logits: [");
    for (int i = 0; i < NUM_CLASSES; i++) {
        printf("%d", ref_logits[i]);
        if (i < NUM_CLASSES - 1) printf(", ");
        if (ref_logits[i] > best_val) {
            best_val = ref_logits[i];
            best_class = i;
        }
    }
    printf("]\n");
    printf("Predicted class: %d (logit=%d)\n", best_class, best_val);

    // ═══════════════════════════════════════════════════════════════════════
    // Cleanup
    // ═══════════════════════════════════════════════════════════════════════
    free(input);
    free(feat1); free(feat2); free(feat3);
    free(tmp1);  free(tmp2);  free(skip_ds);
    free(logits);

    free(params_conv1);
    free(params_l1b0_c1); free(params_l1b0_c2);
    free(params_l1b1_c1); free(params_l1b1_c2);
    free(params_l2b0_c1); free(params_l2b0_c2); free(params_l2_ds);
    free(params_l2b1_c1); free(params_l2b1_c2);
    free(params_l3b0_c1); free(params_l3b0_c2); free(params_l3_ds);
    free(params_l3b1_c1); free(params_l3b1_c2);
    free(params_l4b0_c1); free(params_l4b0_c2); free(params_l4_ds);
    free(params_l4b1_c1); free(params_l4b1_c2);
    free(params_fc);

    free(ref_f1); free(ref_f2); free(ref_f3);
    free(ref_t1); free(ref_t2); free(ref_skip);
    free(ref_logits);

    return 0;
}
