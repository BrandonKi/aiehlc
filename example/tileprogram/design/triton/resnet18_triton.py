###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# AIE Programming Model — Triton-style ResNet18 Inference
#
# Complete ResNet18 with 8 BasicBlocks, residual skip connections,
# 1x1 downsample projections, global average pooling, and FC classifier.
# Scaled-down dimensions: 8x8 input, channels 4->8->16->32, 4 classes.
# Data type: int8 with Q7 fixed-point for BN scale/bias.
#
# This is the Triton Python equivalent of resnet18.cc.
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │                        Compilation Flow                                │
# │                                                                        │
# │  resnet18_triton.py                                                    │
# │    │  Python AST parse (@aie_triton.jit functions)                     │
# │    ▼                                                                   │
# │  routing dialect IR (one buildRoutingIR call per kernel launch)        │
# │    │  TensorParam for feat_in, params, feat_out per kernel             │
# │    ▼                                                                   │
# │  MLIR pipeline: routing → dmap → dmaphop → blueprint → dfschedule     │
# │    ▼                                                                   │
# │  host.cc + kernel.cc + routing.cc + aieml.bcf + aieml.prx             │
# │    │  cross-compile: xchesscc (kernel), aarch64-g++ (host)            │
# │    ▼                                                                   │
# │  Deploy on AIE HW                                                     │
# └─────────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │        ResNet18 Architecture (scaled-down for AIE)                     │
# │                                                                        │
# │  Input: 8x8x1                                                         │
# │    ▼                                                                   │
# │  conv1 (1→4, 3x3, stride=1) + BN + ReLU  →  8x8x4                   │
# │    ▼                                                                   │
# │  layer1: BasicBlock x2 (4→4, 3x3)        →  8x8x4                   │
# │    ▼                                                                   │
# │  layer2: BasicBlock x2 (4→8, stride=2)    →  4x4x8   + 1x1 ds       │
# │    ▼                                                                   │
# │  layer3: BasicBlock x2 (8→16, stride=2)   →  2x2x16  + 1x1 ds       │
# │    ▼                                                                   │
# │  layer4: BasicBlock x2 (16→32, stride=2)  →  1x1x32  + 1x1 ds       │
# │    ▼                                                                   │
# │  Global Average Pool → FC (32→4)          →  4 logits                 │
# └─────────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │            Triton ←→ AIE / resnet18.cc Concept Map                    │
# │                                                                        │
# │  Triton Python                  │ AIE C / resnet18.cc                  │
# │  ──────────────────────────────┼────────────────────────────────────── │
# │  @aie_triton.jit               │ __global__                            │
# │  tl.program_id(0), (1)         │ get_coreid() row/col                  │
# │  tl.load(feat_ptr + ...)       │ acquire_input_window(window_in_feat)  │
# │  tl.load(params_ptr + ...)     │ acquire_input_window(window_in_params)│
# │  tl.store(out_ptr + ...)       │ acquire/write/release output_window   │
# │  Manual conv loop              │ 6-nested loop over oc,oh,ow,ic,kh,kw │
# │  Q7 BN: (sum * scale) >> 7    │ (sum * bn_scale[oc]) >> 7 + bias     │
# │  tl.maximum(x, 0) (ReLU)      │ result < 0 ? 0 : result (clamped)    │
# │  element-wise add              │ main_in[i] + skip_in[i]              │
# │  kernel[grid](...)             │ kernel<<<mesh>>>(...)                 │
# └─────────────────────────────────────────────────────────────────────────┘
#
# Conv param buffer layout (matches resnet18.cc exactly):
#   [config: 6 bytes] [weights: Cin*Cout*K*K] [bn_scale: Cout] [bn_bias: Cout]
#   config = { H, W, Cin, Cout, K, stride }
#
###############################################################################

import aie_triton
import aie_triton.language as tl
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# Dimensions (scaled-down ResNet18, matches resnet18.cc defines)
# ═══════════════════════════════════════════════════════════════════════════
INPUT_H, INPUT_W, INPUT_C = 8, 8, 1
NUM_CLASSES = 4

# Channel progression: 4 → 8 → 16 → 32
CH0, CH1, CH2, CH3 = 4, 8, 16, 32

# Spatial sizes after each downsampling (stride=2)
S0, S1, S2, S3 = 8, 4, 2, 1

# Q7 fixed-point BN defaults
BN_SCALE_DEFAULT = 64   # 0.5 in Q7 (64/128)
BN_BIAS_DEFAULT  = 0

CONFIG_SZ = 6  # { H, W, Cin, Cout, K, stride }


###############################################################################
#  KERNEL 1: conv_bn_relu — Conv + BatchNorm + ReLU (fused)
#
#  resnet18.cc equivalent:
#    __global__ void conv_bn_relu(input_window_int8 *window_in_feat,
#                                  input_window_int8 *window_in_params,
#                                  output_window_int8 *window_out_feat)
#
#  Parameterized by config packed in params buffer header.
#  Handles both 3x3 (K=3) and 1x1 (K=1) convolutions.
###############################################################################

@aie_triton.jit
def conv_bn_relu_kernel(
    feat_ptr,       # → resnet18.cc: window_in_feat (input_window_int8 *)
    params_ptr,     # → resnet18.cc: window_in_params (input_window_int8 *)
    out_ptr,        # → resnet18.cc: window_out_feat (output_window_int8 *)
    # Config passed as scalars (frontend extracts from params buffer)
    H, W, Cin, Cout, K, stride,
    BLOCK_H: tl.constexpr = 8,
    BLOCK_W: tl.constexpr = 8,
):
    """Conv2D + BatchNorm + ReLU fused kernel for one AIE tile.

    Mapping to resnet18.cc:
        @aie_triton.jit          → __global__ void conv_bn_relu(...)
        tl.load(feat_ptr + ...)  → acquire_input_window(window_in_feat)
        tl.load(params_ptr + ...)→ acquire_input_window(window_in_params)
        manual conv loop         → 6-nested for loop (oc, oh, ow, ic, kh, kw)
        (sum * scale) >> 7       → BN: (sum * bn_scale[oc]) >> 7 + bias
        tl.maximum(val, 0)       → ReLU: result < 0 ? 0 : result
        tl.store(out_ptr + ...)  → acquire/write/release output_window
    """
    # ── Tile identity ──
    # resnet18.cc: (not used in kernel — partition handled by DMA)
    tile_row = tl.program_id(axis=0)
    tile_col = tl.program_id(axis=1)

    pad = K // 2     # 1 for 3x3, 0 for 1x1
    outH = H // stride
    outW = W // stride

    # ── Load input feature map ──
    # resnet18.cc: int8_t *feat_in = (int8_t *)acquire_input_window(window_in_feat);
    feat_in = tl.load(feat_ptr + tl.arange(0, Cin * H * W))

    # ── Load params: [config(6)] [weights(Cin*Cout*K*K)] [bn_scale(Cout)] [bn_bias(Cout)] ──
    # resnet18.cc: int8_t *params = (int8_t *)acquire_input_window(window_in_params);
    wt_count = Cin * Cout * K * K
    total_param_sz = CONFIG_SZ + wt_count + Cout * 2
    params = tl.load(params_ptr + tl.arange(0, total_param_sz))

    # Extract sub-arrays from params buffer
    weights  = params[CONFIG_SZ : CONFIG_SZ + wt_count]
    bn_scale = params[CONFIG_SZ + wt_count : CONFIG_SZ + wt_count + Cout]
    bn_bias  = params[CONFIG_SZ + wt_count + Cout : CONFIG_SZ + wt_count + Cout * 2]

    # ── Convolution + BN + ReLU ──
    # resnet18.cc: 6-nested loop over (oc, oh, ow, ic, kh, kw)
    #   int16_t sum = 0;
    #   sum += (int16_t)feat_in[in_idx] * (int16_t)weights[wt_idx];
    #   int16_t bn_out = (sum * (int16_t)bn_scale[oc]) >> 7;
    #   bn_out += bn_bias[oc];
    #   result = (bn_out > 127) ? 127 : (bn_out < 0) ? 0 : (int8_t)bn_out;
    feat_out = tl.zeros((Cout * outH * outW,), dtype=tl.int8)

    for oc in range(Cout):
        for oh in range(outH):
            for ow in range(outW):
                acc = tl.zeros((1,), dtype=tl.int16)
                for ic in range(Cin):
                    for kh in range(K):
                        for kw in range(K):
                            ih = oh * stride + kh - pad
                            iw = ow * stride + kw - pad
                            if ih >= 0 and ih < H and iw >= 0 and iw < W:
                                in_idx = ic * H * W + ih * W + iw
                                wt_idx = oc * Cin * K * K + ic * K * K + kh * K + kw
                                acc += feat_in[in_idx].to(tl.int16) * weights[wt_idx].to(tl.int16)

                # BatchNorm: out = (conv * scale) >> 7 + bias
                bn_out = (acc * bn_scale[oc].to(tl.int16)) >> 7
                bn_out += bn_bias[oc].to(tl.int16)

                # ReLU + saturate
                result = tl.minimum(tl.maximum(bn_out, 0), 127).to(tl.int8)
                feat_out[oc * outH * outW + oh * outW + ow] = result

    # ── Store output feature map ──
    # resnet18.cc: release_output_window(window_out_feat);
    tl.store(out_ptr + tl.arange(0, Cout * outH * outW), feat_out)


###############################################################################
#  KERNEL 2: conv_bn — Conv + BatchNorm (no activation)
#
#  resnet18.cc equivalent:
#    __global__ void conv_bn(input_window_int8 *window_in_feat,
#                             input_window_int8 *window_in_params,
#                             output_window_int8 *window_out_feat)
#
#  Same as conv_bn_relu but no ReLU. Used for the second conv in each
#  BasicBlock (before residual add) and for 1x1 downsample projections.
###############################################################################

@aie_triton.jit
def conv_bn_kernel(
    feat_ptr,       # → resnet18.cc: window_in_feat
    params_ptr,     # → resnet18.cc: window_in_params
    out_ptr,        # → resnet18.cc: window_out_feat
    H, W, Cin, Cout, K, stride,
    BLOCK_H: tl.constexpr = 8,
    BLOCK_W: tl.constexpr = 8,
):
    """Conv2D + BatchNorm (no activation) kernel for one AIE tile.

    Mapping to resnet18.cc:
        Same as conv_bn_relu_kernel but no ReLU at the end.
        Saturates to [-128, 127] instead of [0, 127].
    """
    tile_row = tl.program_id(axis=0)
    tile_col = tl.program_id(axis=1)

    pad = K // 2
    outH = H // stride
    outW = W // stride

    feat_in = tl.load(feat_ptr + tl.arange(0, Cin * H * W))

    wt_count = Cin * Cout * K * K
    total_param_sz = CONFIG_SZ + wt_count + Cout * 2
    params = tl.load(params_ptr + tl.arange(0, total_param_sz))

    weights  = params[CONFIG_SZ : CONFIG_SZ + wt_count]
    bn_scale = params[CONFIG_SZ + wt_count : CONFIG_SZ + wt_count + Cout]
    bn_bias  = params[CONFIG_SZ + wt_count + Cout : CONFIG_SZ + wt_count + Cout * 2]

    feat_out = tl.zeros((Cout * outH * outW,), dtype=tl.int8)

    for oc in range(Cout):
        for oh in range(outH):
            for ow in range(outW):
                acc = tl.zeros((1,), dtype=tl.int16)
                for ic in range(Cin):
                    for kh in range(K):
                        for kw in range(K):
                            ih = oh * stride + kh - pad
                            iw = ow * stride + kw - pad
                            if ih >= 0 and ih < H and iw >= 0 and iw < W:
                                in_idx = ic * H * W + ih * W + iw
                                wt_idx = oc * Cin * K * K + ic * K * K + kh * K + kw
                                acc += feat_in[in_idx].to(tl.int16) * weights[wt_idx].to(tl.int16)

                # BatchNorm only (no ReLU)
                bn_out = (acc * bn_scale[oc].to(tl.int16)) >> 7
                bn_out += bn_bias[oc].to(tl.int16)

                # Saturate to int8 range (NO ReLU — negative values preserved)
                result = tl.minimum(tl.maximum(bn_out, -128), 127).to(tl.int8)
                feat_out[oc * outH * outW + oh * outW + ow] = result

    tl.store(out_ptr + tl.arange(0, Cout * outH * outW), feat_out)


###############################################################################
#  KERNEL 3: residual_add_relu — Element-wise Add + ReLU
#
#  resnet18.cc equivalent:
#    __global__ void residual_add_relu(input_window_int8 *window_in_main,
#                                       input_window_int8 *window_in_skip,
#                                       output_window_int8 *window_out,
#                                       int len)
#
#  Adds main path and skip path tensors, then applies ReLU.
###############################################################################

@aie_triton.jit
def residual_add_relu_kernel(
    main_ptr,       # → resnet18.cc: window_in_main (main path output)
    skip_ptr,       # → resnet18.cc: window_in_skip (skip/identity path)
    out_ptr,        # → resnet18.cc: window_out
    length,         # number of elements
):
    """Element-wise residual add + ReLU for one AIE tile.

    Mapping to resnet18.cc:
        tl.load(main_ptr + ...)  → acquire_input_window(window_in_main)
        tl.load(skip_ptr + ...)  → acquire_input_window(window_in_skip)
        element-wise add + relu  → for(i) sum = main[i]+skip[i]; clamp
        tl.store(out_ptr + ...)  → acquire/write/release output_window
    """
    tile_row = tl.program_id(axis=0)
    tile_col = tl.program_id(axis=1)

    # ── Load both paths ──
    # resnet18.cc: int8_t *main_in = acquire_input_window(window_in_main);
    #              int8_t *skip_in = acquire_input_window(window_in_skip);
    offsets = tl.arange(0, length)
    main_data = tl.load(main_ptr + offsets).to(tl.int16)
    skip_data = tl.load(skip_ptr + offsets).to(tl.int16)

    # ── Add + ReLU + saturate ──
    # resnet18.cc: int16_t sum = main[i] + skip[i];
    #              if (sum > 127) sum = 127;
    #              if (sum < 0)   sum = 0;
    total = main_data + skip_data
    result = tl.minimum(tl.maximum(total, 0), 127).to(tl.int8)

    # ── Store ──
    tl.store(out_ptr + offsets, result)


###############################################################################
#  KERNEL 4: avgpool_fc — Global Average Pooling + Fully-Connected
#
#  resnet18.cc equivalent:
#    __global__ void avgpool_fc(input_window_int8 *window_in_feat,
#                                input_window_int8 *window_in_params,
#                                output_window_int8 *window_out_logits,
#                                int spatial_h, int spatial_w, int channels,
#                                int num_classes)
#
#  FC params layout: [weights: channels*num_classes] [bias: num_classes]
###############################################################################

@aie_triton.jit
def avgpool_fc_kernel(
    feat_ptr,       # → resnet18.cc: window_in_feat
    fc_params_ptr,  # → resnet18.cc: window_in_params ([weights][bias])
    logits_ptr,     # → resnet18.cc: window_out_logits
    spatial_h, spatial_w, channels, num_classes,
):
    """Global average pooling + FC classifier for one AIE tile.

    Mapping to resnet18.cc:
        GAP: pooled[c] = sum(feat[c,:,:]) / spatial_sz
        FC:  logits[j] = sum_i(pooled[i] * w[i*nclass+j]) + bias[j]
    """
    tile_row = tl.program_id(axis=0)
    tile_col = tl.program_id(axis=1)

    spatial_sz = spatial_h * spatial_w

    # ── Load feature map and FC params ──
    feat = tl.load(feat_ptr + tl.arange(0, channels * spatial_sz))
    fc_param_sz = channels * num_classes + num_classes
    fc_params = tl.load(fc_params_ptr + tl.arange(0, fc_param_sz))

    fc_weights = fc_params[0 : channels * num_classes]
    fc_bias    = fc_params[channels * num_classes : fc_param_sz]

    # ── Global Average Pooling ──
    # resnet18.cc: for(c) { sum=0; for(s) sum += feat[c*sz+s]; pooled[c] = sum/sz; }
    pooled = tl.zeros((channels,), dtype=tl.int8)
    for c in range(channels):
        channel_sum = tl.zeros((1,), dtype=tl.int16)
        for s in range(spatial_sz):
            channel_sum += feat[c * spatial_sz + s].to(tl.int16)
        pooled[c] = (channel_sum // spatial_sz).to(tl.int8)

    # ── Fully Connected ──
    # resnet18.cc: for(j) { acc=0; for(i) acc += pooled[i]*w[i*nclass+j]; acc += bias[j]; }
    logits = tl.zeros((num_classes,), dtype=tl.int8)
    for j in range(num_classes):
        acc = tl.zeros((1,), dtype=tl.int16)
        for i in range(channels):
            acc += pooled[i].to(tl.int16) * fc_weights[i * num_classes + j].to(tl.int16)
        acc += fc_bias[j].to(tl.int16)
        logits[j] = tl.minimum(tl.maximum(acc, -128), 127).to(tl.int8)

    # ── Store logits ──
    tl.store(logits_ptr + tl.arange(0, num_classes), logits)


###############################################################################
#  HOST HELPERS
#  Matches resnet18.cc: pack_config, make_conv_params, make_fc_params
###############################################################################

def pack_config(H, W, Cin, Cout, K, stride):
    """Pack conv config into 6-byte header (matches resnet18.cc pack_config)."""
    return np.array([H, W, Cin, Cout, K, stride], dtype=np.int8)


def make_conv_params(H, W, Cin, Cout, K, stride):
    """Allocate + initialize conv param buffer (matches resnet18.cc make_conv_params).

    resnet18.cc:
        pack_config(buf, H, W, Cin, Cout, K, stride);
        init_conv_weights(&buf[CONFIG_SZ], wt_count);   // alternating 1/-1
        init_bn_params(&buf[CONFIG_SZ+wt_count], ..., Cout);  // scale=64, bias=0
    """
    wt_count = Cin * Cout * K * K
    param_sz = CONFIG_SZ + wt_count + Cout * 2

    buf = np.zeros(param_sz, dtype=np.int8)

    # Config header
    buf[:CONFIG_SZ] = pack_config(H, W, Cin, Cout, K, stride)

    # Weights: alternating 1, -1 pattern
    # resnet18.cc: for(i) dst[i] = (i%2==0) ? 1 : -1;
    weights = np.array([1 if i % 2 == 0 else -1 for i in range(wt_count)], dtype=np.int8)
    buf[CONFIG_SZ : CONFIG_SZ + wt_count] = weights

    # BN scale = BN_SCALE_DEFAULT (64), bias = BN_BIAS_DEFAULT (0)
    buf[CONFIG_SZ + wt_count : CONFIG_SZ + wt_count + Cout] = BN_SCALE_DEFAULT
    buf[CONFIG_SZ + wt_count + Cout : CONFIG_SZ + wt_count + Cout * 2] = BN_BIAS_DEFAULT

    return buf


def make_fc_params(Cin, Cout):
    """Allocate + initialize FC param buffer (matches resnet18.cc make_fc_params).

    resnet18.cc:
        for(i<Cin*Cout) buf[i] = 1;      // weights = 1
        for(i<Cout) buf[Cin*Cout+i] = 0;  // bias = 0
    """
    param_sz = Cin * Cout + Cout
    buf = np.zeros(param_sz, dtype=np.int8)
    buf[:Cin * Cout] = 1   # uniform weights
    # bias stays 0
    return buf


###############################################################################
#  CPU REFERENCE FUNCTIONS (for verification)
#  Matches resnet18.cc: cpu_conv_bn_relu, cpu_conv_bn, cpu_add_relu,
#                        cpu_avgpool_fc
###############################################################################

def cpu_conv_bn_relu(feat_in, params):
    """CPU reference: Conv + BN + ReLU (matches resnet18.cc cpu_conv_bn_relu)."""
    H, W  = int(np.uint8(params[0])), int(np.uint8(params[1]))
    Cin    = int(np.uint8(params[2]))
    Cout   = int(np.uint8(params[3]))
    K      = int(np.uint8(params[4]))
    stride = int(np.uint8(params[5]))
    pad    = K // 2

    wt_count = Cin * Cout * K * K
    weights  = params[CONFIG_SZ : CONFIG_SZ + wt_count]
    bn_scale = params[CONFIG_SZ + wt_count : CONFIG_SZ + wt_count + Cout]
    bn_bias  = params[CONFIG_SZ + wt_count + Cout : CONFIG_SZ + wt_count + Cout * 2]

    outH, outW = H // stride, W // stride
    out = np.zeros(Cout * outH * outW, dtype=np.int8)

    for oc in range(Cout):
        for oh in range(outH):
            for ow in range(outW):
                s = np.int16(0)
                for ic in range(Cin):
                    for kh in range(K):
                        for kw in range(K):
                            ih = oh * stride + kh - pad
                            iw = ow * stride + kw - pad
                            if 0 <= ih < H and 0 <= iw < W:
                                in_idx = ic * H * W + ih * W + iw
                                wt_idx = oc * Cin * K * K + ic * K * K + kh * K + kw
                                s += np.int16(feat_in[in_idx]) * np.int16(weights[wt_idx])
                bn_out = (s * np.int16(bn_scale[oc])) >> 7
                bn_out += np.int16(bn_bias[oc])
                r = max(0, min(127, int(bn_out)))
                out[oc * outH * outW + oh * outW + ow] = np.int8(r)
    return out


def cpu_conv_bn(feat_in, params):
    """CPU reference: Conv + BN, no ReLU (matches resnet18.cc cpu_conv_bn)."""
    H, W  = int(np.uint8(params[0])), int(np.uint8(params[1]))
    Cin    = int(np.uint8(params[2]))
    Cout   = int(np.uint8(params[3]))
    K      = int(np.uint8(params[4]))
    stride = int(np.uint8(params[5]))
    pad    = K // 2

    wt_count = Cin * Cout * K * K
    weights  = params[CONFIG_SZ : CONFIG_SZ + wt_count]
    bn_scale = params[CONFIG_SZ + wt_count : CONFIG_SZ + wt_count + Cout]
    bn_bias  = params[CONFIG_SZ + wt_count + Cout : CONFIG_SZ + wt_count + Cout * 2]

    outH, outW = H // stride, W // stride
    out = np.zeros(Cout * outH * outW, dtype=np.int8)

    for oc in range(Cout):
        for oh in range(outH):
            for ow in range(outW):
                s = np.int16(0)
                for ic in range(Cin):
                    for kh in range(K):
                        for kw in range(K):
                            ih = oh * stride + kh - pad
                            iw = ow * stride + kw - pad
                            if 0 <= ih < H and 0 <= iw < W:
                                in_idx = ic * H * W + ih * W + iw
                                wt_idx = oc * Cin * K * K + ic * K * K + kh * K + kw
                                s += np.int16(feat_in[in_idx]) * np.int16(weights[wt_idx])
                bn_out = (s * np.int16(bn_scale[oc])) >> 7
                bn_out += np.int16(bn_bias[oc])
                r = max(-128, min(127, int(bn_out)))
                out[oc * outH * outW + oh * outW + ow] = np.int8(r)
    return out


def cpu_add_relu(main_path, skip_path):
    """CPU reference: element-wise add + ReLU (matches resnet18.cc cpu_add_relu)."""
    out = np.zeros(len(main_path), dtype=np.int8)
    for i in range(len(main_path)):
        s = np.int16(main_path[i]) + np.int16(skip_path[i])
        r = max(0, min(127, int(s)))
        out[i] = np.int8(r)
    return out


def cpu_avgpool_fc(feat, fc_params, spatial_h, spatial_w, channels, num_classes):
    """CPU reference: GAP + FC (matches resnet18.cc cpu_avgpool_fc)."""
    spatial_sz = spatial_h * spatial_w
    fc_weights = fc_params[:channels * num_classes]
    fc_bias    = fc_params[channels * num_classes:]

    pooled = np.zeros(channels, dtype=np.int8)
    for c in range(channels):
        s = np.int16(0)
        for idx in range(spatial_sz):
            s += np.int16(feat[c * spatial_sz + idx])
        pooled[c] = np.int8(int(s) // spatial_sz)

    logits = np.zeros(num_classes, dtype=np.int8)
    for j in range(num_classes):
        acc = np.int16(0)
        for i in range(channels):
            acc += np.int16(pooled[i]) * np.int16(fc_weights[i * num_classes + j])
        acc += np.int16(fc_bias[j])
        logits[j] = np.int8(max(-128, min(127, int(acc))))
    return logits


###############################################################################
#  HOST: main — matches resnet18.cc main() structure
#
#  resnet18.cc:                              Python equivalent:
#    aieSetDevice(0);                         aie_triton.set_device(0)
#    aieDim mesh(2, 2);                       mesh = aie_triton.mesh(rows=2, cols=2)
#    int8_t *input = malloc(input_sz);        input_data = np.array(...)
#    make_conv_params(S0,S0,INPUT_C,CH0,3,1)  make_conv_params(S0,S0,INPUT_C,CH0,3,1)
#    conv_bn_relu<<<mesh>>>(...)              conv_bn_relu_kernel[grid](...)
#    residual_add_relu<<<mesh>>>(...)         residual_add_relu_kernel[grid](...)
#    avgpool_fc<<<mesh>>>(...)                avgpool_fc_kernel[grid](...)
#    aieDeviceSynchronize();                  aie_triton.synchronize()
#    verify loop                              numpy comparison
###############################################################################

def main():
    print("=== Triton-style ResNet18 (scaled-down) on AIE Tile Mesh ===")

    # ── Device + mesh ──
    # resnet18.cc: aieSetDevice(0); aieDim mesh(2, 2);
    aie_triton.set_device(0)
    mesh = aie_triton.mesh(rows=2, cols=2)
    grid = (mesh.rows, mesh.cols)

    # ── Input: 8x8x1, test pattern 1..7 repeating ──
    # resnet18.cc: for(i) input[i] = (i%7) + 1;
    input_sz = INPUT_H * INPUT_W * INPUT_C
    input_data = np.array([(i % 7) + 1 for i in range(input_sz)], dtype=np.int8)

    # ── Feature map buffers (reusable scratch) ──
    # resnet18.cc: max_feat_sz = S0*S0*CH0 = 256
    max_feat_sz = S0 * S0 * CH0   # 256
    feat1   = np.zeros(max_feat_sz, dtype=np.int8)
    feat2   = np.zeros(max_feat_sz, dtype=np.int8)
    feat3   = np.zeros(max_feat_sz, dtype=np.int8)
    tmp1    = np.zeros(max_feat_sz, dtype=np.int8)
    tmp2    = np.zeros(max_feat_sz, dtype=np.int8)
    skip_ds = np.zeros(max_feat_sz, dtype=np.int8)
    logits  = np.zeros(NUM_CLASSES, dtype=np.int8)

    # ── Weight parameter buffers ──
    # resnet18.cc: make_conv_params(...) for each layer
    params_conv1   = make_conv_params(S0, S0, INPUT_C, CH0, 3, 1)  # conv1: 1→4

    # layer1: BasicBlock 0 & 1 (4→4, 3x3, stride=1, 8x8)
    params_l1b0_c1 = make_conv_params(S0, S0, CH0, CH0, 3, 1)
    params_l1b0_c2 = make_conv_params(S0, S0, CH0, CH0, 3, 1)
    params_l1b1_c1 = make_conv_params(S0, S0, CH0, CH0, 3, 1)
    params_l1b1_c2 = make_conv_params(S0, S0, CH0, CH0, 3, 1)

    # layer2: BasicBlock 0 (4→8, stride=2) + downsample, BasicBlock 1 (8→8)
    params_l2b0_c1 = make_conv_params(S0, S0, CH0, CH1, 3, 2)   # 8x8→4x4
    params_l2b0_c2 = make_conv_params(S1, S1, CH1, CH1, 3, 1)
    params_l2_ds   = make_conv_params(S0, S0, CH0, CH1, 1, 2)   # 1x1 downsample
    params_l2b1_c1 = make_conv_params(S1, S1, CH1, CH1, 3, 1)
    params_l2b1_c2 = make_conv_params(S1, S1, CH1, CH1, 3, 1)

    # layer3: BasicBlock 0 (8→16, stride=2) + downsample, BasicBlock 1 (16→16)
    params_l3b0_c1 = make_conv_params(S1, S1, CH1, CH2, 3, 2)   # 4x4→2x2
    params_l3b0_c2 = make_conv_params(S2, S2, CH2, CH2, 3, 1)
    params_l3_ds   = make_conv_params(S1, S1, CH1, CH2, 1, 2)
    params_l3b1_c1 = make_conv_params(S2, S2, CH2, CH2, 3, 1)
    params_l3b1_c2 = make_conv_params(S2, S2, CH2, CH2, 3, 1)

    # layer4: BasicBlock 0 (16→32, stride=2) + downsample, BasicBlock 1 (32→32)
    params_l4b0_c1 = make_conv_params(S2, S2, CH2, CH3, 3, 2)   # 2x2→1x1
    params_l4b0_c2 = make_conv_params(S3, S3, CH3, CH3, 3, 1)
    params_l4_ds   = make_conv_params(S2, S2, CH2, CH3, 1, 2)
    params_l4b1_c1 = make_conv_params(S3, S3, CH3, CH3, 3, 1)
    params_l4b1_c2 = make_conv_params(S3, S3, CH3, CH3, 3, 1)

    # FC: 32→4
    params_fc = make_fc_params(CH3, NUM_CLASSES)

    # ═══════════════════════════════════════════════════════════════════════
    # Forward pass: sequential kernel launches
    # Each launch maps to: kernel<<<mesh>>>(...) in resnet18.cc
    # ═══════════════════════════════════════════════════════════════════════

    feat_sz_s0 = S0 * S0 * CH0   # 256

    # ── conv1: 8x8x1 → 8x8x4 ──
    # resnet18.cc: conv_bn_relu<<<mesh>>>(input, params_conv1, feat1);
    conv_bn_relu_kernel[grid](
        input_data, params_conv1, feat1,
        S0, S0, INPUT_C, CH0, 3, 1,
    )

    # ── layer1: BasicBlock 0 ──
    # resnet18.cc: conv_bn_relu<<<mesh>>>(feat1, params_l1b0_c1, tmp1);
    conv_bn_relu_kernel[grid](feat1, params_l1b0_c1, tmp1, S0, S0, CH0, CH0, 3, 1)
    # resnet18.cc: conv_bn<<<mesh>>>(tmp1, params_l1b0_c2, tmp2);
    conv_bn_kernel[grid](tmp1, params_l1b0_c2, tmp2, S0, S0, CH0, CH0, 3, 1)
    # resnet18.cc: residual_add_relu<<<mesh>>>(tmp2, feat1, feat2, feat_sz_s0);
    residual_add_relu_kernel[grid](tmp2, feat1, feat2, feat_sz_s0)

    # ── layer1: BasicBlock 1 ──
    conv_bn_relu_kernel[grid](feat2, params_l1b1_c1, tmp1, S0, S0, CH0, CH0, 3, 1)
    conv_bn_kernel[grid](tmp1, params_l1b1_c2, tmp2, S0, S0, CH0, CH0, 3, 1)
    residual_add_relu_kernel[grid](tmp2, feat2, feat3, feat_sz_s0)
    # feat3: layer1 output 8x8x4

    # ── layer2: BasicBlock 0 (downsample 4→8, stride=2) ──
    feat_sz_s1 = S1 * S1 * CH1   # 128
    # Main path: 8x8x4 → 4x4x8
    conv_bn_relu_kernel[grid](feat3, params_l2b0_c1, tmp1, S0, S0, CH0, CH1, 3, 2)
    conv_bn_kernel[grid](tmp1, params_l2b0_c2, tmp2, S1, S1, CH1, CH1, 3, 1)
    # Skip path: 1x1 downsample 8x8x4 → 4x4x8
    # resnet18.cc: conv_bn<<<mesh>>>(feat3, params_l2_ds, skip_ds);
    conv_bn_kernel[grid](feat3, params_l2_ds, skip_ds, S0, S0, CH0, CH1, 1, 2)
    # Residual add
    residual_add_relu_kernel[grid](tmp2, skip_ds, feat1, feat_sz_s1)
    # feat1: layer2 block0 output 4x4x8

    # ── layer2: BasicBlock 1 ──
    conv_bn_relu_kernel[grid](feat1, params_l2b1_c1, tmp1, S1, S1, CH1, CH1, 3, 1)
    conv_bn_kernel[grid](tmp1, params_l2b1_c2, tmp2, S1, S1, CH1, CH1, 3, 1)
    residual_add_relu_kernel[grid](tmp2, feat1, feat2, feat_sz_s1)
    # feat2: layer2 output 4x4x8

    # ── layer3: BasicBlock 0 (downsample 8→16, stride=2) ──
    feat_sz_s2 = S2 * S2 * CH2   # 64
    conv_bn_relu_kernel[grid](feat2, params_l3b0_c1, tmp1, S1, S1, CH1, CH2, 3, 2)
    conv_bn_kernel[grid](tmp1, params_l3b0_c2, tmp2, S2, S2, CH2, CH2, 3, 1)
    conv_bn_kernel[grid](feat2, params_l3_ds, skip_ds, S1, S1, CH1, CH2, 1, 2)
    residual_add_relu_kernel[grid](tmp2, skip_ds, feat3, feat_sz_s2)
    # feat3: 2x2x16

    # ── layer3: BasicBlock 1 ──
    conv_bn_relu_kernel[grid](feat3, params_l3b1_c1, tmp1, S2, S2, CH2, CH2, 3, 1)
    conv_bn_kernel[grid](tmp1, params_l3b1_c2, tmp2, S2, S2, CH2, CH2, 3, 1)
    residual_add_relu_kernel[grid](tmp2, feat3, feat1, feat_sz_s2)
    # feat1: 2x2x16

    # ── layer4: BasicBlock 0 (downsample 16→32, stride=2) ──
    feat_sz_s3 = S3 * S3 * CH3   # 32
    conv_bn_relu_kernel[grid](feat1, params_l4b0_c1, tmp1, S2, S2, CH2, CH3, 3, 2)
    conv_bn_kernel[grid](tmp1, params_l4b0_c2, tmp2, S3, S3, CH3, CH3, 3, 1)
    conv_bn_kernel[grid](feat1, params_l4_ds, skip_ds, S2, S2, CH2, CH3, 1, 2)
    residual_add_relu_kernel[grid](tmp2, skip_ds, feat2, feat_sz_s3)
    # feat2: 1x1x32

    # ── layer4: BasicBlock 1 ──
    conv_bn_relu_kernel[grid](feat2, params_l4b1_c1, tmp1, S3, S3, CH3, CH3, 3, 1)
    conv_bn_kernel[grid](tmp1, params_l4b1_c2, tmp2, S3, S3, CH3, CH3, 3, 1)
    residual_add_relu_kernel[grid](tmp2, feat2, feat3, feat_sz_s3)
    # feat3: 1x1x32

    # ── Classifier: avgpool + FC ──
    # resnet18.cc: avgpool_fc<<<mesh>>>(feat3, params_fc, logits, S3, S3, CH3, NUM_CLASSES);
    avgpool_fc_kernel[grid](feat3, params_fc, logits, S3, S3, CH3, NUM_CLASSES)

    # ── Wait for all tiles ──
    # resnet18.cc: aieDeviceSynchronize();
    aie_triton.synchronize()

    # ═══════════════════════════════════════════════════════════════════════
    # CPU reference computation (mirrors resnet18.cc CPU reference section)
    # ═══════════════════════════════════════════════════════════════════════

    # conv1
    ref_f1 = cpu_conv_bn_relu(input_data, params_conv1)

    # layer1 block0
    ref_t1 = cpu_conv_bn_relu(ref_f1, params_l1b0_c1)
    ref_t2 = cpu_conv_bn(ref_t1, params_l1b0_c2)
    ref_f2 = cpu_add_relu(ref_t2, ref_f1)

    # layer1 block1
    ref_t1 = cpu_conv_bn_relu(ref_f2, params_l1b1_c1)
    ref_t2 = cpu_conv_bn(ref_t1, params_l1b1_c2)
    ref_f3 = cpu_add_relu(ref_t2, ref_f2)

    # layer2 block0 (downsample)
    ref_t1 = cpu_conv_bn_relu(ref_f3, params_l2b0_c1)
    ref_t2 = cpu_conv_bn(ref_t1, params_l2b0_c2)
    ref_skip = cpu_conv_bn(ref_f3, params_l2_ds)
    ref_f1 = cpu_add_relu(ref_t2, ref_skip)

    # layer2 block1
    ref_t1 = cpu_conv_bn_relu(ref_f1, params_l2b1_c1)
    ref_t2 = cpu_conv_bn(ref_t1, params_l2b1_c2)
    ref_f2 = cpu_add_relu(ref_t2, ref_f1)

    # layer3 block0 (downsample)
    ref_t1 = cpu_conv_bn_relu(ref_f2, params_l3b0_c1)
    ref_t2 = cpu_conv_bn(ref_t1, params_l3b0_c2)
    ref_skip = cpu_conv_bn(ref_f2, params_l3_ds)
    ref_f3 = cpu_add_relu(ref_t2, ref_skip)

    # layer3 block1
    ref_t1 = cpu_conv_bn_relu(ref_f3, params_l3b1_c1)
    ref_t2 = cpu_conv_bn(ref_t1, params_l3b1_c2)
    ref_f1 = cpu_add_relu(ref_t2, ref_f3)

    # layer4 block0 (downsample)
    ref_t1 = cpu_conv_bn_relu(ref_f1, params_l4b0_c1)
    ref_t2 = cpu_conv_bn(ref_t1, params_l4b0_c2)
    ref_skip = cpu_conv_bn(ref_f1, params_l4_ds)
    ref_f2 = cpu_add_relu(ref_t2, ref_skip)

    # layer4 block1
    ref_t1 = cpu_conv_bn_relu(ref_f2, params_l4b1_c1)
    ref_t2 = cpu_conv_bn(ref_t1, params_l4b1_c2)
    ref_f3 = cpu_add_relu(ref_t2, ref_f2)

    # classifier
    ref_logits = cpu_avgpool_fc(ref_f3, params_fc, S3, S3, CH3, NUM_CLASSES)

    # ═══════════════════════════════════════════════════════════════════════
    # Verification (matches resnet18.cc verification section)
    # ═══════════════════════════════════════════════════════════════════════

    mismatches = 0
    for i in range(NUM_CLASSES):
        if logits[i] != ref_logits[i]:
            # resnet18.cc: printf("MISMATCH logit[%d]: got %d, expected %d\n", ...)
            print(f"MISMATCH logit[{i}]: got {logits[i]}, expected {ref_logits[i]}")
            mismatches += 1

    if mismatches == 0:
        print(f"PASS: all {NUM_CLASSES} logits match.")
    else:
        print(f"FAIL: {mismatches} logit mismatches out of {NUM_CLASSES}.")

    # Print classification result
    best_class = int(np.argmax(ref_logits))
    print(f"Logits: {list(ref_logits)}")
    print(f"Predicted class: {best_class} (logit={ref_logits[best_class]})")


if __name__ == "__main__":
    main()
