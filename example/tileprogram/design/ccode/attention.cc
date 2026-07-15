/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * AIE Programming Model — Scaled Dot-Product Self-Attention
 *
 * Single-head self-attention with linear Q/K/V projections, int8
 * approximate softmax, and residual skip connection.
 * Scaled-down dimensions: seq_len=8, embed_dim=8.
 *
 *   Input (8x8) → Wq,Wk,Wv projections → Q,K,V (8x8 each)
 *   scores = Q * K^T / sqrt(d)  → approx softmax → attn_weights
 *   context = attn_weights * V  → Wo projection → output
 *   result = input + output     (residual connection)
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
// Dimensions
// ═══════════════════════════════════════════════════════════════════════════
#define SEQ_LEN    8     // sequence length (number of tokens)
#define EMBED_DIM  8     // embedding dimension per token
#define FEAT_SZ    (SEQ_LEN * EMBED_DIM)  // 64 elements per matrix

// Scaling factor for dot-product attention: 1/sqrt(8) ≈ 0.354
// In Q7 fixed-point: 0.354 * 128 ≈ 45
#define SCALE_FACTOR  45

// Weight matrix size: EMBED_DIM x EMBED_DIM = 64
#define WEIGHT_SZ  (EMBED_DIM * EMBED_DIM)

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL 1: linear_proj — Matrix multiply: Out = In * W
//
// In:  [SEQ_LEN x EMBED_DIM]   (row-major)
// W:   [EMBED_DIM x EMBED_DIM] (row-major)
// Out: [SEQ_LEN x EMBED_DIM]   (row-major)
//
// Reused for Q, K, V, and output projections.
// ═══════════════════════════════════════════════════════════════════════════
__global__ void linear_proj(input_window_int8 *window_in_feat,
                             input_window_int8 *window_in_weights,
                             output_window_int8 *window_out_feat) {
    int8_t *input   = (int8_t *)acquire_input_window(window_in_feat);
    int8_t *weights = (int8_t *)acquire_input_window(window_in_weights);
    int8_t *output  = acquire_output_window(window_out_feat);

    // Out[i][j] = sum_k( In[i][k] * W[k][j] )
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            int16_t sum = 0;
            for (int k = 0; k < EMBED_DIM; k++) {
                sum += (int16_t)input[i * EMBED_DIM + k] *
                       (int16_t)weights[k * EMBED_DIM + j];
            }
            // Saturate to int8
            if (sum > 127)       sum = 127;
            else if (sum < -128) sum = -128;
            output[i * EMBED_DIM + j] = (int8_t)sum;
        }
    }

    release_input_window(window_in_feat);
    release_input_window(window_in_weights);
    release_output_window(window_out_feat);
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL 2: matmul_qk — Scaled dot-product: scores = Q * K^T / sqrt(d)
//
// Q:      [SEQ_LEN x EMBED_DIM]
// K:      [SEQ_LEN x EMBED_DIM]
// scores: [SEQ_LEN x SEQ_LEN]
//
// scores[i][j] = (sum_k Q[i][k] * K[j][k]) * SCALE_FACTOR >> 7
// Note: K is transposed implicitly (K[j][k] instead of K[k][j]).
// ═══════════════════════════════════════════════════════════════════════════
__global__ void matmul_qk(input_window_int8 *window_in_q,
                           input_window_int8 *window_in_k,
                           output_window_int8 *window_out_scores) {
    int8_t *Q      = (int8_t *)acquire_input_window(window_in_q);
    int8_t *K      = (int8_t *)acquire_input_window(window_in_k);
    int8_t *scores = acquire_output_window(window_out_scores);

    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < SEQ_LEN; j++) {
            int16_t dot = 0;
            for (int k = 0; k < EMBED_DIM; k++) {
                dot += (int16_t)Q[i * EMBED_DIM + k] *
                       (int16_t)K[j * EMBED_DIM + k];  // K^T
            }
            // Scale: dot * (1/sqrt(d)) in Q7: dot * 45 >> 7
            int16_t scaled = (dot * SCALE_FACTOR) >> 7;
            if (scaled > 127)       scaled = 127;
            else if (scaled < -128) scaled = -128;
            scores[i * SEQ_LEN + j] = (int8_t)scaled;
        }
    }

    release_input_window(window_in_q);
    release_input_window(window_in_k);
    release_output_window(window_out_scores);
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL 3: approx_softmax — Int8-friendly softmax approximation
//
// Per row of the score matrix:
//   1. Find row max
//   2. Subtract max from each element (shift to [max-min..0] range)
//   3. Piecewise-linear exp approximation:
//        x >= 0:  exp_approx = 127
//        x >= -32: exp_approx = 127 + 2*x  (linear ramp from 63 to 127)
//        x >= -64: exp_approx = 63 + x     (gentler slope, 0 to 63)
//        x < -64:  exp_approx = 0
//   4. Normalize: scale each value so row sums to ~127
//
// Input:  scores [SEQ_LEN x SEQ_LEN]
// Output: attn_weights [SEQ_LEN x SEQ_LEN]  (pseudo-probabilities in [0,127])
// ═══════════════════════════════════════════════════════════════════════════
__global__ void approx_softmax(input_window_int8 *window_in_scores,
                                output_window_int8 *window_out_attn) {
    int8_t *scores = (int8_t *)acquire_input_window(window_in_scores);
    int8_t *attn   = acquire_output_window(window_out_attn);

    for (int i = 0; i < SEQ_LEN; i++) {
        // Step 1: find row max
        int8_t row_max = scores[i * SEQ_LEN];
        for (int j = 1; j < SEQ_LEN; j++) {
            int8_t val = scores[i * SEQ_LEN + j];
            if (val > row_max) row_max = val;
        }

        // Step 2-3: subtract max, apply piecewise-linear exp approximation
        int16_t exp_vals[8];  // SEQ_LEN = 8
        int16_t exp_sum = 0;
        for (int j = 0; j < SEQ_LEN; j++) {
            int16_t x = (int16_t)scores[i * SEQ_LEN + j] - (int16_t)row_max;
            // x is now in [-255, 0] range
            int16_t e;
            if (x >= 0)         e = 127;
            else if (x >= -32)  e = 127 + 2 * x;    // [63, 127]
            else if (x >= -64)  e = 63 + x;         // [0, 63]  (shifted: -1 to 63)
            else                e = 0;
            if (e < 0) e = 0;
            exp_vals[j] = e;
            exp_sum += e;
        }

        // Step 4: normalize so row sums to ~127
        for (int j = 0; j < SEQ_LEN; j++) {
            if (exp_sum > 0)
                attn[i * SEQ_LEN + j] = (int8_t)((exp_vals[j] * 127) / exp_sum);
            else
                attn[i * SEQ_LEN + j] = (int8_t)(127 / SEQ_LEN);  // uniform fallback
        }
    }

    release_input_window(window_in_scores);
    release_output_window(window_out_attn);
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL 4: attn_apply — Context = attn_weights * V
//
// attn_weights: [SEQ_LEN x SEQ_LEN]  (pseudo-probabilities, scale ~127 per row)
// V:           [SEQ_LEN x EMBED_DIM]
// context:     [SEQ_LEN x EMBED_DIM]
//
// context[i][j] = sum_k( attn[i][k] * V[k][j] ) >> 7
// The >>7 compensates for attn weights being in [0,127] instead of [0,1].
// ═══════════════════════════════════════════════════════════════════════════
__global__ void attn_apply(input_window_int8 *window_in_attn,
                            input_window_int8 *window_in_v,
                            output_window_int8 *window_out_context) {
    int8_t *attn_w  = (int8_t *)acquire_input_window(window_in_attn);
    int8_t *V       = (int8_t *)acquire_input_window(window_in_v);
    int8_t *context = acquire_output_window(window_out_context);

    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            int16_t sum = 0;
            for (int k = 0; k < SEQ_LEN; k++) {
                // attn_w is unsigned [0,127], V is signed int8
                sum += (int16_t)(uint8_t)attn_w[i * SEQ_LEN + k] *
                       (int16_t)V[k * EMBED_DIM + j];
            }
            // Rescale: attn weights summed to ~127, so >>7 to normalize
            sum >>= 7;
            if (sum > 127)       sum = 127;
            else if (sum < -128) sum = -128;
            context[i * EMBED_DIM + j] = (int8_t)sum;
        }
    }

    release_input_window(window_in_attn);
    release_input_window(window_in_v);
    release_output_window(window_out_context);
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL 5: residual_add — Element-wise add (skip connection)
//
// output[i] = saturate(a[i] + b[i])
// Standard transformer residual: input + attention_output
// ═══════════════════════════════════════════════════════════════════════════
__global__ void residual_add(input_window_int8 *window_in_a,
                              input_window_int8 *window_in_b,
                              output_window_int8 *window_out) {
    int8_t *a   = (int8_t *)acquire_input_window(window_in_a);
    int8_t *b   = (int8_t *)acquire_input_window(window_in_b);
    int8_t *out = acquire_output_window(window_out);

    for (int i = 0; i < FEAT_SZ; i++) {
        int16_t sum = (int16_t)a[i] + (int16_t)b[i];
        if (sum > 127)       sum = 127;
        else if (sum < -128) sum = -128;
        out[i] = (int8_t)sum;
    }

    release_input_window(window_in_a);
    release_input_window(window_in_b);
    release_output_window(window_out);
}


// ═══════════════════════════════════════════════════════════════════════════
// CPU REFERENCE FUNCTIONS (for verification)
// ═══════════════════════════════════════════════════════════════════════════

// CPU: matrix multiply Out = In * W, saturate to int8
static void cpu_linear_proj(const int8_t *input, const int8_t *weights,
                              int8_t *output) {
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            int16_t sum = 0;
            for (int k = 0; k < EMBED_DIM; k++) {
                sum += (int16_t)input[i * EMBED_DIM + k] *
                       (int16_t)weights[k * EMBED_DIM + j];
            }
            if (sum > 127)       sum = 127;
            else if (sum < -128) sum = -128;
            output[i * EMBED_DIM + j] = (int8_t)sum;
        }
    }
}

// CPU: scaled dot-product Q * K^T / sqrt(d)
static void cpu_matmul_qk(const int8_t *Q, const int8_t *K, int8_t *scores) {
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < SEQ_LEN; j++) {
            int16_t dot = 0;
            for (int k = 0; k < EMBED_DIM; k++) {
                dot += (int16_t)Q[i * EMBED_DIM + k] *
                       (int16_t)K[j * EMBED_DIM + k];
            }
            int16_t scaled = (dot * SCALE_FACTOR) >> 7;
            if (scaled > 127)       scaled = 127;
            else if (scaled < -128) scaled = -128;
            scores[i * SEQ_LEN + j] = (int8_t)scaled;
        }
    }
}

// CPU: approximate softmax (must match kernel exactly)
static void cpu_approx_softmax(const int8_t *scores, int8_t *attn) {
    for (int i = 0; i < SEQ_LEN; i++) {
        int8_t row_max = scores[i * SEQ_LEN];
        for (int j = 1; j < SEQ_LEN; j++) {
            int8_t val = scores[i * SEQ_LEN + j];
            if (val > row_max) row_max = val;
        }

        int16_t exp_vals[8];
        int16_t exp_sum = 0;
        for (int j = 0; j < SEQ_LEN; j++) {
            int16_t x = (int16_t)scores[i * SEQ_LEN + j] - (int16_t)row_max;
            int16_t e;
            if (x >= 0)         e = 127;
            else if (x >= -32)  e = 127 + 2 * x;
            else if (x >= -64)  e = 63 + x;
            else                e = 0;
            if (e < 0) e = 0;
            exp_vals[j] = e;
            exp_sum += e;
        }

        for (int j = 0; j < SEQ_LEN; j++) {
            if (exp_sum > 0)
                attn[i * SEQ_LEN + j] = (int8_t)((exp_vals[j] * 127) / exp_sum);
            else
                attn[i * SEQ_LEN + j] = (int8_t)(127 / SEQ_LEN);
        }
    }
}

// CPU: context = attn_weights * V, rescale by >>7
static void cpu_attn_apply(const int8_t *attn_w, const int8_t *V,
                             int8_t *context) {
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            int16_t sum = 0;
            for (int k = 0; k < SEQ_LEN; k++) {
                sum += (int16_t)(uint8_t)attn_w[i * SEQ_LEN + k] *
                       (int16_t)V[k * EMBED_DIM + j];
            }
            sum >>= 7;
            if (sum > 127)       sum = 127;
            else if (sum < -128) sum = -128;
            context[i * EMBED_DIM + j] = (int8_t)sum;
        }
    }
}

// CPU: element-wise add with saturation
static void cpu_residual_add(const int8_t *a, const int8_t *b,
                               int8_t *output) {
    for (int i = 0; i < FEAT_SZ; i++) {
        int16_t sum = (int16_t)a[i] + (int16_t)b[i];
        if (sum > 127)       sum = 127;
        else if (sum < -128) sum = -128;
        output[i] = (int8_t)sum;
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// HOST: main
//
// Self-attention flow:
//   Input → Q,K,V projections → Q*K^T/sqrt(d) → softmax → *V
//   → output projection → residual add with input → result
// ═══════════════════════════════════════════════════════════════════════════
int main() {
    printf("=== Scaled Dot-Product Self-Attention on AIE Tile Mesh ===\n");
    printf("    seq_len=%d, embed_dim=%d, int8 quantized\n", SEQ_LEN, EMBED_DIM);

    // --- Device + mesh ---
    aieSetDevice(0);
    aieDim mesh(2, 2);

    // ─── Input: [SEQ_LEN x EMBED_DIM] = 8x8 ───
    int8_t *input = (int8_t *)malloc(FEAT_SZ);
    for (int i = 0; i < SEQ_LEN; i++)
        for (int j = 0; j < EMBED_DIM; j++)
            input[i * EMBED_DIM + j] = (int8_t)((i * EMBED_DIM + j) % 11 - 5);
    // Test pattern: values in [-5, 5], varied enough to exercise attention

    // ─── Weight matrices: Wq, Wk, Wv, Wo [EMBED_DIM x EMBED_DIM] each ───
    int8_t *Wq = (int8_t *)malloc(WEIGHT_SZ);
    int8_t *Wk = (int8_t *)malloc(WEIGHT_SZ);
    int8_t *Wv = (int8_t *)malloc(WEIGHT_SZ);
    int8_t *Wo = (int8_t *)malloc(WEIGHT_SZ);

    // Initialize weights: near-identity for Q/K (slightly perturbed),
    // simple pattern for V/O
    for (int i = 0; i < EMBED_DIM; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            int idx = i * EMBED_DIM + j;
            // Wq: identity + small perturbation
            Wq[idx] = (i == j) ? 2 : 0;
            // Wk: identity (so Q*K^T tests genuine dot-product similarity)
            Wk[idx] = (i == j) ? 2 : 0;
            // Wv: alternating pattern
            Wv[idx] = (i == j) ? 1 : ((i + j) % 2 == 0 ? 1 : 0);
            // Wo: identity (pass-through output projection)
            Wo[idx] = (i == j) ? 1 : 0;
        }
    }

    // ─── Intermediate buffers ───
    int8_t *Q_buf       = (int8_t *)malloc(FEAT_SZ);          // [8x8]
    int8_t *K_buf       = (int8_t *)malloc(FEAT_SZ);          // [8x8]
    int8_t *V_buf       = (int8_t *)malloc(FEAT_SZ);          // [8x8]
    int8_t *scores      = (int8_t *)malloc(SEQ_LEN * SEQ_LEN); // [8x8]
    int8_t *attn_weights = (int8_t *)malloc(SEQ_LEN * SEQ_LEN); // [8x8]
    int8_t *context     = (int8_t *)malloc(FEAT_SZ);          // [8x8]
    int8_t *proj_out    = (int8_t *)malloc(FEAT_SZ);          // [8x8]
    int8_t *result      = (int8_t *)malloc(FEAT_SZ);          // [8x8]

    // ═══════════════════════════════════════════════════════════════════════
    // Forward pass
    // ═══════════════════════════════════════════════════════════════════════

    // Step 1: Linear projections → Q, K, V
    linear_proj<<<mesh>>>(input, Wq, Q_buf);      // Q = input * Wq
    linear_proj<<<mesh>>>(input, Wk, K_buf);      // K = input * Wk
    linear_proj<<<mesh>>>(input, Wv, V_buf);      // V = input * Wv

    // Step 2: Scaled dot-product scores = Q * K^T / sqrt(d)
    matmul_qk<<<mesh>>>(Q_buf, K_buf, scores);

    // Step 3: Approximate softmax → attention weights
    approx_softmax<<<mesh>>>(scores, attn_weights);

    // Step 4: Apply attention: context = attn_weights * V
    attn_apply<<<mesh>>>(attn_weights, V_buf, context);

    // Step 5: Output projection
    linear_proj<<<mesh>>>(context, Wo, proj_out);

    // Step 6: Residual connection: result = input + proj_out
    residual_add<<<mesh>>>(input, proj_out, result);

    // --- Wait for all tiles ---
    aieDeviceSynchronize();

    // ═══════════════════════════════════════════════════════════════════════
    // CPU reference (exact same ops)
    // ═══════════════════════════════════════════════════════════════════════
    int8_t *ref_Q       = (int8_t *)malloc(FEAT_SZ);
    int8_t *ref_K       = (int8_t *)malloc(FEAT_SZ);
    int8_t *ref_V       = (int8_t *)malloc(FEAT_SZ);
    int8_t *ref_scores  = (int8_t *)malloc(SEQ_LEN * SEQ_LEN);
    int8_t *ref_attn    = (int8_t *)malloc(SEQ_LEN * SEQ_LEN);
    int8_t *ref_context = (int8_t *)malloc(FEAT_SZ);
    int8_t *ref_proj    = (int8_t *)malloc(FEAT_SZ);
    int8_t *ref_result  = (int8_t *)malloc(FEAT_SZ);

    cpu_linear_proj(input, Wq, ref_Q);
    cpu_linear_proj(input, Wk, ref_K);
    cpu_linear_proj(input, Wv, ref_V);
    cpu_matmul_qk(ref_Q, ref_K, ref_scores);
    cpu_approx_softmax(ref_scores, ref_attn);
    cpu_attn_apply(ref_attn, ref_V, ref_context);
    cpu_linear_proj(ref_context, Wo, ref_proj);
    cpu_residual_add(input, ref_proj, ref_result);

    // ═══════════════════════════════════════════════════════════════════════
    // Verification
    // ═══════════════════════════════════════════════════════════════════════
    int mismatches = 0;

    // Verify final result
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            int idx = i * EMBED_DIM + j;
            if (result[idx] != ref_result[idx]) {
                printf("MISMATCH result[%d][%d]: got %d, expected %d\n",
                       i, j, result[idx], ref_result[idx]);
                mismatches++;
            }
        }
    }

    if (mismatches == 0)
        printf("PASS: all %d elements match.\n", FEAT_SZ);
    else
        printf("FAIL: %d mismatches out of %d.\n", mismatches, FEAT_SZ);

    // Print attention pattern (which tokens attend to which)
    printf("\nAttention weights (row=query, col=key, scale 0-127):\n");
    printf("     ");
    for (int j = 0; j < SEQ_LEN; j++) printf(" k%d ", j);
    printf("\n");
    for (int i = 0; i < SEQ_LEN; i++) {
        printf("q%d:  ", i);
        for (int j = 0; j < SEQ_LEN; j++) {
            printf("%3d ", (uint8_t)ref_attn[i * SEQ_LEN + j]);
        }
        printf("\n");
    }

    // Print input vs output (first 2 tokens)
    printf("\nInput vs Output (first 2 tokens):\n");
    for (int i = 0; i < 2; i++) {
        printf("  token %d input:  [", i);
        for (int j = 0; j < EMBED_DIM; j++) {
            printf("%3d", input[i * EMBED_DIM + j]);
            if (j < EMBED_DIM - 1) printf(", ");
        }
        printf("]\n");
        printf("  token %d output: [", i);
        for (int j = 0; j < EMBED_DIM; j++) {
            printf("%3d", ref_result[i * EMBED_DIM + j]);
            if (j < EMBED_DIM - 1) printf(", ");
        }
        printf("]\n");
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Cleanup
    // ═══════════════════════════════════════════════════════════════════════
    free(input);
    free(Wq); free(Wk); free(Wv); free(Wo);
    free(Q_buf); free(K_buf); free(V_buf);
    free(scores); free(attn_weights);
    free(context); free(proj_out); free(result);

    free(ref_Q); free(ref_K); free(ref_V);
    free(ref_scores); free(ref_attn);
    free(ref_context); free(ref_proj); free(ref_result);

    return 0;
}
