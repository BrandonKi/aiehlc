/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * AIE Programming Model — KV Cache Autoregressive Decoding
 *
 * Single-head attention with KV cache for autoregressive text generation.
 * Two phases: prefill (process prompt at once) and decode (one token at a
 * time, growing the KV cache each step).
 *
 * Dimensions: MAX_SEQ=16, EMBED_DIM=8, PROMPT_LEN=4, GEN_LEN=4, VOCAB=8
 * Data type: int8, Q7 fixed-point where needed.
 *
 *   PREFILL:  embed prompt → Q,K,V projections → cache K,V
 *             → attention on last prompt token → first generated token
 *
 *   DECODE:   embed new token → project q,k,v → append to cache
 *             → attend over full cache → output proj → residual → next token
 *
 * CUDA concepts kept (honest mapping):
 *   __global__             - kernel runs on AIE tiles
 *   kernel<<<mesh>>>()     - launch kernel across tile mesh
 *   aieDeviceSynchronize() - wait for all tiles to finish
 *   malloc/free            - plain C host memory allocation
 *
 ******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// ═══════════════════════════════════════════════════════════════════════════
// Dimensions
// ═══════════════════════════════════════════════════════════════════════════
#define MAX_SEQ     16    // maximum sequence length (cache capacity)
#define EMBED_DIM   8     // embedding dimension per token
#define PROMPT_LEN  4     // number of prompt tokens
#define GEN_LEN     4     // number of tokens to generate
#define VOCAB       8     // vocabulary size

// Scaling factor for dot-product attention: 1/sqrt(8) ≈ 0.354
// In Q7 fixed-point: 0.354 * 128 ≈ 45
#define SCALE_FACTOR  45

// Weight matrix size: EMBED_DIM x EMBED_DIM = 64
#define WEIGHT_SZ  (EMBED_DIM * EMBED_DIM)

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL 1: linear_proj — Full-sequence matrix multiply: Out = In * W
//
// In:  [seq_len x EMBED_DIM]     (row-major)
// W:   [EMBED_DIM x EMBED_DIM]   (row-major)
// Out: [seq_len x EMBED_DIM]     (row-major)
//
// seq_len passed as parameter for prefill (PROMPT_LEN rows).
// ═══════════════════════════════════════════════════════════════════════════
__global__ void linear_proj(input_window_int8 *window_in_feat,
                             input_window_int8 *window_in_weights,
                             output_window_int8 *window_out_feat,
                             int seq_len) {
    int8_t *input   = (int8_t *)acquire_input_window(window_in_feat);
    int8_t *weights = (int8_t *)acquire_input_window(window_in_weights);
    int8_t *output  = acquire_output_window(window_out_feat);

    for (int i = 0; i < seq_len; i++) {
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

    release_input_window(window_in_feat);
    release_input_window(window_in_weights);
    release_output_window(window_out_feat);
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL 2: linear_proj_token — Single-token matrix multiply: Out = In * W
//
// In:  [1 x EMBED_DIM]
// W:   [EMBED_DIM x EMBED_DIM]
// Out: [1 x EMBED_DIM]
//
// Used during decode phase for projecting one token at a time.
// ═══════════════════════════════════════════════════════════════════════════
__global__ void linear_proj_token(input_window_int8 *window_in_tok,
                                   input_window_int8 *window_in_weights,
                                   output_window_int8 *window_out_tok) {
    int8_t *input   = (int8_t *)acquire_input_window(window_in_tok);
    int8_t *weights = (int8_t *)acquire_input_window(window_in_weights);
    int8_t *output  = acquire_output_window(window_out_tok);

    for (int j = 0; j < EMBED_DIM; j++) {
        int16_t sum = 0;
        for (int k = 0; k < EMBED_DIM; k++) {
            sum += (int16_t)input[k] *
                   (int16_t)weights[k * EMBED_DIM + j];
        }
        if (sum > 127)       sum = 127;
        else if (sum < -128) sum = -128;
        output[j] = (int8_t)sum;
    }

    release_input_window(window_in_tok);
    release_input_window(window_in_weights);
    release_output_window(window_out_tok);
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL 3: kv_cache_append — Copy [1 x EMBED_DIM] into cache at position
//
// in_vec:    [1 x EMBED_DIM]          (new k or v vector)
// cache_out: [MAX_SEQ x EMBED_DIM]    (the KV cache)
// pos:       row index to write into
// ═══════════════════════════════════════════════════════════════════════════
__global__ void kv_cache_append(input_window_int8 *window_in_vec,
                                 output_window_int8 *window_cache_out,
                                 int pos) {
    int8_t *vec   = (int8_t *)acquire_input_window(window_in_vec);
    int8_t *cache = acquire_output_window(window_cache_out);

    for (int j = 0; j < EMBED_DIM; j++) {
        cache[pos * EMBED_DIM + j] = vec[j];
    }

    release_input_window(window_in_vec);
    release_output_window(window_cache_out);
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL 4: qk_scores_cached — q * K_cache^T / sqrt(d)
//
// in_q:       [1 x EMBED_DIM]           (current query vector)
// in_kcache:  [MAX_SEQ x EMBED_DIM]     (cached keys, first cache_len valid)
// out_scores: [MAX_SEQ]                  (first cache_len entries filled)
// cache_len:  number of valid cache rows
//
// out_scores[j] = (sum_k q[k] * K_cache[j][k]) * SCALE_FACTOR >> 7
// ═══════════════════════════════════════════════════════════════════════════
__global__ void qk_scores_cached(input_window_int8 *window_in_q,
                                   input_window_int8 *window_in_kcache,
                                   output_window_int8 *window_out_scores,
                                   int cache_len) {
    int8_t *q      = (int8_t *)acquire_input_window(window_in_q);
    int8_t *kcache = (int8_t *)acquire_input_window(window_in_kcache);
    int8_t *scores = acquire_output_window(window_out_scores);

    for (int j = 0; j < cache_len; j++) {
        int16_t dot = 0;
        for (int k = 0; k < EMBED_DIM; k++) {
            dot += (int16_t)q[k] *
                   (int16_t)kcache[j * EMBED_DIM + k];  // K^T
        }
        int16_t scaled = (dot * SCALE_FACTOR) >> 7;
        if (scaled > 127)       scaled = 127;
        else if (scaled < -128) scaled = -128;
        scores[j] = (int8_t)scaled;
    }
    // Zero remaining entries
    for (int j = cache_len; j < MAX_SEQ; j++) {
        scores[j] = 0;
    }

    release_input_window(window_in_q);
    release_input_window(window_in_kcache);
    release_output_window(window_out_scores);
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL 5: approx_softmax_row — Variable-length single-row softmax
//
// Piecewise-linear exp approximation on a single row of length `len`:
//   x >= 0:   exp_approx = 127
//   x >= -32: exp_approx = 127 + 2*x   (linear ramp, [63, 127])
//   x >= -64: exp_approx = 63 + x      (gentler slope, [0, 63])
//   x < -64:  exp_approx = 0
// Normalize so row sums to ~127.
//
// Input:  in_scores  [MAX_SEQ]   (first `len` entries valid)
// Output: out_attn   [MAX_SEQ]   (first `len` entries filled)
// ═══════════════════════════════════════════════════════════════════════════
__global__ void approx_softmax_row(input_window_int8 *window_in_scores,
                                     output_window_int8 *window_out_attn,
                                     int len) {
    int8_t *scores = (int8_t *)acquire_input_window(window_in_scores);
    int8_t *attn   = acquire_output_window(window_out_attn);

    // Step 1: find max over valid entries
    int8_t row_max = scores[0];
    for (int j = 1; j < len; j++) {
        if (scores[j] > row_max) row_max = scores[j];
    }

    // Step 2-3: subtract max, piecewise-linear exp
    int16_t exp_vals[16];  // MAX_SEQ = 16
    int16_t exp_sum = 0;
    for (int j = 0; j < len; j++) {
        int16_t x = (int16_t)scores[j] - (int16_t)row_max;
        int16_t e;
        if (x >= 0)         e = 127;
        else if (x >= -32)  e = 127 + 2 * x;
        else if (x >= -64)  e = 63 + x;
        else                e = 0;
        if (e < 0) e = 0;
        exp_vals[j] = e;
        exp_sum += e;
    }

    // Step 4: normalize so row sums to ~127
    for (int j = 0; j < len; j++) {
        if (exp_sum > 0)
            attn[j] = (int8_t)((exp_vals[j] * 127) / exp_sum);
        else
            attn[j] = (int8_t)(127 / len);
    }
    // Zero remaining
    for (int j = len; j < MAX_SEQ; j++) {
        attn[j] = 0;
    }

    release_input_window(window_in_scores);
    release_output_window(window_out_attn);
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL 6: attn_apply_cached — context = attn * V_cache
//
// in_attn:     [MAX_SEQ]               (first cache_len entries are weights)
// in_vcache:   [MAX_SEQ x EMBED_DIM]   (first cache_len rows valid)
// out_context: [1 x EMBED_DIM]
//
// context[j] = sum_k( attn[k] * V_cache[k][j] ) >> 7
// ═══════════════════════════════════════════════════════════════════════════
__global__ void attn_apply_cached(input_window_int8 *window_in_attn,
                                    input_window_int8 *window_in_vcache,
                                    output_window_int8 *window_out_context,
                                    int cache_len) {
    int8_t *attn_w  = (int8_t *)acquire_input_window(window_in_attn);
    int8_t *vcache  = (int8_t *)acquire_input_window(window_in_vcache);
    int8_t *context = acquire_output_window(window_out_context);

    for (int j = 0; j < EMBED_DIM; j++) {
        int16_t sum = 0;
        for (int k = 0; k < cache_len; k++) {
            sum += (int16_t)(uint8_t)attn_w[k] *
                   (int16_t)vcache[k * EMBED_DIM + j];
        }
        sum >>= 7;
        if (sum > 127)       sum = 127;
        else if (sum < -128) sum = -128;
        context[j] = (int8_t)sum;
    }

    release_input_window(window_in_attn);
    release_input_window(window_in_vcache);
    release_output_window(window_out_context);
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL 7: residual_add_token — Element-wise add [1 x EMBED_DIM]
//
// out[i] = saturate(a[i] + b[i])
// ═══════════════════════════════════════════════════════════════════════════
__global__ void residual_add_token(input_window_int8 *window_in_a,
                                     input_window_int8 *window_in_b,
                                     output_window_int8 *window_out) {
    int8_t *a   = (int8_t *)acquire_input_window(window_in_a);
    int8_t *b   = (int8_t *)acquire_input_window(window_in_b);
    int8_t *out = acquire_output_window(window_out);

    for (int i = 0; i < EMBED_DIM; i++) {
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
// KERNEL 8: token_embed — Embedding table lookup
//
// in_table: [VOCAB x EMBED_DIM]   (the embedding table)
// out_emb:  [1 x EMBED_DIM]       (embedding for the given token_id)
// token_id: index into the table
// ═══════════════════════════════════════════════════════════════════════════
__global__ void token_embed(input_window_int8 *window_in_table,
                              output_window_int8 *window_out_emb,
                              int token_id) {
    int8_t *table = (int8_t *)acquire_input_window(window_in_table);
    int8_t *emb   = acquire_output_window(window_out_emb);

    for (int j = 0; j < EMBED_DIM; j++) {
        emb[j] = table[token_id * EMBED_DIM + j];
    }

    release_input_window(window_in_table);
    release_output_window(window_out_emb);
}


// ═══════════════════════════════════════════════════════════════════════════
// CPU REFERENCE FUNCTIONS (for verification)
// ═══════════════════════════════════════════════════════════════════════════

// CPU: full-sequence matrix multiply Out = In * W
static void cpu_linear_proj(const int8_t *input, const int8_t *weights,
                              int8_t *output, int seq_len) {
    for (int i = 0; i < seq_len; i++) {
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

// CPU: single-token matrix multiply Out = In * W
static void cpu_linear_proj_token(const int8_t *input, const int8_t *weights,
                                     int8_t *output) {
    for (int j = 0; j < EMBED_DIM; j++) {
        int16_t sum = 0;
        for (int k = 0; k < EMBED_DIM; k++) {
            sum += (int16_t)input[k] *
                   (int16_t)weights[k * EMBED_DIM + j];
        }
        if (sum > 127)       sum = 127;
        else if (sum < -128) sum = -128;
        output[j] = (int8_t)sum;
    }
}

// CPU: append vector to cache at position
static void cpu_kv_cache_append(const int8_t *vec, int8_t *cache, int pos) {
    for (int j = 0; j < EMBED_DIM; j++) {
        cache[pos * EMBED_DIM + j] = vec[j];
    }
}

// CPU: q * K_cache^T / sqrt(d)
static void cpu_qk_scores_cached(const int8_t *q, const int8_t *kcache,
                                    int8_t *scores, int cache_len) {
    for (int j = 0; j < cache_len; j++) {
        int16_t dot = 0;
        for (int k = 0; k < EMBED_DIM; k++) {
            dot += (int16_t)q[k] *
                   (int16_t)kcache[j * EMBED_DIM + k];
        }
        int16_t scaled = (dot * SCALE_FACTOR) >> 7;
        if (scaled > 127)       scaled = 127;
        else if (scaled < -128) scaled = -128;
        scores[j] = (int8_t)scaled;
    }
    for (int j = cache_len; j < MAX_SEQ; j++) {
        scores[j] = 0;
    }
}

// CPU: variable-length single-row softmax (must match kernel exactly)
static void cpu_approx_softmax_row(const int8_t *scores, int8_t *attn,
                                      int len) {
    int8_t row_max = scores[0];
    for (int j = 1; j < len; j++) {
        if (scores[j] > row_max) row_max = scores[j];
    }

    int16_t exp_vals[16];
    int16_t exp_sum = 0;
    for (int j = 0; j < len; j++) {
        int16_t x = (int16_t)scores[j] - (int16_t)row_max;
        int16_t e;
        if (x >= 0)         e = 127;
        else if (x >= -32)  e = 127 + 2 * x;
        else if (x >= -64)  e = 63 + x;
        else                e = 0;
        if (e < 0) e = 0;
        exp_vals[j] = e;
        exp_sum += e;
    }

    for (int j = 0; j < len; j++) {
        if (exp_sum > 0)
            attn[j] = (int8_t)((exp_vals[j] * 127) / exp_sum);
        else
            attn[j] = (int8_t)(127 / len);
    }
    for (int j = len; j < MAX_SEQ; j++) {
        attn[j] = 0;
    }
}

// CPU: context = attn * V_cache
static void cpu_attn_apply_cached(const int8_t *attn_w, const int8_t *vcache,
                                     int8_t *context, int cache_len) {
    for (int j = 0; j < EMBED_DIM; j++) {
        int16_t sum = 0;
        for (int k = 0; k < cache_len; k++) {
            sum += (int16_t)(uint8_t)attn_w[k] *
                   (int16_t)vcache[k * EMBED_DIM + j];
        }
        sum >>= 7;
        if (sum > 127)       sum = 127;
        else if (sum < -128) sum = -128;
        context[j] = (int8_t)sum;
    }
}

// CPU: element-wise add [1 x EMBED_DIM]
static void cpu_residual_add_token(const int8_t *a, const int8_t *b,
                                      int8_t *output) {
    for (int i = 0; i < EMBED_DIM; i++) {
        int16_t sum = (int16_t)a[i] + (int16_t)b[i];
        if (sum > 127)       sum = 127;
        else if (sum < -128) sum = -128;
        output[i] = (int8_t)sum;
    }
}

// CPU: embedding lookup
static void cpu_token_embed(const int8_t *table, int8_t *emb, int token_id) {
    for (int j = 0; j < EMBED_DIM; j++) {
        emb[j] = table[token_id * EMBED_DIM + j];
    }
}

// Host helper: score result against embedding table → argmax → next token
static int host_score_argmax(const int8_t *result, const int8_t *embed_table) {
    int best_id = 0;
    int16_t best_score = -32768;
    for (int v = 0; v < VOCAB; v++) {
        int16_t dot = 0;
        for (int j = 0; j < EMBED_DIM; j++) {
            dot += (int16_t)result[j] *
                   (int16_t)embed_table[v * EMBED_DIM + j];
        }
        if (dot > best_score) {
            best_score = dot;
            best_id = v;
        }
    }
    return best_id;
}


// ═══════════════════════════════════════════════════════════════════════════
// HOST: main
//
// KV cache autoregressive decoding:
//   Prefill: embed prompt → Q,K,V → cache K,V → attention → first token
//   Decode:  embed token → q,k,v → grow cache → attention → next token
// ═══════════════════════════════════════════════════════════════════════════
int main() {
    printf("=== KV Cache Autoregressive Decoding on AIE Tile Mesh ===\n");
    printf("    max_seq=%d, embed_dim=%d, prompt=%d, gen=%d, vocab=%d\n",
           MAX_SEQ, EMBED_DIM, PROMPT_LEN, GEN_LEN, VOCAB);

    // --- Device + mesh ---
    aieSetDevice(0);
    aieDim mesh(2, 2);

    // ─── Embedding table: [VOCAB x EMBED_DIM] ───
    int8_t *embed_table = (int8_t *)malloc(VOCAB * EMBED_DIM);
    for (int v = 0; v < VOCAB; v++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            // Distinct per-token pattern: shifted identity-like rows
            embed_table[v * EMBED_DIM + j] = (int8_t)((v == j) ? 4 :
                                              ((v + j) % 3 == 0 ? 1 : 0));
        }
    }

    // ─── Weight matrices: Wq, Wk, Wv, Wo [EMBED_DIM x EMBED_DIM] each ───
    int8_t *Wq = (int8_t *)malloc(WEIGHT_SZ);
    int8_t *Wk = (int8_t *)malloc(WEIGHT_SZ);
    int8_t *Wv = (int8_t *)malloc(WEIGHT_SZ);
    int8_t *Wo = (int8_t *)malloc(WEIGHT_SZ);
    for (int i = 0; i < EMBED_DIM; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            int idx = i * EMBED_DIM + j;
            Wq[idx] = (i == j) ? 2 : 0;
            Wk[idx] = (i == j) ? 2 : 0;
            Wv[idx] = (i == j) ? 1 : ((i + j) % 2 == 0 ? 1 : 0);
            Wo[idx] = (i == j) ? 1 : 0;
        }
    }

    // ─── Prompt token IDs ───
    int prompt_ids[PROMPT_LEN] = {0, 3, 5, 2};

    // ─── Buffers ───
    // KV caches: [MAX_SEQ x EMBED_DIM]
    int8_t *K_cache = (int8_t *)malloc(MAX_SEQ * EMBED_DIM);
    int8_t *V_cache = (int8_t *)malloc(MAX_SEQ * EMBED_DIM);
    memset(K_cache, 0, MAX_SEQ * EMBED_DIM);
    memset(V_cache, 0, MAX_SEQ * EMBED_DIM);

    // Prefill buffers
    int8_t *prompt_emb = (int8_t *)malloc(PROMPT_LEN * EMBED_DIM);
    int8_t *Q_buf      = (int8_t *)malloc(PROMPT_LEN * EMBED_DIM);
    int8_t *K_buf      = (int8_t *)malloc(PROMPT_LEN * EMBED_DIM);
    int8_t *V_buf      = (int8_t *)malloc(PROMPT_LEN * EMBED_DIM);

    // Per-token decode buffers
    int8_t *q_last       = (int8_t *)malloc(EMBED_DIM);
    int8_t *token_emb    = (int8_t *)malloc(EMBED_DIM);
    int8_t *q_tok        = (int8_t *)malloc(EMBED_DIM);
    int8_t *k_tok        = (int8_t *)malloc(EMBED_DIM);
    int8_t *v_tok        = (int8_t *)malloc(EMBED_DIM);
    int8_t *scores       = (int8_t *)malloc(MAX_SEQ);
    int8_t *attn_weights = (int8_t *)malloc(MAX_SEQ);
    int8_t *context      = (int8_t *)malloc(EMBED_DIM);
    int8_t *proj_out     = (int8_t *)malloc(EMBED_DIM);
    int8_t *result       = (int8_t *)malloc(EMBED_DIM);

    // Generated token IDs
    int generated_ids[GEN_LEN];
    int cache_len = 0;

    // ═══════════════════════════════════════════════════════════════════════
    // PREFILL PHASE
    // ═══════════════════════════════════════════════════════════════════════
    printf("\n--- Prefill Phase (prompt: ");
    for (int i = 0; i < PROMPT_LEN; i++) printf("%d ", prompt_ids[i]);
    printf(") ---\n");

    // Step 1: Embed prompt tokens (host memcpy from embed_table)
    for (int i = 0; i < PROMPT_LEN; i++) {
        memcpy(&prompt_emb[i * EMBED_DIM],
               &embed_table[prompt_ids[i] * EMBED_DIM],
               EMBED_DIM);
    }

    // Step 2: Linear projections for Q, K, V (full prompt batch)
    linear_proj<<<mesh>>>(prompt_emb, Wq, Q_buf, PROMPT_LEN);
    linear_proj<<<mesh>>>(prompt_emb, Wk, K_buf, PROMPT_LEN);
    linear_proj<<<mesh>>>(prompt_emb, Wv, V_buf, PROMPT_LEN);

    // Step 3: Copy K_buf, V_buf into K_cache, V_cache positions 0..PROMPT_LEN-1
    // (host memcpy — bulk fill for prefill)
    aieDeviceSynchronize();
    memcpy(K_cache, K_buf, PROMPT_LEN * EMBED_DIM);
    memcpy(V_cache, V_buf, PROMPT_LEN * EMBED_DIM);
    cache_len = PROMPT_LEN;

    // Step 4: Extract last query row for attention
    memcpy(q_last, &Q_buf[(PROMPT_LEN - 1) * EMBED_DIM], EMBED_DIM);

    // Step 5: Attention on last prompt token over cached K,V
    qk_scores_cached<<<mesh>>>(q_last, K_cache, scores, cache_len);
    approx_softmax_row<<<mesh>>>(scores, attn_weights, cache_len);
    attn_apply_cached<<<mesh>>>(attn_weights, V_cache, context, cache_len);

    // Step 6: Output projection + residual
    linear_proj_token<<<mesh>>>(context, Wo, proj_out);
    residual_add_token<<<mesh>>>(&prompt_emb[(PROMPT_LEN - 1) * EMBED_DIM],
                                  proj_out, result);
    aieDeviceSynchronize();

    // Step 7: Score against vocab → first generated token
    generated_ids[0] = host_score_argmax(result, embed_table);
    printf("  Prefill → first generated token: %d\n", generated_ids[0]);

    // ═══════════════════════════════════════════════════════════════════════
    // DECODE LOOP
    // ═══════════════════════════════════════════════════════════════════════
    printf("\n--- Decode Phase (%d steps) ---\n", GEN_LEN);

    int current_token = generated_ids[0];
    for (int step = 0; step < GEN_LEN; step++) {
        printf("  Step %d: token %d → ", step, current_token);

        // (a) Embed current token
        token_embed<<<mesh>>>(embed_table, token_emb, current_token);

        // (b) Project q, k, v (single token)
        linear_proj_token<<<mesh>>>(token_emb, Wq, q_tok);
        linear_proj_token<<<mesh>>>(token_emb, Wk, k_tok);
        linear_proj_token<<<mesh>>>(token_emb, Wv, v_tok);

        // (c) Append k, v to cache
        kv_cache_append<<<mesh>>>(k_tok, K_cache, cache_len);
        kv_cache_append<<<mesh>>>(v_tok, V_cache, cache_len);
        cache_len++;

        // (d) Attention: q over full cache
        qk_scores_cached<<<mesh>>>(q_tok, K_cache, scores, cache_len);
        approx_softmax_row<<<mesh>>>(scores, attn_weights, cache_len);
        attn_apply_cached<<<mesh>>>(attn_weights, V_cache, context, cache_len);

        // (e) Output projection + residual with embedding
        linear_proj_token<<<mesh>>>(context, Wo, proj_out);
        residual_add_token<<<mesh>>>(token_emb, proj_out, result);

        aieDeviceSynchronize();

        // (f) Score → argmax → next token
        int next_token = host_score_argmax(result, embed_table);

        if (step < GEN_LEN - 1) {
            generated_ids[step + 1] = next_token;
            current_token = next_token;
        }
        printf("next token: %d (cache_len=%d, reused %d cached K,V, "
               "computed only 1 new)\n", next_token, cache_len, cache_len - 1);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // CPU REFERENCE
    // ═══════════════════════════════════════════════════════════════════════
    printf("\n--- CPU Reference ---\n");

    int8_t *ref_K_cache = (int8_t *)malloc(MAX_SEQ * EMBED_DIM);
    int8_t *ref_V_cache = (int8_t *)malloc(MAX_SEQ * EMBED_DIM);
    memset(ref_K_cache, 0, MAX_SEQ * EMBED_DIM);
    memset(ref_V_cache, 0, MAX_SEQ * EMBED_DIM);

    int8_t *ref_prompt_emb = (int8_t *)malloc(PROMPT_LEN * EMBED_DIM);
    int8_t *ref_Q      = (int8_t *)malloc(PROMPT_LEN * EMBED_DIM);
    int8_t *ref_K      = (int8_t *)malloc(PROMPT_LEN * EMBED_DIM);
    int8_t *ref_V      = (int8_t *)malloc(PROMPT_LEN * EMBED_DIM);
    int8_t *ref_q_last = (int8_t *)malloc(EMBED_DIM);
    int8_t *ref_tok_emb  = (int8_t *)malloc(EMBED_DIM);
    int8_t *ref_q_tok    = (int8_t *)malloc(EMBED_DIM);
    int8_t *ref_k_tok    = (int8_t *)malloc(EMBED_DIM);
    int8_t *ref_v_tok    = (int8_t *)malloc(EMBED_DIM);
    int8_t *ref_scores   = (int8_t *)malloc(MAX_SEQ);
    int8_t *ref_attn     = (int8_t *)malloc(MAX_SEQ);
    int8_t *ref_context  = (int8_t *)malloc(EMBED_DIM);
    int8_t *ref_proj     = (int8_t *)malloc(EMBED_DIM);
    int8_t *ref_result   = (int8_t *)malloc(EMBED_DIM);

    int ref_generated[GEN_LEN];
    int ref_cache_len = 0;

    // Prefill
    for (int i = 0; i < PROMPT_LEN; i++) {
        cpu_token_embed(embed_table, &ref_prompt_emb[i * EMBED_DIM],
                        prompt_ids[i]);
    }
    cpu_linear_proj(ref_prompt_emb, Wq, ref_Q, PROMPT_LEN);
    cpu_linear_proj(ref_prompt_emb, Wk, ref_K, PROMPT_LEN);
    cpu_linear_proj(ref_prompt_emb, Wv, ref_V, PROMPT_LEN);

    memcpy(ref_K_cache, ref_K, PROMPT_LEN * EMBED_DIM);
    memcpy(ref_V_cache, ref_V, PROMPT_LEN * EMBED_DIM);
    ref_cache_len = PROMPT_LEN;

    memcpy(ref_q_last, &ref_Q[(PROMPT_LEN - 1) * EMBED_DIM], EMBED_DIM);
    cpu_qk_scores_cached(ref_q_last, ref_K_cache, ref_scores, ref_cache_len);
    cpu_approx_softmax_row(ref_scores, ref_attn, ref_cache_len);
    cpu_attn_apply_cached(ref_attn, ref_V_cache, ref_context, ref_cache_len);
    cpu_linear_proj_token(ref_context, Wo, ref_proj);
    cpu_residual_add_token(&ref_prompt_emb[(PROMPT_LEN - 1) * EMBED_DIM],
                            ref_proj, ref_result);
    ref_generated[0] = host_score_argmax(ref_result, embed_table);

    // Decode
    int ref_current = ref_generated[0];

    // Store attention patterns for printing
    int8_t attn_history[GEN_LEN][MAX_SEQ];

    for (int step = 0; step < GEN_LEN; step++) {
        cpu_token_embed(embed_table, ref_tok_emb, ref_current);
        cpu_linear_proj_token(ref_tok_emb, Wq, ref_q_tok);
        cpu_linear_proj_token(ref_tok_emb, Wk, ref_k_tok);
        cpu_linear_proj_token(ref_tok_emb, Wv, ref_v_tok);

        cpu_kv_cache_append(ref_k_tok, ref_K_cache, ref_cache_len);
        cpu_kv_cache_append(ref_v_tok, ref_V_cache, ref_cache_len);
        ref_cache_len++;

        cpu_qk_scores_cached(ref_q_tok, ref_K_cache, ref_scores,
                              ref_cache_len);
        cpu_approx_softmax_row(ref_scores, ref_attn, ref_cache_len);
        cpu_attn_apply_cached(ref_attn, ref_V_cache, ref_context,
                               ref_cache_len);
        cpu_linear_proj_token(ref_context, Wo, ref_proj);
        cpu_residual_add_token(ref_tok_emb, ref_proj, ref_result);

        // Save attention pattern
        memcpy(attn_history[step], ref_attn, MAX_SEQ);

        int next = host_score_argmax(ref_result, embed_table);
        if (step < GEN_LEN - 1) {
            ref_generated[step + 1] = next;
            ref_current = next;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Verification
    // ═══════════════════════════════════════════════════════════════════════
    int mismatches = 0;
    for (int i = 0; i < GEN_LEN; i++) {
        if (generated_ids[i] != ref_generated[i]) {
            printf("MISMATCH step %d: AIE generated %d, CPU generated %d\n",
                   i, generated_ids[i], ref_generated[i]);
            mismatches++;
        }
    }

    if (mismatches == 0)
        printf("PASS: all %d generated tokens match.\n", GEN_LEN);
    else
        printf("FAIL: %d mismatches out of %d generated tokens.\n",
               mismatches, GEN_LEN);

    // ═══════════════════════════════════════════════════════════════════════
    // Results
    // ═══════════════════════════════════════════════════════════════════════
    printf("\nGenerated sequence: ");
    for (int i = 0; i < PROMPT_LEN; i++) printf("%d ", prompt_ids[i]);
    printf("| ");
    for (int i = 0; i < GEN_LEN; i++) printf("%d ", ref_generated[i]);
    printf("\n");
    printf("  (prompt | generated)\n");

    // Print attention patterns at each decode step (showing growing cache)
    printf("\nAttention patterns (growing KV cache):\n");
    for (int step = 0; step < GEN_LEN; step++) {
        int cl = PROMPT_LEN + step + 1;  // cache_len at this step
        printf("  Decode step %d (cache_len=%d): [", step, cl);
        for (int j = 0; j < cl; j++) {
            printf("%3d", (uint8_t)attn_history[step][j]);
            if (j < cl - 1) printf(",");
        }
        printf("]\n");
    }

    printf("\nKV cache efficiency: each decode step reused cached K,V vectors "
           "and computed only 1 new token projection.\n");
    printf("  Prefill: projected %d tokens at once.\n", PROMPT_LEN);
    printf("  Decode: %d steps, cache grew from %d to %d entries.\n",
           GEN_LEN, PROMPT_LEN, PROMPT_LEN + GEN_LEN);

    // ═══════════════════════════════════════════════════════════════════════
    // Cleanup
    // ═══════════════════════════════════════════════════════════════════════
    free(embed_table);
    free(Wq); free(Wk); free(Wv); free(Wo);
    free(K_cache); free(V_cache);
    free(prompt_emb);
    free(Q_buf); free(K_buf); free(V_buf);
    free(q_last);
    free(token_emb); free(q_tok); free(k_tok); free(v_tok);
    free(scores); free(attn_weights);
    free(context); free(proj_out); free(result);

    free(ref_K_cache); free(ref_V_cache);
    free(ref_prompt_emb);
    free(ref_Q); free(ref_K); free(ref_V);
    free(ref_q_last);
    free(ref_tok_emb); free(ref_q_tok); free(ref_k_tok); free(ref_v_tok);
    free(ref_scores); free(ref_attn);
    free(ref_context); free(ref_proj); free(ref_result);

    return 0;
}
