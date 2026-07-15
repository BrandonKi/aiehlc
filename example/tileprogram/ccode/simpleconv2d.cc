/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * AIE Programming Model — Conv2d via Im2col (Parameterized Kernel API)
 *
 * Conv2d: output[OH,OW,F] = conv2d(input[H,W,C], filter[KH,KW,C,F])
 *
 * Implementation strategy:
 *   1. DMA (shim/memtile) performs im2col using multi-dim BD addressing —
 *      extracts sliding-window patches from the input and streams them as
 *      a flat matrix [OH*OW, KH*KW*C] to the core tiles.
 *   2. Core tiles run standard matmul: C = A * B
 *      where A = im2col matrix [M, K], B = filter [K, N], C = output [M, N]
 *      with M = OH*OW, K = KH*KW*C, N = F.
 *   3. Host orchestrates DMA and kernel launch, then reshapes output.
 *
 * Data distribution (same as GEMM):
 *   A (im2col patches): broadcast per row — each tile row gets TILE_ROWS x K
 *   B (filter):         broadcast per col — each tile col gets TILE_COLS x K
 *   C (output):         gathered per row — merged left-to-right
 *
 * For this example:
 *   Input: [8, 8, 1], Filter: [3, 3, 1, 1], stride=1, pad=0
 *   Output: [6, 6, 1]
 *   Im2col GEMM: [36, 9] x [9, 1] = [36, 1]
 *
 * The kernel is identical to simplematmul.cc — conv2d awareness lives
 * entirely in the DMA descriptor setup (done by the compiler pipeline).
 *
 * DMA im2col descriptor for shim BD (set by DmaphopTodfscheblueprintPass):
 *   dim0: stride=1,  wrap=3   (KW elements per kernel row)
 *   dim1: stride=8,  wrap=3   (KH rows, jumping W=8 per row)
 *   dim2: stride=1,  wrap=6   (OW=6 sliding positions)
 *   iter: step=8 bytes (W*stride*sizeof(int8_t)), wrap=6 (OH=6 output rows)
 *
 ******************************************************************************/
#include "simpleconv2d.h"

#pragma aie_debug_level(2 | AIE_DEBUG_FLAG_DISABLE_PARTITIONTEARDOWN | AIE_DMA_ISSUE_COUNT)

#define USE_SPATIAL_HALO // default

// ═══════════════════════════════════════════════════════════════════════════
// Kernel-visible conv geometry (spatial-halo path)
//
// The conv2d_spatial kernel runs in a SEPARATE translation unit that does NOT
// include simpleconv2d.h (the header pulls in host-only xil_cache.h/xiltimer.h
// that the AIE chess compiler can't build). The kernel extractor only carries
// #defines written DIRECTLY in this .cc into the kernel TU. Because RowBC_spatial
// is now a lean GemmSpace (carrying only the halo split + raw [H,W*C] shape) it
// no longer provides the conv-geometry get_*() builtins, so mirror the geometry
// here as literals. These MUST match simpleconv2d.h.
#define SP_KH 7   // KERNEL_H
#define SP_KW 7   // KERNEL_W
#define SP_C 4    // INPUT_C_ALIGN (channel layout stride; real cin=3, +1 zero pad)
#define SP_S 2    // STRIDE
#define SP_OW 112 // OUTPUT_W
#define SP_K 196  // KH*KW*INPUT_C_ALIGN = 7*7*4
#define SP_WC                                                                                                          \
    896         // INPUT_W * INPUT_C_ALIGN = 224*4 (FULL unpadded row; NOTE:
                // with the 2D width-split the conv2d_spatial kernel uses the
                // PER-CHUNK row width TILE_W*SP_C=244, not SP_WC. SP_WC is kept
                // for host-side/reference geometry only.)
#define PAD_W 3 // PAD
#define PAD_H 3 // PAD
// Output tile geometry as #defines (not constexpr) so the kernel TU — which only
// carries #defines from this .cc — also sees them (TILE_H/SP_OHR expand to use
// OH_T below).
#define OH_T 7  // Output tile High (112 / (HW_ROWS*OH_T) = 112/(4*7) = 4 slabs/core, clean integer tiling)
#define OW_T 28 // Output tile Width
// Spatial-halo tile descriptor (drives both RowBC_spatial.d1 and SP_OHR so the
// per-tile slab supply == kernel output-row demand):
#define PAD_H_LO 3                              // top padding (first tile only)
#define PAD_H_HI 3                              // bottom padding (last tile only)
#define TILE_H ((OH_T - 1) * SP_S + SP_KH)      // 19 per-tile input rows (halo slice): (7-1)*2+7
#define TILE_STRIDE_H (TILE_H - (SP_KH - SP_S)) // 11 halo step between tiles: 16-(7-2)
// Per-tile output rows produced from a TILE_H-row slab. The DDR is pre-padded,
// so the slab is a window of the already-padded buffer (top pad materialized) —
// do NOT re-add PAD_H_LO here:
#define SP_OHR ((TILE_H - SP_KH) / SP_S + 1) // = 7 (= OH_T): (19-7)/2+1
// Spatial-halo WIDTH descriptor (drives RowBC_spatial.d2 — the width dim is now
// split across mesh COLS the same way d1 splits height across mesh ROWS):
#define PAD_W_LO 3                              // left padding (first col tile only)
#define PAD_W_HI 3                              // right padding (last col tile only)
#define TILE_W ((OW_T - 1) * SP_S + SP_KW)      // per-tile input cols (width halo slice)
#define TILE_STRIDE_W (TILE_W - (SP_KW - SP_S)) // halo step between width tiles (overlap = 61-56 = KW-S = 5)
// Per-tile output cols produced from a TILE_W-col slab (DDR pre-padded; do NOT
// re-add PAD_W_LO — mirrors SP_OHR). Unused: kernel uses ow_dim = OW_T directly.
#define SP_OWR ((TILE_W - SP_KW) / SP_S + 1)
// Tiling logic
// ifm
constexpr int IH_T = (OH_T - 1) * SP_S + SP_KH - PAD_H;
constexpr int IW_T = (OW_T - 1) * SP_S + SP_KW - PAD_W;
constexpr int HALO = SP_KH - SP_S; // = 21 ✓
constexpr int OC_PER_G = 16;       // 64 / 4 (tile cols per filter group, for N=64 and HW_COLS=4)
// INPUT_C_ALIGN is #defined in simpleconv2d.h (host TU). Mirror it here as a
// #define so the kernel TU — which does NOT include the header — also sees it
// (used by ColBC/RowBC space descriptors below). Identical-value redefinition
// across the two TUs is harmless.
#ifndef INPUT_C_ALIGN
#define INPUT_C_ALIGN 4
#endif

// kernel
constexpr int KERNEL_ROWS = OC_PER_G;             // 16
constexpr int KERNEL_COLS = SP_KH * SP_KW * SP_C; // 64
// ═══════════════════════════════════════════════════════════════════════════
// Composition-based Spatial Spaces (Conv2dSpace + SpatialPolicy)
//
// Conv2dSpace composes a generic SpatialPolicy with the conv iteration space.
// ih/iw carry the EXACT input spatial dims; the compiler derives OH/OW and the
// shim-DMA layout from these dims + policy.mode (no explicit DmaTransform):
//   policy.mode == Partition -> im2col multi-dim BD (DMA extracts patches)
//   policy.mode == Overlap   -> spatial-halo (DMA sends raw overlapping slabs)
//
// Distribution strategy (same as GEMM):
//   A (input):          Broadcast per Row — all tiles in a row see same input
//   B (filter weights):  Broadcast per Col — all tiles in a column see same filter
//   C (output):          Gather per Row   — tile outputs merged left-to-right
// ═══════════════════════════════════════════════════════════════════════════

// A-input space, im2col model: Partition mode -> compiler derives im2col BD.
constexpr aie::Conv2dSpace RowBC_im2col = {
    .policy = {.map = {.act = aie::Pattern::Broadcast, .layout = aie::Layout::Row},
               .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::Dma},
               .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{4096}}},
    .ih = {.tile_size = INPUT_H},
    .iw = {.tile_size = INPUT_W},
    .ic = {.tile_size = INPUT_C},
    .oc = {.tile_size = NUM_FILTERS},
    .kh = {.tile_size = KERNEL_H},
    .kw = {.tile_size = KERNEL_W},
    .stride = STRIDE,
    .pad = PAD};

// A-input space, spatial-halo model: a DECLARATIVE Conv2dSpace_Spatial. Instead
// of hand-computing the d1/d2/d3 halo split, the user describes the RAW conv
// geometry (geom), the desired OUTPUT tile (out_tile_h/out_tile_w), and an
// objective; the compiler derives the spatial-halo split deterministically:
//   halo_slice = (out_tile_h-1)*S + K - pad_hi = (8-1)*2 + 7 - 3 = 18 input rows
//   halo_step  = halo_slice - (K - S)          = 18 - (7-2)     = 13 input rows
//   raw_h  = in_h               = 224
//   raw_wc = in_w * cin_aligned = 224 * 4 = 896
// The conv2d_spatial kernel reads kernel/stride/output geometry from the
// compile-time SP_* macros. `cin_aligned` is now FUNCTIONAL: it sets the
// channel-layout stride (raw_wc, full_k, filter K). The host DDR pre-pads the
// input and filter with a zero channel (cin=3 -> 4); the padded channel
// contributes 0 to every dot product, so the output is bit-identical to the
// unaligned cin=3 run. `objective` remains metadata only (no autotuner).
// Lean GemmSpace form: RowBC_spatial is now a generic GemmSpace whose d1
// describes the HEIGHT halo split AND carries the conv kernel window (win) +
// stride (win_stride). The compiler's GemmSpace spatial-halo path derives the
// conv tiling (kernel_h/stride/oh_per_row) from d1.{tile_size,stride,fullsize,
// pad_lo,pad_hi,win,win_stride} + d2 — no separate Conv2dSpace_Spatial needed.
//   d1 = HEIGHT halo (split across mesh ROWS):
//        tile_size  = halo_slice = TILE_H        = 18 input rows
//        stride     = halo_step  = TILE_STRIDE_H = 13 input rows
//        fullsize   = INPUT_H = 224 (full input H, raw — no pad)
//        pad_lo/pad_hi = top/bottom padding (boundary slabs)
//        win        = SP_KH = 7 (conv kernel window height)  <- NEW
//        win_stride = SP_S  = 2 (conv stride)                <- NEW
//   d2 = WIDTH halo (split across mesh COLS): per-tile slice of the W dim with
//        its own kernel window (win=KW) + stride. The compiler folds the channel
//        dim (d3) into raw_wc = d2.fullsize * d3.fullsize = 224 * 4 = 896.
//   d3 = CHANNEL: real cin=3 padded to INPUT_C_ALIGN=4 (zero channel). The
//        padded count (fullsize=4) is the channel-layout stride; pad_hi=1 records
//        the single zero pad channel the host materializes in DDR.
// DDR is pre-padded, so the slab already contains materialized top/left pad; the
// per-tile output count does NOT re-add pad_lo:
// Derived oh_per_row = (TILE_H - win)/win_stride + 1 = (19-7)/2+1 = 7,
// which == OH_T so slab supply matches the kernel's per-tile output-row demand.
// Derived ow_per_col = OW_T = 28 (kernel uses ow_dim = OW_T directly).
constexpr aie::GemmSpace RowBC_spatial = {
    .policy = {.map = {.act = aie::Pattern::Broadcast, .layout = aie::Layout::Row},
               .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::Dma},
               .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{4096}}},
    // d1 = HEIGHT halo across mesh ROWS. Outer slice=61 rows stepping 56 over the
    // padded height base=230 (=INPUT_H+2*PAD_H) -> 4 mesh rows. (Symmetric with the
    // width tile below since the input is square.) Each 61-row slab is chunked
    // on-core into 4 L2 rounds of TILE_H=19 rows stepping TILE_STRIDE_H=14
    // (overlap = 19-14 = SP_KH-SP_S). Coverage: (4-1)*14+19 = 61.
    .d1 = {.fullsize = INPUT_H + PAD_H_LO + PAD_H_HI, // 230 padded H
           .tile_round = 4,
           .tile_size = 61,                          // outer height slice (rows per mesh row)
           .stride = 56,                             // outer height step
           .slice_tiling = {.tile_size = TILE_H,     // 19 rows per on-core round
                            .stride = TILE_STRIDE_H, // 14 row step between rounds
                            .rounds = 4}},
    // d2 = WIDTH K-accum split across mesh COLS: 61-col chunks stepping 56 over the
    // padded width base=230; combined with the channel stride (d3) this yields the
    // d1 routing.level slice=244/step=224/rounds=4 and padded row pitch = 230*4=920.
    .d2 = {.fullsize = INPUT_W + PAD_W_LO + PAD_W_HI, // 230 padded W
           .tile_round = 4,
           .tile_size = TILE_W,                  // 61 per-chunk input cols (width halo slice)
           .stride = TILE_STRIDE_W},             // 56 width halo step between chunk
    .d3 = {.fullsize = INPUT_C_ALIGN,            // 4 channel not split (per-tile == full stride)
           .tile_size = INPUT_C_ALIGN,           // 4 padded per-tile channel coverage (layout stride)
           .padsize = INPUT_C_ALIGN - INPUT_C}}; // 1 zero pad channel (3 -> 4)

// Filter (B) and output (C) carry generic policies (not conv inputs).
// B=[K, tile_N] described per-port via d1/d2 (a 2D operand has only 2 dims):
//   d1 = N-tile, d2 = K.
constexpr aie::GemmSpace ColBC = {
    .policy = {.map = {.wgt = aie::Pattern::Broadcast, .layout = aie::Layout::Col},
               .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::Dma},
               .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{4096}}},
    .d1 = {.fullsize = OC_PER_G * HW_COLS,
           .tile_round = HW_COLS,
           .tile_size = OC_PER_G,
           .stride = OC_PER_G},                                     // N-tile (divides K for B^T layout)
    .d2 = {.fullsize = SP_KW, .tile_size = SP_KH, .stride = SP_KW}, // N-tile (divides K for B^T layout)
    .d3 = {.fullsize = INPUT_C_ALIGN,
           .tile_size = INPUT_C_ALIGN,
           .stride = INPUT_C_ALIGN,
           .padsize = INPUT_C_ALIGN - INPUT_C}}; // K

constexpr aie::GemmSpace LtoR_Merge = {
    .policy = {.map = {.layout = aie::Layout::Row, .merge_order = aie::Flow::LeftToRight},
               .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
               .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{4096}}},
    .d1 = {.tile_size = OH_T, .stride = OH_T}, // M-tile (output tile rows)
    .d2 = {.tile_size = OW_T, .stride = OW_T},
    .d3 = {.tile_size = OC_PER_G, .stride = OC_PER_G},
}; // N-tile (output tile cols)
// Note: C output space must be described with the FULL output tile size, not the per-tile sub-split, because the
// compiler needs to know the full tile coverage

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL: conv2d_spatial
//
// Receives the RAW input slab in win_a as [halo_slice, W*C] (overlapping
// row-block owned by this tile-row), the filter [K, tile_N] in win_b, and
// produces [oh_per_row * OW, tile_N] output rows. The kernel does the
// sliding-window extraction (on-chip im2col) itself, then matmul.
//
// The spatial-halo DMA (raw overlapping row-blocks) is DERIVED by the compiler
// from RowBC_spatial (Overlap mode + exact ih/iw): each tile-row owns OH/R
// output rows, needing ((OH/R)-1)*S + KH input rows (halo_slice), advancing by
// (OH/R)*S input rows per tile-row (halo_step). The shim BD stays flat; overlap
// is realized via per-tile DDR base offsets.
//
// Conv params are resolved to integer literals at compile time:
//   get_kernel_h/get_kernel_w/get_input_c/get_stride/get_ow/get_oh_per_row.
// ═══════════════════════════════════════════════════════════════════════════
constexpr aie::GlobalPolicy conv_policy = {.fullconnect_auto = 0};
__global__(conv_policy) void conv2d_spatial(
    aie::port<input_window_int8 *, RowBC_spatial> win_a, // raw slab [halo_slice, W*C] (derived)
    aie::port<input_window_int8 *, ColBC> win_b,         // filter [K, tile_N]
    aie::port<output_window_int8 *, LtoR_Merge> win_c    // output [oh_per_row*OW, tile_N]
) {
    // Conv geometry comes from the kernel-visible SP_* literals defined at the
    // top of this file, NOT from conv-specific port helpers — RowBC_spatial is
    // now a lean GemmSpace that only carries the spatial-halo split
    // (size/stride) + raw [H, W*C] shape.
    const int kh_dim = SP_KH;
    const int kw_dim = SP_KW;
    const int c_dim = SP_C;
    const int stride = SP_S;
    // 2D WIDTH-SPLIT: the conv WIDTH is chunked into on-core rounds (NOT a mesh
    // axis). Each round delivers a NARROW slab [halo_slice, TILE_W*C] cut from the
    // PADDED DDR buffer and produces a per-chunk output tile of OW_T cols. The host
    // streams H_chunks*W_chunks slabs (m_rounds = spatialMRounds * w_rounds) and the
    // 2D shim BDs handle the (hc,wc) base-offset stepping, so every round is uniform
    // here (the left/top pad is already baked into the pre-padded DDR buffer).
    const int ow_dim = OW_T;                    // per-chunk output cols (28), was SP_OW=112
    const int oh_per_row = SP_OHR;              // OUTPUT_H / HW_ROWS
    const int k_dim = SP_K;                     // KH*KW*C
    const int tile_cols = aie::get_tile_cols(); // tile_N (GEMM N tiling)

    const int num_a_rounds = aie::get_num_rounds(win_a);
    const int num_b_rounds = aie::get_num_rounds(win_b);
    const int num_c_rounds = aie::get_num_rounds(win_c);
    const int buf_sz_a = aie::get_buffer_size(win_a); // halo_slice * (W*C)
    const int buf_sz_c = aie::get_buffer_size(win_c);

    // Number of spatial M sub-tiles (slabs) this core processes. Each slab is a
    // contiguous overlapping halo block that yields one [oh_per_row*OW, tile_N]
    // output tile. The host streams `m_rounds` slabs and expects `m_rounds`
    // output tiles back, so the whole receive->compute->emit sequence repeats
    // once per slab (mirrors the matmul kernel's outer mr loop).
    const int m_rounds = aie::get_spatial_multiple_rounds(win_a);

    // Per-chunk input row width (TILE_W*C). With the 2D width-split each round's
    // slab is the NARROW block [halo_slice, TILE_W*C] = [19, 61*4=244], delivered
    // by a 2D shim BD (contiguous run 244, row pitch = padded INPUT_W_PAD*C). The
    // im2col indexing below reads within this narrow slab: ih = oh*S+kh < halo_slice,
    // iw = ow*S+kw < TILE_W. (Was SP_WC=896, the full unpadded row.)
    const int raw_wc = TILE_W * SP_C;

    const int buf_sz_b = aie::get_buffer_size(win_b);

    int8_t slab[buf_sz_a]; // raw input slab [halo_slice, raw_wc]
    int8_t local_out[oh_per_row * ow_dim * tile_cols];

    // ===== Receive B (filter) ONCE, before the slab loop =====
    // The filter is streamed a single time (win_b num_rounds == 1). Acquiring it
    // inside the mr loop would over-acquire the lock (m_rounds times) against a
    // producer that releases it once -> DMA/lock stall. Copy it into a persistent
    // local buffer so every slab reuses the same filter without re-acquiring.
    int8_t B_local[num_b_rounds * buf_sz_b];
    for (int rb = 0; rb < num_b_rounds; rb++) {
        int8_t *B_ptr = (int8_t *)acquire_input_window(win_b);
        for (int i = 0; i < buf_sz_b; i++)
            B_local[rb * buf_sz_b + i] = B_ptr[i];
        release_input_window(win_b);
    }

    // ===== M sub-tile loop: one slab -> one output tile per iteration =====
    for (int mr = 0; mr < m_rounds; mr++) {

        // ===== Phase 1: receive the raw input slab =====
        for (int ra = 0; ra < num_a_rounds; ra++) {
            int8_t *A_ptr = (int8_t *)acquire_input_window(win_a);
            for (int i = 0; i < buf_sz_a; i++)
                slab[ra * buf_sz_a + i] = A_ptr[i];
            release_input_window(win_a);
        }

        // ===== Phase 2: on-chip windowing (im2col) + matmul (reads B_local) =====
        for (int oh = 0; oh < oh_per_row; oh++) {
            for (int ow = 0; ow < ow_dim; ow++) {
                for (int j = 0; j < tile_cols; j++) {
                    int16_t sum = 0;
                    // local im2col: gather KH*KW*C patch from the slab
                    int kk = 0;
                    for (int kh = 0; kh < kh_dim; kh++) {
                        for (int kw = 0; kw < kw_dim; kw++) {
                            for (int c = 0; c < c_dim; c++) {
                                int ih = oh * stride + kh;
                                int iw = ow * stride + kw;
                                int8_t iv = slab[ih * raw_wc + iw * c_dim + c];
                                int8_t fv = B_local[j * k_dim + kk];
                                sum += (int16_t)iv * (int16_t)fv;
                                kk++;
                            }
                        }
                    }
                    if (sum > 127)
                        sum = 127;
                    else if (sum < -128)
                        sum = -128;
                    local_out[(oh * ow_dim + ow) * tile_cols + j] = (int8_t)sum;
                }
            }
        }

        // ===== Phase 3: output =====
        for (int rc = 0; rc < num_c_rounds; rc++) {
            int8_t *out = (int8_t *)acquire_output_window(win_c);
            for (int i = 0; i < buf_sz_c; i++)
                out[i] = local_out[rc * buf_sz_c + i];
            release_output_window(win_c);
        }
    } // end m_rounds
}

// ═══════════════════════════════════════════════════════════════════════════
// HOST
// ═══════════════════════════════════════════════════════════════════════════
int main() {
    printf("=== Conv2d via Im2col on AIE %dx%d Mesh ===\n", HW_ROWS, HW_COLS);
    printf("    Input:  [%d, %d, %d]\n", INPUT_H, INPUT_W, INPUT_C);
    printf("    Filter: [%d, %d, %d, %d]\n", KERNEL_H, KERNEL_W, INPUT_C, NUM_FILTERS);
    printf("    Output: [%d, %d, %d]\n", OUTPUT_H, OUTPUT_W, NUM_FILTERS);
    printf("    Im2col GEMM: [%d, %d] x [%d, %d] = [%d, %d]\n", M, K, K, N, M, N);

    // --- Device + mesh ---
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({3, 6, 0, 6}, HW_ROWS, HW_COLS);

    // --- Allocate host memory (DDR pre-pad: spatial border + channel layout) ---
    // Two independent pads are materialized in DDR (AIE DMA can only relocate
    // bytes, so any zero region must exist physically):
    //   * channel pad: cin (=3) -> INPUT_C_ALIGN (=4) by inserting a zero channel.
    //   * spatial pad: PAD zero pixels on every H/W border -> [INPUT_H_PAD,
    //     INPUT_W_PAD, INPUT_C_ALIGN]. Real pixel (h,w) sits at (h+PAD, w+PAD), so
    //     the CPU reference windows (oh*S+kh, ow*S+kw) index the padded buffer
    //     directly and implement true padded conv (no out-of-bounds reads).
    int8_t *input = (int8_t *)malloc(INPUT_H_PAD * INPUT_W_PAD * INPUT_C_ALIGN * sizeof(int8_t));
    // Filter in B^T [N, K] layout with K = KH*KW*INPUT_C_ALIGN (matches kernel
    // B_ptr[f*K + ((kh*KW+kw)*INPUT_C_ALIGN + c)]).
    int8_t *filter = (int8_t *)malloc(KERNEL_H * KERNEL_W * INPUT_C_ALIGN * NUM_FILTERS * sizeof(int8_t));
    // Output in [OH, OW, F] layout = [M, N] for GEMM (unchanged by alignment)
    int8_t *output = (int8_t *)malloc(OUTPUT_H * OUTPUT_W * NUM_FILTERS * sizeof(int8_t));

    // --- Initialize test data ---
    // Input: sequential values for the real channels (identical to the cin=3 run),
    // zero for the padding channel and the spatial-border pad. Real value at
    // (h,w,c) keeps the old flat index ((h*W+w)*INPUT_C + c) + 1 so the output is
    // bit-identical to the cin=3 case; the pixel is written at the padded position
    // (h+PAD, w+PAD) with the padded row pitch INPUT_W_PAD.
    memset(input, 0, INPUT_H_PAD * INPUT_W_PAD * INPUT_C_ALIGN * sizeof(int8_t));
    for (int h = 0; h < INPUT_H; h++) {
        for (int w = 0; w < INPUT_W; w++) {
            for (int c = 0; c < INPUT_C; c++) {
                int real_idx = (h * INPUT_W + w) * INPUT_C + c;
                input[((h + PAD) * INPUT_W_PAD + (w + PAD)) * INPUT_C_ALIGN + c] = (int8_t)(real_idx + 1);
            }
            // padding channels [INPUT_C, INPUT_C_ALIGN) and spatial border left at 0
        }
    }

    // Filter: all 1s for the real channels, 0 for the padding channel. Stored in
    // B^T [N, K] layout. The padding weight is zeroed for defense-in-depth (the
    // zero input channel already nulls every padding-channel product).
    memset(filter, 0, KERNEL_H * KERNEL_W * INPUT_C_ALIGN * NUM_FILTERS * sizeof(int8_t));
    for (int f = 0; f < NUM_FILTERS; f++) {
        for (int kh = 0; kh < KERNEL_H; kh++) {
            for (int kw = 0; kw < KERNEL_W; kw++) {
                for (int c = 0; c < INPUT_C; c++) {
                    int kk = (kh * KERNEL_W + kw) * INPUT_C_ALIGN + c;
                    filter[f * K + kk] = 1;
                }
                // padding channel c == INPUT_C..INPUT_C_ALIGN-1 left at 0
            }
        }
    }

    memset(output, 0, OUTPUT_H * OUTPUT_W * NUM_FILTERS * sizeof(int8_t));

    // --- CPU sanity check: im2col + matmul == naive conv2d ---
    printf("\n--- CPU Sanity Check ---\n");
    int sanity = verify_im2col_equivalence(input, filter);
    if (sanity != 0) {
        printf("ERROR: im2col equivalence check failed, aborting.\n");
        free(input);
        free(filter);
        free(output);
        return 1;
    }

    // --- Launch kernel on AIE mesh ---
    // The compiler pipeline (buildConv2dRoutingIR) will:
    //   1. Map conv2d params to GEMM: A[36,9], B[9,1], C[36,1]
    //   2. Attach conv2d.im2col_config module attribute
    //   3. DmaphopTodfscheblueprintPass sets multi-dim BD on input shim DMA
    //   4. Core tiles run this kernel as standard matmul
    // --- Select tiling model ---
    //   Model 1 (default): im2col — DMA extracts patches, kernel is pure matmul.
    //   Model 2 (-DUSE_SPATIAL_HALO): spatial halo — DMA sends raw IFM slabs,
    //                                 kernel does on-chip windowing + matmul.
    printf("\n--- Launching conv2d_spatial (spatial-halo) on AIE ---\n");
    conv2d_spatial<<<mesh>>>(input, filter, output, M, N, K);

    // --- Wait and verify ---
    // device.synchronize();

    printf("\n--- Verification ---\n");
    int result = verify_conv2d(input, filter, output);
    uint32_t timeout = 0;
    while (result > 0) {
        sleep(1);
        timeout++;
        printf("ERROR: conv2d output mismatch,app waiting for debug... ( %u second) \n", timeout);
    }
    // --- Cleanup ---
    free(input);
    free(filter);
    free(output);

    return result;
}
