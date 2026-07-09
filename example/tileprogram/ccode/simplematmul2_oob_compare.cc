/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * AIE OOB Comparison Benchmark — int32 GEMM, 8 matrices.
 *
 * Matches the OOB example_oob_4x4 workload parameters:
 *   - Data type:    int32  (OOB uses uint32; same 32-bit width, signed variant)
 *   - Matrix size:  256x256x256 per matrix  (OOB PANEL_SIZE=256, NUM_PANELS=1)
 *   - Tile array:   4x4 = 16 compute tiles  (OOB NUM_HW_ROWS=4, NUM_HW_COLS=4)
 *   - Matrices:     8 back-to-back GEMMs    (OOB NUM_MATRICES=8)
 *   - GFLOPS:       8 * 2 * 256^3 / wall_s / 1e9 (exact OOB formula)
 *
 * Kernel: int32 mac+reduce_add with int64 accumulator (same structure as exp58,
 * scaled to int32). Uses col-major B (no in-kernel packing). KCHUNK=4 gives
 * 4-wide int32 vectors; aie::mul(int32x4, int32x4) → acc64x4 on AIE2PS.
 *
 * Primary metric: wall GFLOPS (same as OOB) + PMCCNTR cycles per matrix.
 * [PERF] summary block matches simplematmul2_prof.cc format for easy grep.
 ******************************************************************************/
#define M 256
#define K 256
#define N 256
#define HW_ROWS 4
#define HW_COLS 4
#define NUM_MATRICES 8
// Per-DMA-round sub-tile granularity (int32 = 4 bytes per element).
// Per-tile sub-problem: (M/HW_ROWS)=64 rows, (N/HW_COLS)=64 cols, K=256.
// A window: TILE_M=4 rows × KCHUNK=4 K-elements = 16 int32 = 64 bytes (row-major)
// B window: TILE_N=4 cols × KCHUNK=4 K-elements = 16 int32 = 64 bytes (col-major: B[col][k])
// C window: TILE_M=4 rows × TILE_N=4 cols        = 16 int32 = 64 bytes
#define TILE_M 4
#define TILE_N 4
#define KCHUNK 4
#include "simplematmul.h"
#pragma aie_debug_level(0 | AIE_DEBUG_FLAG_DISABLE_PARTITIONTEARDOWN | AIE_DEBUG_FLAG_MM2SBDFINISH_COUNTER |           \
                        AIE_DEBUG_FLAG_CORE_PERF_COUNTER)

extern void __Runtime_core_perf_read_probe(uint32_t *active, uint32_t *vec_instr, uint32_t *stream_stall,
                                           uint32_t *lock_stall);
extern void __Runtime_perfcnt_read_mm2s_probe(uint32_t *ch0, uint32_t *ch1);
extern int __Runtime_core_perf_probe_valid(void);
extern void __Runtime_wait_io_cycles(unsigned long long *cycles, unsigned int *calls);
extern void __Runtime_phase_cycles(unsigned long long *cyc, unsigned int *calls);
extern void __Runtime_kload_split_cycles(unsigned long long *elf_cyc, unsigned int *elf_n, unsigned long long *rst_cyc,
                                         unsigned int *rst_n);
extern void __Runtime_wait_io_iters(unsigned long long *iters);

// Spatial policies: same broadcast layout as simplematmul2_prof.cc, scaled for int32.
//   win_a: A[M,K] -> d1=TILE_M rows, d2=KCHUNK K-elements, int32
//   win_b: B[N,K] -> d1=TILE_N cols, d2=KCHUNK K-elements, int32 (K-major: b[k][n])
//   win_c: C[M,N] -> d1=TILE_M rows, d2=TILE_N cols, int32
constexpr aie::GemmSpace RowBA = {
    .policy = {.map = {.act = aie::Pattern::Broadcast, .layout = aie::Layout::Row},
               .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
               .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{4096}}},
    .d1 = {.tile_size = TILE_M, .stride = TILE_M, .fullsize = M, .pad_hi = 0, .pad_lo = 0},
    .d2 = {.tile_size = KCHUNK, .stride = KCHUNK, .fullsize = K, .pad_hi = 0, .pad_lo = 0}};
constexpr aie::GemmSpace ColBB = {.policy = {.map = {.wgt = aie::Pattern::Broadcast, .layout = aie::Layout::Col},
                                             .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
                                             .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{4096}}},
                                  .d1 = {.tile_size = TILE_N, .stride = TILE_N, .fullsize = N},
                                  .d2 = {.tile_size = KCHUNK, .stride = KCHUNK, .fullsize = K}};
constexpr aie::GemmSpace LtoR_Merge = {
    .policy = {.map = {.layout = aie::Layout::Row, .merge_order = aie::Flow::LeftToRight},
               .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
               .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{4096}}},
    .d1 = {.tile_size = TILE_M, .stride = TILE_M, .fullsize = M},
    .d2 = {.tile_size = TILE_N, .stride = TILE_N, .fullsize = N}};

// ─── KERNEL: per-tile int32 GEMM — mac+reduce_add, matching exp58 structure ──
//
// Uses int32 operands and int64 accumulator instead of int8/int32 (exp58).
// B arrives col-major: B[j*K + k] delivered as B_ptr[col * eff_k + k].
// No in-kernel packing needed — same layout as the working int8 kernel.
//
// Per output element (i, j):
//   acc = sum_k A[i][k] * B[j][k]
//
// KCHUNK=4 gives 4 K elements per window — use aie::mul(a_vec, b_vec) where
// a_vec and b_vec are 4-wide int32 vectors, accumulating acc32 (or acc64).
// int32×int32 → acc64 on AIE2PS; aie::mul<int32,int32> returns acc64.
//
__global__ void matmul(aie::port<input_window_int32 *, RowBA> win_a, aie::port<input_window_int32 *, ColBB> win_b,
                       aie::port<output_window_int32 *, LtoR_Merge> win_c) {
    const int tile_rows = aie::get_tile_rows(); // TILE_M = 4
    const int tile_cols = aie::get_tile_cols(); // TILE_N = 4
    const int eff_k = aie::get_effective_k();   // KCHUNK = 4
    const int k_rounds = aie::get_k_rounds();   // K / KCHUNK = 64
    const int num_a_rounds = aie::get_num_rounds(win_a);
    const int num_b_rounds = aie::get_num_rounds(win_b);
    const int num_c_rounds = aie::get_num_rounds(win_c);
    const int buf_sz_a = aie::get_buffer_size(win_a); // TILE_M * KCHUNK = 16 int32
    const int buf_sz_c = aie::get_buffer_size(win_c); // TILE_M * TILE_N = 16 int32
    const int m_rounds = aie::get_spatial_multiple_rounds(win_a);
    const int n_rounds = aie::get_spatial_multiple_rounds(win_b);
    const int cols_per_b = aie::get_buffer_size(win_b) / eff_k; // TILE_N = 4

    // Cached A tile: TILE_M rows × KCHUNK K-elements, row-major
    alignas(aie::vector_decl_align) int32_t all_A[tile_rows * eff_k]; // 4×4 = 16 int32
    // int64 accumulator — avoids overflow for int32 * int32 partial sums across 256 K-steps
    int64_t acc_buf[tile_rows * tile_cols];

    for (int mr = 0; mr < m_rounds * n_rounds; mr++) {
        for (int idx = 0; idx < tile_rows * tile_cols; idx++)
            acc_buf[idx] = 0LL;

        for (int kr = 0; kr < k_rounds; kr++) {
            // Cache A tile: TILE_M=4 rows × KCHUNK=4 K-elements
            for (int ra = 0; ra < num_a_rounds; ra++) {
                int32_t *A_ptr = (int32_t *)acquire_input_window(win_a);
                for (int i = 0; i < buf_sz_a; i++)
                    all_A[ra * buf_sz_a + i] = A_ptr[i];
                release_input_window(win_a);
            }

            for (int rb = 0; rb < num_b_rounds; rb++) {
                int32_t *B_ptr = (int32_t *)acquire_input_window(win_b);
                // B_ptr: col-major B[cols_per_b=4][eff_k=4]: B_ptr[col * eff_k + k]

                for (int i = 0; i < tile_rows; i++) {
                    // Load 4-wide A slice for row i: all_A[i][0:4]
                    aie::vector<int32, 4> a_row = aie::load_v<4>(all_A + i * eff_k);

                    for (int j = 0; j < cols_per_b; j++) {
                        // Load 4-wide B col slice: B_ptr[j][0:4]
                        aie::vector<int32, 4> b_col = aie::load_v<4>(B_ptr + j * eff_k);
                        // int32×int32 → acc64, reduce 4 lanes to scalar
                        aie::accum<acc64, 4> prod = aie::mul(a_row, b_col);
                        acc_buf[i * tile_cols + j] += aie::reduce_add(prod.to_vector<int64>());
                    }
                }
                release_input_window(win_b);
            }
        }

        // Saturate int64 → int32 and write output
        int32_t local_out[tile_rows * tile_cols];
        for (int idx = 0; idx < tile_rows * tile_cols; idx++) {
            int64_t v = acc_buf[idx];
            local_out[idx] = (v > (int64_t)0x7FFFFFFF)      ? (int32_t)0x7FFFFFFF
                             : (v < (int64_t)-0x80000000LL) ? (int32_t)-0x80000000LL
                                                            : (int32_t)v;
        }

        for (int rc = 0; rc < num_c_rounds; rc++) {
            int32_t *out = (int32_t *)acquire_output_window(win_c);
            const int rows_per_c_round = buf_sz_c / tile_cols;
            for (int i = 0; i < rows_per_c_round; i++)
                for (int j = 0; j < tile_cols; j++)
                    out[i * tile_cols + j] = local_out[rc * rows_per_c_round * tile_cols + i * tile_cols + j];
            release_output_window(win_c);
        }
    }
}

// Correctness check for int32 output.
// B is col-major: B[j*K + k].
// Golden: C[i][j] = saturate(sum_k A[i][k] * B[j][k])
static int oob_verify(const int32_t *A, const int32_t *B, const int32_t *C) {
    int mismatches = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int64_t s = 0;
            for (int k = 0; k < K; k++)
                s += (int64_t)A[i * K + k] * (int64_t)B[j * K + k];
            int32_t expected;
            if (s > (int64_t)0x7FFFFFFF)
                expected = (int32_t)0x7FFFFFFF;
            else if (s < (int64_t)-0x80000000LL)
                expected = (int32_t)-0x80000000LL;
            else
                expected = (int32_t)s;
            if (C[i * N + j] != expected) {
                if (mismatches < 8)
                    printf("  mismatch C[%d,%d] got %d exp %d\n", i, j, C[i * N + j], expected);
                mismatches++;
            }
        }
    }
    if (mismatches == 0)
        printf("RESULT: PASS (all %d elements match)\n", M * N);
    else
        printf("RESULT: FAIL (%d / %d mismatches)\n", mismatches, M * N);
    return mismatches;
}

// HOST
int main() {
    printf("\n=== aiehlc OOB Comparison Benchmark (int32, 8 matrices, mmul<4,4,4>) ===\n");
    printf("  C[%dx%d] = A[%dx%d] * B[%dx%d], int32, %dx%d mesh (%d tiles), %d matrices\n", M, N, M, K, K, N, HW_ROWS,
           HW_COLS, HW_ROWS * HW_COLS, NUM_MATRICES);
    printf("  OOB reference: example_oob_4x4 (PANEL_SIZE=256, NUM_MATRICES=8, aie::mmul<4,4,4,uint32,uint32>)\n");
    printf("  Kernel: int32 mac+reduce_add (col-major B, no in-kernel packing)\n");

    __ps_pmccntr_enable();
    unsigned long long pc_init0 = __ps_pmccntr();
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({0, 3, 0, 5}, HW_ROWS, HW_COLS);
    unsigned long long pc_init1 = __ps_pmccntr();

    unsigned long long pc_setup0 = __ps_pmccntr();
    // Single A and B for all 8 matrices (same data repeated, matching OOB single-panel 8× repeat).
    // B is col-major B[j*K + k] — no pre-transposition needed; kernel handles col-major directly.
    int32_t *A = (int32_t *)device.alloc(M * K * sizeof(int32_t));
    int32_t *B = (int32_t *)device.alloc(K * N * sizeof(int32_t));
    int32_t *C = (int32_t *)device.alloc(M * N * sizeof(int32_t));

    // Initialize A row-major: A[i][k] = (i*K+k) % 7 - 3
    for (int i = 0; i < M * K; i++)
        A[i] = (int32_t)((i % 7) - 3);

    // Initialize B col-major: B[j*K + k] = (j*K+k) % 5 - 2
    for (int i = 0; i < K * N; i++)
        B[i] = (int32_t)((i % 5) - 2);

    // Poison C with 0x5A5A5A5A and flush to DDR
    extern void __Runtime_sync_for_dev(XAie_DevInst * dev, void *ptr, __SIZE_TYPE__ size);
    for (int i = 0; i < M * N; i++)
        C[i] = (int32_t)0x5A5A5A5A;
    __Runtime_sync_for_dev(device._dev, C, M * N * sizeof(int32_t));
    printf("[oob] poisoned C and flushed to DDR\n");

    // Golden reference: C[i][j] = saturate(sum_k A[i][k] * B[j][k])
    // B is col-major: B[j][k] = B[j*K + k]
    static int32_t golden[M * N];
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int64_t s = 0;
            for (int k = 0; k < K; k++)
                s += (int64_t)A[i * K + k] * (int64_t)B[j * K + k];
            if (s > (int64_t)0x7FFFFFFF)
                s = (int64_t)0x7FFFFFFF;
            else if (s < (int64_t)-0x80000000LL)
                s = (int64_t)-0x80000000LL;
            golden[i * N + j] = (int32_t)s;
        }
    }
    unsigned long long pc_setup1 = __ps_pmccntr();

    // ─── Timed window: NUM_MATRICES back-to-back GEMMs ───────────────────────
    // Mirrors OOB's: for (int m = 0; m < NUM_MATRICES; m++) { run_graph(); }
    // ELF loading stays inside the window (charged to the metric, matching our convention).
    const uint64_t MAX_POLL = 500000000ULL;
    XTime t0, t1;
    unsigned long long cv0 = __ps_cntvct();
    unsigned long long pc0 = __ps_pmccntr();
    XTime_GetTime(&t0);

    for (int mat = 0; mat < NUM_MATRICES; mat++) {
        matmul<<<mesh>>>(A, B, C, M, N, K);
        // Each matrix uses the same data (matching OOB's single-panel 8× repeat).
        // The completion barrier below covers the last matrix; intermediate launches
        // are async — exactly as OOB runs them (no per-matrix sync between launches).
    }

    unsigned long long pc_mid = __ps_pmccntr();
    uint64_t polls = 0;
    int complete = 0;
    unsigned long long poll_sync_cyc = 0ULL, poll_cmp_cyc = 0ULL;
    do {
        unsigned long long __ps0 = __ps_pmccntr();
        device.synchronizecpu(C, M * N * sizeof(int32_t));
        poll_sync_cyc += (__ps_pmccntr() - __ps0);
        unsigned long long __pc0 = __ps_pmccntr();
        complete = 1;
        for (int idx = 0; idx < M * N; idx++) {
            if (C[idx] != golden[idx]) {
                complete = 0;
                break;
            }
        }
        poll_cmp_cyc += (__ps_pmccntr() - __pc0);
        polls++;
    } while (!complete && polls < MAX_POLL);
    XTime_GetTime(&t1);
    unsigned long long cv1 = __ps_cntvct();
    unsigned long long pc1 = __ps_pmccntr();

    if (!complete)
        printf("  WARNING: completion barrier hit MAX_POLL without full result\n");

    // ─── Metrics ─────────────────────────────────────────────────────────────
    uint64_t raw_counts = (uint64_t)(t1 - t0);
    uint64_t timer_hz = (uint64_t)COUNTS_PER_SECOND;
    double wall_ms = 1000.0 * (double)raw_counts / (double)timer_hz;
    double wall_us = 1.0e6 * (double)raw_counts / (double)timer_hz;
    double tick_ns = 1.0e9 / (double)timer_hz;
    // OOB GFLOPS formula: NUM_MATRICES * 2 * M * N * K / wall_s / 1e9
    double total_flops = (double)NUM_MATRICES * 2.0 * (double)M * (double)N * (double)K;
    double gflops_wall = (wall_ms > 0.0) ? total_flops / (wall_ms * 1e-3) / 1e9 : 0.0;
    double gflops_per_mat = gflops_wall / NUM_MATRICES; // per-matrix rate (OOB also reports single)

    uint32_t active = 0, vec = 0, sstall = 0, lstall = 0, mm0 = 0, mm1 = 0;
    int have_core = __Runtime_core_perf_probe_valid();
    __Runtime_core_perf_read_probe(&active, &vec, &sstall, &lstall);
    __Runtime_perfcnt_read_mm2s_probe(&mm0, &mm1);

    // Per-tile budget
    double tile_flops_per_mat = 2.0 * (double)(M / HW_ROWS) * (double)(N / HW_COLS) * (double)K;
    double total_budget = (double)active + (double)sstall + (double)lstall;
    double compute_pct = total_budget ? 100.0 * (double)active / total_budget : 0.0;
    double stream_pct = total_budget ? 100.0 * (double)sstall / total_budget : 0.0;
    double lock_pct = total_budget ? 100.0 * (double)lstall / total_budget : 0.0;
    double vec_util = active ? 100.0 * (double)vec / (double)active : 0.0;

    // OOB array peak: 184 TOPS INT8 / 4 = 46 TOPS INT32, scaled to 16/144 tiles = ~5.1 TOPS array
    const double DEVICE_INT8_TOPS = 184.0;
    const int DEVICE_TILES = 144;
    int array_tiles = HW_ROWS * HW_COLS;
    // INT32 peak = INT8_peak / 4 (per AMD AIE2PS spec: INT32 throughput is 1/4 of INT8)
    double array_peak_int32_gops = (DEVICE_INT8_TOPS / 4.0) * 1000.0 * (double)array_tiles / (double)DEVICE_TILES;
    double util_pct = array_peak_int32_gops ? 100.0 * gflops_wall / array_peak_int32_gops : 0.0;

    printf("\n--- Layer 0: pre-launch setup (outside timed window) ---\n");
    printf("  [pmccntr] device_init: %llu cyc\n", (unsigned long long)(pc_init1 - pc_init0));
    printf("  [pmccntr] data_setup:  %llu cyc  (alloc + A/B init + B-transpose + poison-C + golden)\n",
           (unsigned long long)(pc_setup1 - pc_setup0));

    printf("\n--- Layer 1: PS wall-clock (%d matrices, OOB-formula GFLOPS) ---\n", NUM_MATRICES);
    printf("  raw counts:        %llu\n", (unsigned long long)raw_counts);
    printf("  timer freq:        %llu Hz  (1 tick = %.3f ns)\n", (unsigned long long)timer_hz, tick_ns);
    printf("  total time:        %.6f ms  (%.3f us)\n", wall_ms, wall_us);
    printf("  per-matrix time:   %.6f ms\n", wall_ms / NUM_MATRICES);
    printf("  completion polls:  %llu\n", (unsigned long long)polls);
    printf("  wall GFLOPS:       %.3f GOPS  (%d matrices, OOB formula: N*2*M*N*K/wall_s/1e9)\n", gflops_wall,
           NUM_MATRICES);
    printf("  per-matrix GFLOPS: %.3f GOPS  (= wall GFLOPS / %d)\n", gflops_per_mat, NUM_MATRICES);
    {
        unsigned long long cv_raw = (cv1 >= cv0) ? (cv1 - cv0) : 0ULL;
        unsigned long long cv_hz = __ps_cntfrq();
        double cv_ms = cv_hz ? 1000.0 * (double)cv_raw / (double)cv_hz : 0.0;
        printf("  [cntvct] raw: %llu counts  freq: %llu Hz  wall: %.6f ms\n", cv_raw, cv_hz, cv_ms);
    }
    {
        unsigned long long pc_raw = (pc1 >= pc0) ? (pc1 - pc0) : 0ULL;
        unsigned long long pmcr = __ps_pmcr();
        unsigned int d_bit = (unsigned int)((pmcr >> 3) & 1ULL);
        printf("  [pmccntr] raw: %llu cycles  pmcr:0x%llx (D=%u, %s)\n", pc_raw, pmcr, d_bit,
               d_bit ? "counts=CPUcyc/64" : "counts=CPUcyc");
        printf("  [pmccntr] per-matrix: %llu cycles\n", pc_raw / NUM_MATRICES);

        unsigned long long pc_launch = (pc_mid >= pc0) ? (pc_mid - pc0) : 0ULL;
        unsigned long long pc_poll = (pc1 >= pc_mid) ? (pc1 - pc_mid) : 0ULL;
        printf("  [pmccntr] launch: %llu cycles  poll: %llu cycles\n", pc_launch, pc_poll);

        unsigned long long wio_cyc = 0ULL;
        unsigned int wio_calls = 0U;
        __Runtime_wait_io_cycles(&wio_cyc, &wio_calls);
        double wio_pct = (pc_launch > 0) ? 100.0 * (double)wio_cyc / (double)pc_launch : 0.0;
        printf("  [phase] wait_io: %llu cycles over %u calls  (=%.1f%% of launch)\n", wio_cyc, wio_calls, wio_pct);

        unsigned long long wio_iters = 0ULL;
        __Runtime_wait_io_iters(&wio_iters);
        printf("  [wait_io] poll iters: %llu  (%.1f cyc/iter)\n", wio_iters,
               wio_iters ? (double)wio_cyc / (double)wio_iters : 0.0);

        double sync_pct = (pc_poll > 0) ? 100.0 * (double)poll_sync_cyc / (double)pc_poll : 0.0;
        printf("  [poll] synchronizecpu: %llu cyc over %llu calls  (=%.1f%% of poll)\n", poll_sync_cyc,
               (unsigned long long)polls, sync_pct);

        unsigned long long ph[4] = {0, 0, 0, 0};
        unsigned int phc[4] = {0, 0, 0, 0};
        __Runtime_phase_cycles(ph, phc);
        const char *phn[4] = {"kload  ", "bdcfg  ", "coreen ", "startio"};
        for (int i = 0; i < 4; i++) {
            double p = (pc_launch > 0) ? 100.0 * (double)ph[i] / (double)pc_launch : 0.0;
            printf("  [phase] %s: %llu cycles over %u calls  (=%.1f%% of launch)\n", phn[i], ph[i], phc[i], p);
        }

        unsigned long long kelf = 0ULL, krst = 0ULL;
        unsigned int kelfn = 0U, krstn = 0U;
        __Runtime_kload_split_cycles(&kelf, &kelfn, &krst, &krstn);
        double kep = (ph[0] > 0) ? 100.0 * (double)kelf / (double)ph[0] : 0.0;
        double krp = (ph[0] > 0) ? 100.0 * (double)krst / (double)ph[0] : 0.0;
        printf("  [kload] loadelf: %llu cyc  (=%.1f%% of kload)\n", kelf, kep);
        printf("  [kload] corerst: %llu cyc  (=%.1f%% of kload)\n", krst, krp);

        unsigned long long ph_total = ph[0] + ph[1] + ph[2] + ph[3] + wio_cyc;
        unsigned long long unacct = (pc_launch > ph_total) ? (pc_launch - ph_total) : 0ULL;
        printf("  [launch] unaccounted (lock_init+glue): %llu cyc  (=%.1f%%)\n", unacct,
               pc_launch ? 100.0 * (double)unacct / (double)pc_launch : 0.0);
        printf("  [launch] BUDGET (%d matrices):\n", NUM_MATRICES);
        printf("    kload    %10llu  (%.1f%%)\n", ph[0], pc_launch ? 100.0 * (double)ph[0] / (double)pc_launch : 0.0);
        printf("    bdcfg    %10llu  (%.1f%%)\n", ph[1], pc_launch ? 100.0 * (double)ph[1] / (double)pc_launch : 0.0);
        printf("    lockinit %10llu  (%.1f%%)\n", unacct, pc_launch ? 100.0 * (double)unacct / (double)pc_launch : 0.0);
        printf("    startio  %10llu  (%.1f%%)\n", ph[3], pc_launch ? 100.0 * (double)ph[3] / (double)pc_launch : 0.0);
        printf("    coreen   %10llu  (%.1f%%)\n", ph[2], pc_launch ? 100.0 * (double)ph[2] / (double)pc_launch : 0.0);
        printf("    wait_io  %10llu  (%.1f%%)\n", wio_cyc,
               pc_launch ? 100.0 * (double)wio_cyc / (double)pc_launch : 0.0);
        printf("    TOTAL    %10llu  (launch=%llu)\n", ph_total + unacct, pc_launch);
    }

    printf("\n--- Layer 2: DMA stream ---\n");
    printf("  MM2S ch0 BDs done: %u\n", mm0);
    printf("  MM2S ch1 BDs done: %u\n", mm1);

    printf("\n--- Layer 3: AIE core tile cycle budget ---\n");
    if (!have_core)
        printf("  [no probe tile armed]\n");
    printf("  total budget:    %.0f cycles\n", total_budget);
    printf("  active:          %u  (%.2f%%)\n", active, compute_pct);
    printf("  stream stall:    %u  (%.2f%%)\n", sstall, stream_pct);
    printf("  lock stall:      %u  (%.2f%%)\n", lstall, lock_pct);
    printf("  vector instrs:   %u\n", vec);
    printf("  vec utilization: %.1f%%\n", vec_util);

    printf("\n--- Hardware utilization (INT32, OOB yardstick) ---\n");
    printf("  array INT32 peak: %.1f GOPS  (INT8_peak/4 * %d/%d tiles)\n", array_peak_int32_gops, array_tiles,
           DEVICE_TILES);
    printf("  measured (wall):  %.3f GOPS  ->  %.4f%% of array INT32 peak\n", gflops_wall, util_pct);
    printf("  OOB-equivalent:   %.3f GOPS (8 matrices, OOB formula)\n", gflops_wall);

    printf("\n--- Correctness ---\n");
    unsigned long long pc_verify0 = __ps_pmccntr();
    int result = oob_verify(A, B, C);
    unsigned long long pc_verify1 = __ps_pmccntr();

    device.free(A);
    device.free(B);
    device.free(C);

    printf("  [pmccntr] verify: %llu cyc\n", (unsigned long long)(pc_verify1 - pc_verify0));

    // ── Compact [PERF] summary ────────────────────────────────────────────────
    {
        unsigned long long ph[4] = {0, 0, 0, 0};
        unsigned int phc[4] = {0, 0, 0, 0};
        __Runtime_phase_cycles(ph, phc);
        unsigned long long wio_cyc2 = 0ULL;
        unsigned int wio_calls2 = 0U;
        __Runtime_wait_io_cycles(&wio_cyc2, &wio_calls2);
        unsigned long long pc_launch2 = (pc_mid >= pc0) ? (pc_mid - pc0) : 0ULL;
        unsigned long long pc_raw2 = (pc1 >= pc0) ? (pc1 - pc0) : 0ULL;
        double kp = pc_launch2 ? 100.0 * (double)ph[0] / (double)pc_launch2 : 0.0;
        double bp = pc_launch2 ? 100.0 * (double)ph[1] / (double)pc_launch2 : 0.0;
        double wp = pc_launch2 ? 100.0 * (double)wio_cyc2 / (double)pc_launch2 : 0.0;
        double vp = active ? 100.0 * (double)vec / (double)active : 0.0;
        double tb2 = (double)active + (double)sstall + (double)lstall;
        double lsp = tb2 ? 100.0 * (double)lstall / tb2 : 0.0;
        double ssp = tb2 ? 100.0 * (double)sstall / tb2 : 0.0;
        unsigned long long ph_tot = ph[0] + ph[1] + ph[2] + ph[3] + wio_cyc2;
        unsigned long long unacct2 = pc_launch2 > ph_tot ? pc_launch2 - ph_tot : 0ULL;
        printf("\n[PERF] exp=oob_compare dtype=int32 matrices=%d\n", NUM_MATRICES);
        printf("[PERF] launch_cyc=%llu\n", pc_launch2);
        printf("[PERF] launch_cyc_per_mat=%llu\n", pc_launch2 / NUM_MATRICES);
        printf("[PERF] total_cyc=%llu\n", pc_raw2);
        printf("[PERF] total_cyc_per_mat=%llu\n", pc_raw2 / NUM_MATRICES);
        printf("[PERF] wall_ms=%.6f\n", wall_ms);
        printf("[PERF] gflops_wall=%.3f\n", gflops_wall);
        printf("[PERF] gflops_per_mat=%.3f\n", gflops_per_mat);
        printf("[PERF] util_pct_int32=%.4f\n", util_pct);
        printf("[PERF] kload_cyc=%llu kload_pct=%.1f\n", ph[0], kp);
        printf("[PERF] bdcfg_cyc=%llu bdcfg_pct=%.1f\n", ph[1], bp);
        printf("[PERF] lockinit_cyc=%llu lockinit_pct=%.1f\n", unacct2,
               pc_launch2 ? 100.0 * (double)unacct2 / (double)pc_launch2 : 0.0);
        printf("[PERF] coreen_cyc=%llu coreen_pct=%.1f\n", ph[2],
               pc_launch2 ? 100.0 * (double)ph[2] / (double)pc_launch2 : 0.0);
        printf("[PERF] startio_cyc=%llu startio_pct=%.1f\n", ph[3],
               pc_launch2 ? 100.0 * (double)ph[3] / (double)pc_launch2 : 0.0);
        printf("[PERF] wait_io_cyc=%llu wait_io_pct=%.1f\n", wio_cyc2, wp);
        printf("[PERF] core_active=%u core_sstall=%u core_lstall=%u\n", active, sstall, lstall);
        printf("[PERF] vec_instr=%u vec_util_pct=%.1f\n", vec, vp);
        printf("[PERF] lock_stall_pct=%.1f stream_stall_pct=%.1f\n", lsp, ssp);
        printf("[PERF] result=%s\n", result == 0 ? "PASS" : "FAIL");
    }

    printf("\n[prof] device_teardown done\n");
    return result;
}
