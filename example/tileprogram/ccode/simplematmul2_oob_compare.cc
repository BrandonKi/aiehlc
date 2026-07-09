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
 * exp59 — aie::mmul<4,4,4,int32,int32> kernel with K-major B (Option B).
 *
 * Kernel: aie::mmul<4,4,4,int32,int32> with K-major B DDR layout.
 *   - B pre-transposed host-side: B_t[k*N+n] = B[n*K+k] (col-major → K-major)
 *   - GemmSpace KmajBB: d1={KCHUNK,K} (K-tiles), d2={TILE_N,N} (N-tiles), Layout::Row
 *   - Kernel: C.mul(av,bv) / C.mac(av,bv) / C.to_vector<int32>()
 *   - Eliminates all inner i/j scalar loops — mmul computes all 16 C elements per call
 *   - No in-kernel B transpose needed (layout matches mmul operand format directly)
 *
 * Previous baseline (exp_oob_compare):
 *   - mac+reduce_add, col-major B, 16 scalar reduce ops per KCHUNK slice
 *   - Logged in autoperf/results/retime/exp_oob_compare.log (git: dada830)
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
// B_t window: KCHUNK=4 K-elements × TILE_N=4 N-elements = 16 int32 = 64 bytes (K-major)
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

// Spatial policies:
//   win_a: A[M,K] -> d1=TILE_M rows, d2=KCHUNK K-elements, int32, row-major
//   win_b: B_t[K,N] -> d1=KCHUNK K-slices, d2=TILE_N N-elements, int32, K-major (row-major layout)
//   win_c: C[M,N] -> d1=TILE_M rows, d2=TILE_N cols, int32
constexpr aie::GemmSpace RowBA = {
    .policy = {.map = {.act = aie::Pattern::Broadcast, .layout = aie::Layout::Row},
               .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
               .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{4096}}},
    .d1 = {.tile_size = TILE_M, .stride = TILE_M, .fullsize = M, .pad_hi = 0, .pad_lo = 0},
    .d2 = {.tile_size = KCHUNK, .stride = KCHUNK, .fullsize = K, .pad_hi = 0, .pad_lo = 0}};
// K-major B: B_t[k][n] stored in DDR as B_t[k*N+n].
// d1 iterates over K (KCHUNK slices), d2 iterates over N (TILE_N tiles).
// Layout::Row: data contiguous along d2 (N dimension) — matches K-major row storage.
constexpr aie::GemmSpace KmajBB = {.policy = {.map = {.wgt = aie::Pattern::Broadcast, .layout = aie::Layout::Row},
                                              .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
                                              .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{4096}}},
                                   .d1 = {.tile_size = KCHUNK, .stride = KCHUNK, .fullsize = K},
                                   .d2 = {.tile_size = TILE_N, .stride = TILE_N, .fullsize = N}};
constexpr aie::GemmSpace LtoR_Merge = {
    .policy = {.map = {.layout = aie::Layout::Row, .merge_order = aie::Flow::LeftToRight},
               .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
               .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{4096}}},
    .d1 = {.tile_size = TILE_M, .stride = TILE_M, .fullsize = M},
    .d2 = {.tile_size = TILE_N, .stride = TILE_N, .fullsize = N}};

// ─── KERNEL: per-tile int32 GEMM — aie::mmul<4,4,4,int32,int32> ─────────────
//
// B arrives K-major: B_t[k*N+n] → window = B_ptr[0..KCHUNK*TILE_N-1] row-major.
// A arrives row-major: A[i][k] → window = A_ptr[0..TILE_M*KCHUNK-1] row-major.
//
// One mmul<4,4,4> call computes the full TILE_M×TILE_N (4×4) output tile
// from a TILE_M×KCHUNK A slice and a KCHUNK×TILE_N B_t slice, accumulating
// across k_rounds K-chunks.
//
// av: TILE_M*KCHUNK = 16 int32 (A tile, row-major)
// bv: KCHUNK*TILE_N = 16 int32 (B_t tile, K-major = row-major in K)
// cv: TILE_M*TILE_N = 16 int32 (C tile, row-major)
//
__global__ void matmul(aie::port<input_window_int32 *, RowBA> win_a, aie::port<input_window_int32 *, KmajBB> win_b,
                       aie::port<output_window_int32 *, LtoR_Merge> win_c) {
    const int k_rounds = aie::get_k_rounds(); // K / KCHUNK = 64
    const int m_rounds = aie::get_spatial_multiple_rounds(win_a);
    const int n_rounds = aie::get_spatial_multiple_rounds(win_b);

    using MMUL = aie::mmul<TILE_M, KCHUNK, TILE_N, int32, int32>;

    for (int mr = 0; mr < m_rounds * n_rounds; mr++) {
        MMUL C;

        for (int kr = 0; kr < k_rounds; kr++) {
            // Load TILE_M×KCHUNK A slice (row-major, 16 int32)
            int32_t *A_ptr = (int32_t *)acquire_input_window(win_a);
            aie::vector<int32, TILE_M * KCHUNK> av = aie::load_v<TILE_M * KCHUNK>(A_ptr);
            release_input_window(win_a);

            // Load KCHUNK×TILE_N B_t slice (K-major = row-major, 16 int32)
            int32_t *B_ptr = (int32_t *)acquire_input_window(win_b);
            aie::vector<int32, KCHUNK * TILE_N> bv = aie::load_v<KCHUNK * TILE_N>(B_ptr);
            release_input_window(win_b);

            // Accumulate: first round uses mul (resets accumulator), rest use mac
            if (kr == 0)
                C.mul(av, bv);
            else
                C.mac(av, bv);
        }

        // Write TILE_M×TILE_N (4×4) result — values safe for int32 (max |C[i][j]|=1536)
        int32_t *out = (int32_t *)acquire_output_window(win_c);
        aie::store_v(out, C.to_vector<int32>());
        release_output_window(win_c);
    }
}

// Correctness check for int32 output.
// A is row-major: A[i][k] = A[i*K + k].
// B_orig is col-major: B[j][k] = B[j*K + k].
// Golden: C[i][j] = saturate(sum_k A[i][k] * B[j][k])  (same math regardless of B_t layout)
static int oob_verify(const int32_t *A, const int32_t *B_orig, const int32_t *C) {
    int mismatches = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int64_t s = 0;
            for (int k = 0; k < K; k++)
                s += (int64_t)A[i * K + k] * (int64_t)B_orig[j * K + k];
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
    printf("\n=== aiehlc OOB Comparison Benchmark exp59 (int32, 8 matrices, mmul<4,4,4>, K-major B) ===\n");
    printf("  C[%dx%d] = A[%dx%d] * B[%dx%d], int32, %dx%d mesh (%d tiles), %d matrices\n", M, N, M, K, K, N, HW_ROWS,
           HW_COLS, HW_ROWS * HW_COLS, NUM_MATRICES);
    printf("  OOB reference: example_oob_4x4 (PANEL_SIZE=256, NUM_MATRICES=8, aie::mmul<4,4,4,uint32,uint32>)\n");
    printf("  Kernel: aie::mmul<4,4,4,int32,int32> (K-major B pre-transposed host-side)\n");

    __ps_pmccntr_enable();
    unsigned long long pc_init0 = __ps_pmccntr();
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({0, 3, 0, 5}, HW_ROWS, HW_COLS);
    unsigned long long pc_init1 = __ps_pmccntr();

    unsigned long long pc_setup0 = __ps_pmccntr();
    // A: row-major A[i*K+k]
    int32_t *A = (int32_t *)device.alloc(M * K * sizeof(int32_t));
    // B_orig: col-major B[j*K+k] — kept for golden reference computation
    static int32_t B_orig[K * N];
    // B_t: K-major B_t[k*N+n] — pre-transposed, passed to kernel via device.alloc
    int32_t *B_t = (int32_t *)device.alloc(K * N * sizeof(int32_t));
    int32_t *C = (int32_t *)device.alloc(M * N * sizeof(int32_t));

    // Initialize A row-major: A[i][k] = (i*K+k) % 7 - 3
    for (int i = 0; i < M * K; i++)
        A[i] = (int32_t)((i % 7) - 3);

    // Initialize B col-major: B[j*K + k] = (j*K+k) % 5 - 2
    for (int i = 0; i < K * N; i++)
        B_orig[i] = (int32_t)((i % 5) - 2);

    // Pre-transpose B to K-major: B_t[k*N+n] = B_orig[n*K+k]
    // This matches the KmajBB GemmSpace layout (d1=KCHUNK over K, d2=TILE_N over N).
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++)
            B_t[k * N + n] = B_orig[n * K + k];

    // Poison C with 0x5A5A5A5A and flush to DDR
    extern void __Runtime_sync_for_dev(XAie_DevInst * dev, void *ptr, __SIZE_TYPE__ size);
    for (int i = 0; i < M * N; i++)
        C[i] = (int32_t)0x5A5A5A5A;
    __Runtime_sync_for_dev(device._dev, C, M * N * sizeof(int32_t));
    printf("[oob] poisoned C and flushed to DDR\n");

    // Golden reference: C[i][j] = saturate(sum_k A[i][k] * B_orig[j][k])
    // Use B_orig (col-major) for the reference — math is identical to mmul(A_row, B_t_col)
    static int32_t golden[M * N];
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int64_t s = 0;
            for (int k = 0; k < K; k++)
                s += (int64_t)A[i * K + k] * (int64_t)B_orig[j * K + k];
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
        matmul<<<mesh>>>(A, B_t, C, M, N, K);
        // Each matrix uses the same data (matching OOB's single-panel 8× repeat).
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
    double gflops_per_mat = gflops_wall / NUM_MATRICES;

    uint32_t active = 0, vec = 0, sstall = 0, lstall = 0, mm0 = 0, mm1 = 0;
    int have_core = __Runtime_core_perf_probe_valid();
    __Runtime_core_perf_read_probe(&active, &vec, &sstall, &lstall);
    __Runtime_perfcnt_read_mm2s_probe(&mm0, &mm1);

    // Per-tile budget
    double total_budget = (double)active + (double)sstall + (double)lstall;
    double compute_pct = total_budget ? 100.0 * (double)active / total_budget : 0.0;
    double stream_pct = total_budget ? 100.0 * (double)sstall / total_budget : 0.0;
    double lock_pct = total_budget ? 100.0 * (double)lstall / total_budget : 0.0;
    double vec_util = active ? 100.0 * (double)vec / (double)active : 0.0;

    // OOB array peak: 184 TOPS INT8 / 4 = 46 TOPS INT32, scaled to 16/144 tiles
    const double DEVICE_INT8_TOPS = 184.0;
    const int DEVICE_TILES = 144;
    int array_tiles = HW_ROWS * HW_COLS;
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
    int result = oob_verify(A, B_orig, C);
    unsigned long long pc_verify1 = __ps_pmccntr();

    device.free(A);
    device.free(B_t);
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
        printf("\n[PERF] exp=oob_mmul dtype=int32 matrices=%d kernel=mmul<4,4,4,int32,int32> B=K-major\n",
               NUM_MATRICES);
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
