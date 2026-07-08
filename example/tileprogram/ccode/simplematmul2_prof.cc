/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * AIE Programming Model — Matrix Multiplication, HARDWARE PROFILING build.
 *
 * Same int8 GEMM kernel as simplematmul2.cc, configured for a 4x4 (16-tile)
 * mesh on a 256x256x256 problem so it matches the AEG example_oob_4x4 workload
 * (256^3, 4x4 array). Adds a 3-layer profiling report to main(), inspired by
 * example_oob_4x4/src/graph.cpp:
 *
 *   Layer 1 — PS wall-clock: end-to-end launch time (XTime) and wall GFLOPS.
 *   Layer 2 — DMA stream:    probe-tile MM2S BD-finished counts (aiehlc's
 *                            XAie-native analog of the AEG GMIO stream cycles).
 *   Layer 3 — AIE core tile:  active / stream-stall / lock-stall / vector-instr
 *                            cycle budget on the first compute tile, plus
 *                            frequency-free FLOP/cycle efficiency.
 ******************************************************************************/
#define M 256
#define K 256
#define N 256
#define HW_ROWS 4
#define HW_COLS 4
// Per-DMA-round sub-tile granularity (recovered from the prior 256^3 config:
// 16-row sub-tile, 64-element K chunk -> 4 m/n sub-tile rounds, 4 K rounds).
#define TILE_M 16
#define TILE_N 16
#define KCHUNK 64
#include "simplematmul.h"
// Debug level 0 (no verbose UART snapshot — keeps the timed launch region clean)
// but with the profiling flags enabled: disable partition teardown, set up MM2S
// BD-finished DMA counters (Layer 2) and arm core-tile perf counters (Layer 3).
#pragma aie_debug_level(0 | AIE_DEBUG_FLAG_DISABLE_PARTITIONTEARDOWN | AIE_DEBUG_FLAG_MM2SBDFINISH_COUNTER |             \
                        AIE_DEBUG_FLAG_CORE_PERF_COUNTER)

// Profiling hooks implemented in src/mlir/runtime/aie_runtime.c (compiled as
// C++), declared here so the aiehlc front-end can parse main() referencing them.
extern void __Runtime_core_perf_read_probe(uint32_t *active, uint32_t *vec_instr, uint32_t *stream_stall,
                                           uint32_t *lock_stall);
extern void __Runtime_perfcnt_read_mm2s_probe(uint32_t *ch0, uint32_t *ch1);
extern int __Runtime_core_perf_probe_valid(void);
/* [exp44] accumulated PS PMU cycles spent inside wait_io (output-DMA drain, gates completion) */
extern void __Runtime_wait_io_cycles(unsigned long long *cycles, unsigned int *calls);
/* [exp45] setup-phase PMU cycles [KLOAD, BDCFG, COREEN, STARTIO]; caller passes len-4 arrays */
extern void __Runtime_phase_cycles(unsigned long long *cyc, unsigned int *calls);
/* [exp46] sub-split bdcfg: DmaDescInit vs DmaWriteBd cycles */
extern void __Runtime_bd_subphase_cycles(unsigned long long *init_cyc, unsigned int *init_n,
                                         unsigned long long *write_cyc, unsigned int *write_n);
/* [exp47] localize the rest of bdcfg: mid (post-descinit->pre-writebd) + tail (post-writebd->return) */
extern void __Runtime_bd_midtail_cycles(unsigned long long *mid_cyc, unsigned int *mid_n, unsigned long long *tail_cyc,
                                        unsigned int *tail_n);
/* [exp48] split the mid region into GetTileTypefromLoc / DmaSetAddr / DmaEnableBd */
extern void __Runtime_bd_mid3_cycles(unsigned long long *gtt_cyc, unsigned int *gtt_n, unsigned long long *saddr_cyc,
                                     unsigned int *saddr_n, unsigned long long *en_cyc, unsigned int *en_n);
/* [exp51] split the dominant kload phase: XAie_LoadElfMem (elf) vs Core Disable/Reset/Unreset (rst) */
extern void __Runtime_kload_split_cycles(unsigned long long *elf_cyc, unsigned int *elf_n, unsigned long long *rst_cyc,
                                         unsigned int *rst_n);

// Composition-based spatial spaces (per-port 2D iteration space), identical in
// structure to simplematmul2.cc; only the tile/full sizes scale to 4x4 / 256^3.
//   win_a A=[M,K] -> d1 = M-tile,  d2 = K-chunk
//   win_b B=[N,K] -> d1 = N-tile,  d2 = K-chunk
//   win_c C=[M,N] -> d1 = M-tile,  d2 = N-tile
constexpr aie::GemmSpace RowBA = {
    .policy = {.map = {.act = aie::Pattern::Broadcast, .layout = aie::Layout::Row},
               .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
               .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{4096}}},
    .d1 = {.tile_size = TILE_M, .stride = TILE_M, .fullsize = M, .pad_hi = 0, .pad_lo = 0},
    .d2 = {.tile_size = KCHUNK, .stride = KCHUNK, .fullsize = K, .pad_hi = 0, .pad_lo = 0}};
constexpr aie::GemmSpace ColBB = {
    .policy = {.map = {.wgt = aie::Pattern::Broadcast, .layout = aie::Layout::Col},
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

// ─── KERNEL: per-tile int8 GEMM (cache-A / stream-B), no debug logging ───────
__global__ void matmul(aie::port<input_window_int8 *, RowBA> win_a, aie::port<input_window_int8 *, ColBB> win_b,
                       aie::port<output_window_int8 *, LtoR_Merge> win_c) {
    const int tile_rows = aie::get_tile_rows();
    const int tile_cols = aie::get_tile_cols();
    const int eff_k = aie::get_effective_k();
    const int k_rounds = aie::get_k_rounds();
    const int num_a_rounds = aie::get_num_rounds(win_a);
    const int num_b_rounds = aie::get_num_rounds(win_b);
    const int num_c_rounds = aie::get_num_rounds(win_c);
    const int buf_sz_a = aie::get_buffer_size(win_a);
    const int buf_sz_c = aie::get_buffer_size(win_c);
    const int m_rounds = aie::get_spatial_multiple_rounds(win_a);
    const int n_rounds = aie::get_spatial_multiple_rounds(win_b);
    const int cols_per_round = aie::get_buffer_size(win_b) / eff_k;

    int8_t all_A[tile_rows * eff_k];
    int16_t accum[tile_rows * tile_cols];
    int8_t local_out[tile_rows * tile_cols];

    for (int mr = 0; mr < m_rounds * n_rounds; mr++) {
        for (int i = 0; i < tile_rows * tile_cols; i++)
            accum[i] = 0;

        for (int kr = 0; kr < k_rounds; kr++) {
            for (int ra = 0; ra < num_a_rounds; ra++) {
                int8_t *A_ptr = (int8_t *)acquire_input_window(win_a);
                for (int i = 0; i < buf_sz_a; i++)
                    all_A[ra * buf_sz_a + i] = A_ptr[i];
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
            release_output_window(win_c);
        }
    }
}

// Compact correctness check (avoids the huge matrix dumps in simplematmul.h's
// verify_matmul, which would flood the slow UART for a 256^3 run).
static int prof_verify(const int8_t *A, const int8_t *B, const int8_t *C) {
    int mismatches = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int16_t s = 0;
            for (int k = 0; k < K; k++)
                s += (int16_t)A[i * K + k] * (int16_t)B[j * K + k];
            if (s > 127)
                s = 127;
            else if (s < -128)
                s = -128;
            if (C[i * N + j] != (int8_t)s) {
                if (mismatches < 8)
                    printf("  mismatch C[%d,%d] got %d exp %d\n", i, j, (int)C[i * N + j], (int)(int8_t)s);
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
    printf("\n=== aiehlc GEMM profiling ===\n");
    printf("  C[%dx%d] = A[%dx%d] * B^T[%dx%d], int8, %dx%d mesh (%d tiles)\n", M, N, M, K, N, K, HW_ROWS, HW_COLS,
           HW_ROWS * HW_COLS);

    aieSetDevice(0);
    aieArray device;
    // 4x4 mesh: cols 0-3, rows 0-5 (shim row0, memtile row1, 4 compute rows 2-5).
    aieMesh mesh = device.partition({0, 3, 0, 5}, HW_ROWS, HW_COLS);

    int8_t *A = (int8_t *)device.alloc(M * K * sizeof(int8_t) * 4);
    int8_t *B = (int8_t *)device.alloc(K * N * sizeof(int8_t) * 4);
    int8_t *C = (int8_t *)device.alloc(M * N * sizeof(int8_t) * 4);
    for (int i = 0; i < M * K; i++)
        A[i] = (int8_t)((i % 7) - 3);
    for (int i = 0; i < K * N; i++)
        B[i] = (int8_t)((i % 5) - 2);
    // [exp07 poison] forward-decl (front-end only declares sync_for_cpu, not sync_for_dev)
    extern void __Runtime_sync_for_dev(XAie_DevInst *dev, void *ptr, __SIZE_TYPE__ size);
    // Write a poison pattern to C AND FLUSH it to device DRAM before the
    // launch. If the kernel actually computes, it overwrites C with the correct result
    // (PASS). If the workload deadlocks (suspected), the poison survives in device DRAM
    // and verification FAILS showing "got 90" — proving prior PASSes were stale-DDR.
    for (int i = 0; i < M * N; i++)
        C[i] = (int8_t)0x5A; /* 90 */
    __Runtime_sync_for_dev(device._dev, C, M * N * sizeof(int8_t) * 4);
    printf("[exp07] poisoned device C with 0x5A and flushed to DDR\n");

    // Golden reference for the completion barrier, computed BEFORE the timed region (outside
    // t0..t1) so the GEMM reference cost is not charged to the metric. Same saturated-int8
    // formula as prof_verify().
    static int8_t golden[M * N];
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int16_t s = 0;
            for (int k = 0; k < K; k++)
                s += (int16_t)A[i * K + k] * (int16_t)B[j * K + k];
            if (s > 127)
                s = 127;
            else if (s < -128)
                s = -128;
            golden[i * N + j] = (int8_t)s;
        }
    }

    // ── Layer 1: time the WHOLE process until the full result lands in DDR. The launch is
    // asynchronous (posted PS->AIE writes + non-blocking waits), so timing the launch call
    // alone reads ~0. Instead, after issuing the launch we poll the output buffer in device
    // DRAM (invalidate cache via synchronizecpu, then compare against the golden reference)
    // and stop the timer the instant C holds the complete correct result. This is the true
    // end-to-end "launch -> results readable" wall, independent of DMA-status quirks.
    const uint64_t MAX_POLL = 500000000ULL; /* safety bound on a genuine hang */
    XTime t0, t1;
    /* [exp40] Independent wall counter (CNTVCT_EL0) bracketing the SAME window, in case
     * XTime's xiltimer source is frozen this boot (raw counts read 0). Read-only. */
    unsigned long long cv0 = __ps_cntvct();
    /* [exp41] PMU cycle counter (PMCCNTR_EL0) — CPU-core-clock, independent of the frozen
     * system generic timer that stalls both XTime and CNTVCT. One-time enable, then bracket
     * the SAME window. If [pmccntr] raw advances while XTime+CNTVCT read 0, the host-side
     * wall is measurable again. (Enable is a system-register write; safe at EL1+.) */
    __ps_pmccntr_enable();
    unsigned long long pc0 = __ps_pmccntr();
    XTime_GetTime(&t0);
    matmul<<<mesh>>>(A, B, C, M, N, K);
    unsigned long long pc_mid = __ps_pmccntr(); /* [exp43] boundary: launch-call vs poll-loop */
    uint64_t polls = 0;
    int complete = 0;
    do {
        device.synchronizecpu(C, M * N * sizeof(int8_t) * 4); /* invalidate -> fresh DDR read */
        complete = 1;
        for (int idx = 0; idx < M * N; idx++) {
            if (C[idx] != golden[idx]) {
                complete = 0;
                break;
            }
        }
        polls++;
    } while (!complete && polls < MAX_POLL);
    XTime_GetTime(&t1);
    unsigned long long cv1 = __ps_cntvct();  /* [exp40] */
    unsigned long long pc1 = __ps_pmccntr(); /* [exp41] */
    if (!complete)
        printf("  WARNING: completion barrier hit MAX_POLL=%llu without full result\n", (unsigned long long)MAX_POLL);

    // [exp20] After removing the dead Core_Done poll the launch wall dropped below the old
    // %.3f-ms print's resolution (read 0.000 ms). Relate raw XTime counts -> wall directly:
    // wall_seconds = counts / COUNTS_PER_SECOND. We print the raw delta + the tick frequency
    // so the true sub-ms wall (and the timer's own resolution = 1/freq) is always recoverable.
    uint64_t raw_counts = (uint64_t)(t1 - t0);
    uint64_t timer_hz = (uint64_t)COUNTS_PER_SECOND;
    double wall_ms = 1000.0 * (double)raw_counts / (double)timer_hz;
    double wall_us = 1.0e6 * (double)raw_counts / (double)timer_hz;
    double tick_ns = 1.0e9 / (double)timer_hz;                           // one count in ns
    double total_flops = 2.0 * (double)M * (double)N * (double)K;        // MACs counted as 2 ops
    double gflops_wall = (wall_ms > 0.0) ? total_flops / (wall_ms * 1e-3) / 1e9 : 0.0;

    // ── Read profiling counters (probe = first compute tile of the group).
    uint32_t active = 0, vec = 0, sstall = 0, lstall = 0, mm0 = 0, mm1 = 0;
    int have_core = __Runtime_core_perf_probe_valid();
    __Runtime_core_perf_read_probe(&active, &vec, &sstall, &lstall);
    __Runtime_perfcnt_read_mm2s_probe(&mm0, &mm1);

    // Per-tile compute work (frequency-free; all tiles run concurrently).
    double tile_flops = 2.0 * (double)(M / HW_ROWS) * (double)(N / HW_COLS) * (double)K;
    // In this counter set the "active" counter measures pure compute cycles while
    // the stream/lock-stall counters measure (disjoint) stall cycles, so the core
    // cycle budget is their sum. Percentages are taken against that total budget.
    double total_budget = (double)active + (double)sstall + (double)lstall;
    double compute_pct = total_budget ? 100.0 * (double)active / total_budget : 0.0;
    double stream_pct = total_budget ? 100.0 * (double)sstall / total_budget : 0.0;
    double lock_pct = total_budget ? 100.0 * (double)lstall / total_budget : 0.0;
    double vec_util = active ? 100.0 * (double)vec / (double)active : 0.0;
    double flop_per_active = active ? tile_flops / (double)active : 0.0;

    // Device yardstick (same as AEG report so the two are comparable).
    const double DEVICE_INT8_TOPS = 184.0;
    const int DEVICE_TILES = 144;
    int array_tiles = HW_ROWS * HW_COLS;
    double array_peak_gops = DEVICE_INT8_TOPS * 1000.0 * (double)array_tiles / (double)DEVICE_TILES;
    double util_pct = array_peak_gops ? 100.0 * gflops_wall / array_peak_gops : 0.0;

    printf("\n--- Layer 1: PS wall-clock (end-to-end launch) ---\n");
    printf("  raw counts:        %llu  (t1 - t0)\n", (unsigned long long)raw_counts);
    printf("  timer freq:        %llu Hz  (COUNTS_PER_SECOND; 1 tick = %.3f ns)\n", (unsigned long long)timer_hz,
           tick_ns);
    printf("  total time:        %.6f ms  (%.3f us)\n", wall_ms, wall_us);
    printf("  completion polls:  %llu  (DDR read-back until full result present)\n", (unsigned long long)polls);
    printf("  wall GFLOPS:       %.3f GOPS  (2*M*N*K / total_ms)\n", gflops_wall);
    printf("  note: launch -> full result in DDR (async launch + poll-to-result barrier)\n");
    {
        /* [exp40] Independent CNTVCT_EL0 wall — bypasses a frozen xiltimer source. */
        unsigned long long cv_raw = (cv1 >= cv0) ? (cv1 - cv0) : 0ULL;
        unsigned long long cv_hz = __ps_cntfrq();
        double cv_ms = cv_hz ? 1000.0 * (double)cv_raw / (double)cv_hz : 0.0;
        printf("  [cntvct] raw:      %llu counts  freq: %llu Hz  wall: %.6f ms\n", cv_raw, cv_hz, cv_ms);
    }
    {
        /* [exp41] PMU cycle-counter wall — CPU-core-clock, survives a frozen generic timer.
         * Primary signal is raw cycles (nonzero => the PS-side wall is finally measurable).
         * ms = raw / CPU_Hz done offline (CPU_Hz not known at compile time); raw is enough
         * to decide alive-vs-frozen and to compare host-side levers relatively. */
        unsigned long long pc_raw = (pc1 >= pc0) ? (pc1 - pc0) : 0ULL;
        unsigned long long pmcr = __ps_pmcr(); /* [exp42] */
        unsigned int d_bit = (unsigned int)((pmcr >> 3) & 1ULL);
        printf("  [pmccntr] raw:     %llu cycles  pmcr:0x%llx (D=%u, %s)\n", pc_raw, pmcr, d_bit,
               d_bit ? "counts=CPUcyc/64" : "counts=CPUcyc");
        /* [exp43] split: launch = matmul<<<>>> call (input-DMA+load+config+enable+wait_io drain);
         * poll = DDR poll-to-result loop. */
        unsigned long long pc_launch = (pc_mid >= pc0) ? (pc_mid - pc0) : 0ULL;
        unsigned long long pc_poll = (pc1 >= pc_mid) ? (pc1 - pc_mid) : 0ULL;
        printf("  [pmccntr] launch:  %llu cycles  poll: %llu cycles\n", pc_launch, pc_poll);
        /* [exp44] how much of the launch is the wait_io output-DMA drain (array-gated)? */
        unsigned long long wio_cyc = 0ULL;
        unsigned int wio_calls = 0U;
        __Runtime_wait_io_cycles(&wio_cyc, &wio_calls);
        double wio_pct = (pc_launch > 0) ? 100.0 * (double)wio_cyc / (double)pc_launch : 0.0;
        printf("  [phase] wait_io:   %llu cycles over %u calls  (=%.1f%% of launch)\n", wio_cyc, wio_calls, wio_pct);
        /* [exp45] split the ~98% setup portion of the launch into runtime sub-phases. */
        unsigned long long ph[4] = {0, 0, 0, 0};
        unsigned int phc[4] = {0, 0, 0, 0};
        __Runtime_phase_cycles(ph, phc);
        const char *phn[4] = {"kload  ", "bdcfg  ", "coreen ", "startio"};
        for (int i = 0; i < 4; i++) {
            double p = (pc_launch > 0) ? 100.0 * (double)ph[i] / (double)pc_launch : 0.0;
            printf("  [phase] %s: %llu cycles over %u calls  (=%.1f%% of launch)\n", phn[i], ph[i], phc[i], p);
        }
        /* [exp46] which XAie call dominates bdcfg: DmaDescInit (HW config read) or DmaWriteBd? */
        unsigned long long bi = 0ULL, bw = 0ULL;
        unsigned int bin = 0U, bwn = 0U;
        __Runtime_bd_subphase_cycles(&bi, &bin, &bw, &bwn);
        double bip = (ph[1] > 0) ? 100.0 * (double)bi / (double)ph[1] : 0.0;
        double bwp = (ph[1] > 0) ? 100.0 * (double)bw / (double)ph[1] : 0.0;
        printf("  [bdcfg] descinit: %llu cyc over %u  (=%.1f%% of bdcfg)\n", bi, bin, bip);
        printf("  [bdcfg] writebd:  %llu cyc over %u  (=%.1f%% of bdcfg)\n", bw, bwn, bwp);
        /* [exp47] localize the ~99.85% that is neither descinit nor writebd */
        unsigned long long bmid = 0ULL, btail = 0ULL;
        unsigned int bmidn = 0U, btailn = 0U;
        __Runtime_bd_midtail_cycles(&bmid, &bmidn, &btail, &btailn);
        double bmp = (ph[1] > 0) ? 100.0 * (double)bmid / (double)ph[1] : 0.0;
        double btp = (ph[1] > 0) ? 100.0 * (double)btail / (double)ph[1] : 0.0;
        unsigned long long bacct = bi + bw + bmid + btail;
        double bap = (ph[1] > 0) ? 100.0 * (double)bacct / (double)ph[1] : 0.0;
        printf("  [bdcfg] mid:      %llu cyc over %u  (=%.1f%% of bdcfg)\n", bmid, bmidn, bmp);
        printf("  [bdcfg] tail:     %llu cyc over %u  (=%.1f%% of bdcfg)\n", btail, btailn, btp);
        printf("  [bdcfg] accounted (init+write+mid+tail): %llu cyc  (=%.1f%% of bdcfg)\n", bacct, bap);
        /* [exp48] split mid into GetTileTypefromLoc / DmaSetAddr / DmaEnableBd (+ residual = Set*+printf) */
        unsigned long long bgtt = 0ULL, bsa = 0ULL, ben = 0ULL;
        unsigned int bgttn = 0U, bsan = 0U, benn = 0U;
        __Runtime_bd_mid3_cycles(&bgtt, &bgttn, &bsa, &bsan, &ben, &benn);
        double gttp = (bmid > 0) ? 100.0 * (double)bgtt / (double)bmid : 0.0;
        double sap = (bmid > 0) ? 100.0 * (double)bsa / (double)bmid : 0.0;
        double enp = (bmid > 0) ? 100.0 * (double)ben / (double)bmid : 0.0;
        unsigned long long bres = (bmid > bgtt + bsa + ben) ? (bmid - bgtt - bsa - ben) : 0ULL;
        double resp = (bmid > 0) ? 100.0 * (double)bres / (double)bmid : 0.0;
        printf("  [mid] gettiletype: %llu cyc over %u  (=%.1f%% of mid)\n", bgtt, bgttn, gttp);
        printf("  [mid] setaddr:     %llu cyc over %u  (=%.1f%% of mid)\n", bsa, bsan, sap);
        printf("  [mid] enablebd:    %llu cyc over %u  (=%.1f%% of mid)\n", ben, benn, enp);
        printf("  [mid] residual (setlock/nextbd/pkt/ooo+printf): %llu cyc  (=%.1f%% of mid)\n", bres, resp);
        /* [exp51] split the dominant kload phase: XAie_LoadElfMem vs Core Disable/Reset/Unreset */
        unsigned long long kelf = 0ULL, krst = 0ULL;
        unsigned int kelfn = 0U, krstn = 0U;
        __Runtime_kload_split_cycles(&kelf, &kelfn, &krst, &krstn);
        double kep = (ph[0] > 0) ? 100.0 * (double)kelf / (double)ph[0] : 0.0;
        double krp = (ph[0] > 0) ? 100.0 * (double)krst / (double)ph[0] : 0.0;
        printf("  [kload] loadelf:  %llu cyc over %u  (=%.1f%% of kload)\n", kelf, kelfn, kep);
        printf("  [kload] corerst:  %llu cyc over %u  (=%.1f%% of kload)\n", krst, krstn, krp);
    }

    printf("\n--- Layer 2: DMA stream (probe tile MM2S BD finished) ---\n");
    printf("  MM2S ch0 BDs done: %u\n", mm0);
    printf("  MM2S ch1 BDs done: %u\n", mm1);

    printf("\n--- Layer 3: AIE core tile cycle budget (probe = first compute tile) ---\n");
    if (!have_core)
        printf("  [no probe tile armed]\n");
    printf("  total budget:      %.0f cycles  (active + stream-stall + lock-stall)\n", total_budget);
    printf("  active/compute:    %u  (%.2f%% of budget)\n", active, compute_pct);
    printf("  stream stall:      %u  (%.2f%% of budget)  [waiting for window data]\n", sstall, stream_pct);
    printf("  lock stall:        %u  (%.2f%% of budget)  [waiting for buffer lock/DMA]\n", lstall, lock_pct);
    printf("  vector instrs:     %u\n", vec);
    printf("  vec utilization:   %.1f %%  (vec instrs / active cycle; 0 => scalar kernel)\n", vec_util);

    printf("\n--- Kernel efficiency (frequency-free) ---\n");
    printf("  tile FLOP/active-cyc:  %.3f  (2*tileM*tileN*K / active compute cycles)\n", flop_per_active);
    printf("  note: cores are lock/DMA-bound — active compute is a tiny fraction of the budget\n");

    printf("\n--- Hardware utilization (INT8, same yardstick as AEG) ---\n");
    printf("  array INT8 peak:   %.1f GOPS  (%d/%d tiles of %.0f TOPS device)\n", array_peak_gops, array_tiles,
           DEVICE_TILES, DEVICE_INT8_TOPS);
    printf("  measured (wall):   %.3f GOPS  ->  %.4f %% of array peak\n", gflops_wall, util_pct);

    printf("\n--- Correctness ---\n");
    int result = prof_verify(A, B, C);

    device.free(A);
    device.free(B);
    device.free(C);

    // Sentinel so the board harness (appvek385.py) detects completion promptly
    // even though partition teardown is disabled for profiling.
    printf("\n[prof] device_teardown done\n");
    return result;
}
