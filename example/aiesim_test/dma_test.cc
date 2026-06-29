/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: MIT
 ******************************************************************************/
/*
 * example/aiesim_test/dma_test.cc — DMA roundtrip test using the aiehlc
 * routing engine (XAie_InitRoutingHandler / XAie_Route / XAie_MoveData*).
 *
 * This matches the pattern used in actual application code and avoids
 * manually calling XAie_StrmConnCctEnable for every stream-switch hop.
 * The routing engine handles path finding (shim→mem→core) automatically
 * across all AIE generations.
 *
 * PHASE 1: DDR → core DM
 *   XAie_MoveDataExternal2Aie — fills core DM @ CORE_DM_RX from DDR
 *   Verify via XAie_DataMemBlockRead.
 *
 * PHASE 2: core DM → DDR
 *   XAie_DataMemBlockWrite — seed core DM @ CORE_DM_TX with a pattern
 *   XAie_MoveDataAie2External — reads core DM out to DDR
 *   Verify via XAie_MemGetVAddr after XAie_MemSyncForCPU.
 *
 * Build: see script/sim/Makefile
 */

#include "aie_device_map.h"   /* HW_GEN, XAIE_NUM_ROWS/COLS, geometry macros */
#include "aie_runtime.h"
#include "aie_runtime_debug.h"
#include "xaiengine.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

/* ── Optional kernel embedding ────────────────────────────────────── */
/* When KERNEL_SRC=matmul_kernel.cc is passed to the Makefile, the AIE ELF
 * is embedded in the PS .so via 'ld -r -b binary'.  These symbols are then
 * available for XAie_LoadElfMem.  A weak declaration lets the test compile
 * without the kernel — Phase 3 is skipped if the symbol is absent. */
/* Symbols produced by: ld -r -b binary -o matmul_kernel.o kernel
 * (name is always _binary_kernel_{start,end,size} because ld is run from
 * the kernel build dir where the ELF is named simply "kernel") */
__attribute__((weak)) extern unsigned char _binary_kernel_start[];
__attribute__((weak)) extern unsigned char _binary_kernel_end[];

/* ── Test parameters ──────────────────────────────────────────────── */
/* Shim tile that feeds the AIE array — must match aieshim_solution.aiesol.
 * Column 6 = M01_AXI for gen2 (aie-ml/VC2802).
 * Column 4 = M00_AXI for gen5 (aie2ps/XC2VE3858).
 * Column 3 = M00_AXI for gen1 (aie/VC1902). */
/* ── Tile selection ──────────────────────────────────────────────────────
 * Change these to target a different tile. No other changes needed.
 * The Work/ directory (generated once with --stub-all) covers all tiles,
 * so only the PS .so needs to be rebuilt when you change these.
 *
 * SHIM_COL : NOC-connected shim column for DMA (VC2802 M00_AXI = col 3)
 * CORE_COL : kernel tile column
 * CORE_ROW : kernel tile absolute row = TILE_ROW_START(3) + adf_row - 1
 *            e.g. adf_row=1 → abs_row=3, adf_row=2 → abs_row=4, etc.
 * ────────────────────────────────────────────────────────────────────── */
static const int SHIM_COL = 3;
static const int CORE_COL = 4;
static const int CORE_ROW = 4;   /* col=4, adf_row=1 */

#define TEST_WORDS  16
#define TEST_BYTES  (TEST_WORDS * (int)sizeof(int32_t))

/* Core data-memory offsets */
#define CORE_DM_RX  0x2000   /* Phase 1: core S2MM writes here */
#define CORE_DM_TX  0x3000   /* Phase 2: core MM2S reads from here */

/* Phase 3: matmul kernel memory layout (matches matmul_kernel.cc defines) */
#define MAT_N          4
#define MAT_BYTES      (MAT_N * MAT_N * (int)sizeof(int32_t))  /* 64 bytes */
#define MAT_A_OFFSET   0x2000    /* A matrix in tile DM */
#define MAT_B_OFFSET   0x2040    /* B matrix in tile DM */
#define MAT_C_OFFSET   0x2080    /* C = A*B result */

/*
 * routing() — manual stream-switch configuration, matching the pattern
 * the MLIR pipeline generates into routing.cc.
 *
 * XAie_MoveDataExternal2Aie / XAie_MoveDataAie2External use the routing
 * instance for DMA BD setup but do NOT require XAie_Route to have been
 * called — the stream switch is configured here independently, exactly as
 * the real pipeline does (routing.cc + __Runtime_routing_init).
 */
/* routing() is called by __Runtime_routing_init.  Stream switch is configured
 * by XAie_Route in main() which handles both same-column and cross-column paths
 * via BFS.  This stub satisfies the extern declaration in aie_runtime.c. */
void routing(XAie_DevInst *) {}

/* ================================================================== */
int main(int argc, char **argv)
{
    (void)argc; (void)argv;

    printf("=== DMA Roundtrip Test ===\n");
    printf("  Shim: (%d,0)  Core: (%d,%d)  %d words (%d B)\n",
           SHIM_COL, CORE_COL, CORE_ROW, TEST_WORDS, TEST_BYTES);

    /* ─── 1. Inline device init (matches test.cc — avoids __Runtime_explicit_init
     *        which calls XAie_InitRoutingHandler internally and may conflict) ── */
    XAie_Config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.AieGen          = HW_GEN;
    cfg.BaseAddr        = XAIE_BASE_ADDR;
    cfg.ColShift        = XAIE_COL_SHIFT;
    cfg.RowShift        = XAIE_ROW_SHIFT;
    cfg.NumRows         = (uint8_t)XAIE_NUM_ROWS;
    cfg.NumCols         = (uint8_t)XAIE_NUM_COLS;
    cfg.ShimRowNum      = XAIE_SHIM_ROW;
    cfg.MemTileRowStart = XAIE_RES_TILE_ROW_START;
    cfg.MemTileNumRows  = XAIE_RES_TILE_NUM_ROWS;
    cfg.AieTileRowStart = XAIE_AIE_TILE_ROW_START;
    cfg.AieTileNumRows  = XAIE_AIE_TILE_NUM_ROWS;

    XAie_DevInst *dev = (XAie_DevInst *)calloc(1, sizeof(XAie_DevInst));
    if (!dev) { printf("FAIL: calloc\n"); return 1; }

    AieRC rc = XAie_CfgInitialize(dev, &cfg);
    if (rc != XAIE_OK) { printf("FAIL: CfgInitialize %d\n", (int)rc); free(dev); return 1; }

    XAie_SetIOBackend(dev, XAIE_IO_BACKEND_SIM);
    printf("[dma_test] Device init OK, backend=SIM\n");

    XAie_LocType shim = XAie_TileLoc(SHIM_COL, 0);
    XAie_LocType core = XAie_TileLoc(CORE_COL, CORE_ROW);

    /* ─── 2. Routing engine: configures stream switch + DMA channel selection.
     * XAie_Route does BFS path finding and configures both the hardware stream
     * switch and the software routing instance that XAie_MoveData* uses.
     * routing() (called by __Runtime_routing_init) is a stub here — the real
     * pipeline's routing.cc uses manual XAie_StrmConnCctEnable calls instead,
     * but for this test the routing engine handles everything. */
    XAie_RoutingInstance *ri = XAie_InitRoutingHandler(dev);
    XAie_Route(ri, NULL, shim, core);   /* input path:  shim → core */
    XAie_Route(ri, NULL, core, shim);   /* output path: core → shim */
    printf("[routing] Paths configured via routing engine\n");

    /* ─── 3. Allocate DMA-capable DDR buffers ─────────────────────── */
    XAie_MemInst *in_mem  = XAie_MemAllocate(dev, TEST_BYTES, XAIE_MEM_CACHEABLE);
    XAie_MemInst *out_mem = XAie_MemAllocate(dev, TEST_BYTES, XAIE_MEM_CACHEABLE);
    if (!in_mem || !out_mem) { printf("FAIL: alloc\n"); return 1; }

    int32_t *in_ptr  = (int32_t *)XAie_MemGetVAddr(in_mem);
    int32_t *out_ptr = (int32_t *)XAie_MemGetVAddr(out_mem);

    /* ═══════════════════════════════════════════════════════════════
     * PHASE 1: DDR → core DM
     * ════════════════════════════════════════════════════════════════ */
    printf("\n── Phase 1: DDR → core DM ──────────────────────────────\n");

    for (int i = 0; i < TEST_WORDS; i++) in_ptr[i]  = (int32_t)(0xA1E00000u | (uint32_t)i);
    for (int i = 0; i < TEST_WORDS; i++) out_ptr[i] = -1;
    XAie_MemSyncForDev(in_mem);

    XAie_MoveDataExternal2Aie(ri, shim, in_mem, TEST_BYTES, CORE_DM_RX, core);
    printf("[phase1] MoveDataExternal2Aie complete\n");

    /* Read back from core DM and verify */
    int32_t rx[TEST_WORDS] = {0};
    XAie_DataMemBlockRead(dev, core, CORE_DM_RX, rx, TEST_BYTES);

    int pass1 = 1;
    for (int i = 0; i < TEST_WORDS && pass1; i++) {
        if (rx[i] != in_ptr[i]) {
            printf("FAIL phase1 word[%d]: got=0x%08x want=0x%08x\n",
                   i, (unsigned)rx[i], (unsigned)in_ptr[i]);
            pass1 = 0;
        }
    }
    printf("Phase 1: %s (%d words DDR→core DM)\n",
           pass1 ? "PASS" : "FAIL", TEST_WORDS);

    /* ═══════════════════════════════════════════════════════════════
     * PHASE 2: core DM → DDR
     * ════════════════════════════════════════════════════════════════ */
    printf("\n── Phase 2: core DM → DDR ──────────────────────────────\n");

    /* Seed core DM with a different pattern via direct register write */
    int32_t tx[TEST_WORDS];
    for (int i = 0; i < TEST_WORDS; i++) tx[i] = (int32_t)(0x0C0D0000u | (uint32_t)i);
    XAie_DataMemBlockWrite(dev, core, CORE_DM_TX, tx, TEST_BYTES);

    XAie_MemSyncForDev(out_mem);   /* ensure out_mem is clean in DDR before DMA */
    XAie_MoveDataAie2External(ri, core, CORE_DM_TX, TEST_BYTES, out_mem, shim);
    printf("[phase2] MoveDataAie2External complete\n");

    XAie_MemSyncForCPU(out_mem);

    int pass2 = 1;
    for (int i = 0; i < TEST_WORDS && pass2; i++) {
        if (out_ptr[i] != tx[i]) {
            printf("FAIL phase2 word[%d]: got=0x%08x want=0x%08x\n",
                   i, (unsigned)out_ptr[i], (unsigned)tx[i]);
            pass2 = 0;
        }
    }
    printf("Phase 2: %s (%d words core DM→DDR)\n",
           pass2 ? "PASS" : "FAIL", TEST_WORDS);

    /* ═══════════════════════════════════════════════════════════════
     * PHASE 3: matmul kernel (only if ELF was embedded at build time)
     * ════════════════════════════════════════════════════════════════ */
    int pass3 = 1;
    XAie_MemInst *ma = NULL, *mb = NULL, *mc = NULL;  /* freed at teardown */
    if (_binary_kernel_start != NULL &&
        _binary_kernel_start != _binary_kernel_end) {

        printf("\n── Phase 3: matmul kernel (4x4 int32) ──────────────────\n");

        /* Allocate DDR buffers for A, B, C */
        ma = XAie_MemAllocate(dev, MAT_BYTES, XAIE_MEM_CACHEABLE);
        mb = XAie_MemAllocate(dev, MAT_BYTES, XAIE_MEM_CACHEABLE);
        mc = XAie_MemAllocate(dev, MAT_BYTES, XAIE_MEM_CACHEABLE);
        if (!ma || !mb || !mc) { printf("FAIL phase3: alloc\n"); pass3 = 0; goto teardown; }

        int32_t *A = (int32_t *)XAie_MemGetVAddr(ma);
        int32_t *B = (int32_t *)XAie_MemGetVAddr(mb);
        int32_t *C = (int32_t *)XAie_MemGetVAddr(mc);

        /* Fill A and B with distinct values, zero C, compute expected on host */
        for (int i = 0; i < MAT_N * MAT_N; i++) {
            A[i] = i + 1;       /* 1..16  */
            B[i] = i + 17;      /* 17..32 */
            C[i] = 0;
        }
        /* Compute expected C = A * B on host for verification */
        int32_t C_expected[MAT_N * MAT_N] = {0};
        for (int i = 0; i < MAT_N; i++)
            for (int j = 0; j < MAT_N; j++)
                for (int k = 0; k < MAT_N; k++)
                    C_expected[i * MAT_N + j] += A[i * MAT_N + k] * B[k * MAT_N + j];
        XAie_MemSyncForDev(ma);
        XAie_MemSyncForDev(mb);
        XAie_MemSyncForDev(mc);

        /* Reset core, load ELF, DMA data in — matching __Runtime_load_kernel_group
         * pattern: CoreReset → CoreUnreset → LoadElfMem ensures the ISS starts
         * the kernel fresh from our ELF, not from any prior run. */
        XAie_CoreReset(dev, core);
        XAie_CoreUnreset(dev, core);
        AieRC krc = XAie_LoadElfMem(dev, core, _binary_kernel_start);
        if (krc != XAIE_OK) {
            printf("FAIL phase3: XAie_LoadElfMem rc=%d\n", (int)krc);
            pass3 = 0; goto teardown;
        }
        printf("[phase3] ELF loaded into core (%d,%d)\n", CORE_COL, CORE_ROW);

        /* DMA A and B into tile DM before enabling core */
        XAie_MoveDataExternal2Aie(ri, shim, ma, MAT_BYTES, MAT_A_OFFSET, core);
        XAie_MoveDataExternal2Aie(ri, shim, mb, MAT_BYTES, MAT_B_OFFSET, core);
        printf("[phase3] Matrices DMA'd to tile DM: A@0x%X B@0x%X\n",
               MAT_A_OFFSET, MAT_B_OFFSET);

        /* Enable core using transaction batching — matches __Runtime_core_run pattern.
         * Each XAie_CoreWaitForDone(timeout=0) does one AXI register read which
         * advances the ISS clock, giving the core cycles to execute.
         * printf() from the kernel appears in AIESimulator.log. */
        XAie_StartTransaction(dev, XAIE_TRANSACTION_ENABLE_AUTO_FLUSH);
        XAie_CoreEnable(dev, core);
        XAie_SubmitTransaction(dev, NULL);
        printf("[phase3] Core enabled (transaction-batched), waiting...\n");

        /* Poll CoreWaitForDone — same as __Runtime_wait_event */
        int kernel_done = 0;
        for (int iter = 0; iter < 10000 && !kernel_done; iter++) {
            AieRC rc = XAie_CoreWaitForDone(dev, core, 0);
            if (rc == XAIE_OK)
                kernel_done = 1;
        }
        printf("[phase3] CoreWaitForDone loop done: kernel_done=%d\n", kernel_done);

        /* DMA result C back from tile DM to DDR */
        XAie_MoveDataAie2External(ri, core, MAT_C_OFFSET, MAT_BYTES, mc, shim);
        XAie_MemSyncForCPU(mc);

        /* Verify C matches host-computed C_expected — print full matrix */
        printf("[phase3] Result C (4x4):\n");
        for (int i = 0; i < MAT_N; i++) {
            printf("  [");
            for (int j = 0; j < MAT_N; j++)
                printf(" %6d", C[i * MAT_N + j]);
            printf(" ]\n");
        }
        printf("[phase3] Expected C (4x4):\n");
        for (int i = 0; i < MAT_N; i++) {
            printf("  [");
            for (int j = 0; j < MAT_N; j++)
                printf(" %6d", C_expected[i * MAT_N + j]);
            printf(" ]\n");
        }
        for (int i = 0; i < MAT_N * MAT_N; i++) {
            if (C[i] != C_expected[i]) {
                printf("FAIL phase3 C[%d][%d]: got=%d want=%d\n",
                       i / MAT_N, i % MAT_N, C[i], C_expected[i]);
                pass3 = 0;
            }
        }
        printf("Phase 3: %s (4x4 int32 matmul)\n", pass3 ? "PASS" : "FAIL");
    } else {
        printf("\n── Phase 3: skipped (no kernel embedded; rebuild with KERNEL_SRC=...)\n");
    }

teardown:
    /* ─── Teardown: free MemInst objects before XAie_Finish to avoid
     *     "Freeing SimIO while MemInst not freed" error in sim backend. ── */
    if (ma)  XAie_MemFree(ma);
    if (mb)  XAie_MemFree(mb);
    if (mc)  XAie_MemFree(mc);
    XAie_MemFree(in_mem);
    XAie_MemFree(out_mem);
    XAie_Finish(dev);
    free(dev);
    free(ri);

    int all_pass = pass1 && pass2 && pass3;
    printf("\n=== DMA Test: %s ===\n", all_pass ? "PASS" : "FAIL");
    return all_pass ? 0 : 1;
}
