/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: MIT
 ******************************************************************************/
/*
 * matmul_global.cc — dma_test.cc Phase 3 rewritten to use __global__ style.
 *
 * Instead of a separate matmul_kernel.cc, the kernel is defined inline with
 * __global__ so aiehlc.sh --platform sim extracts and compiles it.
 *
 * The DMA offsets are chosen to match aiehlc's BCF layout:
 *   in[]  placed at BCF 0x71000  → DM offset 0x1000  (CORE_IN_OFFSET)
 *   out[] placed at BCF 0x72000  → DM offset 0x2000  (CORE_OUT_OFFSET)
 *
 * Run:
 *   source script/aiehlc.sh --aie-version 2 --platform sim \
 *       --runtime-source-file example/aiesim_test/matmul_global.cc
 */

#include "xaiengine.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

/* Device geometry (aie-ml / gen2, VC2802) — mirrors aie_device_map.h */
#if AIE_GEN <= 2
#  define HW_GEN XAIE_DEV_GEN_AIEML
#else
#  define HW_GEN XAIE_DEV_GEN_AIE2PS
#endif
#define XAIE_BASE_ADDR         0x20000000000ULL
#define XAIE_COL_SHIFT         25
#define XAIE_ROW_SHIFT         20
#define XAIE_NUM_ROWS          11
#define XAIE_NUM_COLS          38
#define XAIE_SHIM_ROW          0
#define XAIE_RES_TILE_ROW_START 1
#define XAIE_RES_TILE_NUM_ROWS  2
#define XAIE_AIE_TILE_ROW_START 3
#define XAIE_AIE_TILE_NUM_ROWS  8

#define SHIM_COL  3
#define CORE_COL  4
#define CORE_ROW  4   /* absolute row; adf_row = CORE_ROW - XAIE_AIE_TILE_ROW_START + 1 = 2 */

#define MAT_N        4
#define MAT_BYTES    (MAT_N * MAT_N * (int)sizeof(int32_t))   /* 64 bytes */

/* DM offsets used by XAie_MoveData*: BCF address - 0x70000 */
#define CORE_IN_OFFSET   0x1000   /* BCF: in[]  @ 0x71000 */
#define CORE_OUT_OFFSET  0x2000   /* BCF: out[] @ 0x72000 */

/* ── Kernel ──────────────────────────────────────────────────────────────── */
/* DM base address for the AIE-ML core tile's own data memory (East direction).
 * Matches what host-side DM offsets map to from inside the core.
 * Reference: example/aiesim_test/matmul_kernel.cc                          */
#define DM_BASE    0x70000

/* aiehlc extracts this __global__ function, compiles it with xchesscc,
 * and embeds the ELF as _binary_kernel_start in the PS .so.               */
__global__
void matmul_kernel(int *in, int *out) {
    /* Direct DM access — same pattern as matmul_kernel.cc (the working     *
     * reference). Uses DM_BASE absolute addresses, not window pointers,    *
     * so the host-side DM offsets (CORE_IN_OFFSET, CORE_OUT_OFFSET)        *
     * map exactly to the addresses written/read here.                      */
    volatile int *a = (volatile int *)(DM_BASE + CORE_IN_OFFSET);       /* 0x71000 */
    volatile int *b = a + MAT_N * MAT_N;                                  /* 0x71040 */
    volatile int *c = (volatile int *)(DM_BASE + CORE_OUT_OFFSET);       /* 0x72000 */
    const int n = MAT_N;

    printf("[kernel] a[0]=%d b[0]=%d\n", (int)a[0], (int)b[0]);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            int s = 0;
            for (int k = 0; k < n; k++)
                s += a[i*n+k] * b[k*n+j];
            c[i*n+j] = s;
        }

    printf("[kernel] c[0]=%d c[15]=%d\n", (int)c[0], (int)c[15]);
}

/* ── routing() stub ─────────────────────────────────────────────────────── */
/* Called by __Runtime_routing_init.  Stream switch is set up by XAie_Route
 * in main(), so this stub is intentionally empty.                          */
void routing(XAie_DevInst *) {}

/* ── main ────────────────────────────────────────────────────────────────── */
int main(int argc, char **argv) {
    (void)argc; (void)argv;
    printf("=== matmul_global: 4x4 int32 matmul (aiehlc flow) ===\n");

    /* Device init */
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
    if (rc != XAIE_OK) { printf("FAIL: CfgInitialize\n"); free(dev); return 1; }
    XAie_SetIOBackend(dev, XAIE_IO_BACKEND_SIM);

    XAie_LocType shim = XAie_TileLoc(SHIM_COL, 0);
    XAie_LocType core = XAie_TileLoc(CORE_COL, CORE_ROW);

    /* Build A, B matrices */
    int32_t a[MAT_N*MAT_N], b[MAT_N*MAT_N], c_cpu[MAT_N*MAT_N];
    for (int i = 0; i < MAT_N*MAT_N; i++) { a[i] = i + 1; b[i] = (i % MAT_N == i / MAT_N) ? 1 : 0; }
    /* b = identity so C == A */
    for (int i = 0; i < MAT_N; i++)
        for (int j = 0; j < MAT_N; j++) {
            int s = 0;
            for (int k = 0; k < MAT_N; k++) s += a[i*MAT_N+k] * b[k*MAT_N+j];
            c_cpu[i*MAT_N+j] = s;
        }

    /* Use direct DM access — simpler and avoids routing/DMA path issues */
    int32_t in_buf[2 * MAT_N * MAT_N];
    int32_t out_buf[MAT_N * MAT_N];
    memcpy(in_buf,            a, MAT_BYTES);
    memcpy(in_buf + MAT_N*MAT_N, b, MAT_BYTES);
    memset(out_buf, 0, MAT_BYTES);
    int32_t *in_ptr = in_buf;
    int32_t *out_ptr = out_buf;
    /* (no DDR sync needed with direct DM access) */

    /* Load kernel ELF — aiehlc replaces (unsigned char*)matmul_kernel with
     * the embedded ELF pointer (_binary_kernel_start)                      */
    XAie_CoreReset(dev, core);
    XAie_CoreUnreset(dev, core);
    XAie_LoadElfMem(dev, core, (unsigned char *)matmul_kernel);

    /* Write [A|B] to tile DM at CORE_IN_OFFSET (BCF places in[] there) */
    rc = XAie_DataMemBlockWrite(dev, core, CORE_IN_OFFSET, in_ptr, 2 * MAT_BYTES);
    if (rc != XAIE_OK) { printf("FAIL: DataMemBlockWrite rc=%d\n", (int)rc); return 1; }

    /* Run kernel — use transaction batching pattern required by ISS
     * (matches __Runtime_core_run / dma_test.cc Phase 3 pattern)    */
    XAie_StartTransaction(dev, XAIE_TRANSACTION_ENABLE_AUTO_FLUSH);
    XAie_CoreEnable(dev, core);
    XAie_SubmitTransaction(dev, NULL);
    printf("[matmul_global] Core enabled, waiting for done...\n");
    int kernel_done = 0;
    for (int iter = 0; iter < 10000 && !kernel_done; iter++) {
        if (XAie_CoreWaitForDone(dev, core, 0) == XAIE_OK)
            kernel_done = 1;
    }
    printf("[matmul_global] Kernel done (kernel_done=%d)\n", kernel_done);

    /* Read C from tile DM directly — BCF places out[] at 0x72000 = DM offset 0x2000 */
    rc = XAie_DataMemBlockRead(dev, core, CORE_OUT_OFFSET, out_ptr, MAT_BYTES);
    if (rc != XAIE_OK) { printf("FAIL: DataMemBlockRead rc=%d\n", (int)rc); return 1; }

    /* Verify */
    int mismatches = 0;
    for (int i = 0; i < MAT_N*MAT_N; i++) {
        if (out_ptr[i] != c_cpu[i]) {
            printf("FAIL: C[%d] expected=%d got=%d\n", i, c_cpu[i], out_ptr[i]);
            mismatches++;
        }
    }
    if (mismatches == 0)
        printf("PASS: 4x4 matmul result matches CPU\n");
    else
        printf("FAIL: %d mismatches\n", mismatches);

    XAie_Finish(dev);
    free(dev);
    return mismatches ? 1 : 0;
}
