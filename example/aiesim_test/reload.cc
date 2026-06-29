// Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
// SPDX-License-Identifier: MIT
//
// reload.cc — runtime kernel reload test.
//
// Demonstrates loading two different kernels onto the same AIE tile at
// runtime without restarting the simulator:
//
//   Pass 1: matmul_kernel — 4x4 int32 matrix multiply  C = A * B
//   Pass 2: add_kernel    — 16-element int32 vector add C = A + B
//
// Both kernels share the same tile DM layout:
//   offset 0x2000 : A  (16 x int32, 64 B)
//   offset 0x2040 : B  (16 x int32, 64 B)
//   offset 0x2080 : C  (16 x int32, 64 B)  — written by kernel
//   offset 0x20C0 : done flag (0xCAFECAFE when kernel finishes)
//
// Build:
//   make -C script/sim AIEHLC_HOST_SRC=example/aiesim_test/reload.cc  \
//        KERNEL_SRC=example/aiesim_test/matmul_kernel.cc               \
//        KERNEL2_SRC=example/aiesim_test/add_kernel.cc  AIE_ARCH=20
//
// Run:
//   bash script/sim/runsim.sh                              \
//       --host-src   example/aiesim_test/reload.cc         \
//       --kernel-src example/aiesim_test/matmul_kernel.cc  \
//       --kernel-src2 example/aiesim_test/add_kernel.cc    \
//       --aie-gen 2 --stub-all [--skip-work]

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "xaiengine.h"
#include "aie_device_map.h"

/* ── Tile selection ──────────────────────────────────────────────────────────
 * Change CORE_COL / CORE_ROW to target a different tile.
 * CORE_ROW = TILE_ROW_START(3) + adf_row - 1
 * The Work/ directory generated with --stub-all covers all tiles. */
static const int SHIM_COL = 3;
static const int CORE_COL = 4;
static const int CORE_ROW = 4;   /* adf_row=1 */

/* Shared kernel DM layout (both kernels use the same offsets) */
#define N           4
#define ELEMS       (N * N)
#define MAT_BYTES   (ELEMS * (int)sizeof(int32_t))
#define A_OFFSET    0x2000
#define B_OFFSET    0x2040
#define C_OFFSET    0x2080
#define DONE_OFFSET 0x20C0
#define DONE_MAGIC  0xCAFECAFEU

/* Embedded kernel ELFs — provided by the Makefile via ld -r -b binary */
extern unsigned char _binary_kernel_start[];
extern unsigned char _binary_kernel_end[];
extern unsigned char _binary_kernel2_start[];
extern unsigned char _binary_kernel2_end[];

void routing(XAie_DevInst *) {}

/* ── run_kernel ──────────────────────────────────────────────────────────────
 * Reset core, load ELF, enable, poll done flag. Returns 1 on success. */
static int run_kernel(XAie_DevInst *dev, XAie_LocType core,
                      unsigned char *elf, const char *label)
{
    XAie_CoreReset(dev, core);
    XAie_CoreUnreset(dev, core);
    AieRC rc = XAie_LoadElfMem(dev, core, elf);
    if (rc != XAIE_OK) {
        printf("FAIL [%s]: XAie_LoadElfMem rc=%d\n", label, (int)rc);
        return 0;
    }

    XAie_StartTransaction(dev, XAIE_TRANSACTION_ENABLE_AUTO_FLUSH);
    XAie_CoreEnable(dev, core);
    XAie_SubmitTransaction(dev, NULL);

    for (int i = 0; i < 10000; i++) {
        AieRC wrc = XAie_CoreWaitForDone(dev, core, 0);
        if (wrc == XAIE_OK) return 1;
    }
    printf("FAIL [%s]: kernel timed out\n", label);
    return 0;
}

/* ── main ─────────────────────────────────────────────────────────────────── */
int main(int argc, char **argv)
{
    (void)argc; (void)argv;

    printf("=== Kernel Reload Test ===\n");
    printf("  Tile: shim=(%d,0) core=(%d,%d)\n", SHIM_COL, CORE_COL, CORE_ROW);

    int has_k1 = (_binary_kernel_start  != _binary_kernel_end);
    int has_k2 = (_binary_kernel2_start != _binary_kernel2_end);
    if (!has_k1 || !has_k2) {
        printf("ERROR: both KERNEL_SRC and KERNEL2_SRC must be set at build time.\n");
        printf("  has_k1=%d  has_k2=%d\n", has_k1, has_k2);
        return 1;
    }

    /* ── Device init (matches dma_test.cc pattern) ───────────────────────── */
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
    XAie_CfgInitialize(dev, &cfg);
    XAie_SetIOBackend(dev, XAIE_IO_BACKEND_SIM);

    XAie_LocType shim = XAie_TileLoc(SHIM_COL, 0);
    XAie_LocType core = XAie_TileLoc(CORE_COL, CORE_ROW);

    XAie_RoutingInstance *ri = XAie_InitRoutingHandler(dev);
    XAie_Route(ri, NULL, shim, core);
    XAie_Route(ri, NULL, core, shim);

    printf("[init] Device ready, routing configured\n");

    /* ── Allocate DDR buffers ─────────────────────────────────────────────── */
    XAie_MemInst *ma = XAie_MemAllocate(dev, MAT_BYTES, XAIE_MEM_CACHEABLE);
    XAie_MemInst *mb = XAie_MemAllocate(dev, MAT_BYTES, XAIE_MEM_CACHEABLE);
    XAie_MemInst *mc = XAie_MemAllocate(dev, MAT_BYTES, XAIE_MEM_CACHEABLE);
    int32_t *A = (int32_t *)XAie_MemGetVAddr(ma);
    int32_t *B = (int32_t *)XAie_MemGetVAddr(mb);
    int32_t *C = (int32_t *)XAie_MemGetVAddr(mc);

    for (int i = 0; i < ELEMS; i++) {
        A[i] = i + 1;    /* 1..16  */
        B[i] = i + 17;   /* 17..32 */
    }
    XAie_MemSyncForDev(ma);
    XAie_MemSyncForDev(mb);

    /* DMA A and B into tile DM — both kernels read from the same offsets */
    XAie_MoveDataExternal2Aie(ri, shim, ma, MAT_BYTES, A_OFFSET, core);
    XAie_MoveDataExternal2Aie(ri, shim, mb, MAT_BYTES, B_OFFSET, core);
    printf("[init] A and B DMA'd to tile DM\n");

    int pass = 1;

    /* ════════════════════════════════════════════════════════════════════════
     * Pass 1: matmul_kernel — C = A * B  (4x4 matrix multiply)
     * ════════════════════════════════════════════════════════════════════════ */
    printf("\n── Pass 1: matmul_kernel (C = A*B) ─────────────────────────\n");
    if (!run_kernel(dev, core, _binary_kernel_start, "matmul")) {
        pass = 0;
        goto teardown;
    }

    XAie_MoveDataAie2External(ri, core, C_OFFSET, MAT_BYTES, mc, shim);
    XAie_MemSyncForCPU(mc);

    {
        int32_t C_exp[ELEMS] = {0};
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                for (int k = 0; k < N; k++)
                    C_exp[i * N + j] += A[i * N + k] * B[k * N + j];

        int p1 = 1;
        for (int i = 0; i < ELEMS && p1; i++) {
            if (C[i] != C_exp[i]) {
                printf("FAIL matmul C[%d]: got=%d want=%d\n", i, C[i], C_exp[i]);
                p1 = 0;
            }
        }
        printf("Pass 1 matmul: %s  (C[0]=%d C[15]=%d)\n",
               p1 ? "PASS" : "FAIL", C[0], C[15]);
        if (!p1) pass = 0;
    }

    /* ════════════════════════════════════════════════════════════════════════
     * Pass 2: add_kernel — C = A + B  (element-wise, same tile, new ELF)
     * ════════════════════════════════════════════════════════════════════════ */
    printf("\n── Pass 2: add_kernel   (C = A+B) ─────────────────────────\n");
    if (!run_kernel(dev, core, _binary_kernel2_start, "add")) {
        pass = 0;
        goto teardown;
    }

    XAie_MoveDataAie2External(ri, core, C_OFFSET, MAT_BYTES, mc, shim);
    XAie_MemSyncForCPU(mc);

    {
        int p2 = 1;
        for (int i = 0; i < ELEMS && p2; i++) {
            int32_t exp = A[i] + B[i];
            if (C[i] != exp) {
                printf("FAIL add C[%d]: got=%d want=%d\n", i, C[i], exp);
                p2 = 0;
            }
        }
        printf("Pass 2 add:    %s  (C[0]=%d C[15]=%d)\n",
               p2 ? "PASS" : "FAIL", C[0], C[15]);
        if (!p2) pass = 0;
    }

teardown:
    XAie_MemFree(ma);
    XAie_MemFree(mb);
    XAie_MemFree(mc);
    XAie_Finish(dev);
    free(dev);

    printf("\n=== Reload Test: %s ===\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
