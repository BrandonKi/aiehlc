/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "xaiengine.h"

#include "xil_printf.h"
#include "xil_cache.h"

#if AIE_GEN <= 2
#define HW_GEN XAIE_DEV_GEN_AIEML
#include "xtime_l.h"
#else
#define HW_GEN XAIE_DEV_GEN_AIE2PS
#include "xiltimer.h"
#endif

#include <stdio.h>

#define XAIE_BASE_ADDR 0x20000000000
#define XAIE_COL_SHIFT 25
#define XAIE_ROW_SHIFT 20

#define XAIE_NUM_ROWS 11
#define XAIE_NUM_COLS 38
#define XAIE_SHIM_ROW 0
#define XAIE_RES_TILE_ROW_START 1
#define XAIE_RES_TILE_NUM_ROWS 2
#define XAIE_AIE_TILE_ROW_START 3
#define XAIE_AIE_TILE_NUM_ROWS 8

// default memory addresses for input and output
#define CORE_IP_MEM 0x1000
#define CORE_OP_MEM 0x5000

#define DISABLE_CACHE

__global__ void k1(int *in, int *out) {
    for (int i = 0; i < 20; ++i)
        *(out++) = *in + 1;
}

__global__ void k2(int *in, int *out) {
    for (int i = 0; i < 20; ++i)
        *(out++) = *in + 10;
}

#define IN_COL 2
#define IN_ROW 0

#ifdef __AIELINUX__
#define OUT_COL 35
#else
#define OUT_COL 2
#endif
#define OUT_ROW 0

#define K1_COL 1
#define K1_ROW 3

#define K2_COL 1
#define K2_ROW 4

int test_col(XAie_DevInst *DevInst, int test_col) {
    printf("Checking status of all tiles...\n");
    int start_row = -1, end_row = -1;
    // for (int col = 0; col < XAIE_NUM_COLS; ++col) {
        for (int row = 0; row < XAIE_NUM_ROWS; ++row) {
            XAie_LocType tileLoc = XAie_TileLoc(test_col, row);
            uint32_t enabled = 0;
            XAie_CoreGetStatus(DevInst, tileLoc, &enabled);
            printf("Tile (%d, %d): %s\n", test_col, row, enabled ? "ENABLED" : "DISABLED");
            if(enabled) {
                if (start_row == -1) {
                    start_row = row;
                }
                end_row = row;
            }
        }
    // }
    printf("Finished checking tile status.\n\n");

    printf("\nLoading column of kernels %d...\n", test_col);
    for(int i = start_row; i <= end_row; i++) {
        XAie_LocType tileLoc = XAie_TileLoc(test_col, i);
        XAie_CoreReset(DevInst, tileLoc);
        XAie_CoreUnreset(DevInst, tileLoc);
        printf("Tile (%d, %d) reset and unreset.\n", test_col, i);
        XAie_LoadElfMem(DevInst, XAie_TileLoc(test_col, i), (unsigned char *)k1);
        printf("Tile (%d, %d) loaded elf\n", test_col, i);
    }
    printf("Finished. Continuing...\n");

    printf("\nRouting...\n");
    XAie_RoutingInstance* routingInstance = XAie_InitRoutingHandler(DevInst);
    XAie_Route(routingInstance, NULL, /*src=*/ XAie_TileLoc(IN_COL, IN_ROW), /*dest=*/ XAie_TileLoc(test_col, start_row));
    for(int i = start_row; i < end_row; i++) {
        XAie_LocType srcLoc = XAie_TileLoc(test_col, i);
        XAie_LocType destLoc = XAie_TileLoc(test_col, i + 1);
        XAie_Route(routingInstance, NULL, srcLoc, destLoc);
        printf("Routing from (%d, %d) to (%d, %d)\n", test_col, i, destLoc.Col, destLoc.Row);
    }
    XAie_Route(routingInstance, NULL, /*src=*/ XAie_TileLoc(test_col, end_row), /*dest=*/ XAie_TileLoc(OUT_COL, OUT_ROW));
    printf("Finished. Continuing...\n");
    
    printf("\nAllocating DDR memory...\n");
    int *in_ptr = 0, *out_ptr = 0;
    const int len = 20;
    XAie_MemInst *in = XAie_MemAllocate(DevInst, len * sizeof(u32), XAIE_MEM_CACHEABLE);
    in_ptr = (int*)XAie_MemGetVAddr(in);
    XAie_MemInst *out = XAie_MemAllocate(DevInst, len * sizeof(u32), XAIE_MEM_CACHEABLE);
    out_ptr = (int*)XAie_MemGetVAddr(out);
    printf("Finished. Continuing...\n");

    printf("\nInitializing DDR memory...\n");
    for(int i = 0; i < len; i++) {
        in_ptr[i] = 15;
        out_ptr[i] = 9999;
    }

    XAie_MemSyncForDev(in);
    XAie_MemSyncForCPU(out);
    printf("Finished. Continuing...\n");

    printf("\nMoving input data to AIE...\n");
    XAie_MoveDataExternal2Aie(routingInstance, /*src=*/ XAie_TileLoc(IN_COL, IN_ROW),
                              in, len*sizeof(u32),
                              CORE_IP_MEM, /*dest=*/ XAie_TileLoc(K1_COL, K1_ROW));
    printf("Finished. Continuing...\n");
    
    for(int i = start_row; i <= end_row; i++) {
        printf("\nRunning kernel %d...\n", i);
        XTime tStart, tEnd;
        XTime_GetTime(&tStart);
        XAie_Run(routingInstance, 1);

        while(XAie_CoreWaitForDone(DevInst, XAie_TileLoc(test_col, i), 0) != XAIE_OK) {
            // wait until core done
        }

        XTime_GetTime(&tEnd);
        printf("Output took %.2f us.\n", 1.0 * (tEnd - tStart) / (COUNTS_PER_SECOND/1000000));
        printf("Finished. Continuing...\n");

        if (i != end_row) {
            printf("\nMoving output from kernel 1 to kernel 2 input...\n");
            XAie_MoveData(routingInstance, /*src=*/ XAie_TileLoc(test_col, i),
                                    (void*)CORE_OP_MEM, len*sizeof(u32),
                                    (void*)CORE_IP_MEM, /*dest=*/ XAie_TileLoc(test_col, i+1));
        }
        printf("Finished. Continuing...\n");
    }

    printf("\nMoving output data to CPU...\n");
    XAie_MoveDataAie2External(routingInstance, /*src=*/ XAie_TileLoc(test_col, end_row),
                              CORE_OP_MEM, len*sizeof(u32),
                              out, /*dest=*/ XAie_TileLoc(OUT_COL, OUT_ROW));	
    printf("Finished. Continuing...\n");

    printf("\nVerifying output data...\n");
    int mismatches = 0;
    const int c = 23;
    for (int i = 0; i < len; i++) {
        if (c != out_ptr[i]) {
            printf("Mismatch at index %d: CPU=%d, AIE=%d\n", i, c, out_ptr[i]);
            mismatches++;
        } else {
            printf("%d: CPU=%d, AIE=%d\n", i, c, out_ptr[i]);
        }
    }

    if (mismatches == 0) {
        printf("Sucess: CPU result matches AIE.\n");
    } else {
        printf("Failure: There were %d mismatches.\n", mismatches);
    }

    printf("\n\nDone with all kernels!\n\n");
    return 0;
}

int main(int argc, char* argv[]) {
#ifdef DISABLE_CACHE
    Xil_DCacheDisable();
    Xil_ICacheDisable();
    printf("Cache disabled\n");
#else
    printf("Cache enabled\n");
#endif

    XAie_SetupConfig(ConfigPtr, HW_GEN, XAIE_BASE_ADDR,
                     XAIE_COL_SHIFT, XAIE_ROW_SHIFT,
                     XAIE_NUM_COLS, XAIE_NUM_ROWS, XAIE_SHIM_ROW,
                     XAIE_RES_TILE_ROW_START, XAIE_RES_TILE_NUM_ROWS,
                     XAIE_AIE_TILE_ROW_START, XAIE_AIE_TILE_NUM_ROWS);

    XAie_InstDeclare(DevInst, &ConfigPtr);

    AieRC RC = XAie_CfgInitialize(&DevInst, &ConfigPtr);
    if(RC != XAIE_OK) {
        printf("Driver initialization failed.\n");
        return -1;
    }

    XAie_SetIOBackend(&DevInst, XAIE_IO_BACKEND_BAREMETAL);

#if AIE_GEN >= 2
    if(DevInst.Backend->Type == XAIE_IO_BACKEND_BAREMETAL) {
        RC = XAie_UpdateNpiAddr(&DevInst, 0xF6D10000);
        if(RC != XAIE_OK) {
            printf("Failed to update NPI address\n");
            return -1;
        }
    }
    RC = XAie_PartitionInitialize(&DevInst, NULL);
    if(RC != XAIE_OK) {
        printf("Failed to initialize partition\n");
        return -1;
    }
#else
    XAie_PmRequestTiles(&DevInst, NULL, 0); 
#endif

    if(test_col(&DevInst, 1) == 0) {
        printf("\nKernel test passed!\n");
        return 0;
    } else {
        printf("\nKernel test failed!\n");
        return -1;
    }
    
}
