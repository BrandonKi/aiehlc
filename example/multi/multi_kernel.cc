/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "xaiengine.h"

#include "aie_device_map.h"
#include "aie_timer.h"

#ifndef __AIESIM__
#include "xil_cache.h"
#include "xil_printf.h"
#endif

#ifdef __AIELINUX__
#define OUT_COL 35
#elif defined(__AIESIM__)
#define OUT_COL 3
#else
#define OUT_COL 2
#endif

#include <stdio.h>

// these are fallbacks, the actual values are injected via dm_offsets.h by aiehlc
#ifndef CORE_IP_MEM
#define CORE_IP_MEM 0x9000
#endif
#ifndef CORE_OP_MEM
#define CORE_OP_MEM 0xA000
#endif

#define DISABLE_CACHE

__global__ void k1(int *in, int *out) {
    for (int i = 0; i < 20; ++i)
        *(out++) = 100 + i;
}

__global__ void k2(int *in, int *out) {
    for (int i = 0; i < 20; ++i)
        *(out++) = 200 + i;
}

int test_kernel1(XAie_DevInst *DevInst) {
    printf("\nLoading kernel 1...\n");
    XAie_CoreReset(DevInst, XAie_TileLoc(4,4));
    XAie_CoreUnreset(DevInst, XAie_TileLoc(4,4));
    XAie_LoadElfMem(DevInst, XAie_TileLoc(4,4), (unsigned char *)k1);
    printf("Finished. Continuing...\n");
    
    printf("\nRouting...\n");
    XAie_RoutingInstance* routingInstance = XAie_InitRoutingHandler(DevInst);
    XAie_Route(routingInstance, NULL, /*src=*/ XAie_TileLoc(2,0), /*dest=*/ XAie_TileLoc(4,4));
    XAie_Route(routingInstance, NULL, /*src=*/ XAie_TileLoc(4,4), /*dest=*/ XAie_TileLoc(OUT_COL,0));
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
        in_ptr[i] = 0;
        out_ptr[i] = 0;
    }

    XAie_MemSyncForDev(in);
    XAie_MemSyncForCPU(out);
    printf("Finished. Continuing...\n");

    printf("\nMoving input data to AIE...\n");
    XAie_MoveDataExternal2Aie(routingInstance, /*src=*/ XAie_TileLoc(2,0),
                              in, len*sizeof(u32),
                              CORE_IP_MEM, /*dest=*/ XAie_TileLoc(4,4));
    printf("Finished. Continuing...\n");
    
    printf("\nRunning kernel 1...\n");
    XTime tStart, tEnd;
    XTime_GetTime(&tStart);
    XAie_Run(routingInstance, 1);

    while(XAie_CoreWaitForDone(DevInst, XAie_TileLoc(4,4), 0) != XAIE_OK) {
        // wait until core done
    }
    XTime_GetTime(&tEnd);
    printf("Output took %.2f us.\n", 1.0 * (tEnd - tStart) / (COUNTS_PER_SECOND/1000000));
    printf("Finished. Continuing...\n");

    printf("\nMoving output data to CPU...\n");
    XAie_MoveDataAie2External(routingInstance, /*src=*/XAie_TileLoc(4, 4), CORE_OP_MEM, len * sizeof(u32), out,
                              /*dest=*/XAie_TileLoc(OUT_COL, 0));
    XAie_MemSyncForCPU(out);
    printf("Finished. Continuing...\n");

    printf("\nVerifying output data...\n");
    int mismatches = 0;
    for (int i = 0; i < len; i++) {
        if (100 + i != out_ptr[i]) {
            printf("Mismatch at index %d: CPU=%d, AIE=%d\n", i, 100 + i, out_ptr[i]);
            mismatches++;
        } else {
            printf("%d: CPU=%d, AIE=%d\n", i, 100 + i, out_ptr[i]);
        }
    }

    if (mismatches == 0) {
        printf("Sucess: CPU result matches AIE.\n");
        printf("\n\nDone with kernel 1!\n\n");
        return 0;
    } else {
        printf("Failure: There were %d mismatches.\n", mismatches);
        printf("\n\nDone with kernel 1!\n\n");
        return -1;
    }
}

int test_kernel2(XAie_DevInst *DevInst) {
    printf("\nLoading kernel 2...\n");
    XAie_CoreReset(DevInst, XAie_TileLoc(4,4));
    XAie_CoreUnreset(DevInst, XAie_TileLoc(4,4));
    XAie_LoadElfMem(DevInst, XAie_TileLoc(4,4), (unsigned char *)k2);
    printf("Finished. Continuing...\n");
    
    printf("\nRouting...\n");
    XAie_RoutingInstance* routingInstance = XAie_InitRoutingHandler(DevInst);
    XAie_Route(routingInstance, NULL, /*src=*/ XAie_TileLoc(2,0), /*dest=*/ XAie_TileLoc(4,4));
    XAie_Route(routingInstance, NULL, /*src=*/ XAie_TileLoc(4,4), /*dest=*/ XAie_TileLoc(OUT_COL,0));
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
        in_ptr[i] = 0;
        out_ptr[i] = 0;
    }

    XAie_MemSyncForDev(in);
    XAie_MemSyncForCPU(out);
    printf("Finished. Continuing...\n");

    printf("\nMoving input data to AIE...\n");
    XAie_MoveDataExternal2Aie(routingInstance, /*src=*/ XAie_TileLoc(2,0),
                              in, len*sizeof(u32),
                              CORE_IP_MEM, /*dest=*/ XAie_TileLoc(4,4));
    printf("Finished. Continuing...\n");
    
    printf("\nRunning kernel 2...\n");
    XTime tStart, tEnd;
    XTime_GetTime(&tStart);
    XAie_Run(routingInstance, 1);

    while(XAie_CoreWaitForDone(DevInst, XAie_TileLoc(4,4), 0) != XAIE_OK) {
    //wait until core done
    }
    XTime_GetTime(&tEnd);
    printf("Output took %.2f us.\n", 1.0 * (tEnd - tStart) / (COUNTS_PER_SECOND/1000000));
    printf("Finished. Continuing...\n");

    printf("\nMoving output data to CPU...\n");
    XAie_MoveDataAie2External(routingInstance, /*src=*/XAie_TileLoc(4, 4), CORE_OP_MEM, len * sizeof(u32), out,
                              /*dest=*/XAie_TileLoc(OUT_COL, 0));
    XAie_MemSyncForCPU(out);
    printf("Finished. Continuing...\n");

    printf("\nVerifying output data...\n");
    int mismatches = 0;
    for (int i = 0; i < len; i++) {
        if (200 + i != out_ptr[i]) {
            printf("Mismatch at index %d: CPU=%d, AIE=%d\n", i, 200 + i, out_ptr[i]);
            mismatches++;
        } else {
            printf("%d: CPU=%d, AIE=%d\n", i, 200 + i, out_ptr[i]);
        }
    }

    if (mismatches == 0) {
        printf("Sucess: CPU result matches AIE.\n");
        printf("\n\nDone with kernel 2!\n\n");
        return 0;
    } else {
        printf("Failure: There were %d mismatches.\n", mismatches);
        printf("\n\nDone with kernel 2!\n\n");
        return -1;
    }
}

int main(int argc, char* argv[]) {
#if defined(DISABLE_CACHE) && !defined(__AIESIM__)
    Xil_DCacheDisable();
    Xil_ICacheDisable();
    printf("Cache disabled\n");
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

#ifdef __AIESIM__
    XAie_SetIOBackend(&DevInst, XAIE_IO_BACKEND_SIM);
#else
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
#endif /* __AIESIM__ */

    if(test_kernel1(&DevInst) == 0 && test_kernel2(&DevInst) == 0) {
        printf("\nKernel test passed!\n");
        return 0;
    } else {
        printf("\nKernel test failed!\n");
        return -1;
    }
    
}
