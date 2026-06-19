/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "xaiengine.h"
#include <math.h>
#include <stdio.h>

#ifndef __AIESIM__
#include "xil_cache.h"
#include "xil_printf.h"
#if AIE_GEN <= 2
#include "xtime_l.h"
#else
#include "xiltimer.h"
#endif
#endif /* __AIESIM__ */

#define uint_TYPE uint32_t

#if AIE_GEN <= 2
#define HW_GEN XAIE_DEV_GEN_AIEML
#else
#define HW_GEN XAIE_DEV_GEN_AIE2PS
#endif

#if AIE_GEN <= 2

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
#else

#define XAIE_BASE_ADDR 0x20000000000
#define XAIE_COL_SHIFT 25
#define XAIE_ROW_SHIFT 20

#define XAIE_NUM_ROWS 7
#define XAIE_NUM_COLS 36
#define XAIE_SHIM_ROW 0
#define XAIE_RES_TILE_ROW_START 1
#define XAIE_RES_TILE_NUM_ROWS 2
#define XAIE_AIE_TILE_ROW_START 3
#define XAIE_AIE_TILE_NUM_ROWS 4

#endif

// real offsets are injected by aiehlc
#ifndef CORE_IP_MEM
#define CORE_IP_MEM 0x1000
#endif
#ifndef CORE_OP_MEM
#define CORE_OP_MEM 0x6000
#endif

// Reduced for simulator
// #define N 16
#define N 4
#define MAT_SIZE (N * N)

// #define DISABLE_CACHE
//__attribute__((annotate("streaming")))
__global__ void perf(input_window_int32 *win __attribute__((annotate("mem_address:0x1000"), annotate("size_hint:512"))),
                     output_window_int32 *out
                     __attribute__((annotate("mem_address:0x6000"), annotate("size_hint:512")))) {
#define N 4
#define MAT_SIZE (N * N)
#define DATA_SIZE (MAT_SIZE * 2)
#define VECTOR_LENGTH 16
	//aie::vector<int32_t, VECTOR_LENGTH> temp_a = window_readincr_v<VECTOR_LENGTH>(win);
	//aie::store_unaligned_v<VECTOR_LENGTH>(A_mat + (w*VECTOR_LENGTH), temp_a);
	uint32_t * ptr_out = (uint32_t *)(0x70000 + 0x6000);
	uint32_t * ptr_in = (uint32_t *)(0x70000 + 0x1000);

    uint32_t * vec1 = ((uint32_t*)ptr_in), * vec2 = ((uint32_t*)ptr_in + MAT_SIZE);

    for (int i = 0; i < N; i++) {
        for (uint32_t j = 0; j < N; j++) {
            uint32_t ret = 0;
            for (int k = 0; k < N; k++) {
                ret += vec1[i * N + k] * vec2[j * N + k];
            }
            ptr_out[i * N + j] = ret;
        }
    }
}
void blockread(XAie_DevInst *DevInst, uint64_t addr)
{
#define DSIZE  512
  uint32_t odata[DSIZE];
	XAie_DataMemBlockRead(DevInst, XAie_TileLoc(4,4),  addr,
			 (void*)odata, DSIZE * sizeof(uint32_t));

	// step 5 validate data
	printf("addr = 0x%lx\n", addr);
	for(uint32_t i = 0U; i < DSIZE; i++) {
		printf("odata[%d] = %x\n", i, odata[i]);
	}

}
void breakprint(char *info) {
	return;
	char input;
	printf("input key to run %s\n", info);
	scanf("%c", input);
	printf("log--- %s done\n", info);
}
int test_routing(XAie_DevInst *DevInst)
{
	AieRC RC = XAIE_OK;
	XAie_RoutingInstance* routingInstance;
#ifndef __AIESIM__
    XTime tStart, tEnd;
#endif
    printf("Starting test_routing 02/2 -1\n");
    breakprint("core reset--");
#ifdef __AIESIM__
    int shimcol = 3;
#elif AIE_GEN == 5
    int shimcol = 10;
#else
    int shimcol = 6;
#endif
    XAie_CoreReset(DevInst, XAie_TileLoc(4, 4));
    XAie_LoadElfMem(DevInst, XAie_TileLoc(4, 4), (unsigned char *)perf);
    XAie_CoreUnreset(DevInst, XAie_TileLoc(4, 4));

    routingInstance = XAie_InitRoutingHandler(DevInst);
    XAie_Route(routingInstance, NULL, XAie_TileLoc(shimcol, 0) /* Source*/, XAie_TileLoc(4, 4) /* destination*/);
    XAie_Route(routingInstance, NULL, XAie_TileLoc(4, 4) /* Source*/, XAie_TileLoc(shimcol, 0) /* destination*/);

    u32 mlen = MAT_SIZE * 2;
    const u32 recv_len = MAT_SIZE;
	
	//Prepare DDR data
    XAie_MemInst *in = XAie_MemAllocate(DevInst, mlen * sizeof(u32), XAIE_MEM_CACHEABLE);
    XAie_MemInst *out = XAie_MemAllocate(DevInst, mlen * sizeof(u32), XAIE_MEM_CACHEABLE);

    breakprint(" XAie_MemSyncForDev---\n");

    int32_t *vmem = (int32_t *)XAie_MemGetVAddr(in);
    int32_t *vmem_out = (int32_t *)XAie_MemGetVAddr(out);

    for (int j = 0; j < mlen; j++) {
        vmem[j] = 1 + j;
        vmem_out[j] = j;
    }

    const int count = 2; // iterations for perf measurement

#ifndef __AIESIM__
    XTime_GetTime(&tStart);
#endif
    for (int i = 0; i < count; i++) {

#ifdef __AIESIM__
        if (i > 0) {
            XAie_CoreReset(DevInst, XAie_TileLoc(4, 4));
            XAie_LoadElfMem(DevInst, XAie_TileLoc(4, 4), (unsigned char *)perf);
            XAie_CoreUnreset(DevInst, XAie_TileLoc(4, 4));
        }
#endif

        XAie_MemSyncForDev(in);

        breakprint("Starting to Move data\n");
        // step 3: move data to destination tile
        // XTime_GetTime(&tStart);
        // printf("vmem = 0x%p\n",vmem);

        XAie_MoveDataExternal2Aie(routingInstance, /*src=*/XAie_TileLoc(shimcol, 0), in, mlen * sizeof(u32),
                                  CORE_IP_MEM, /*dest=*/XAie_TileLoc(4, 4));
        XAie_RouteDmaWait(routingInstance, XAie_TileLoc(shimcol, 0), XAie_TileLoc(4, 4), true);
        XAie_Run(routingInstance, 1);

#ifdef __AIESIM__
        while (XAie_CoreWaitForDone(DevInst, XAie_TileLoc(4, 4), 1) != XAIE_OK) {
        }
#else
        XAie_CoreWaitForDone(DevInst, XAie_TileLoc(4, 4), 0);
#endif

        breakprint("fflush\n");

#ifndef __AIESIM__
        Xil_DCacheFlushRange((INTPTR)vmem_out, mlen * sizeof(int32_t));
#endif
        XAie_MoveDataAie2External(routingInstance, XAie_TileLoc(4, 4), CORE_OP_MEM, mlen * sizeof(u32), out,
                                  XAie_TileLoc(shimcol, 0));
        XAie_RouteDmaWait(routingInstance, XAie_TileLoc(4, 4), XAie_TileLoc(shimcol, 0), false);
        XAie_MemSyncForCPU(out);
#ifndef __AIESIM__
        Xil_DCacheInvalidateRange((INTPTR)vmem_out, mlen * sizeof(int32_t));
#endif

        ///*
        // XTime_GetTime(&tEnd);
        // printf("Output took %.2f us.\n", 1.0 * (tEnd - tStart) / (COUNTS_PER_SECOND/1000000));

        //printf("\nFinished moving data back to DDR\n");
        // step 5 validate data
        int32_t vmem_out_cpu[recv_len];

        // vmem contains the input (128 samples, 64 of matrix A and 64 of matrix B, in row major and column major forms
        // respectively) and vmem_out contains the output samples (64 of result)
        // compute CPU Result for softmax
        int32_t A_mat[N][N];  // Matrix A
        int32_t B_mat[N][N];  // Matrix B
        int32_t result[N][N] = {0};  // Result matrix

        // Extract matrix A (row major)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A_mat[i][j] = vmem[i * N + j];
            }
        }

        // Extract matrix B (column major)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                B_mat[i][j] = vmem[MAT_SIZE + i * N + j];
            }
        }

        // Perform matrix multiplication
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    result[i][j] += A_mat[i][k] * B_mat[j][k];
                }
            }
        }

        // Store the result in vmem_out_cpu
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                vmem_out_cpu[i * N + j] = result[i][j];
            }
        }

        int mismatches = 0;
        for (int i = 0; i < MAT_SIZE; i++) {
            if (vmem_out_cpu[i] != vmem_out[i]) {
                if (mismatches < 4)
                    printf("[MISMATCH] [%d]: CPU=%d AIE=%d\n", i, vmem_out_cpu[i], vmem_out[i]);
                mismatches++;
            }
        }

        if (mismatches == 0) {
            printf("\n[PASS] Iteration %d: all %d outputs match CPU reference.\n\n", i, MAT_SIZE);
        } else {
            printf("\n[FAIL] Iteration %d: %d / %d mismatches.\n\n", i, mismatches, MAT_SIZE);
        }
        fflush(stdout);
        //*/
    }
#ifndef __AIESIM__
    XTime_GetTime(&tEnd);
    printf("Transferred %u bytes (%d iter) in %.2f us.\n", (unsigned)(count * mlen * sizeof(u32)), count,
           1.0 * (tEnd - tStart) / (COUNTS_PER_SECOND / 1000000));
#endif
    return 0;
}

int main(int argc, char* argv[]) {
#ifndef __AIESIM__
#ifdef DISABLE_CACHE
	Xil_DCacheDisable();
	Xil_ICacheDisable();
    printf("Cache disabled: expect a big perf drop (>350 us for 8x8 and 16x16).\n");
#else
    printf("Cache enabled: expect ~15 us for 8x8, ~215 us for 16x16.\n");
#endif
#endif /* __AIESIM__ */

    XAie_SetupConfig(ConfigPtr, HW_GEN, XAIE_BASE_ADDR, XAIE_COL_SHIFT, XAIE_ROW_SHIFT, XAIE_NUM_COLS, XAIE_NUM_ROWS,
                     XAIE_SHIM_ROW, XAIE_RES_TILE_ROW_START, XAIE_RES_TILE_NUM_ROWS, XAIE_AIE_TILE_ROW_START,
                     XAIE_AIE_TILE_NUM_ROWS);

    ///*

	XAie_InstDeclare(DevInst, &ConfigPtr);

	int partitonnum = 34;
    int startcol = 1;
    int colnum = (startcol + partitonnum <= XAIE_NUM_COLS) ? partitonnum : (XAIE_NUM_COLS- startcol);

	AieRC RC;
    /*
    RC = XAie_SetupPartitionConfig(&DevInst, XAIE_BASE_ADDR + (startcol<<XAIE_COL_SHIFT),
                                       startcol, colnum);
    if(RC != XAIE_OK) {
        printf("Driver XAie_SetupPartitionConfig failed.\n");
        return -1;
    }
    //*/

    RC = XAie_CfgInitialize(&DevInst, &ConfigPtr);
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
		printf("XAie_UpdateNpiAddr()\n");
#if AIE_GEN == 5
        printf("XAie_UpdateNpiAddr(0xf6d50000)\n");
        RC = XAie_UpdateNpiAddr(&DevInst, 0xf6d50000);
#else
        RC = XAie_UpdateNpiAddr(&DevInst, 0xF6D10000);
#endif
        if(RC != XAIE_OK) {
			printf("Failed to update NPI address\n");
			return -1;
		}
	}
    printf("before XAie_PartitionInitialize-2--\n");
    RC = XAie_PartitionInitialize(&DevInst, NULL);
#else
    XAie_PmRequestTiles(&DevInst, NULL, 0);
#endif
#endif /* __AIESIM__ */

    test_routing(&DevInst);

    RC = XAie_PartitionTeardown(&DevInst);
    if(RC != XAIE_OK) {
        printf("Failed to Teardown partition\n");
        return -1;
	}

	return 1;
}
