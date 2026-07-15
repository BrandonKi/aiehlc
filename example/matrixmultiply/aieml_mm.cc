/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <iostream>
#include <sstream>
#include "xil_printf.h"
#include "xaiengine.h"
// #include "xil_io.h"
#include "xil_cache.h"
#include <stdio.h>
#include <math.h>
#include "xtime_l.h"

#define AIE_GEN 2
#define uint_TYPE uint32_t

#define HW_GEN XAIE_DEV_GEN_AIEML

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

#define CORE_IP_MEM 0x1000
#define CORE_OP_MEM 0x6000

#define MAT_SIZE 256 // Size of each matrix (A and B)
#define N 16         // Dimension of the square matrices (16x16)

#define DISABLE_CACHE

__global__ void mm(input_window_int32 * win __attribute__((annotate("mem_address:0x1000"), annotate("size_hint:512"))),
        output_window_int32 *out __attribute__((annotate("mem_address:0x6000"), annotate("size_hint:512"))))
{
#define DATA_SIZE 512 
#define MAT_SIZE 256
#define N 16          // Dimension of the square matrices
#define VECTOR_LENGTH 16
///*
    int32_t A_mat[MAT_SIZE];  //A contains 64 samples denoting a 8*8 matrix
    int32_t B_mat[MAT_SIZE];  //B contains 64 samples denoting a 8*8 matrix
    int32_t result[MAT_SIZE];
    for (unsigned w = 0; w < DATA_SIZE / VECTOR_LENGTH; w++)
    chess_prepare_for_pipelining{
        if (w*VECTOR_LENGTH < MAT_SIZE)
        {
          //first half of data goes into matrix A
          aie::vector<int32_t, VECTOR_LENGTH> temp_a = window_readincr_v<VECTOR_LENGTH>(win);
          aie::store_unaligned_v<VECTOR_LENGTH>(A_mat + (w*VECTOR_LENGTH), temp_a);
        }
        else
        {
          aie::vector<int32_t, VECTOR_LENGTH> temp_b = window_readincr_v<VECTOR_LENGTH>(win);
          aie::store_unaligned_v<VECTOR_LENGTH>(B_mat + (w*VECTOR_LENGTH-MAT_SIZE), temp_b);
        }
    }

    // Multiply A_mat and B_mat
    for (int i = 0; i < N; i++) {     // Iterate over rows of A_mat
        for (int j = 0; j < N; j++) { // Iterate over columns of B_mat
            for (int k = 0; k < N; k++) {
                // A_mat is in row-major order, so its index is i*N + k
                // B_mat is in column-major order, so its index is j*N + k
                result[i*N + j] += A_mat[i*N + k] * B_mat[j*N + k];
            }
        }
    }

    for (unsigned z = 0; z < MAT_SIZE/VECTOR_LENGTH; z++) chess_prepare_for_pipelining{
  		window_writeincr(out, aie::load_v<VECTOR_LENGTH>(result + (z * VECTOR_LENGTH)));
    }
		//log
		/*
		uint32_t* p = (uint32_t *)(0x70000 + 0x6000);
		for (int i = 0; i < 512; i++) {
			if (i < 256) {
				*(p++) = result[i]+0x10000;
			} else {
				*(p++) = B_mat[i-256]+0x20000;

			}
		}
		*/
}
void blockread(XAie_DevInst *DevInst, uint64_t addr)
{
#define DSIZE  512
  uint32_t odata[DSIZE];
	XAie_DataMemBlockRead(DevInst, XAie_TileLoc(4,4),  addr,
			 (void*)odata, DSIZE * sizeof(uint32_t));

	// step 5 validate data
	printf("addr = 0x%x\n", addr);
	for(uint32_t i = 0U; i < DSIZE; i++) {
		printf("odata[%d] = %x\n", i, odata[i]);
	}

}
int test_routing(XAie_DevInst *DevInst)
{
	AieRC RC = XAIE_OK;
	XAie_RoutingInstance* routingInstance;
  XTime tStart, tEnd;

	XAie_CoreReset(DevInst, XAie_TileLoc(4,4));
	XAie_CoreUnreset(DevInst, XAie_TileLoc(4,4));
	XAie_LoadElfMem(DevInst, XAie_TileLoc(4,4), (unsigned char *)mm);

	routingInstance = XAie_InitRoutingHandler(DevInst);
	XAie_Route(routingInstance, NULL, XAie_TileLoc(2,0) /* Source*/,
				 XAie_TileLoc(4,4) /* destination*/	);
	XAie_Route(routingInstance, NULL, XAie_TileLoc(4,4) /* Source*/,
				 XAie_TileLoc(2,0) /* destination*/	);

	//printf("Routing successful\n");
	u64 phy = 0, phy_out = 0;
	u32 mlen = MAT_SIZE*2;
	const u32 recv_len = MAT_SIZE;
	
	//Prepare DDR data
	XAie_MemInst *in = XAie_MemAllocate(DevInst, mlen * sizeof(u32), XAIE_MEM_CACHEABLE);
	phy = (u32)XAie_MemGetDevAddr(in);
	XAie_MemInst *out = XAie_MemAllocate(DevInst, mlen * sizeof(u32), XAIE_MEM_CACHEABLE);
	phy_out = (u32)XAie_MemGetDevAddr(out);

	u64 vmem =    phy;
	u64 vmem_out = phy_out;
	for(int i = 0; i < mlen; i++) {
		((int32_t*)vmem)[i] = i;
		((int32_t*)vmem_out)[i] = 0;
	}

	XAie_MemSyncForDev(in);
	XAie_MemSyncForDev(out);
	//printf("aiebaremetal alloc memory \n");

	//printf("Starting to Move data\n");
	// step 3: move data to destination tile
	XTime_GetTime(&tStart);
	XAie_MoveData(routingInstance,  XAie_TileLoc(2,0) /* Source*/,
		((void*)(phy)), mlen*sizeof(uint32_t),
		(void*)(CORE_IP_MEM), XAie_TileLoc(4,4) /* destination*/);					

  //uint32_t odata[32];
	//XAie_DataMemBlockRead(DevInst, XAie_TileLoc(4,4),  0x4000,
	//		 (void*)odata, 16 * sizeof(uint32_t));

	// step 5 validate data
	/*
	for(uint8_t i = 0U; i < 16; i++) {
		printf("odata[%d] = %x\n", i, odata[i]);
	}
	*/
  //blockread(DevInst, CORE_IP_MEM);

	XAie_Run(routingInstance, 1);

	//wait until core done
	u8 allDone  = 0;
	uint32_t CoreStatus = 0;
	do {
		allDone = 1; // Assume all cores are done initially
		uint32_t coreStatCharWritten = 0;
		for (int i = 0; i < 1; i++) { // Iterate over the specified tiles
			RC = XAie_CoreWaitForDone(DevInst, XAie_TileLoc(4,4), 0);

			if (RC != XAIE_OK) {
				allDone = 0;
			}
		}
	} while (!allDone);

  //blockread(DevInst, CORE_OP_MEM);

	XAie_MoveData(routingInstance,  XAie_TileLoc(4,4) /* Source*/,
		(void*)(CORE_OP_MEM), recv_len*sizeof(uint32_t),
		(void*)(phy_out), XAie_TileLoc(2,0) /* destination*/);					

	XAie_MemSyncForCPU(out);
	XTime_GetTime(&tEnd);
	printf("Output took %.2f us.\n", 1.0 * (tEnd - tStart) / (COUNTS_PER_SECOND/1000000));

	//printf("\nFinished moving data back to DDR\n");
	// step 5 validate data
  int32_t vmem_out_cpu[recv_len];

	//vmem contains the input (128 samples, 64 of matrix A and 64 of matrix B, in row major and column major forms respectively) and vmem_out contains the output samples (64 of result) 
	//compute CPU Result for softmax
    int32_t A_mat[N][N];  // Matrix A
    int32_t B_mat[N][N];  // Matrix B
    int32_t result[N][N] = {0};  // Result matrix

    // Extract matrix A (row major)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A_mat[i][j] = ((int32_t*)vmem)[i * N + j];
        }
    }

    // Extract matrix B (column major)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            B_mat[i][j] = ((int32_t*)vmem)[MAT_SIZE + i * N + j];  // Adjust index for column major
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
        if (vmem_out_cpu[i] != ((int32_t*)vmem_out)[i]) {

            printf("Mismatch at index %d: CPU=%d, vmem_out=%d\n", i, vmem_out_cpu[i], ((int32_t*)vmem_out)[i]);
            mismatches++;
        }
    }

    if (mismatches == 0) {
        printf("CPU result matches vmem_out.\n");
    } else {
        printf("There were %d mismatches.\n", mismatches);
    }

	printf("\nDone\n");
	return 0;
}

int main(int argc, char* argv[]) {
#ifdef DISABLE_CACHE
	Xil_DCacheDisable();
	Xil_ICacheDisable();
	printf("Cache Disabled performance will have big drop (this test should >350us(8*8 and 16*16)\n ");
#else
	printf("cache enabled, this test should be 15us(8*8) 215 us(16*16)");
#endif
	XAie_SetupConfig(ConfigPtr, HW_GEN, XAIE_BASE_ADDR,
			XAIE_COL_SHIFT, XAIE_ROW_SHIFT,
			XAIE_NUM_COLS, XAIE_NUM_ROWS, XAIE_SHIM_ROW,
			XAIE_RES_TILE_ROW_START, XAIE_RES_TILE_NUM_ROWS,
			XAIE_AIE_TILE_ROW_START, XAIE_AIE_TILE_NUM_ROWS);

	///*

	XAie_InstDeclare(DevInst, &ConfigPtr);

	AieRC RC = XAie_CfgInitialize(&DevInst, &ConfigPtr);
	if(RC != XAIE_OK) {
		printf("Driver initialization failed.\n");
		return -1;
	}
	printf("1\n");

	XAie_SetIOBackend(&DevInst, XAIE_IO_BACKEND_BAREMETAL);
	printf("1.0\n");

#if AIE_GEN >= 2
	if(DevInst.Backend->Type == XAIE_IO_BACKEND_BAREMETAL) {
		RC = XAie_UpdateNpiAddr(&DevInst, 0xF6D10000);
		if(RC != XAIE_OK) {
			printf("Failed to update NPI address\n");
			return -1;
		}
	}
	//fix in aie2 the shim dma not work issue
	RC = XAie_PartitionInitialize(&DevInst, NULL);
#else
	//fix in aie1 shimd dma not work issue
	XAie_PmRequestTiles(&DevInst, NULL, 0); 
#endif
  	test_routing(&DevInst);

	return 1;
}
