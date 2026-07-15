/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <iostream>
//#include <boost/property_tree/ptree.hpp>
//#include <boost/property_tree/json_parser.hpp>
#include <iostream>
#include <sstream>
#include "xil_printf.h"
#include "xaiengine.h"
#include "xil_io.h"
#include "xil_cache.h"
#include <stdio.h>
#include <math.h>

#define uint_TYPE uint32_t
#define AIE_GEN 2
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

#define CORE_IP_MEM 0x4000
#define CORE_OP_MEM 0x5000
//unsigned char* mm;
__global__ void mm(input_window_int32 * win __attribute__((annotate("mem_address:0x4000"))), output_window_int32 * out __attribute__((annotate("mem_address:0x5000"))))
{
    #define DATA_SIZE 128 
    #define MAT_SIZE 64
    #define N 8          // Dimension of the square matrices
    #define VECTOR_LENGTH 8

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
    return;
}
void test2(XAie_DevInst* inst ){
	return;
}
int test_routing(XAie_DevInst *DevInst)
{
	AieRC RC = XAIE_OK;
	XAie_RoutingInstance* routingInstance;
	
	XAie_CoreReset(DevInst, XAie_TileLoc(4,4));
	XAie_CoreUnreset(DevInst, XAie_TileLoc(4,4));
	XAie_LoadElfMem(DevInst, XAie_TileLoc(4,4), (unsigned char*)mm);

	routingInstance = XAie_InitRoutingHandler(DevInst);
	XAie_Route(routingInstance, NULL, XAie_TileLoc(2,0),
				 XAie_TileLoc(4,4) 	);
	XAie_Route(routingInstance, NULL, XAie_TileLoc(4,4) ,
				 XAie_TileLoc(2,0) );

	printf("Routing successful\n");
	u64 phy = 0, phy_out = 0;
	u32 mlen = 256;
	const u32 recv_len = 256;
	
	//Prepare DDR data
	XAie_MemInst *in = XAie_MemAllocate(DevInst, mlen * sizeof(float), XAIE_MEM_CACHEABLE);
	phy = (u32)XAie_MemGetDevAddr(in);
	XAie_MemInst *out = XAie_MemAllocate(DevInst, mlen * sizeof(float), XAIE_MEM_CACHEABLE);
	phy_out = (u32)XAie_MemGetDevAddr(out);

	u64 vmem =    phy;
	u64 vmem_out = phy_out;
	//XAie_MemSyncForCPU(in);
	//XAie_MemSyncForCPU(out);
	for(int i = 0; i < mlen; i++) {
		((float*)vmem)[i] = (float)rand() / RAND_MAX;
		((float*)vmem_out)[i] = 0;
	}

	printf("aiebaremetal alloc memory \n");

	printf("Starting to Move data\n");
	// step 3: move data to destination tile
	XAie_MoveData(routingInstance,  XAie_TileLoc(2,0) ,
		((void*)(phy)), mlen*sizeof(float),
		(void*)(CORE_IP_MEM), XAie_TileLoc(4,4) );					

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

	XAie_MoveData(routingInstance,  XAie_TileLoc(4,4) ,
		(void*)(CORE_OP_MEM), mlen*sizeof(float),
		(void*)(phy_out), XAie_TileLoc(2,0) );					

	printf("\nFinished moving data back to DDR\n");
	// step 5 validate data
    float vmem_out_cpu[mlen];
    float sum = 0.0;
	//vmem contains the input (256 samples) and vmem_out contains the output sample 
	//compute CPU Result for softmax
	for (int i = 0; i < mlen; i++) {
        //float exp_val = (float)exp(double(((float*)vmem)[i]));
        vmem_out_cpu[i] = ((float*)vmem)[i] * ((float*)vmem)[i];
  }

    printf("Input Parameter:\n");
    for (int i = 0; i < mlen; i++) {
        printf("%f ", ((float*)vmem)[i]);
        if ((i + 1) % 16 == 0) {
            printf("\n");
        }
    }
    printf("\n");

    // Verify the output
    printf("CPU Softmax Output:\n");
    for (int i = 0; i < mlen; i++) {
        printf("%f ", vmem_out_cpu[i]);
        if ((i + 1) % 16 == 0) {
            printf("\n");
        }
    }
    printf("\n");

	// Verify the output
    printf("Kernel Softmax Output:\n");
    for (int i = 0; i < mlen; i++) {
        printf("%f ", ((float*)vmem_out)[i]);
        if ((i + 1) % 16 == 0) {
            printf("\n");
        }
    }
    printf("\n");

    // Compare the output from the kernel (vmem_out) with the CPU result
    printf("Comparing Kernel Output with CPU Result:\n");
    for (int i = 0; i < mlen; i++) {
        if (fabs(vmem_out_cpu[i] - ((float*)vmem_out)[i]) > 1e-2) {
            printf("Mismatch at index %d: CPU=%f, Kernel=%f\n", i, vmem_out_cpu[i], ((float*)vmem_out)[i]);
            return 1;
        }
    }

    printf("Kernel output matches CPU result.\n");

	printf("\nsuccess\n");
	return 0;
}
//*/
void test3();
int main(int argc, char* argv[]) {
	Xil_DCacheDisable();
	Xil_ICacheDisable();
	XAie_SetupConfig(ConfigPtr, HW_GEN, XAIE_BASE_ADDR,
			XAIE_COL_SHIFT, XAIE_ROW_SHIFT,
			XAIE_NUM_COLS, XAIE_NUM_ROWS, XAIE_SHIM_ROW,
			XAIE_RES_TILE_ROW_START, XAIE_RES_TILE_NUM_ROWS,
			XAIE_AIE_TILE_ROW_START, XAIE_AIE_TILE_NUM_ROWS);

	/*

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
*/
  test_routing(&DevInst);
	test2(&DevInst);
	//*/
	test3();
	return 1;
}
