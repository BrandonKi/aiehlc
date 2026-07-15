/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "xaie_placeholder.h"
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

#define DATA_SIZE 128
#define MAT_SIZE 64 // Size of each matrix (A and B)
#define N 8         // Dimension of the square matrices (8x8)

void test() {
	int a =2;
	return;
}
__global__ void mm(input_window_int32 * win, output_window_int32 *out)
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

int main() {
		test();
    size_t bytes = DATA_SIZE * sizeof(uint32_t);
    uint32_t recv_len = DATA_SIZE/2;
    // Allocate memory on the CPU
    uint32_t *h_a = (uint32_t *)malloc(bytes);
    uint32_t *h_c = (uint32_t *)malloc(bytes);

    // Initialize matrices
    for (int i = 0; i < DATA_SIZE; i++) {
        h_a[i] = i;  // Example values
        h_c[i] = 0.0;
    }

    XAie_Kernel_Inst* kinst = XAie_LoadKernel("mm", 12, 3);
    XAie_RoutingHostToAie(kinst,  XAie_TileLoc(2,0));
    XAie_RoutingAieToHost(kinst,  XAie_TileLoc(35,0));
    //as per xgemm, you run and then you start moving the data
    XAie_MoveDataToAie(kinst, 1, h_a, bytes);
    XAie_RunCore(kinst);
    XAie_WaitOutData(kinst);
    XAie_MoveAieToData(kinst, 1, h_c, bytes);
  
    int32_t vmem_out_cpu[recv_len];

	//vmem contains the input (128 samples, 64 of matrix A and 64 of matrix B, in row major and column major forms respectively) and vmem_out contains the output samples (64 of result) 
	//compute CPU Result for softmax
    int32_t A_mat[N][N];  // Matrix A
    int32_t B_mat[N][N];  // Matrix B
    int32_t result[N][N] = {0};  // Result matrix

    // Extract matrix A (row major)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A_mat[i][j] = h_a[i * N + j];
        }
    }

    // Extract matrix B (column major)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            B_mat[i][j] = h_a[MAT_SIZE + i * N + j];  // Adjust index for column major
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
        if (vmem_out_cpu[i] != h_c[i]) {
            printf("Mismatch at index %d: CPU=%d, vmem_out=%d\n", i, vmem_out_cpu[i], h_c[i]);
            mismatches++;
        }
    }

    if (mismatches == 0) {
        printf("CPU result matches vmem_out.\n");
    } else {
        printf("There were %d mismatches.\n", mismatches);
    }

	printf("\nsuccess\n");

    free(h_a);
    free(h_c);
    return 0;
}

