/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

/*
    Test for running kernels on all tiles in parallel and verifiying the output:

    1. Runs all the tiles in parallel for selected columns or all columns
    2. Kernels run infinitely
    3. Kernel takes some input, and outputs the result of a matrix multiplication
    4. Output for each kernel is verified by the host 
    5. Option to stop and start the kernels
*/

#include "xaiengine.h"
#include "sleep.h"

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

#define XAIE_NUM_ROWS 7
#define XAIE_NUM_COLS 36
#define XAIE_SHIM_ROW 0
#define XAIE_RES_TILE_ROW_START 1
#define XAIE_RES_TILE_NUM_ROWS 2
#define XAIE_AIE_TILE_ROW_START 3
#define XAIE_AIE_TILE_NUM_ROWS 8

#define START_ROW 3
#define END_ROW 6

#define N 3

// default memory addresses for input and output
#define CORE_IP_MEM 0x1000
#define CORE_OP_MEM 0x5000

#define DISABLE_CACHE

// multiply "in" matrix by itself
__global__ void mm(uint64_t *in, uint64_t *out) {
    #define N 3
    uint64_t count = 0;
    while(true) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                int sum = 0;
                for (int k = 0; k < N; k++) {
                    sum += in[i * N + k] * in[k * N + j];
                }
                out[i * N + j] = sum;
            }
        }
        out[N * N] = ++count;
    }
}

void check_col_status(XAie_DevInst *DevInst, int test_col) {
    printf("\nChecking Core Status...\n");
    for (int row = START_ROW; row <= END_ROW; ++row) {
        XAie_LocType tileLoc = XAie_TileLoc(test_col, row);
        u8 enabled = 0;
        XAie_CoreReadDoneBit(DevInst, tileLoc, &enabled);
        printf("Tile (%d, %d): %s\n", test_col, row, enabled ? "DONE" : "EXECUTING");
    }
    printf("Finished. Continuing...\n");
}

bool check_tile_result(XAie_DevInst *DevInst, XAie_LocType Loc, u32 Value) {
    uint64_t output[N*N+1];
    XAie_DataMemBlockRead(DevInst, Loc, CORE_OP_MEM, (void*)&output, sizeof(output));
    bool matches = true;
    for(int i = 0; i < N * N; i++) {
        if(output[i] != Value) {
            matches = false;
        }
    }
    if(matches) {
        printf("Tile (%d, %d) output matches expected value: %d\n", Loc.Col, Loc.Row, Value);
        printf("Iterations: %d\n", output[N*N]);
        return true;
    } else {
        printf("Error: Tile (%d, %d) output mismatch: expected %d, got %d\n", Loc.Col, Loc.Row, Value, output[0]);
        printf("Iterations: %d\n", output[N*N]);
        return false;
    }
}

void start_col(XAie_DevInst *DevInst, int test_col) {
    printf("\nLoading column of kernels %d...\n", test_col);
    for(int i = START_ROW; i <= END_ROW; i++) {
        XAie_LocType tileLoc = XAie_TileLoc(test_col, i);
        XAie_CoreReset(DevInst, tileLoc);
        XAie_CoreUnreset(DevInst, tileLoc);
        printf("Tile (%d, %d) reset and unreset.\n", test_col, i);
        XAie_LoadElfMem(DevInst, XAie_TileLoc(test_col, i), (unsigned char *)mm);
        printf("Tile (%d, %d) loaded elf\n", test_col, i);
    }
    printf("Finished. Continuing...\n");

    printf("\nSetting up input data...\n");
    for(uint64_t i = START_ROW; i <= END_ROW; i++) {
        XAie_LocType Loc = XAie_TileLoc(test_col, (int)i);
        printf("Setting up input data for tile (%d, %d)\n", test_col, (int)i);

        uint64_t input[9] = {i, i, i, i, i, i, i, i, i};
        XAie_DataMemBlockWrite(DevInst, Loc, CORE_IP_MEM, (void*)&input, sizeof(input));
    }
    printf("Finished. Continuing...\n");
    
    printf("\nStarting kernels...\n");
    for(int i = START_ROW; i <= END_ROW; i++) {
        printf("Starting kernel (%d, %d)...\n", test_col, i);
        XAie_CoreEnable(DevInst, XAie_TileLoc(test_col, i));
        printf("Finished. Continuing...\n");
    }

    printf("\n\nStarted all kernels in col %d!\n\n", test_col);
}

bool check_col_output(XAie_DevInst *DevInst, int test_col) {
    printf("\nChecking tile results...\n");
    int expected_results[END_ROW - START_ROW + 1] = {27, 48, 75, 108};
    int mismatches = 0;
    for(int i = 0; i <= END_ROW - START_ROW; i++) {
        printf("\n\nReading Tile (%d, %d) output...\n", test_col, i);
        if(!check_tile_result(DevInst, XAie_TileLoc(test_col, START_ROW + i), expected_results[i])) {
            mismatches++;
        }
    }
    printf("Finished. Continuing...\n");

    if (mismatches == 0) {
        printf("\nSucess: CPU result matches AIE.\n");
        return true;
    } else {
        printf("\nFailure: There were %d mismatches.\n", mismatches);
        return false;
    }
}

void stop_col(XAie_DevInst *DevInst, int test_col) {
    printf("\nStopping column of kernels %d...\n", test_col);
    for(int i = START_ROW; i <= END_ROW; i++) {
        XAie_LocType tileLoc = XAie_TileLoc(test_col, i);
        XAie_CoreDisable(DevInst, tileLoc);
        printf("Tile (%d, %d) disabled.\n", test_col, i);
    }
    printf("Finished. Continuing...\n");
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

    int num_cols = XAIE_NUM_COLS;
    // int num_cols = 2; // can also test with a smaller number of columns
    for(int i = 0; i < num_cols; i++) {
        start_col(&DevInst, i);
    }

    printf("\n\nWaiting while kernels run...\n");
    sleep(5); // wait for kernels to run for 5 seconds

    for(int i = 0; i < num_cols; i++) {
        if(check_col_output(&DevInst, i)) {
            printf("\nCol %d test passed!\n", i);
        } else {
            printf("\nCol %d test failed!\n", i);
        }
    }

    // can also stop columns
    stop_col(&DevInst, 1);
    // stop_col(&DevInst, 2);
    
}
