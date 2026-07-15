/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

// #include <iostream>
// #include <sstream>
#include "xaiengine.h"
#include "xil_printf.h"
// #include "xil_io.h"
#include "xil_cache.h"
#include <math.h>
#include <stdio.h>
#if AIE_GEN <= 2
#define HW_GEN XAIE_DEV_GEN_AIEML
#include "xtime_l.h"
#else
#define HW_GEN XAIE_DEV_GEN_AIE2PS
#include "xiltimer.h"
#endif
// #include "unistd.h"
#define uint_TYPE uint32_t

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

#define CORE_IP_MEM 0x1000
#define CORE_OP_MEM 0x2000

/* Ping-pong buffer addresses for streaming */
/* With 1024-byte buffers (256 words), pong must be offset by at least 0x400 */
#define CORE_IP_PING 0x1000
#define CORE_IP_PONG 0x2000
#define CORE_OP_PING 0x3000
#define CORE_OP_PONG 0x4000

/* Buffer sizes for streaming */
#define PING_PONG_SIZE 1024 // Size of each ping/pong buffer in WORDS (1024 * 4 = 4096 bytes)

/* BD IDs for streaming */
#define SHIM_IN_BD_ID 0
#define AIE_IN_PING_BD 0
#define AIE_IN_PONG_BD 1
#define AIE_OUT_PING_BD 2
#define AIE_OUT_PONG_BD 3
#define SHIM_OUT_BD_ID 1

/* Lock IDs for streaming (AIE-ML has 16 locks: 0-15) */
// the east lock is local lock and start from 48
#define IN_ACCQUIRE_LOCK_ID 0 // kernel access lock for local start from 16, so 0 is the first lock
#define IN_RELEASE_LOCK_ID 1  // kernel access lock for local start from 16, so 1 is the second lock
#define OUT_ACQUIRE_LOCK_ID 2 // kernel access lock for local start from 16, so 2 is the third lock
#define OUT_RELEASE_LOCK_ID 3 // kernel access lock for local start from 16, so 3 is the fourth lock

#define MAT_SIZE 512 // Size of each matrix (A and B)
#define N 16         // Dimension of the square matrices (16x16)

// #define DISABLE_CACHE
__attribute__((annotate("streaming"))) __global__ void perf(
    input_window_int32 *win
    __attribute__((annotate("mem_ping_address:0x1000"), annotate("mem_pong_address:0x2000"), annotate("size_hint:4096"),
                   annotate("lock_acquire_id:48"), annotate("lock_release_id:49"))),
    output_window_int32 *out
    __attribute__((annotate("mem_ping_address:0x3000"), annotate("mem_pong_address:0x4000"), annotate("size_hint:4096"),
                   annotate("lock_acquire_id:51"), // kernel acquire lock should be dma release lock cooperate with
                                                   // window_acquire(outpointer ) logic
                   annotate("lock_release_id:50")))) {
#define DATA_SIZE 1024
#define MAT_SIZE 128
#define N 16 // Dimension of the square matrices
#define VECTOR_LENGTH 32
#define COUNT 25000

    // Acquire windows before accessing data
    for (int iter = 0; iter < COUNT; iter++) {
        window_acquire(win);
        window_acquire(out);

        int32_t *ptr_out = (int32_t *)out->ptr; //(0x70000 + 0x2000);
        int32_t *ptr_in = (int32_t *)win->ptr;  //(0x70000 + 0x1000);

        // Use vector operations for maximum throughput
        // Process 32 elements at a time using AIE vector intrinsics with loop unrolling
        for (int j = 0; j < DATA_SIZE; j += VECTOR_LENGTH * 2) {
            // Process first 32 elements
            aie::vector<int32_t, VECTOR_LENGTH> v0 = aie::load_v<VECTOR_LENGTH>(ptr_in + j);
            aie::vector<int32_t, VECTOR_LENGTH> v1 = aie::load_v<VECTOR_LENGTH>(ptr_in + j + VECTOR_LENGTH);

            // Use add instead of multiply (v * 2 = v + v)
            v0 = aie::add(v0, v0);
            v1 = aie::add(v1, v1);

            aie::store_v(ptr_out + j, v0);
            aie::store_v(ptr_out + j + VECTOR_LENGTH, v1);
        }

        // Release windows after processing
        window_release(out);
        window_release(win);
    }
}
void blockread(XAie_DevInst *DevInst, uint64_t addr) {
#define DSIZE 512
    uint32_t odata[DSIZE];
    XAie_DataMemBlockRead(DevInst, XAie_TileLoc(4, 4), addr, (void *)odata, DSIZE * sizeof(uint32_t));

    // step 5 validate data
    printf("addr = 0x%lx\n", addr);
    for (uint32_t i = 0U; i < DSIZE; i++) {
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

/**
 * Debug function to print AIE core and DMA status
 */
void printAieDebugStatus(XAie_DevInst *DevInst, XAie_LocType tile, const char *context) {
    return;
    u32 coreStatus = 0;
    u32 lockValue = 0;
    u8 dmaS2mmStatus = 0;
    u8 dmaMm2sStatus = 0;

    printf("\n===== AIE DEBUG STATUS [%s] =====\n", context);
    printf("Tile: (%d, %d)\n", tile.Col, tile.Row);

    // Read core status
    AieRC rc = XAie_CoreGetStatus(DevInst, tile, &coreStatus);
    if (rc == XAIE_OK) {
        printf("Core Status: 0x%08X\n", coreStatus);
        printf("  - Core Enable: %s\n", (coreStatus & (1 << 0)) ? "Yes" : "No");
        printf("  - Core Reset: %s\n", (coreStatus & (1 << 1)) ? "Yes" : "No");
        printf("  - Core Done: %s\n", (coreStatus & (1 << 20)) ? "Yes" : "No");
        printf("  - Core Stuck: %s\n", (coreStatus & (1 << 21)) ? "Yes" : "No");
        printf("  - ECC Error: %s\n", (coreStatus & (1 << 17)) ? "Yes" : "No");
    } else {
        printf("Core Status: Failed to read (rc=%d)\n", rc);
    }

    // Read lock values for streaming locks (0-7)
    printf("Lock Status:\n");
    for (int lockId = 0; lockId < 8; lockId++) {
        rc = XAie_LockGetValue(DevInst, tile, XAie_LockInit(lockId, 0), &lockValue);
        if (rc == XAIE_OK) {
            printf("  Lock %d: value=%d\n", lockId, lockValue);
        } else {
            printf("  Lock %d: read failed (rc=%d)\n", lockId, rc);
        }
    }

    // Read DMA channel status
    printf("DMA Status:\n");
    for (int ch = 0; ch < 2; ch++) {
        AieRC rc1 = XAie_DmaGetPendingBdCount(DevInst, tile, ch, DMA_S2MM, &dmaS2mmStatus);
        AieRC rc2 = XAie_DmaGetPendingBdCount(DevInst, tile, ch, DMA_MM2S, &dmaMm2sStatus);
        if (rc1 == XAIE_OK && rc2 == XAIE_OK) {
            printf("  Channel %d: S2MM pending=%d, MM2S pending=%d\n", ch, dmaS2mmStatus, dmaMm2sStatus);
        } else {
            printf("  Channel %d: read failed\n", ch);
        }
    }

    printf("===== END DEBUG STATUS =====\n\n");
    fflush(stdout);
}
int test_routing(XAie_DevInst *DevInst) {
    AieRC RC = XAIE_OK;
    XAie_RoutingInstance *routingInstance;
    // XTime tStart, tEnd;
    breakprint("core reset-- 3");
    // printf("core test_routing-- start ---6\n");
#if AIE_GEN == XAIE_DEV_GEN_AIE2PS
    int shimcol = 10; // 33;
#else
    int shimcol = 33;
#endif
    XAie_CoreReset(DevInst, XAie_TileLoc(4, 4));
    XAie_CoreUnreset(DevInst, XAie_TileLoc(4, 4));
    XAie_LoadElfMem(DevInst, XAie_TileLoc(4, 4), (unsigned char *)perf);

    breakprint("  XAie_InitRoutingHandler---\n");
    routingInstance = XAie_InitRoutingHandler(DevInst);
    breakprint("  XAie_Route-4--\n");
    XAie_Route(routingInstance, NULL, XAie_TileLoc(shimcol, 0) /* Source*/, XAie_TileLoc(4, 4) /* destination*/);
    XAie_Route(routingInstance, NULL, XAie_TileLoc(4, 4) /* Source*/, XAie_TileLoc(shimcol, 0) /* destination*/);

    breakprint("  XAie_MemAllocate---\n");

    // printf("Routing successful\n");
    u64 phy = 0, phy_out = 0;
    u32 mlen = (MAT_SIZE * 2) * 25000;
    const u32 recv_len = MAT_SIZE;

    // Prepare DDR data
    XAie_MemInst *in = XAie_MemAllocate(DevInst, mlen * sizeof(u32), XAIE_MEM_CACHEABLE);
    phy = (u32)XAie_MemGetDevAddr(in);
    XAie_MemInst *out = XAie_MemAllocate(DevInst, mlen * sizeof(u32), XAIE_MEM_CACHEABLE);
    phy_out = (u32)XAie_MemGetDevAddr(out);

    breakprint(" XAie_MemSyncForDev---\n");

    u64 vmem = phy;
    u64 vmem_out = phy_out;
    for (int i = 0; i < mlen; i++) {
        ((int32_t *)vmem)[i] = i + 2;
        ((int32_t *)vmem_out)[i] = 0;
    }

    XTime tStart, tEnd;
    XTime_GetTime(&tStart);

    //((u32*)vmem)[0] = 1024*1024;

    XAie_MemSyncForDev(in);
    XAie_MemSyncForCPU(out);

    breakprint("Starting to Move data using Streaming APIs\n");
    // step 3: Setup streaming DMA with ping-pong buffers
    // XTime_GetTime(&tStart);
    // printf("vmem = 0x%p\n", vmem);

    // Initialize locks for input streaming (DMA writes to AIE memory)
    // S2MM: DMA acquires with value 0, releases with value 1
    // So initial value should be 0 (buffer empty, DMA can start writing)
    XAie_LockSetValue(DevInst, XAie_TileLoc(4, 4), XAie_LockInit(IN_ACCQUIRE_LOCK_ID, 2));
    XAie_LockSetValue(DevInst, XAie_TileLoc(4, 4), XAie_LockInit(IN_RELEASE_LOCK_ID, 0));

    breakprint("after input lock set value\n");

    // Initialize locks for output streaming (DMA reads from AIE memory)
    // MM2S: DMA acquires with value 1, releases with value 0
    // So initial value should be 0 (buffer empty, AIE can start writing)
    // out release lock should be 2 because release lock should have initial value to let kernel to prepare data
    // and the acquire lock should be only can be set after kernel have data ready hence initial value should be 0
    XAie_LockSetValue(DevInst, XAie_TileLoc(4, 4), XAie_LockInit(OUT_ACQUIRE_LOCK_ID, 0));
    XAie_LockSetValue(DevInst, XAie_TileLoc(4, 4), XAie_LockInit(OUT_RELEASE_LOCK_ID, 2));
    printf("Locks initialized for streaming\n");
    breakprint("after output lock set value\n");

    // Setup input streaming: External(single) -> AIE(ping-pong)
    // printf("Setting up input streaming: SHIM -> AIE with ping-pong\n");
    RC = XAie_MoveDataStreamingIn(routingInstance, XAie_TileLoc(shimcol, 0), // shimTile
                                  in,                                        // extBuffer
                                  mlen * sizeof(u32),                        // extBufferSize (total data)
                                  SHIM_IN_BD_ID,                             // shimBdId
                                  XAie_TileLoc(4, 4),                        // aieTile
                                  CORE_IP_PING,                              // aiePingAddr
                                  CORE_IP_PONG,                              // aiePongAddr
                                  PING_PONG_SIZE * sizeof(u32),              // aieBufferSize (per ping/pong)
                                  AIE_IN_PING_BD,                            // aiePingBdId
                                  AIE_IN_PONG_BD,                            // aiePongBdId
                                  IN_ACCQUIRE_LOCK_ID,                       // pingLockId
                                  IN_RELEASE_LOCK_ID);                       // pongLockId
    if (RC != XAIE_OK) {
        printf("XAie_MoveDataStreamingIn failed!\n");
        return -1;
    }

    // Setup output streaming: AIE(ping-pong) -> External(single)
    // printf("Setting up output streaming: AIE -> SHIM with ping-pong\n");
    RC = XAie_MoveDataStreamingOut(routingInstance, XAie_TileLoc(4, 4), // aieTile
                                   CORE_OP_PING,                        // aiePingAddr
                                   CORE_OP_PONG,                        // aiePongAddr
                                   PING_PONG_SIZE * sizeof(u32),        // aieBufferSize (per ping/pong)
                                   AIE_OUT_PING_BD,                     // aiePingBdId
                                   AIE_OUT_PONG_BD,                     // aiePongBdId
                                   XAie_TileLoc(shimcol, 0),            // shimTile
                                   out,                                 // extBuffer
                                   mlen * sizeof(u32),                  // extBufferSize (total data)
                                   SHIM_OUT_BD_ID,                      // shimBdId
                                   OUT_ACQUIRE_LOCK_ID,                 // pingLockId
                                   OUT_RELEASE_LOCK_ID);                // pongLockId
    if (RC != XAIE_OK) {
        printf("XAie_MoveDataStreamingOut failed!\n");
        return -1;
    }

    // Print status after DMA setup
    printAieDebugStatus(DevInst, XAie_TileLoc(4, 4), "After DMA Setup");

    breakprint("Streaming DMAs configured, starting kernel\n");

    // Print initial status before starting
    printAieDebugStatus(DevInst, XAie_TileLoc(4, 4), "Before XAie_Run");

    XAie_Run(routingInstance, 1);

    breakprint("XAie_CoreWaitForDone\n");
    // wait until core done
    u8 allDone = 0;
    uint32_t CoreStatus = 0;
    int waitIterations = 0;
    const int maxWaitIterations = 20; // Max iterations before printing debug
    const int debugInterval = 2;      // Print debug every N iterations

    do {
        allDone = 1; // Assume all cores are done initially
        uint32_t coreStatCharWritten = 0;
        for (int i = 0; i < 1; i++) { // Iterate over the specified tiles
            // Use timeout version to avoid infinite blocking
            RC = XAie_CoreWaitForDone(DevInst, XAie_TileLoc(4, 4), 1000); // 100ms timeout

            if (RC != XAIE_OK) {
                allDone = 0;
                waitIterations++;

                // Print debug status periodically
                if (waitIterations % debugInterval == 0) {
                    char debugMsg[64];
                    snprintf(debugMsg, sizeof(debugMsg), "Wait iteration %d", waitIterations);
                    printAieDebugStatus(DevInst, XAie_TileLoc(4, 4), debugMsg);
                }

                // Safety exit after too many iterations
                if (waitIterations >= maxWaitIterations) {
                    printf("\n!!! TIMEOUT: Core not completing after %d iterations !!!\n", waitIterations);
                    printAieDebugStatus(DevInst, XAie_TileLoc(4, 4), "TIMEOUT - Final Status");
                    printf("Aborting wait loop...\n");
                    break;
                }
            }
        }

        if (waitIterations >= maxWaitIterations) {
            break;
        }
    } while (!allDone);

    if (allDone) {
        // printf("Core completed successfully!\n");
        printAieDebugStatus(DevInst, XAie_TileLoc(4, 4), "After Core Done");
    }
    breakprint("fflush\n");
    fflush(stdout);

    // Note: Output data transfer is already configured via XAie_MoveDataStreamingOut
    // The ping-pong DMA will automatically transfer data as the kernel produces it
    // Wait for output DMA to complete
#if AIE_GEN == XAIE_DEV_GEN_AIE2PS
    breakprint("XAie_RouteDmaWait for output\n");
    XAie_RouteDmaWait(routingInstance, XAie_TileLoc(4, 4), XAie_TileLoc(shimcol, 0), true);
#endif

    // XTime_GetTime(&tEnd);
    // printf("Output took %.2f us.\n", 1.0 * (tEnd - tStart) / (COUNTS_PER_SECOND/1000000));

    // Sync output memory for CPU access
    XAie_MemSyncForCPU(out);

    XTime_GetTime(&tEnd);
    printf("-%d KB Output took %.2f us.\n", (mlen * sizeof(u32)) / 1024,
           1.0 * (tEnd - tStart) / (COUNTS_PER_SECOND / 1000000));
    printf("\nFinished streaming data back to DDR\n");

    // step 5 validate data
    int32_t vmem_out_cpu[recv_len];

    // vmem contains the input (128 samples, 64 of matrix A and 64 of matrix B, in row major and column major forms
    // respectively) and vmem_out contains the output samples (64 of result) compute CPU Result for softmax

    for (int i = 0; i < 16; i++) {
        printf("vmem_out[%d] = %d\n", i, ((int32_t *)vmem_out)[i]);
    }
    for (int i = mlen - 16; i < mlen; i++) {
        printf("last vmem_out[%d] = %d\n", i, ((int32_t *)vmem_out)[i]);
    }
    printf("\nDone---\n");
    breakprint("press key to clear the partition\n");
    return 0;
}

int main(int argc, char *argv[]) {
#ifdef DISABLE_CACHE
    Xil_DCacheDisable();
    Xil_ICacheDisable();
    printf("1Cache Disabled performance will have big drop (this test should >350us(8*8 and 16*16)\n ");
#else
    printf("1cache enabled, this test should be 15us(8*8) 215 us(16*16)");
#endif

    XAie_SetupConfig(ConfigPtr, HW_GEN, XAIE_BASE_ADDR, XAIE_COL_SHIFT, XAIE_ROW_SHIFT, XAIE_NUM_COLS, XAIE_NUM_ROWS,
                     XAIE_SHIM_ROW, XAIE_RES_TILE_ROW_START, XAIE_RES_TILE_NUM_ROWS, XAIE_AIE_TILE_ROW_START,
                     XAIE_AIE_TILE_NUM_ROWS);

    ///*

    XAie_InstDeclare(DevInst, &ConfigPtr);

    int partitonnum = 34;
    int startcol = 2;
    int colnum = (startcol + partitonnum <= XAIE_NUM_COLS) ? partitonnum : (XAIE_NUM_COLS - startcol);

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
    if (RC != XAIE_OK) {
        printf("Driver initialization failed.\n");
        return -1;
    }

    XAie_SetIOBackend(&DevInst, XAIE_IO_BACKEND_BAREMETAL);

#if AIE_GEN >= 2
    if (DevInst.Backend->Type == XAIE_IO_BACKEND_BAREMETAL) {
        printf("XAie_UpdateNpiAddr()\n");
#if AIE_GEN == 5 // aie2ps
        printf("XAie_UpdateNpiAddr(0xf6d50000)\n");
        RC = XAie_UpdateNpiAddr(&DevInst, 0xf6d50000);
#else
        RC = XAie_UpdateNpiAddr(&DevInst, 0xF6D10000);
#endif
        if (RC != XAIE_OK) {
            printf("Failed to update NPI address\n");
            return -1;
        }
    }
    printf("before XAie_PartitionInitialize-2--\n");
    // fix in aie2 the shim dma not work issue
    RC = XAie_PartitionInitialize(&DevInst, NULL);
#else
    // fix in aie1 shimd dma not work issue
    XAie_PmRequestTiles(&DevInst, NULL, 0);
#endif

    test_routing(&DevInst);

    RC = XAie_PartitionTeardown(&DevInst);
    if (RC != XAIE_OK) {
        printf("Failed to Teardown partition\n");
        return -1;
    }

    return 1;
}
