/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

// #include <iostream>
// #include <sstream>
#include "xaiengine.h"
#include "xil_printf.h"
// #include "xil_io.h"
#include "aie_runtime_common.h"
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
#define CORE_OP_MEM 0x6000

__global__ void dummy_kernel(input_window_int32 *in
                             __attribute__((annotate("mem_address:0x1000"), annotate("size_hint:16"))),
                             output_window_int32 *out
                             __attribute__((annotate("mem_address:0x6000"), annotate("size_hint:16")))) {}

#define TEST_DATA_SIZE 64 // number of uint32_t words to send
#define TEST_DATA_BYTES (TEST_DATA_SIZE * sizeof(uint32_t))

// ---------------------------------------------------------------------------
// Include the generated routing configuration.
// routing.cc calls getOrCreateDeviceInstance(), so we provide the impl here.
// ---------------------------------------------------------------------------
static XAie_DevInst *g_DevInst = nullptr;
XAie_DevInst *getOrCreateDeviceInstance() { return g_DevInst; }
#include "./routing4x4.cc"

// ---------------------------------------------------------------------------
// test_routing_packet:
//   Tests the output path from tile(0,3) → Shim(3,0) using the
//   packet-switched route established by routing() round-0 row-split.
// ---------------------------------------------------------------------------
int test_routing_packet(XAie_DevInst *DevInst) {
    AieRC RC = XAIE_OK;
    printf("05/06=== test_routing_packet: tile(0,3) → Shim(3,0) ===\n");

    // --- Step 1: Apply the full routing configuration ---
    g_DevInst = DevInst;
    printf("[1] Configuring stream switch routing...\n");
    routing();
    printf("[1] Routing configured.\n");

    // --- Step 2: Write test data into tile(0,3) data memory at 0x0 ---

    // --- Step 3: Move data from tile(0,3) → Shim(3,0) via Runtime_Movedata_ManyToOne ---
    // Many-to-one with a single source (same test, new API).
    // stride/wrap receive dims: [[4,1],[8,8],[4,2]]
    printf("abc[3] Moving data (ManyToOne): tile(0,3) → Shim(3,0)...\n");
#define SRCNUM 3
#define SRCNUM_DUP_ROUND 2
#define SRCNUM_DUP (SRCNUM * SRCNUM_DUP_ROUND)
    // Allocate one DDR buffer for all sources
    uint32_t total_bytes = TEST_DATA_BYTES * SRCNUM_DUP;
    XAie_MemInst *ddr_buf = XAie_MemAllocate(DevInst, total_bytes, XAIE_MEM_CACHEABLE);
    u64 ddr_phy = (u64)XAie_MemGetDevAddr(ddr_buf);
    for (uint32_t w = 0; w < total_bytes / sizeof(uint32_t); w++)
        ((uint32_t *)ddr_phy)[w] = 0xDEADBEEF;
    XAie_MemSyncForDev(ddr_buf);

    MovedataSrcDesc srcs[SRCNUM * SRCNUM_DUP_ROUND];
    for (int k = 0; k < SRCNUM_DUP_ROUND; k++) {
        for (int i = 0; i < SRCNUM; i++) {
            auto idx = k * SRCNUM + i;
            srcs[idx].src_tile = XAie_TileLoc(i, 3);
            srcs[idx].src_addr = 0x0 + (k * TEST_DATA_BYTES); // offset for each duplicate round
            srcs[idx].src_ch = 0;
            srcs[idx].src_bd = k;
            srcs[idx].dst_bd = 2 + i; // dst BDs 2,3,4 on shim
            srcs[idx].data_bytes = TEST_DATA_BYTES;
            srcs[idx].dst_len = 0;        // 0 = use data_bytes
            srcs[idx].src_pkt_id = 1 + i; // pkt_id 1,2,3
            srcs[idx].dst_num_dims = 0;   // stride/wrap receive dims: [[4,1],[8,8],[4,2]]
            // srcs[idx].dst_dims[0] = {.AieMlDimDesc = {.StepSize = 1, .Wrap = 4}};
            // srcs[idx].dst_dims[1] = {.AieMlDimDesc = {.StepSize = 8, .Wrap = 8}};
            // srcs[idx].dst_dims[2] = {.AieMlDimDesc = {.StepSize = 4, .Wrap = 2}};
            srcs[idx].recv_buf = ddr_buf; // all share the same MemInst for sync
            srcs[idx].recv_phy = ddr_phy + i * TEST_DATA_BYTES * SRCNUM_DUP_ROUND;
            printf("[2] Writing test data to tile(%d,3) data memory at 0x0...\n", i);
            {
                uint32_t test_data[TEST_DATA_SIZE];
                for (uint32_t j = 0; j < TEST_DATA_SIZE; j++) {
                    test_data[j] = ((k * 0x10000000) | (i * 0x1000000) | j);
                }
                RC = XAie_DataMemBlockWrite(DevInst, XAie_TileLoc(i, 3), srcs[idx].src_addr, (void *)test_data,
                                            TEST_DATA_BYTES);
                if (RC != XAIE_OK) {
                    printf("ERROR: DataMemBlockWrite to tile(%d,3) failed: %d\n", i, RC);
                    return -1;
                }
            }
            printf("---[2] Test data written.\n");
        }
    }
    RC = Runtime_Movedata_ManyToOne(DevInst, srcs, SRCNUM_DUP, // all sources
                                    XAie_TileLoc(3, 0),        // dst: Shim(3,0)
                                    0);                        // dst_ch
    if (RC != XAIE_OK) {
        printf("ERROR: Runtime_Movedata_ManyToOne failed: %d\n", RC);
        return -1;
    }
    printf("[3] ManyToOne data move enqueued, waiting for completion...\n");
    RC = Runtime_Movedata_WaitAll(DevInst);
    if (RC != XAIE_OK) {
        printf("ERROR: Runtime_Movedata_WaitAll failed: %d\n", RC);
        return -1;
    }
    printf("[3] Data move completed.\n");

    // --- Step 4: Verify received data ---
    printf("[4] Verifying received data...\n");
    XAie_MemSyncForCPU(ddr_buf);

    int mismatches = 0;
    for (int i = 0; i < SRCNUM_DUP_ROUND; i++) {

        for (int src = 0; src < SRCNUM; src++) {
            auto idx = (i * SRCNUM) + src;
            uint32_t *recv = (uint32_t *)(srcs[idx].recv_phy + i * TEST_DATA_BYTES); // offset for each duplicate round
            for (uint32_t j = 0; j < TEST_DATA_SIZE; j++) {
                uint32_t expected = ((i * 0x10000000) | (src * 0x1000000) | j);
                uint32_t actual = recv[j];
                if (actual != expected) {
                    // if (mismatches < 256) {
                    printf("  MISMATCH src%d [%2d]: expected=0x%08x, got=0x%08x\n", idx, j, expected, actual);
                    //}
                    mismatches++;
                } else {
                    printf("  match src%d [%2d]: expected=0x%08x, got=0x%08x\n", idx, j, expected, actual);
                }
            }
        }
    }

    if (mismatches == 0) {
        printf("[4] SUCCESS: All %d words (%d sources x %d each) match.\n", TEST_DATA_SIZE * SRCNUM_DUP, SRCNUM_DUP,
               TEST_DATA_SIZE);
    } else {
        printf("[4] FAIL: %d / %d mismatches.\n", mismatches, TEST_DATA_SIZE * SRCNUM_DUP);
    }

    // Print first 8 words of each source for visual inspection
    for (int src = 0; src < SRCNUM; src++) {
        uint32_t *recv = (uint32_t *)srcs[src].recv_phy;
        printf("  Source %d %p first 8 received words:\n", src, recv);
        for (uint32_t j = 0; j < 8 && j < TEST_DATA_SIZE; j++) {
            printf("    [%d] = 0x%08x\n", j, recv[j]);
        }
    }

    return (mismatches == 0) ? 0 : -1;
}

// ---------------------------------------------------------------------------
// test_routing_packet2:
//   Same as test_routing_packet but uses Runtime_Movedata_ManyToOne_SingleDstBd
//   so that only ONE destination BD is consumed on the shim tile.
//   The single BD self-loops (NextBd → self) with iteration wrap = SRCNUM
//   so it replays exactly SRCNUM times and then stops.
// ---------------------------------------------------------------------------
int test_routing_packet2(XAie_DevInst *DevInst) {
    AieRC RC = XAIE_OK;
    printf("=== test_routing_packet2: tile(0..2,3) → Shim(3,0) single-dst-BD ===\n");

    // --- Step 1: Apply the full routing configuration ---
    g_DevInst = DevInst;
    printf("[1] Configuring stream switch routing...\n");
    routing();
    printf("[1] Routing configured.\n");

    // --- Step 2: Allocate one DDR buffer for all sources ---
#define SRCNUM2 3
    uint32_t total_bytes2 = TEST_DATA_BYTES * SRCNUM2;
    XAie_MemInst *ddr_buf2 = XAie_MemAllocate(DevInst, total_bytes2, XAIE_MEM_CACHEABLE);
    u64 ddr_phy2 = (u64)XAie_MemGetDevAddr(ddr_buf2);
    for (uint32_t w = 0; w < total_bytes2 / sizeof(uint32_t); w++)
        ((uint32_t *)ddr_phy2)[w] = 0xDEADBEEF;
    XAie_MemSyncForDev(ddr_buf2);

    // --- Step 3: Write test data into each source tile and fill descriptors ---
    MovedataSrcDesc srcs2[SRCNUM2];
    for (int i = 0; i < SRCNUM2; i++) {
        srcs2[i].src_tile = XAie_TileLoc(i, 3);
        srcs2[i].src_addr = 0x0;
        srcs2[i].src_ch = 0;
        srcs2[i].src_bd = 0;
        srcs2[i].dst_bd = 0; // unused by SingleDstBd API
        srcs2[i].data_bytes = TEST_DATA_BYTES;
        srcs2[i].dst_len = 0;
        srcs2[i].src_pkt_id = 1 + i; // pkt_id 1,2,3
        srcs2[i].dst_num_dims = 0;   // unused by SingleDstBd API
        srcs2[i].recv_buf = ddr_buf2;
        srcs2[i].recv_phy = ddr_phy2 + i * TEST_DATA_BYTES;

        printf("[2] Writing test data to tile(%d,3) data memory at 0x0...\n", i);
        {
            uint32_t test_data[TEST_DATA_SIZE];
            for (uint32_t j = 0; j < TEST_DATA_SIZE; j++) {
                test_data[j] = ((i * 0x10000000) | j);
            }
            RC = XAie_DataMemBlockWrite(DevInst, XAie_TileLoc(i, 3), 0x0, (void *)test_data, TEST_DATA_BYTES);
            if (RC != XAIE_OK) {
                printf("ERROR: DataMemBlockWrite to tile(%d,3) failed: %d\n", i, RC);
                return -1;
            }
        }
        printf("---[2] Test data written.\n");
    }

    // --- Step 4: Move data using SingleDstBd (one BD, self-loop + iteration) ---
    // Same multi-dim dims as test_routing_packet: [[1,4],[8,8],[4,2]]
    XAie_DmaDimDesc dst_dims2[3] = {
        {.AieMlDimDesc = {.StepSize = 1, .Wrap = 4}},
        {.AieMlDimDesc = {.StepSize = 8, .Wrap = 8}},
        {.AieMlDimDesc = {.StepSize = 4, .Wrap = 2}},
    };

    // iter_step_size: advance by TEST_DATA_SIZE words between replays
    // so source 0 → ddr_phy2, source 1 → ddr_phy2 + TEST_DATA_BYTES, etc.
    int iter_step = TEST_DATA_SIZE; // in 32-bit word units

    printf("[3] Moving data (ManyToOne_SingleDstBd): tile(0..2,3) → Shim(3,0) bd2...\n");
    RC = Runtime_Movedata_ManyToOne_SingleDstBd(DevInst, srcs2, SRCNUM2, XAie_TileLoc(3, 0), // dst: Shim(3,0)
                                                0,                                           // dst_ch
                                                2,                                           // single dst BD id
                                                ddr_phy2,                                    // DDR base address
                                                TEST_DATA_BYTES,                             // per_src_bytes
                                                0,                                           // dst_num_dims
                                                dst_dims2,                                   // multi-dim descriptors
                                                iter_step);                                  // iteration step (words)
    if (RC != XAIE_OK) {
        printf("ERROR: Runtime_Movedata_ManyToOne_SingleDstBd failed: %d\n", RC);
        return -1;
    }
    printf("[3] SingleDstBd data move enqueued, waiting for completion...\n");
    RC = Runtime_Movedata_WaitAll(DevInst);
    if (RC != XAIE_OK) {
        printf("ERROR: Runtime_Movedata_WaitAll failed: %d\n", RC);
        return -1;
    }
    printf("[3] Data move completed.\n");

    // --- Step 5: Verify received data ---
    printf("[4] Verifying received data...\n");
    XAie_MemSyncForCPU(ddr_buf2);

    int mismatches2 = 0;
    for (int src = 0; src < SRCNUM2; src++) {
        uint32_t *recv = (uint32_t *)(ddr_phy2 + src * TEST_DATA_BYTES);
        for (uint32_t j = 0; j < TEST_DATA_SIZE; j++) {
            uint32_t expected = ((src * 0x10000000) | j);
            uint32_t actual = recv[j];
            if (actual != expected) {
                if (mismatches2 < 256) {
                    printf("  MISMATCH src%d [%2d]: expected=0x%08x, got=0x%08x\n", src, j, expected, actual);
                }
                mismatches2++;
            }
        }
    }

    if (mismatches2 == 0) {
        printf("[4] SUCCESS: All %d words (%d sources x %d each) match.\n", TEST_DATA_SIZE * SRCNUM2, SRCNUM2,
               TEST_DATA_SIZE);
    } else {
        printf("[4] FAIL: %d / %d mismatches.\n", mismatches2, TEST_DATA_SIZE * SRCNUM2);
    }

    // Print first 8 words of each source for visual inspection
    for (int src = 0; src < SRCNUM2; src++) {
        uint32_t *recv = (uint32_t *)(ddr_phy2 + src * TEST_DATA_BYTES);
        printf("  Source %d %p first 8 received words:\n", src, recv);
        for (uint32_t j = 0; j < 8 && j < TEST_DATA_SIZE; j++) {
            printf("    [%d] = 0x%08x\n", j, recv[j]);
        }
    }

    return (mismatches2 == 0) ? 0 : -1;
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
    int startcol = 1;
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

    test_routing_packet(&DevInst);
    // test_routing_packet2(&DevInst);

    if (0) {
        RC = XAie_PartitionTeardown(&DevInst);
        if (RC != XAIE_OK) {
            printf("Failed to Teardown partition\n");
            return -1;
        }
    } else {
        printf("Not tearing down partition to allow re-running test without reloading\n");
    }

    return 1;
}
