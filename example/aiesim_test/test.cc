/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "aie_device_map.h"   /* HW_GEN, XAIE_NUM_ROWS/COLS, XAIE_AIE_TILE_ROW_START, etc. */
#include "xaiengine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define TEST_COL   2
#define TEST_ROW   XAIE_AIE_TILE_ROW_START

#define TEST_DM_OFFSET   0x1000
#define TEST_WORDS       8

static XAie_DevInst *init_device(void) {
    static XAie_Config cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.AieGen       = HW_GEN;
    cfg.BaseAddr     = XAIE_BASE_ADDR;
    cfg.ColShift     = XAIE_COL_SHIFT;
    cfg.RowShift     = XAIE_ROW_SHIFT;
    cfg.NumRows      = XAIE_NUM_ROWS;
    cfg.NumCols      = XAIE_NUM_COLS;
    cfg.ShimRowNum       = XAIE_SHIM_ROW;
    cfg.MemTileRowStart  = XAIE_RES_TILE_ROW_START;
    cfg.MemTileNumRows   = XAIE_RES_TILE_NUM_ROWS;
    cfg.AieTileRowStart  = XAIE_AIE_TILE_ROW_START;
    cfg.AieTileNumRows   = XAIE_AIE_TILE_NUM_ROWS;
    memset(&cfg.PartProp, 0, sizeof(cfg.PartProp));

    XAie_DevInst *dev = (XAie_DevInst *)calloc(1, sizeof(XAie_DevInst));
    if (!dev) { printf("[test] calloc failed\n"); return NULL; }

    AieRC rc = XAie_CfgInitialize(dev, &cfg);
    if (rc != XAIE_OK) {
        printf("[test] CfgInitialize failed: %d\n", (int)rc);
        free(dev);
        return NULL;
    }

#ifdef __AIESIM__
    printf("[test] Using XAIE_IO_BACKEND_SIM\n");
    XAie_SetIOBackend(dev, XAIE_IO_BACKEND_SIM);
#else
    printf("[test] Using XAIE_IO_BACKEND_BAREMETAL\n");
    XAie_SetIOBackend(dev, XAIE_IO_BACKEND_BAREMETAL);
#  if AIE_GEN == 5
    XAie_UpdateNpiAddr(dev, 0xf6d50000);
#  else
    XAie_UpdateNpiAddr(dev, 0xF6D10000);
#  endif
#endif

#ifndef __AIESIM__
    rc = XAie_PartitionInitialize(dev, NULL);
    if (rc != XAIE_OK) {
        printf("[test] PartitionInitialize failed: %d\n", (int)rc);
        free(dev);
        return NULL;
    }
#endif

    return dev;
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    printf("=== AIE Sim Test ===\n");
    printf("Target tile: col=%d row=%d\n", TEST_COL, TEST_ROW);

    XAie_DevInst *dev = init_device();
    if (!dev) {
        printf("FAIL: device init\n");
        return 1;
    }
    printf("[test] Device initialized OK\n");

    XAie_LocType tile = XAie_TileLoc(TEST_COL, TEST_ROW);

    uint32_t write_buf[TEST_WORDS];
    for (int i = 0; i < TEST_WORDS; i++)
        write_buf[i] = 0xA1E00000u | (uint32_t)i;

    printf("[test] Writing %d words to tile DM @ 0x%x\n", TEST_WORDS, TEST_DM_OFFSET);
    AieRC rc = XAie_DataMemBlockWrite(dev, tile, TEST_DM_OFFSET,
                                      write_buf, TEST_WORDS * sizeof(uint32_t));
    if (rc != XAIE_OK) {
        printf("FAIL: DataMemBlockWrite rc=%d\n", (int)rc);
        XAie_Finish(dev);
        free(dev);
        return 1;
    }

    uint32_t read_buf[TEST_WORDS];
    memset(read_buf, 0, sizeof(read_buf));
    printf("[test] Reading back...\n");
    rc = XAie_DataMemBlockRead(dev, tile, TEST_DM_OFFSET,
                               read_buf, TEST_WORDS * sizeof(uint32_t));
    if (rc != XAIE_OK) {
        printf("FAIL: DataMemBlockRead rc=%d\n", (int)rc);
        XAie_Finish(dev);
        free(dev);
        return 1;
    }

    int pass = 1;
    for (int i = 0; i < TEST_WORDS; i++) {
        if (read_buf[i] != write_buf[i]) {
            printf("FAIL: word[%d] expected=0x%08x got=0x%08x\n",
                   i, write_buf[i], read_buf[i]);
            pass = 0;
        }
    }

    XAie_Finish(dev);
    free(dev);

    if (pass) {
        printf("PASS: all %d words match\n", TEST_WORDS);
        return 0;
    } else {
        printf("FAIL: data mismatch\n");
        return 1;
    }
}
