/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "aie_runtime_stream_debug.h"
#include "aie_runtime_debug.h"
#include <stdio.h>

/* --------------------------------------------------------------------------
 * Implementation
 *
 * Delegates single-tile printing to AieRt_PrintStreamSwitchConfig which is
 * already defined in aie_runtime_debug.c/.h.  This file adds the range and
 * list wrappers requested by the caller.
 * -------------------------------------------------------------------------- */

void AieRtSS_PrintTile(XAie_DevInst *dev, XAie_LocType tile, int print_all) {
    AieRt_PrintStreamSwitchConfig(dev, tile, print_all);
}

void AieRtSS_PrintRange(XAie_DevInst *dev, uint8_t start_col, uint8_t end_col, uint8_t start_row, uint8_t end_row,
                        int print_all) {
    printf("[AieRtSS] ===== Stream Switch Dump: col[%u..%u] row[%u..%u] =====\n", (unsigned)start_col,
           (unsigned)end_col, (unsigned)start_row, (unsigned)end_row);

    for (uint8_t col = start_col; col <= end_col; col++) {
        for (uint8_t row = start_row; row <= end_row; row++) {
            AieRt_PrintStreamSwitchConfig(dev, XAie_TileLoc(col, row), print_all);
        }
    }

    printf("[AieRtSS] ===== End Stream Switch Dump =====\n");
}

void AieRtSS_PrintTiles(XAie_DevInst *dev, const XAie_LocType *tiles, uint32_t num_tiles, int print_all) {
    printf("[AieRtSS] ===== Stream Switch Dump: %u tiles =====\n", (unsigned)num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        AieRt_PrintStreamSwitchConfig(dev, tiles[i], print_all);
    }
    printf("[AieRtSS] ===== End Stream Switch Dump =====\n");
}
