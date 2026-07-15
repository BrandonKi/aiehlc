/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

/**
 * @file aie_runtime_stream_debug.h
 * @brief Stream switch register debug utilities for AIE tile ranges.
 *
 * Provides range-based functions to dump and print stream switch configuration
 * (master ports, slave ports, slave slots) by directly reading hardware
 * registers via XAie_Read32 for a rectangular tile range.
 *
 * Parameters follow the convention: start_col..end_col (inclusive),
 * start_row..end_row (inclusive).
 *
 * Register layout (tile-relative offsets, same for core and shim tiles):
 *   Master config base : 0x3F000  (+4 per port)
 *   Slave config base  : 0x3F100  (+4 per port)
 *   Slave slot base    : 0x3F200  (+16 per slave-port, 4 slots each at +0/4/8/C)
 *
 * Master config register bits:
 *   [31] MASTER_ENABLE  [30] PACKET_ENABLE  [7] DROP_HEADER  [6:0] CONFIG
 *
 * Slave config register bits:
 *   [31] SLAVE_ENABLE
 *
 * Slave slot register bits:
 *   [28:24] PktId  [20:16] PktMask  [8] ENABLE  [5:4] MSEL  [2:0] ARBIT
 */

#ifndef AIE_RUNTIME_STREAM_DEBUG_H
#define AIE_RUNTIME_STREAM_DEBUG_H

#include "aie_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Print stream switch configuration for a single tile by directly reading
 * hardware registers.
 *
 * For each master port: enable, packet-switch mode, drop-header, config field.
 * For each slave port: enable, and for each of the 4 slots: PktId, Mask,
 * MSel, Arbitor.
 *
 * When print_all=0, only enabled or non-zero entries are printed.
 * When print_all=1, every register is printed regardless of value.
 *
 * @param dev        Device instance.
 * @param tile       Tile location (core tile or shim tile).
 * @param print_all  0 = print only enabled/active; 1 = print all registers.
 */
void AieRtSS_PrintTile(XAie_DevInst *dev, XAie_LocType tile, int print_all);

/**
 * Print stream switch configuration for all tiles in the range
 * [start_col..end_col] x [start_row..end_row] (inclusive on both ends).
 *
 * Iterates col then row, calling AieRtSS_PrintTile for each tile.
 *
 * @param dev        Device instance.
 * @param start_col  First column (inclusive).
 * @param end_col    Last column (inclusive).
 * @param start_row  First row (inclusive).
 * @param end_row    Last row (inclusive).
 * @param print_all  0 = print only enabled/active; 1 = print all registers.
 */
void AieRtSS_PrintRange(XAie_DevInst *dev, uint8_t start_col, uint8_t end_col, uint8_t start_row, uint8_t end_row,
                        int print_all);

/**
 * Print stream switch configuration for a list of tiles.
 *
 * @param dev        Device instance.
 * @param tiles      Array of tile locations.
 * @param num_tiles  Number of entries in tiles[].
 * @param print_all  0 = print only enabled/active; 1 = print all registers.
 */
void AieRtSS_PrintTiles(XAie_DevInst *dev, const XAie_LocType *tiles, uint32_t num_tiles, int print_all);

#ifdef __cplusplus
}
#endif

#endif /* AIE_RUNTIME_STREAM_DEBUG_H */
