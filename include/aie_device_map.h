/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef AIE_DEVICE_MAP_H
#define AIE_DEVICE_MAP_H

// AIE tile memory map (reference: aieml_perf.cc lines 53-54)
#define CORE_IP_MEM 0x1000 // Input memory offset
#define CORE_OP_MEM 0x6000 // Output memory offset
#define DEBUG_LOG_BASE 0x73000
#define SYNC_BUFFER_BASE 0x74000

// DMA buffer sizes
#define DEFAULT_BUFFER_SIZE 256
#define ALLOC_LIST_MAX_SIZE 256

// Device layout (reference: aieml_perf.cc lines 26-50)
#define XAIE_BASE_ADDR 0x20000000000
#define XAIE_COL_SHIFT 25
#define XAIE_ROW_SHIFT 20
#define XAIE_SHIM_ROW 0
#define XAIE_RES_TILE_ROW_START 1
#define XAIE_RES_TILE_NUM_ROWS 2

// Architecture-specific defines
#if AIE_GEN <= 2
#define XAIE_NUM_ROWS 11
#define XAIE_NUM_COLS 38
#define XAIE_AIE_TILE_ROW_START 3
#define XAIE_AIE_TILE_NUM_ROWS 8
#else
#define XAIE_NUM_ROWS 7
#define XAIE_NUM_COLS 36
#define XAIE_AIE_TILE_ROW_START 3
#define XAIE_AIE_TILE_NUM_ROWS 4
#endif

#endif // AIE_DEVICE_MAP_H
