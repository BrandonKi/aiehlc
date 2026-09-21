/******************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <stdint.h>
#include <stdio.h>

int __Runtime_ctrl_row_single_tile_write_ack_verify(void *dev, uint8_t shim_col, uint8_t row, uint32_t tile_addr,
                                                    uint32_t value, uint8_t ctrl_id, int32_t bd_id, int32_t mm2s_ch,
                                                    int32_t s2mm_ch, uint32_t *readback);

/*
 * Keep the generated data route off the row-control resources, then replace
 * the tile's individual lock initialization with a control-packet broadcast.
 * A correct result proves that the control write initialized the DMA locks and
 * that the kernel data path continued to operate on the remaining resources.
 */
#pragma control_plan_op_control_packet
#pragma CONTROL_PLAN_GROUP_REG_WRITE

#define ELEMENTS 256
#define TILE_ROWS 16
#define TILE_COLS 16
#define HW_ROWS 1
#define HW_COLS 1
#define CONTROL_ROW 3u
#define CONTROL_COL 0u
#define CONTROL_SCRATCH_ADDR 0x4000u
#define CONTROL_SENTINEL 0xC0DEC001u
#define CONTROL_BD_ID 8
#define CONTROL_DMA_CH 1

constexpr aie::GemmSpace InputSpace = {
    .policy = {.map = {.act = aie::Pattern::Broadcast, .layout = aie::Layout::Row},
               .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
               .sched = {.pp_depth = 1, .l1_budget = aie::Bytes{4096}}},
    .d1 = {.fullsize = TILE_ROWS, .tile_size = TILE_ROWS, .stride = TILE_ROWS},
    .d2 = {.fullsize = TILE_COLS, .tile_size = TILE_COLS, .stride = TILE_COLS}};

constexpr aie::GemmSpace OutputSpace = {
    .policy = {.map = {.layout = aie::Layout::Row, .merge_order = aie::Flow::LeftToRight},
               .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
               .sched = {.pp_depth = 1, .l1_budget = aie::Bytes{4096}}},
    .d1 = {.fullsize = TILE_ROWS, .tile_size = TILE_ROWS, .stride = TILE_ROWS},
    .d2 = {.fullsize = TILE_COLS, .tile_size = TILE_COLS, .stride = TILE_COLS}};

__global__ void add_one(aie::port<input_window_int8 *, InputSpace> input,
                        aie::port<output_window_int8 *, OutputSpace> output) {
    int8_t *in = (int8_t *)acquire_input_window(input);
    int8_t *out = (int8_t *)acquire_output_window(output);

    for (int i = 0; i < ELEMENTS; ++i)
        out[i] = (int8_t)(in[i] + 1);

    release_input_window(input);
    release_output_window(output);
}

static int verify_control_path(void *dev) {
    uint32_t read_value = 0u;
    int rc = __Runtime_ctrl_row_single_tile_write_ack_verify(dev, CONTROL_COL, CONTROL_ROW, CONTROL_SCRATCH_ADDR,
                                                             CONTROL_SENTINEL, 1u, CONTROL_BD_ID, CONTROL_DMA_CH,
                                                             CONTROL_DMA_CH, &read_value);
    if (rc != 0) {
        printf("[control-tiling] control write/ack/readback failed: rc=%d got=0x%08x expected=0x%08x\n", rc,
               read_value, CONTROL_SENTINEL);
        return 1;
    }

    printf("[control-tiling] control write/ack/readback PASS: 0x%08x\n", read_value);
    return 0;
}

int main() {
    printf("[control-tiling] one-tile control/data coexistence test\n");

    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({0, 3, 0, 6}, HW_ROWS, HW_COLS);
    int errors = 0;

    int8_t *input = (int8_t *)device.alloc(ELEMENTS * sizeof(int8_t) * 4);
    int8_t *output = (int8_t *)device.alloc(ELEMENTS * sizeof(int8_t) * 4);
    if (!input || !output) {
        printf("[control-tiling] allocation failed\n");
        if (input)
            device.free(input);
        if (output)
            device.free(output);
        return 1;
    }

    for (int i = 0; i < ELEMENTS; ++i) {
        input[i] = (int8_t)((i % 31) - 16);
        output[i] = 0;
    }

    add_one<<<mesh>>>(input, output);
    device.synchronizecpu(output, ELEMENTS * sizeof(int8_t));

    for (int i = 0; i < ELEMENTS; ++i) {
        int8_t expected = (int8_t)(input[i] + 1);
        if (output[i] != expected) {
            if (errors < 8)
                printf("[control-tiling] mismatch %d: got %d expected %d\n", i, (int)output[i], (int)expected);
            ++errors;
        }
    }
    errors += verify_control_path(device._dev);

    device.free(input);
    device.free(output);

    if (errors) {
        printf("[control-tiling] FAIL: %d mismatches\n", errors);
        return 1;
    }

    printf("[control-tiling] PASS: control initialization and kernel data path both worked\n");
    return 0;
}
