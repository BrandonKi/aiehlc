// ******************************************************************************
// * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
// * SPDX-License-Identifier: Apache-2.0
// ******************************************************************************

module attributes {codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"]} {
  emitc.verbatim "#include \22aie_runtime.h\22"
  emitc.verbatim "#include \22aie_runtime_debug.h\22"
  func.func @main(%arg0: memref<256x256xi8>, %arg1: memref<256x256xi8>, %arg2: memref<256x256xi8>) {
    emitc.call_opaque "host_canonicalized"() : () -> ()
    return
  }
  emitc.func @host_canonicalized(%arg0: !emitc.ptr<!emitc.opaque<"void">>, %arg1: !emitc.ptr<!emitc.opaque<"void">>, %arg2: !emitc.ptr<!emitc.opaque<"void">>) {
    %0 = emitc.call_opaque "__runtime_buffer_offset"(%arg1) {args = [0 : index, 0]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1 = emitc.call_opaque "XAie_TileLoc"() {args = [2 : i8, 0 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=16384, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %2 = emitc.call_opaque "__runtime_buffer_arg"(%0) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %3 = emitc.call_opaque "__Runtime_dma_bd_config"(%1, %2) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 0 : i32, 16384 : i32, -1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=0, tile=(2,0), direction=MM2S */"
    %4 = emitc.call_opaque "__Runtime_dma_createio_4"(%1, %3) {args = [0 : index, 1 : index, 0 : i32, 0 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %5 = emitc.call_opaque "XAie_TileLoc"() {args = [0 : i8, 3 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %6 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)36864">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %7 = emitc.call_opaque "__Runtime_dma_bd_config"(%5, %6) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 1 : i32, 4096 : i32, 0 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(0,3) lock=2 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(0, 3), XAie_LockInit(2, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %8 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)32768">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %9 = emitc.call_opaque "__Runtime_dma_bd_config"(%5, %8) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 0 : i32, 4096 : i32, 1 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(0,3), direction=S2MM */"
    %10 = emitc.call_opaque "__Runtime_dma_createio_4"(%5, %9) {args = [0 : index, 1 : index, 1 : i32, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 0 for tile (0,3) */"
    %11 = emitc.call_opaque "XAie_TileLoc"() {args = [0 : i8, 4 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %12 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)36864">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %13 = emitc.call_opaque "__Runtime_dma_bd_config"(%11, %12) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 1 : i32, 4096 : i32, 0 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(0,4) lock=2 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(0, 4), XAie_LockInit(2, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %14 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)32768">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %15 = emitc.call_opaque "__Runtime_dma_bd_config"(%11, %14) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 0 : i32, 4096 : i32, 1 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(0,4), direction=S2MM */"
    %16 = emitc.call_opaque "__Runtime_dma_createio_4"(%11, %15) {args = [0 : index, 1 : index, 1 : i32, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 0 for tile (0,4) */"
    %17 = emitc.call_opaque "XAie_TileLoc"() {args = [0 : i8, 5 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %18 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)36864">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %19 = emitc.call_opaque "__Runtime_dma_bd_config"(%17, %18) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 1 : i32, 4096 : i32, 0 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(0,5) lock=2 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(0, 5), XAie_LockInit(2, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %20 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)32768">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %21 = emitc.call_opaque "__Runtime_dma_bd_config"(%17, %20) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 0 : i32, 4096 : i32, 1 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(0,5), direction=S2MM */"
    %22 = emitc.call_opaque "__Runtime_dma_createio_4"(%17, %21) {args = [0 : index, 1 : index, 1 : i32, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 0 for tile (0,5) */"
    %23 = emitc.call_opaque "XAie_TileLoc"() {args = [0 : i8, 6 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %24 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)36864">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %25 = emitc.call_opaque "__Runtime_dma_bd_config"(%23, %24) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 1 : i32, 4096 : i32, 0 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(0,6) lock=2 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(0, 6), XAie_LockInit(2, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %26 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)32768">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %27 = emitc.call_opaque "__Runtime_dma_bd_config"(%23, %26) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 0 : i32, 4096 : i32, 1 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(0,6), direction=S2MM */"
    %28 = emitc.call_opaque "__Runtime_dma_createio_4"(%23, %27) {args = [0 : index, 1 : index, 1 : i32, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 0 for tile (0,6) */"
    emitc.verbatim "/* Allocated BD ID 0 for tile (2,0) */"
    %29 = emitc.call_opaque "__Runtime_startio"(%4) {args = [0 : index, 0 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %30 = emitc.call_opaque "__runtime_buffer_offset"(%arg1) {args = [0 : index, 16384]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=16384, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %31 = emitc.call_opaque "__runtime_buffer_arg"(%30) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %32 = emitc.call_opaque "__Runtime_dma_bd_config"(%1, %31) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 1 : i32, 16384 : i32, -1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=1, tile=(2,0), direction=MM2S */"
    %33 = emitc.call_opaque "__Runtime_dma_createio_4"(%1, %32) {args = [0 : index, 1 : index, 1 : i32, 1 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %34 = emitc.call_opaque "XAie_TileLoc"() {args = [1 : i8, 3 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %35 = emitc.call_opaque "__runtime_buffer_offset"(%30) {args = [0 : index, 16384]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %36 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)36864">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %37 = emitc.call_opaque "__Runtime_dma_bd_config"(%34, %36) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 1 : i32, 4096 : i32, 0 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(1,3) lock=2 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(1, 3), XAie_LockInit(2, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %38 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)32768">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %39 = emitc.call_opaque "__Runtime_dma_bd_config"(%34, %38) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 0 : i32, 4096 : i32, 1 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(1,3), direction=S2MM */"
    %40 = emitc.call_opaque "__Runtime_dma_createio_4"(%34, %39) {args = [0 : index, 1 : index, 1 : i32, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 0 for tile (1,3) */"
    %41 = emitc.call_opaque "XAie_TileLoc"() {args = [1 : i8, 4 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %42 = emitc.call_opaque "__runtime_buffer_offset"(%30) {args = [0 : index, 16384]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %43 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)36864">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %44 = emitc.call_opaque "__Runtime_dma_bd_config"(%41, %43) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 1 : i32, 4096 : i32, 0 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(1,4) lock=2 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(1, 4), XAie_LockInit(2, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %45 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)32768">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %46 = emitc.call_opaque "__Runtime_dma_bd_config"(%41, %45) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 0 : i32, 4096 : i32, 1 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(1,4), direction=S2MM */"
    %47 = emitc.call_opaque "__Runtime_dma_createio_4"(%41, %46) {args = [0 : index, 1 : index, 1 : i32, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 0 for tile (1,4) */"
    %48 = emitc.call_opaque "XAie_TileLoc"() {args = [1 : i8, 5 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %49 = emitc.call_opaque "__runtime_buffer_offset"(%30) {args = [0 : index, 16384]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %50 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)36864">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %51 = emitc.call_opaque "__Runtime_dma_bd_config"(%48, %50) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 1 : i32, 4096 : i32, 0 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(1,5) lock=2 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(1, 5), XAie_LockInit(2, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %52 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)32768">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %53 = emitc.call_opaque "__Runtime_dma_bd_config"(%48, %52) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 0 : i32, 4096 : i32, 1 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(1,5), direction=S2MM */"
    %54 = emitc.call_opaque "__Runtime_dma_createio_4"(%48, %53) {args = [0 : index, 1 : index, 1 : i32, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 0 for tile (1,5) */"
    %55 = emitc.call_opaque "XAie_TileLoc"() {args = [1 : i8, 6 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %56 = emitc.call_opaque "__runtime_buffer_offset"(%30) {args = [0 : index, 16384]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %57 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)36864">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %58 = emitc.call_opaque "__Runtime_dma_bd_config"(%55, %57) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 1 : i32, 4096 : i32, 0 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(1,6) lock=2 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(1, 6), XAie_LockInit(2, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %59 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)32768">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %60 = emitc.call_opaque "__Runtime_dma_bd_config"(%55, %59) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 0 : i32, 4096 : i32, 1 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(1,6), direction=S2MM */"
    %61 = emitc.call_opaque "__Runtime_dma_createio_4"(%55, %60) {args = [0 : index, 1 : index, 1 : i32, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 0 for tile (1,6) */"
    emitc.verbatim "/* Allocated BD ID 1 for tile (2,0) */"
    %62 = emitc.call_opaque "__Runtime_startio"(%33) {args = [0 : index, 1 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %63 = emitc.call_opaque "__runtime_buffer_offset"(%arg1) {args = [0 : index, 32768]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %64 = emitc.call_opaque "XAie_TileLoc"() {args = [3 : i8, 0 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=16384, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %65 = emitc.call_opaque "__runtime_buffer_arg"(%63) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %66 = emitc.call_opaque "__Runtime_dma_bd_config"(%64, %65) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 0 : i32, 16384 : i32, -1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=0, tile=(3,0), direction=MM2S */"
    %67 = emitc.call_opaque "__Runtime_dma_createio_4"(%64, %66) {args = [0 : index, 1 : index, 0 : i32, 0 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %68 = emitc.call_opaque "XAie_TileLoc"() {args = [2 : i8, 3 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %69 = emitc.call_opaque "__runtime_buffer_offset"(%63) {args = [0 : index, 32768]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %70 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)36864">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %71 = emitc.call_opaque "__Runtime_dma_bd_config"(%68, %70) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 1 : i32, 4096 : i32, 0 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(2,3) lock=2 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(2, 3), XAie_LockInit(2, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %72 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)32768">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %73 = emitc.call_opaque "__Runtime_dma_bd_config"(%68, %72) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 0 : i32, 4096 : i32, 1 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(2,3), direction=S2MM */"
    %74 = emitc.call_opaque "__Runtime_dma_createio_4"(%68, %73) {args = [0 : index, 1 : index, 1 : i32, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 0 for tile (2,3) */"
    %75 = emitc.call_opaque "XAie_TileLoc"() {args = [2 : i8, 4 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %76 = emitc.call_opaque "__runtime_buffer_offset"(%63) {args = [0 : index, 32768]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %77 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)36864">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %78 = emitc.call_opaque "__Runtime_dma_bd_config"(%75, %77) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 1 : i32, 4096 : i32, 0 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(2,4) lock=2 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(2, 4), XAie_LockInit(2, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %79 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)32768">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %80 = emitc.call_opaque "__Runtime_dma_bd_config"(%75, %79) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 0 : i32, 4096 : i32, 1 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(2,4), direction=S2MM */"
    %81 = emitc.call_opaque "__Runtime_dma_createio_4"(%75, %80) {args = [0 : index, 1 : index, 1 : i32, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 0 for tile (2,4) */"
    %82 = emitc.call_opaque "XAie_TileLoc"() {args = [2 : i8, 5 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %83 = emitc.call_opaque "__runtime_buffer_offset"(%63) {args = [0 : index, 32768]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %84 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)36864">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %85 = emitc.call_opaque "__Runtime_dma_bd_config"(%82, %84) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 1 : i32, 4096 : i32, 0 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(2,5) lock=2 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(2, 5), XAie_LockInit(2, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %86 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)32768">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %87 = emitc.call_opaque "__Runtime_dma_bd_config"(%82, %86) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 0 : i32, 4096 : i32, 1 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(2,5), direction=S2MM */"
    %88 = emitc.call_opaque "__Runtime_dma_createio_4"(%82, %87) {args = [0 : index, 1 : index, 1 : i32, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 0 for tile (2,5) */"
    %89 = emitc.call_opaque "XAie_TileLoc"() {args = [2 : i8, 6 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %90 = emitc.call_opaque "__runtime_buffer_offset"(%63) {args = [0 : index, 32768]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %91 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)36864">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %92 = emitc.call_opaque "__Runtime_dma_bd_config"(%89, %91) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 1 : i32, 4096 : i32, 0 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(2,6) lock=2 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(2, 6), XAie_LockInit(2, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %93 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)32768">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %94 = emitc.call_opaque "__Runtime_dma_bd_config"(%89, %93) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 0 : i32, 4096 : i32, 1 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(2,6), direction=S2MM */"
    %95 = emitc.call_opaque "__Runtime_dma_createio_4"(%89, %94) {args = [0 : index, 1 : index, 1 : i32, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 0 for tile (2,6) */"
    emitc.verbatim "/* Allocated BD ID 0 for tile (3,0) */"
    %96 = emitc.call_opaque "__Runtime_startio"(%67) {args = [0 : index, 0 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %97 = emitc.call_opaque "__runtime_buffer_offset"(%arg1) {args = [0 : index, 49152]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=16384, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %98 = emitc.call_opaque "__runtime_buffer_arg"(%97) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %99 = emitc.call_opaque "__Runtime_dma_bd_config"(%64, %98) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 1 : i32, 16384 : i32, -1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=1, tile=(3,0), direction=MM2S */"
    %100 = emitc.call_opaque "__Runtime_dma_createio_4"(%64, %99) {args = [0 : index, 1 : index, 1 : i32, 1 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %101 = emitc.call_opaque "XAie_TileLoc"() {args = [3 : i8, 3 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %102 = emitc.call_opaque "__runtime_buffer_offset"(%97) {args = [0 : index, 49152]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %103 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)36864">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %104 = emitc.call_opaque "__Runtime_dma_bd_config"(%101, %103) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 1 : i32, 4096 : i32, 0 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(3,3) lock=2 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(3, 3), XAie_LockInit(2, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %105 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)32768">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %106 = emitc.call_opaque "__Runtime_dma_bd_config"(%101, %105) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 0 : i32, 4096 : i32, 1 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(3,3), direction=S2MM */"
    %107 = emitc.call_opaque "__Runtime_dma_createio_4"(%101, %106) {args = [0 : index, 1 : index, 1 : i32, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 0 for tile (3,3) */"
    %108 = emitc.call_opaque "XAie_TileLoc"() {args = [3 : i8, 4 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %109 = emitc.call_opaque "__runtime_buffer_offset"(%97) {args = [0 : index, 49152]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %110 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)36864">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %111 = emitc.call_opaque "__Runtime_dma_bd_config"(%108, %110) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 1 : i32, 4096 : i32, 0 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(3,4) lock=2 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(3, 4), XAie_LockInit(2, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %112 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)32768">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %113 = emitc.call_opaque "__Runtime_dma_bd_config"(%108, %112) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 0 : i32, 4096 : i32, 1 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(3,4), direction=S2MM */"
    %114 = emitc.call_opaque "__Runtime_dma_createio_4"(%108, %113) {args = [0 : index, 1 : index, 1 : i32, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 0 for tile (3,4) */"
    %115 = emitc.call_opaque "XAie_TileLoc"() {args = [3 : i8, 5 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %116 = emitc.call_opaque "__runtime_buffer_offset"(%97) {args = [0 : index, 49152]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %117 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)36864">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %118 = emitc.call_opaque "__Runtime_dma_bd_config"(%115, %117) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 1 : i32, 4096 : i32, 0 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(3,5) lock=2 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(3, 5), XAie_LockInit(2, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %119 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)32768">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %120 = emitc.call_opaque "__Runtime_dma_bd_config"(%115, %119) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 0 : i32, 4096 : i32, 1 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(3,5), direction=S2MM */"
    %121 = emitc.call_opaque "__Runtime_dma_createio_4"(%115, %120) {args = [0 : index, 1 : index, 1 : i32, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 0 for tile (3,5) */"
    %122 = emitc.call_opaque "XAie_TileLoc"() {args = [3 : i8, 6 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    %123 = emitc.call_opaque "__runtime_buffer_offset"(%97) {args = [0 : index, 49152]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %124 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)36864">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %125 = emitc.call_opaque "__Runtime_dma_bd_config"(%122, %124) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 1 : i32, 4096 : i32, 0 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(3,6) lock=2 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(3, 6), XAie_LockInit(2, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %126 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)32768">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %127 = emitc.call_opaque "__Runtime_dma_bd_config"(%122, %126) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 0 : i32, 4096 : i32, 1 : i32, 0 : i32, 0 : i32, 2 : i32, -1 : i32, 3 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(3,6), direction=S2MM */"
    %128 = emitc.call_opaque "__Runtime_dma_createio_4"(%122, %127) {args = [0 : index, 1 : index, 1 : i32, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 0 for tile (3,6) */"
    emitc.verbatim "/* Allocated BD ID 1 for tile (3,0) */"
    %129 = emitc.call_opaque "__Runtime_startio"(%100) {args = [0 : index, 1 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %130 = emitc.call_opaque "__runtime_buffer_offset"(%arg0) {args = [0 : index, 0]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %131 = emitc.call_opaque "XAie_TileLoc"() {args = [6 : i8, 0 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=16384, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %132 = emitc.call_opaque "__runtime_buffer_arg"(%130) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %133 = emitc.call_opaque "__Runtime_dma_bd_config"(%131, %132) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 0 : i32, 16384 : i32, -1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=0, tile=(6,0), direction=MM2S */"
    %134 = emitc.call_opaque "__Runtime_dma_createio_4"(%131, %133) {args = [0 : index, 1 : index, 0 : i32, 0 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* DMA BD Config: bd_id=3, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %135 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)45056">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %136 = emitc.call_opaque "__Runtime_dma_bd_config"(%5, %135) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 3 : i32, 4096 : i32, 2 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(0,3) lock=0 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(0, 3), XAie_LockInit(0, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=2, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %137 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)40960">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %138 = emitc.call_opaque "__Runtime_dma_bd_config"(%5, %137) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 2 : i32, 4096 : i32, 3 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(0,3), direction=S2MM */"
    %139 = emitc.call_opaque "__Runtime_dma_createio_4"(%5, %138) {args = [0 : index, 1 : index, 0 : i32, 2 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 1 for tile (0,3) */"
    emitc.verbatim "/* DMA BD Config: bd_id=3, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %140 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)45056">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %141 = emitc.call_opaque "__Runtime_dma_bd_config"(%34, %140) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 3 : i32, 4096 : i32, 2 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(1,3) lock=0 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(1, 3), XAie_LockInit(0, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=2, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %142 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)40960">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %143 = emitc.call_opaque "__Runtime_dma_bd_config"(%34, %142) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 2 : i32, 4096 : i32, 3 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(1,3), direction=S2MM */"
    %144 = emitc.call_opaque "__Runtime_dma_createio_4"(%34, %143) {args = [0 : index, 1 : index, 0 : i32, 2 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 1 for tile (1,3) */"
    emitc.verbatim "/* DMA BD Config: bd_id=3, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %145 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)45056">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %146 = emitc.call_opaque "__Runtime_dma_bd_config"(%68, %145) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 3 : i32, 4096 : i32, 2 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(2,3) lock=0 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(2, 3), XAie_LockInit(0, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=2, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %147 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)40960">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %148 = emitc.call_opaque "__Runtime_dma_bd_config"(%68, %147) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 2 : i32, 4096 : i32, 3 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(2,3), direction=S2MM */"
    %149 = emitc.call_opaque "__Runtime_dma_createio_4"(%68, %148) {args = [0 : index, 1 : index, 0 : i32, 2 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 1 for tile (2,3) */"
    emitc.verbatim "/* DMA BD Config: bd_id=3, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %150 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)45056">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %151 = emitc.call_opaque "__Runtime_dma_bd_config"(%101, %150) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 3 : i32, 4096 : i32, 2 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(3,3) lock=0 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(3, 3), XAie_LockInit(0, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=2, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %152 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)40960">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %153 = emitc.call_opaque "__Runtime_dma_bd_config"(%101, %152) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 2 : i32, 4096 : i32, 3 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(3,3), direction=S2MM */"
    %154 = emitc.call_opaque "__Runtime_dma_createio_4"(%101, %153) {args = [0 : index, 1 : index, 0 : i32, 2 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 1 for tile (3,3) */"
    emitc.verbatim "/* Allocated BD ID 0 for tile (6,0) */"
    %155 = emitc.call_opaque "__Runtime_startio"(%134) {args = [0 : index, 0 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %156 = emitc.call_opaque "__runtime_buffer_offset"(%arg2) {args = [0 : index, 0]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=6, offset=192, len=2048, enable_packet=false, packet_id=4, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %157 = emitc.call_opaque "__runtime_buffer_arg"(%156) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %158 = emitc.call_opaque "__runtime_buffer_offset"(%157) {args = [0 : index, 192]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %159 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%64, %158) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 6 : i32, 2048 : i32, -1 : i32, 0 : i32, 4 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 2 : i32, 4 : i32, 16 : i32, 256 : i32, 32 : i32, 0 : i32, 0 : i32, 8192 : i32, 2 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=5, offset=128, len=2048, enable_packet=false, packet_id=3, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %160 = emitc.call_opaque "__runtime_buffer_arg"(%156) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %161 = emitc.call_opaque "__runtime_buffer_offset"(%160) {args = [0 : index, 128]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %162 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%64, %161) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 5 : i32, 2048 : i32, -1 : i32, 0 : i32, 3 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 2 : i32, 4 : i32, 16 : i32, 256 : i32, 32 : i32, 0 : i32, 0 : i32, 8192 : i32, 2 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, offset=64, len=2048, enable_packet=false, packet_id=2, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %163 = emitc.call_opaque "__runtime_buffer_arg"(%156) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %164 = emitc.call_opaque "__runtime_buffer_offset"(%163) {args = [0 : index, 64]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %165 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%64, %164) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 4 : i32, 2048 : i32, -1 : i32, 0 : i32, 2 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 2 : i32, 4 : i32, 16 : i32, 256 : i32, 32 : i32, 0 : i32, 0 : i32, 8192 : i32, 2 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=3, offset=0, len=2048, enable_packet=false, packet_id=1, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %166 = emitc.call_opaque "__runtime_buffer_arg"(%156) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %167 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%64, %166) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 3 : i32, 2048 : i32, -1 : i32, 0 : i32, 1 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 2 : i32, 4 : i32, 16 : i32, 256 : i32, 32 : i32, 0 : i32, 0 : i32, 8192 : i32, 2 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=3, tile=(3,0), direction=S2MM */"
    emitc.verbatim "/* Enable out-of-order BD on tile(3,0) ch=0 dir=S2MM */"
    emitc.call_opaque "__Runtime_dma_channel_enable_ooo"(%64) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">) -> ()
    %168 = emitc.call_opaque "__Runtime_dma_createio_4"(%64, %167) {args = [0 : index, 1 : index, 0 : i32, 3 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %169 = emitc.call_opaque "__runtime_buffer_offset"(%156) {args = [0 : index, 0]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=5, offset=0, len=2048, enable_packet=true, packet_id=1, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=3 */"
    %170 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)53248">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %171 = emitc.call_opaque "__Runtime_dma_bd_config"(%5, %170) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 5 : i32, 2048 : i32, 4 : i32, 1 : i32, 1 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 3 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(0,3) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(0, 3), XAie_LockInit(4, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=4, offset=0, len=2048, enable_packet=true, packet_id=1, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=3 */"
    %172 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)49152">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %173 = emitc.call_opaque "__Runtime_dma_bd_config"(%5, %172) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 4 : i32, 2048 : i32, 5 : i32, 1 : i32, 1 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 3 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(0,3), direction=MM2S */"
    %174 = emitc.call_opaque "__Runtime_dma_createio_4"(%5, %173) {args = [0 : index, 1 : index, 0 : i32, 4 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 2 for tile (0,3) */"
    %175 = emitc.call_opaque "__runtime_buffer_offset"(%156) {args = [0 : index, 4096]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=5, offset=0, len=2048, enable_packet=true, packet_id=2, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=4 */"
    %176 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)53248">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %177 = emitc.call_opaque "__Runtime_dma_bd_config"(%34, %176) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 5 : i32, 2048 : i32, 4 : i32, 1 : i32, 2 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 4 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(1,3) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(1, 3), XAie_LockInit(4, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=4, offset=0, len=2048, enable_packet=true, packet_id=2, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=4 */"
    %178 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)49152">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %179 = emitc.call_opaque "__Runtime_dma_bd_config"(%34, %178) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 4 : i32, 2048 : i32, 5 : i32, 1 : i32, 2 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 4 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(1,3), direction=MM2S */"
    %180 = emitc.call_opaque "__Runtime_dma_createio_4"(%34, %179) {args = [0 : index, 1 : index, 0 : i32, 4 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 2 for tile (1,3) */"
    %181 = emitc.call_opaque "__runtime_buffer_offset"(%156) {args = [0 : index, 8192]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=5, offset=0, len=2048, enable_packet=true, packet_id=3, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=5 */"
    %182 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)53248">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %183 = emitc.call_opaque "__Runtime_dma_bd_config"(%68, %182) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 5 : i32, 2048 : i32, 4 : i32, 1 : i32, 3 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 5 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(2,3) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(2, 3), XAie_LockInit(4, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=4, offset=0, len=2048, enable_packet=true, packet_id=3, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=5 */"
    %184 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)49152">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %185 = emitc.call_opaque "__Runtime_dma_bd_config"(%68, %184) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 4 : i32, 2048 : i32, 5 : i32, 1 : i32, 3 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 5 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(2,3), direction=MM2S */"
    %186 = emitc.call_opaque "__Runtime_dma_createio_4"(%68, %185) {args = [0 : index, 1 : index, 0 : i32, 4 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 2 for tile (2,3) */"
    %187 = emitc.call_opaque "__runtime_buffer_offset"(%156) {args = [0 : index, 12288]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=5, offset=0, len=2048, enable_packet=true, packet_id=4, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=6 */"
    %188 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)53248">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %189 = emitc.call_opaque "__Runtime_dma_bd_config"(%101, %188) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 5 : i32, 2048 : i32, 4 : i32, 1 : i32, 4 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 6 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(3,3) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(3, 3), XAie_LockInit(4, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=4, offset=0, len=2048, enable_packet=true, packet_id=4, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=6 */"
    %190 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)49152">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %191 = emitc.call_opaque "__Runtime_dma_bd_config"(%101, %190) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 4 : i32, 2048 : i32, 5 : i32, 1 : i32, 4 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 6 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(3,3), direction=MM2S */"
    %192 = emitc.call_opaque "__Runtime_dma_createio_4"(%101, %191) {args = [0 : index, 1 : index, 0 : i32, 4 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 2 for tile (3,3) */"
    emitc.verbatim "/* Allocated BD ID 2 for tile (3,0) */"
    %193 = emitc.call_opaque "__Runtime_startio"(%168) {args = [0 : index, 2 : i32, 8 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %194 = emitc.call_opaque "__runtime_buffer_offset"(%arg0) {args = [0 : index, 16384]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=16384, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %195 = emitc.call_opaque "__runtime_buffer_arg"(%194) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %196 = emitc.call_opaque "__Runtime_dma_bd_config"(%131, %195) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 1 : i32, 16384 : i32, -1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=1, tile=(6,0), direction=MM2S */"
    %197 = emitc.call_opaque "__Runtime_dma_createio_4"(%131, %196) {args = [0 : index, 1 : index, 1 : i32, 1 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %198 = emitc.call_opaque "__runtime_buffer_offset"(%194) {args = [0 : index, 16384]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=3, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %199 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)45056">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %200 = emitc.call_opaque "__Runtime_dma_bd_config"(%11, %199) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 3 : i32, 4096 : i32, 2 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(0,4) lock=0 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(0, 4), XAie_LockInit(0, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=2, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %201 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)40960">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %202 = emitc.call_opaque "__Runtime_dma_bd_config"(%11, %201) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 2 : i32, 4096 : i32, 3 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(0,4), direction=S2MM */"
    %203 = emitc.call_opaque "__Runtime_dma_createio_4"(%11, %202) {args = [0 : index, 1 : index, 0 : i32, 2 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 1 for tile (0,4) */"
    %204 = emitc.call_opaque "__runtime_buffer_offset"(%194) {args = [0 : index, 16384]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=3, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %205 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)45056">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %206 = emitc.call_opaque "__Runtime_dma_bd_config"(%41, %205) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 3 : i32, 4096 : i32, 2 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(1,4) lock=0 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(1, 4), XAie_LockInit(0, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=2, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %207 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)40960">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %208 = emitc.call_opaque "__Runtime_dma_bd_config"(%41, %207) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 2 : i32, 4096 : i32, 3 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(1,4), direction=S2MM */"
    %209 = emitc.call_opaque "__Runtime_dma_createio_4"(%41, %208) {args = [0 : index, 1 : index, 0 : i32, 2 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 1 for tile (1,4) */"
    %210 = emitc.call_opaque "__runtime_buffer_offset"(%194) {args = [0 : index, 16384]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=3, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %211 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)45056">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %212 = emitc.call_opaque "__Runtime_dma_bd_config"(%75, %211) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 3 : i32, 4096 : i32, 2 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(2,4) lock=0 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(2, 4), XAie_LockInit(0, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=2, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %213 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)40960">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %214 = emitc.call_opaque "__Runtime_dma_bd_config"(%75, %213) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 2 : i32, 4096 : i32, 3 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(2,4), direction=S2MM */"
    %215 = emitc.call_opaque "__Runtime_dma_createio_4"(%75, %214) {args = [0 : index, 1 : index, 0 : i32, 2 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 1 for tile (2,4) */"
    %216 = emitc.call_opaque "__runtime_buffer_offset"(%194) {args = [0 : index, 16384]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=3, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %217 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)45056">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %218 = emitc.call_opaque "__Runtime_dma_bd_config"(%108, %217) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 3 : i32, 4096 : i32, 2 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(3,4) lock=0 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(3, 4), XAie_LockInit(0, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=2, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %219 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)40960">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %220 = emitc.call_opaque "__Runtime_dma_bd_config"(%108, %219) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 2 : i32, 4096 : i32, 3 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(3,4), direction=S2MM */"
    %221 = emitc.call_opaque "__Runtime_dma_createio_4"(%108, %220) {args = [0 : index, 1 : index, 0 : i32, 2 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 1 for tile (3,4) */"
    emitc.verbatim "/* Allocated BD ID 1 for tile (6,0) */"
    %222 = emitc.call_opaque "__Runtime_startio"(%197) {args = [0 : index, 1 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %223 = emitc.call_opaque "__runtime_buffer_offset"(%arg2) {args = [0 : index, 16384]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=11, offset=192, len=2048, enable_packet=false, packet_id=8, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %224 = emitc.call_opaque "__runtime_buffer_arg"(%223) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %225 = emitc.call_opaque "__runtime_buffer_offset"(%224) {args = [0 : index, 192]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %226 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%64, %225) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 11 : i32, 2048 : i32, -1 : i32, 0 : i32, 8 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 2 : i32, 4 : i32, 16 : i32, 256 : i32, 32 : i32, 0 : i32, 0 : i32, 8192 : i32, 2 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=10, offset=128, len=2048, enable_packet=false, packet_id=7, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %227 = emitc.call_opaque "__runtime_buffer_arg"(%223) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %228 = emitc.call_opaque "__runtime_buffer_offset"(%227) {args = [0 : index, 128]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %229 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%64, %228) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 10 : i32, 2048 : i32, -1 : i32, 0 : i32, 7 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 2 : i32, 4 : i32, 16 : i32, 256 : i32, 32 : i32, 0 : i32, 0 : i32, 8192 : i32, 2 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=9, offset=64, len=2048, enable_packet=false, packet_id=6, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %230 = emitc.call_opaque "__runtime_buffer_arg"(%223) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %231 = emitc.call_opaque "__runtime_buffer_offset"(%230) {args = [0 : index, 64]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %232 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%64, %231) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 9 : i32, 2048 : i32, -1 : i32, 0 : i32, 6 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 2 : i32, 4 : i32, 16 : i32, 256 : i32, 32 : i32, 0 : i32, 0 : i32, 8192 : i32, 2 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=8, offset=0, len=2048, enable_packet=false, packet_id=5, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %233 = emitc.call_opaque "__runtime_buffer_arg"(%223) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %234 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%64, %233) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 8 : i32, 2048 : i32, -1 : i32, 0 : i32, 5 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 2 : i32, 4 : i32, 16 : i32, 256 : i32, 32 : i32, 0 : i32, 0 : i32, 8192 : i32, 2 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=8, tile=(3,0), direction=S2MM */"
    emitc.verbatim "/* Enable out-of-order BD on tile(3,0) ch=1 dir=S2MM */"
    emitc.call_opaque "__Runtime_dma_channel_enable_ooo"(%64) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">) -> ()
    %235 = emitc.call_opaque "__Runtime_dma_createio_4"(%64, %234) {args = [0 : index, 1 : index, 1 : i32, 8 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %236 = emitc.call_opaque "__runtime_buffer_offset"(%223) {args = [0 : index, 0]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=5, offset=0, len=2048, enable_packet=true, packet_id=5, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=8 */"
    %237 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)53248">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %238 = emitc.call_opaque "__Runtime_dma_bd_config"(%11, %237) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 5 : i32, 2048 : i32, 4 : i32, 1 : i32, 5 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 8 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(0,4) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(0, 4), XAie_LockInit(4, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=4, offset=0, len=2048, enable_packet=true, packet_id=5, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=8 */"
    %239 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)49152">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %240 = emitc.call_opaque "__Runtime_dma_bd_config"(%11, %239) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 4 : i32, 2048 : i32, 5 : i32, 1 : i32, 5 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 8 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(0,4), direction=MM2S */"
    %241 = emitc.call_opaque "__Runtime_dma_createio_4"(%11, %240) {args = [0 : index, 1 : index, 0 : i32, 4 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 2 for tile (0,4) */"
    %242 = emitc.call_opaque "__runtime_buffer_offset"(%223) {args = [0 : index, 4096]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=5, offset=0, len=2048, enable_packet=true, packet_id=6, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=9 */"
    %243 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)53248">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %244 = emitc.call_opaque "__Runtime_dma_bd_config"(%41, %243) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 5 : i32, 2048 : i32, 4 : i32, 1 : i32, 6 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 9 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(1,4) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(1, 4), XAie_LockInit(4, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=4, offset=0, len=2048, enable_packet=true, packet_id=6, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=9 */"
    %245 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)49152">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %246 = emitc.call_opaque "__Runtime_dma_bd_config"(%41, %245) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 4 : i32, 2048 : i32, 5 : i32, 1 : i32, 6 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 9 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(1,4), direction=MM2S */"
    %247 = emitc.call_opaque "__Runtime_dma_createio_4"(%41, %246) {args = [0 : index, 1 : index, 0 : i32, 4 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 2 for tile (1,4) */"
    %248 = emitc.call_opaque "__runtime_buffer_offset"(%223) {args = [0 : index, 8192]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=5, offset=0, len=2048, enable_packet=true, packet_id=7, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=10 */"
    %249 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)53248">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %250 = emitc.call_opaque "__Runtime_dma_bd_config"(%75, %249) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 5 : i32, 2048 : i32, 4 : i32, 1 : i32, 7 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 10 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(2,4) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(2, 4), XAie_LockInit(4, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=4, offset=0, len=2048, enable_packet=true, packet_id=7, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=10 */"
    %251 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)49152">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %252 = emitc.call_opaque "__Runtime_dma_bd_config"(%75, %251) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 4 : i32, 2048 : i32, 5 : i32, 1 : i32, 7 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 10 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(2,4), direction=MM2S */"
    %253 = emitc.call_opaque "__Runtime_dma_createio_4"(%75, %252) {args = [0 : index, 1 : index, 0 : i32, 4 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 2 for tile (2,4) */"
    %254 = emitc.call_opaque "__runtime_buffer_offset"(%223) {args = [0 : index, 12288]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=5, offset=0, len=2048, enable_packet=true, packet_id=8, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=11 */"
    %255 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)53248">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %256 = emitc.call_opaque "__Runtime_dma_bd_config"(%108, %255) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 5 : i32, 2048 : i32, 4 : i32, 1 : i32, 8 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 11 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(3,4) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(3, 4), XAie_LockInit(4, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=4, offset=0, len=2048, enable_packet=true, packet_id=8, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=11 */"
    %257 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)49152">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %258 = emitc.call_opaque "__Runtime_dma_bd_config"(%108, %257) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 4 : i32, 2048 : i32, 5 : i32, 1 : i32, 8 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 11 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(3,4), direction=MM2S */"
    %259 = emitc.call_opaque "__Runtime_dma_createio_4"(%108, %258) {args = [0 : index, 1 : index, 0 : i32, 4 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 2 for tile (3,4) */"
    emitc.verbatim "/* Allocated BD ID 3 for tile (3,0) */"
    %260 = emitc.call_opaque "__Runtime_startio"(%235) {args = [0 : index, 3 : i32, 8 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %261 = emitc.call_opaque "__runtime_buffer_offset"(%arg0) {args = [0 : index, 32768]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %262 = emitc.call_opaque "XAie_TileLoc"() {args = [7 : i8, 0 : i8]} : () -> !emitc.opaque<"XAie_LocType">
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=16384, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %263 = emitc.call_opaque "__runtime_buffer_arg"(%261) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %264 = emitc.call_opaque "__Runtime_dma_bd_config"(%262, %263) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 0 : i32, 16384 : i32, -1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=0, tile=(7,0), direction=MM2S */"
    %265 = emitc.call_opaque "__Runtime_dma_createio_4"(%262, %264) {args = [0 : index, 1 : index, 0 : i32, 0 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %266 = emitc.call_opaque "__runtime_buffer_offset"(%261) {args = [0 : index, 32768]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=3, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %267 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)45056">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %268 = emitc.call_opaque "__Runtime_dma_bd_config"(%17, %267) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 3 : i32, 4096 : i32, 2 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(0,5) lock=0 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(0, 5), XAie_LockInit(0, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=2, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %269 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)40960">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %270 = emitc.call_opaque "__Runtime_dma_bd_config"(%17, %269) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 2 : i32, 4096 : i32, 3 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(0,5), direction=S2MM */"
    %271 = emitc.call_opaque "__Runtime_dma_createio_4"(%17, %270) {args = [0 : index, 1 : index, 0 : i32, 2 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 1 for tile (0,5) */"
    %272 = emitc.call_opaque "__runtime_buffer_offset"(%261) {args = [0 : index, 32768]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=3, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %273 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)45056">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %274 = emitc.call_opaque "__Runtime_dma_bd_config"(%48, %273) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 3 : i32, 4096 : i32, 2 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(1,5) lock=0 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(1, 5), XAie_LockInit(0, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=2, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %275 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)40960">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %276 = emitc.call_opaque "__Runtime_dma_bd_config"(%48, %275) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 2 : i32, 4096 : i32, 3 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(1,5), direction=S2MM */"
    %277 = emitc.call_opaque "__Runtime_dma_createio_4"(%48, %276) {args = [0 : index, 1 : index, 0 : i32, 2 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 1 for tile (1,5) */"
    %278 = emitc.call_opaque "__runtime_buffer_offset"(%261) {args = [0 : index, 32768]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=3, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %279 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)45056">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %280 = emitc.call_opaque "__Runtime_dma_bd_config"(%82, %279) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 3 : i32, 4096 : i32, 2 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(2,5) lock=0 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(2, 5), XAie_LockInit(0, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=2, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %281 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)40960">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %282 = emitc.call_opaque "__Runtime_dma_bd_config"(%82, %281) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 2 : i32, 4096 : i32, 3 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(2,5), direction=S2MM */"
    %283 = emitc.call_opaque "__Runtime_dma_createio_4"(%82, %282) {args = [0 : index, 1 : index, 0 : i32, 2 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 1 for tile (2,5) */"
    %284 = emitc.call_opaque "__runtime_buffer_offset"(%261) {args = [0 : index, 32768]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=3, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %285 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)45056">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %286 = emitc.call_opaque "__Runtime_dma_bd_config"(%115, %285) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 3 : i32, 4096 : i32, 2 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(3,5) lock=0 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(3, 5), XAie_LockInit(0, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=2, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %287 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)40960">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %288 = emitc.call_opaque "__Runtime_dma_bd_config"(%115, %287) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 2 : i32, 4096 : i32, 3 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(3,5), direction=S2MM */"
    %289 = emitc.call_opaque "__Runtime_dma_createio_4"(%115, %288) {args = [0 : index, 1 : index, 0 : i32, 2 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 1 for tile (3,5) */"
    emitc.verbatim "/* Allocated BD ID 0 for tile (7,0) */"
    %290 = emitc.call_opaque "__Runtime_startio"(%265) {args = [0 : index, 0 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %291 = emitc.call_opaque "__runtime_buffer_offset"(%arg2) {args = [0 : index, 32768]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=6, offset=192, len=2048, enable_packet=false, packet_id=12, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %292 = emitc.call_opaque "__runtime_buffer_arg"(%291) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %293 = emitc.call_opaque "__runtime_buffer_offset"(%292) {args = [0 : index, 192]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %294 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%1, %293) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 6 : i32, 2048 : i32, -1 : i32, 0 : i32, 12 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 2 : i32, 4 : i32, 16 : i32, 256 : i32, 32 : i32, 0 : i32, 0 : i32, 8192 : i32, 2 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=5, offset=128, len=2048, enable_packet=false, packet_id=11, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %295 = emitc.call_opaque "__runtime_buffer_arg"(%291) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %296 = emitc.call_opaque "__runtime_buffer_offset"(%295) {args = [0 : index, 128]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %297 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%1, %296) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 5 : i32, 2048 : i32, -1 : i32, 0 : i32, 11 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 2 : i32, 4 : i32, 16 : i32, 256 : i32, 32 : i32, 0 : i32, 0 : i32, 8192 : i32, 2 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, offset=64, len=2048, enable_packet=false, packet_id=10, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %298 = emitc.call_opaque "__runtime_buffer_arg"(%291) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %299 = emitc.call_opaque "__runtime_buffer_offset"(%298) {args = [0 : index, 64]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %300 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%1, %299) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 4 : i32, 2048 : i32, -1 : i32, 0 : i32, 10 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 2 : i32, 4 : i32, 16 : i32, 256 : i32, 32 : i32, 0 : i32, 0 : i32, 8192 : i32, 2 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=3, offset=0, len=2048, enable_packet=false, packet_id=9, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %301 = emitc.call_opaque "__runtime_buffer_arg"(%291) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %302 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%1, %301) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 3 : i32, 2048 : i32, -1 : i32, 0 : i32, 9 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 2 : i32, 4 : i32, 16 : i32, 256 : i32, 32 : i32, 0 : i32, 0 : i32, 8192 : i32, 2 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=3, tile=(2,0), direction=S2MM */"
    emitc.verbatim "/* Enable out-of-order BD on tile(2,0) ch=0 dir=S2MM */"
    emitc.call_opaque "__Runtime_dma_channel_enable_ooo"(%1) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 0 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">) -> ()
    %303 = emitc.call_opaque "__Runtime_dma_createio_4"(%1, %302) {args = [0 : index, 1 : index, 0 : i32, 3 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %304 = emitc.call_opaque "__runtime_buffer_offset"(%291) {args = [0 : index, 0]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=5, offset=0, len=2048, enable_packet=true, packet_id=9, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=3 */"
    %305 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)53248">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %306 = emitc.call_opaque "__Runtime_dma_bd_config"(%17, %305) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 5 : i32, 2048 : i32, 4 : i32, 1 : i32, 9 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 3 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(0,5) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(0, 5), XAie_LockInit(4, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=4, offset=0, len=2048, enable_packet=true, packet_id=9, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=3 */"
    %307 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)49152">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %308 = emitc.call_opaque "__Runtime_dma_bd_config"(%17, %307) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 4 : i32, 2048 : i32, 5 : i32, 1 : i32, 9 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 3 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(0,5), direction=MM2S */"
    %309 = emitc.call_opaque "__Runtime_dma_createio_4"(%17, %308) {args = [0 : index, 1 : index, 0 : i32, 4 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 2 for tile (0,5) */"
    %310 = emitc.call_opaque "__runtime_buffer_offset"(%291) {args = [0 : index, 4096]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=5, offset=0, len=2048, enable_packet=true, packet_id=10, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=4 */"
    %311 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)53248">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %312 = emitc.call_opaque "__Runtime_dma_bd_config"(%48, %311) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 5 : i32, 2048 : i32, 4 : i32, 1 : i32, 10 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 4 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(1,5) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(1, 5), XAie_LockInit(4, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=4, offset=0, len=2048, enable_packet=true, packet_id=10, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=4 */"
    %313 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)49152">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %314 = emitc.call_opaque "__Runtime_dma_bd_config"(%48, %313) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 4 : i32, 2048 : i32, 5 : i32, 1 : i32, 10 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 4 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(1,5), direction=MM2S */"
    %315 = emitc.call_opaque "__Runtime_dma_createio_4"(%48, %314) {args = [0 : index, 1 : index, 0 : i32, 4 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 2 for tile (1,5) */"
    %316 = emitc.call_opaque "__runtime_buffer_offset"(%291) {args = [0 : index, 8192]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=5, offset=0, len=2048, enable_packet=true, packet_id=11, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=5 */"
    %317 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)53248">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %318 = emitc.call_opaque "__Runtime_dma_bd_config"(%82, %317) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 5 : i32, 2048 : i32, 4 : i32, 1 : i32, 11 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 5 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(2,5) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(2, 5), XAie_LockInit(4, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=4, offset=0, len=2048, enable_packet=true, packet_id=11, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=5 */"
    %319 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)49152">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %320 = emitc.call_opaque "__Runtime_dma_bd_config"(%82, %319) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 4 : i32, 2048 : i32, 5 : i32, 1 : i32, 11 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 5 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(2,5), direction=MM2S */"
    %321 = emitc.call_opaque "__Runtime_dma_createio_4"(%82, %320) {args = [0 : index, 1 : index, 0 : i32, 4 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 2 for tile (2,5) */"
    %322 = emitc.call_opaque "__runtime_buffer_offset"(%291) {args = [0 : index, 12288]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=5, offset=0, len=2048, enable_packet=true, packet_id=12, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=6 */"
    %323 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)53248">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %324 = emitc.call_opaque "__Runtime_dma_bd_config"(%115, %323) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 5 : i32, 2048 : i32, 4 : i32, 1 : i32, 12 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 6 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(3,5) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(3, 5), XAie_LockInit(4, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=4, offset=0, len=2048, enable_packet=true, packet_id=12, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=6 */"
    %325 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)49152">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %326 = emitc.call_opaque "__Runtime_dma_bd_config"(%115, %325) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 4 : i32, 2048 : i32, 5 : i32, 1 : i32, 12 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 6 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(3,5), direction=MM2S */"
    %327 = emitc.call_opaque "__Runtime_dma_createio_4"(%115, %326) {args = [0 : index, 1 : index, 0 : i32, 4 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 2 for tile (3,5) */"
    emitc.verbatim "/* Allocated BD ID 2 for tile (2,0) */"
    %328 = emitc.call_opaque "__Runtime_startio"(%303) {args = [0 : index, 2 : i32, 8 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %329 = emitc.call_opaque "__runtime_buffer_offset"(%arg0) {args = [0 : index, 49152]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=16384, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %330 = emitc.call_opaque "__runtime_buffer_arg"(%329) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %331 = emitc.call_opaque "__Runtime_dma_bd_config"(%262, %330) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 1 : i32, 16384 : i32, -1 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=1, tile=(7,0), direction=MM2S */"
    %332 = emitc.call_opaque "__Runtime_dma_createio_4"(%262, %331) {args = [0 : index, 1 : index, 1 : i32, 1 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %333 = emitc.call_opaque "__runtime_buffer_offset"(%329) {args = [0 : index, 49152]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=3, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %334 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)45056">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %335 = emitc.call_opaque "__Runtime_dma_bd_config"(%23, %334) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 3 : i32, 4096 : i32, 2 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(0,6) lock=0 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(0, 6), XAie_LockInit(0, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=2, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %336 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)40960">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %337 = emitc.call_opaque "__Runtime_dma_bd_config"(%23, %336) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 2 : i32, 4096 : i32, 3 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(0,6), direction=S2MM */"
    %338 = emitc.call_opaque "__Runtime_dma_createio_4"(%23, %337) {args = [0 : index, 1 : index, 0 : i32, 2 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 1 for tile (0,6) */"
    %339 = emitc.call_opaque "__runtime_buffer_offset"(%329) {args = [0 : index, 49152]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=3, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %340 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)45056">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %341 = emitc.call_opaque "__Runtime_dma_bd_config"(%55, %340) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 3 : i32, 4096 : i32, 2 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(1,6) lock=0 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(1, 6), XAie_LockInit(0, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=2, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %342 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)40960">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %343 = emitc.call_opaque "__Runtime_dma_bd_config"(%55, %342) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 2 : i32, 4096 : i32, 3 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(1,6), direction=S2MM */"
    %344 = emitc.call_opaque "__Runtime_dma_createio_4"(%55, %343) {args = [0 : index, 1 : index, 0 : i32, 2 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 1 for tile (1,6) */"
    %345 = emitc.call_opaque "__runtime_buffer_offset"(%329) {args = [0 : index, 49152]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=3, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %346 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)45056">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %347 = emitc.call_opaque "__Runtime_dma_bd_config"(%89, %346) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 3 : i32, 4096 : i32, 2 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(2,6) lock=0 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(2, 6), XAie_LockInit(0, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=2, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %348 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)40960">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %349 = emitc.call_opaque "__Runtime_dma_bd_config"(%89, %348) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 2 : i32, 4096 : i32, 3 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(2,6), direction=S2MM */"
    %350 = emitc.call_opaque "__Runtime_dma_createio_4"(%89, %349) {args = [0 : index, 1 : index, 0 : i32, 2 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 1 for tile (2,6) */"
    %351 = emitc.call_opaque "__runtime_buffer_offset"(%329) {args = [0 : index, 49152]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=3, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %352 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)45056">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %353 = emitc.call_opaque "__Runtime_dma_bd_config"(%122, %352) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 3 : i32, 4096 : i32, 2 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(3,6) lock=0 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(3, 6), XAie_LockInit(0, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=2, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %354 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)40960">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %355 = emitc.call_opaque "__Runtime_dma_bd_config"(%122, %354) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 2 : i32, 4096 : i32, 3 : i32, 0 : i32, 0 : i32, 0 : i32, -1 : i32, 1 : i32, 1 : i32, -1 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(3,6), direction=S2MM */"
    %356 = emitc.call_opaque "__Runtime_dma_createio_4"(%122, %355) {args = [0 : index, 1 : index, 0 : i32, 2 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 1 for tile (3,6) */"
    emitc.verbatim "/* Allocated BD ID 1 for tile (7,0) */"
    %357 = emitc.call_opaque "__Runtime_startio"(%332) {args = [0 : index, 1 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %358 = emitc.call_opaque "__runtime_buffer_offset"(%arg2) {args = [0 : index, 49152]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=11, offset=192, len=2048, enable_packet=false, packet_id=16, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %359 = emitc.call_opaque "__runtime_buffer_arg"(%358) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %360 = emitc.call_opaque "__runtime_buffer_offset"(%359) {args = [0 : index, 192]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %361 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%1, %360) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 11 : i32, 2048 : i32, -1 : i32, 0 : i32, 16 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 2 : i32, 4 : i32, 16 : i32, 256 : i32, 32 : i32, 0 : i32, 0 : i32, 8192 : i32, 2 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=10, offset=128, len=2048, enable_packet=false, packet_id=15, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %362 = emitc.call_opaque "__runtime_buffer_arg"(%358) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %363 = emitc.call_opaque "__runtime_buffer_offset"(%362) {args = [0 : index, 128]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %364 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%1, %363) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 10 : i32, 2048 : i32, -1 : i32, 0 : i32, 15 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 2 : i32, 4 : i32, 16 : i32, 256 : i32, 32 : i32, 0 : i32, 0 : i32, 8192 : i32, 2 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=9, offset=64, len=2048, enable_packet=false, packet_id=14, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %365 = emitc.call_opaque "__runtime_buffer_arg"(%358) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %366 = emitc.call_opaque "__runtime_buffer_offset"(%365) {args = [0 : index, 64]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %367 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%1, %366) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 9 : i32, 2048 : i32, -1 : i32, 0 : i32, 14 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 2 : i32, 4 : i32, 16 : i32, 256 : i32, 32 : i32, 0 : i32, 0 : i32, 8192 : i32, 2 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=8, offset=0, len=2048, enable_packet=false, packet_id=13, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %368 = emitc.call_opaque "__runtime_buffer_arg"(%358) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %369 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%1, %368) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 8 : i32, 2048 : i32, -1 : i32, 0 : i32, 13 : i32, -1 : i32, 0 : i32, -1 : i32, 0 : i32, -1 : i32, 2 : i32, 4 : i32, 16 : i32, 256 : i32, 32 : i32, 0 : i32, 0 : i32, 8192 : i32, 2 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=8, tile=(2,0), direction=S2MM */"
    emitc.verbatim "/* Enable out-of-order BD on tile(2,0) ch=1 dir=S2MM */"
    emitc.call_opaque "__Runtime_dma_channel_enable_ooo"(%1) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">) -> ()
    %370 = emitc.call_opaque "__Runtime_dma_createio_4"(%1, %369) {args = [0 : index, 1 : index, 1 : i32, 8 : i32, #emitc.opaque<"DMA_S2MM">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    %371 = emitc.call_opaque "__runtime_buffer_offset"(%358) {args = [0 : index, 0]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=5, offset=0, len=2048, enable_packet=true, packet_id=13, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=8 */"
    %372 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)53248">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %373 = emitc.call_opaque "__Runtime_dma_bd_config"(%23, %372) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 5 : i32, 2048 : i32, 4 : i32, 1 : i32, 13 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 8 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(0,6) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(0, 6), XAie_LockInit(4, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=4, offset=0, len=2048, enable_packet=true, packet_id=13, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=8 */"
    %374 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)49152">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %375 = emitc.call_opaque "__Runtime_dma_bd_config"(%23, %374) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 4 : i32, 2048 : i32, 5 : i32, 1 : i32, 13 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 8 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(0,6), direction=MM2S */"
    %376 = emitc.call_opaque "__Runtime_dma_createio_4"(%23, %375) {args = [0 : index, 1 : index, 0 : i32, 4 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 2 for tile (0,6) */"
    %377 = emitc.call_opaque "__runtime_buffer_offset"(%358) {args = [0 : index, 4096]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=5, offset=0, len=2048, enable_packet=true, packet_id=14, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=9 */"
    %378 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)53248">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %379 = emitc.call_opaque "__Runtime_dma_bd_config"(%55, %378) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 5 : i32, 2048 : i32, 4 : i32, 1 : i32, 14 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 9 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(1,6) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(1, 6), XAie_LockInit(4, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=4, offset=0, len=2048, enable_packet=true, packet_id=14, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=9 */"
    %380 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)49152">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %381 = emitc.call_opaque "__Runtime_dma_bd_config"(%55, %380) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 4 : i32, 2048 : i32, 5 : i32, 1 : i32, 14 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 9 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(1,6), direction=MM2S */"
    %382 = emitc.call_opaque "__Runtime_dma_createio_4"(%55, %381) {args = [0 : index, 1 : index, 0 : i32, 4 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 2 for tile (1,6) */"
    %383 = emitc.call_opaque "__runtime_buffer_offset"(%358) {args = [0 : index, 8192]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=5, offset=0, len=2048, enable_packet=true, packet_id=15, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=10 */"
    %384 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)53248">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %385 = emitc.call_opaque "__Runtime_dma_bd_config"(%89, %384) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 5 : i32, 2048 : i32, 4 : i32, 1 : i32, 15 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 10 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(2,6) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(2, 6), XAie_LockInit(4, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=4, offset=0, len=2048, enable_packet=true, packet_id=15, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=10 */"
    %386 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)49152">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %387 = emitc.call_opaque "__Runtime_dma_bd_config"(%89, %386) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 4 : i32, 2048 : i32, 5 : i32, 1 : i32, 15 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 10 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(2,6), direction=MM2S */"
    %388 = emitc.call_opaque "__Runtime_dma_createio_4"(%89, %387) {args = [0 : index, 1 : index, 0 : i32, 4 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 2 for tile (2,6) */"
    %389 = emitc.call_opaque "__runtime_buffer_offset"(%358) {args = [0 : index, 12288]} : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=5, offset=0, len=2048, enable_packet=true, packet_id=16, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=11 */"
    %390 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)53248">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %391 = emitc.call_opaque "__Runtime_dma_bd_config"(%122, %390) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 5 : i32, 2048 : i32, 4 : i32, 1 : i32, 16 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 11 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(3,6) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(3, 6), XAie_LockInit(4, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=4, offset=0, len=2048, enable_packet=true, packet_id=16, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=11 */"
    %392 = emitc.call_opaque "__runtime_buffer_arg"() {args = [#emitc.opaque<"(void*)49152">]} : () -> !emitc.ptr<!emitc.opaque<"void">>
    %393 = emitc.call_opaque "__Runtime_dma_bd_config"(%122, %392) {args = [#emitc.opaque<"g_DevInst">, 0 : index, 1 : index, 4 : i32, 2048 : i32, 5 : i32, 1 : i32, 16 : i32, 5 : i32, -1 : i32, 4 : i32, 1 : i32, 11 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(3,6), direction=MM2S */"
    %394 = emitc.call_opaque "__Runtime_dma_createio_4"(%122, %393) {args = [0 : index, 1 : index, 0 : i32, 4 : i32, #emitc.opaque<"DMA_MM2S">]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 2 for tile (3,6) */"
    emitc.verbatim "/* Allocated BD ID 3 for tile (2,0) */"
    %395 = emitc.call_opaque "__Runtime_startio"(%370) {args = [0 : index, 3 : i32, 8 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* Load Kernel Group: 16 tile(s) */"
    %396 = emitc.call_opaque "__Runtime_load_kernel_group_16t"(%5, %11, %17, %23, %34, %41, %48, %55, %68, %75, %82, %89, %101, %108, %115, %122) {args = [0 : index, 1 : index, 2 : index, 3 : index, 4 : index, 5 : index, 6 : index, 7 : index, 8 : index, 9 : index, 10 : index, 11 : index, 12 : index, 13 : index, 14 : index, 15 : index, 16 : i32]} : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">) -> !emitc.opaque<"kernel_group">
    emitc.verbatim "/* Launch Kernel Group */"
    %397 = emitc.call_opaque "__Runtime_launch_kernel_group"(%396) : (!emitc.opaque<"kernel_group">) -> !emitc.opaque<"event">
    %398 = emitc.call_opaque "__Runtime_startio"(%10) {args = [0 : index, 0 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %399 = emitc.call_opaque "__Runtime_startio"(%16) {args = [0 : index, 0 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %400 = emitc.call_opaque "__Runtime_startio"(%22) {args = [0 : index, 0 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %401 = emitc.call_opaque "__Runtime_startio"(%28) {args = [0 : index, 0 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %402 = emitc.call_opaque "__Runtime_startio"(%40) {args = [0 : index, 0 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %403 = emitc.call_opaque "__Runtime_startio"(%47) {args = [0 : index, 0 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %404 = emitc.call_opaque "__Runtime_startio"(%54) {args = [0 : index, 0 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %405 = emitc.call_opaque "__Runtime_startio"(%61) {args = [0 : index, 0 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %406 = emitc.call_opaque "__Runtime_startio"(%74) {args = [0 : index, 0 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %407 = emitc.call_opaque "__Runtime_startio"(%81) {args = [0 : index, 0 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %408 = emitc.call_opaque "__Runtime_startio"(%88) {args = [0 : index, 0 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %409 = emitc.call_opaque "__Runtime_startio"(%95) {args = [0 : index, 0 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %410 = emitc.call_opaque "__Runtime_startio"(%107) {args = [0 : index, 0 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %411 = emitc.call_opaque "__Runtime_startio"(%114) {args = [0 : index, 0 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %412 = emitc.call_opaque "__Runtime_startio"(%121) {args = [0 : index, 0 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %413 = emitc.call_opaque "__Runtime_startio"(%128) {args = [0 : index, 0 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %414 = emitc.call_opaque "__Runtime_startio"(%139) {args = [0 : index, 1 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %415 = emitc.call_opaque "__Runtime_startio"(%144) {args = [0 : index, 1 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %416 = emitc.call_opaque "__Runtime_startio"(%149) {args = [0 : index, 1 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %417 = emitc.call_opaque "__Runtime_startio"(%154) {args = [0 : index, 1 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %418 = emitc.call_opaque "__Runtime_startio"(%174) {args = [0 : index, 2 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %419 = emitc.call_opaque "__Runtime_startio"(%180) {args = [0 : index, 2 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %420 = emitc.call_opaque "__Runtime_startio"(%186) {args = [0 : index, 2 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %421 = emitc.call_opaque "__Runtime_startio"(%192) {args = [0 : index, 2 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %422 = emitc.call_opaque "__Runtime_startio"(%203) {args = [0 : index, 1 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %423 = emitc.call_opaque "__Runtime_startio"(%209) {args = [0 : index, 1 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %424 = emitc.call_opaque "__Runtime_startio"(%215) {args = [0 : index, 1 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %425 = emitc.call_opaque "__Runtime_startio"(%221) {args = [0 : index, 1 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %426 = emitc.call_opaque "__Runtime_startio"(%241) {args = [0 : index, 2 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %427 = emitc.call_opaque "__Runtime_startio"(%247) {args = [0 : index, 2 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %428 = emitc.call_opaque "__Runtime_startio"(%253) {args = [0 : index, 2 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %429 = emitc.call_opaque "__Runtime_startio"(%259) {args = [0 : index, 2 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %430 = emitc.call_opaque "__Runtime_startio"(%271) {args = [0 : index, 1 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %431 = emitc.call_opaque "__Runtime_startio"(%277) {args = [0 : index, 1 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %432 = emitc.call_opaque "__Runtime_startio"(%283) {args = [0 : index, 1 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %433 = emitc.call_opaque "__Runtime_startio"(%289) {args = [0 : index, 1 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %434 = emitc.call_opaque "__Runtime_startio"(%309) {args = [0 : index, 2 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %435 = emitc.call_opaque "__Runtime_startio"(%315) {args = [0 : index, 2 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %436 = emitc.call_opaque "__Runtime_startio"(%321) {args = [0 : index, 2 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %437 = emitc.call_opaque "__Runtime_startio"(%327) {args = [0 : index, 2 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %438 = emitc.call_opaque "__Runtime_startio"(%338) {args = [0 : index, 1 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %439 = emitc.call_opaque "__Runtime_startio"(%344) {args = [0 : index, 1 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %440 = emitc.call_opaque "__Runtime_startio"(%350) {args = [0 : index, 1 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %441 = emitc.call_opaque "__Runtime_startio"(%356) {args = [0 : index, 1 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %442 = emitc.call_opaque "__Runtime_startio"(%376) {args = [0 : index, 2 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %443 = emitc.call_opaque "__Runtime_startio"(%382) {args = [0 : index, 2 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %444 = emitc.call_opaque "__Runtime_startio"(%388) {args = [0 : index, 2 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    %445 = emitc.call_opaque "__Runtime_startio"(%394) {args = [0 : index, 2 : i32, 1 : i32]} : (!emitc.opaque<"io">) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* Wait for 13 event(s) */"
    emitc.call_opaque "__Runtime_wait"(%397) : (!emitc.opaque<"event">) -> ()
    emitc.call_opaque "__Runtime_wait"(%29) : (!emitc.opaque<"ioevent">) -> ()
    emitc.call_opaque "__Runtime_wait"(%62) : (!emitc.opaque<"ioevent">) -> ()
    emitc.call_opaque "__Runtime_wait"(%96) : (!emitc.opaque<"ioevent">) -> ()
    emitc.call_opaque "__Runtime_wait"(%129) : (!emitc.opaque<"ioevent">) -> ()
    emitc.call_opaque "__Runtime_wait"(%155) : (!emitc.opaque<"ioevent">) -> ()
    emitc.call_opaque "__Runtime_wait"(%193) : (!emitc.opaque<"ioevent">) -> ()
    emitc.call_opaque "__Runtime_wait"(%222) : (!emitc.opaque<"ioevent">) -> ()
    emitc.call_opaque "__Runtime_wait"(%260) : (!emitc.opaque<"ioevent">) -> ()
    emitc.call_opaque "__Runtime_wait"(%290) : (!emitc.opaque<"ioevent">) -> ()
    emitc.call_opaque "__Runtime_wait"(%328) : (!emitc.opaque<"ioevent">) -> ()
    emitc.call_opaque "__Runtime_wait"(%357) : (!emitc.opaque<"ioevent">) -> ()
    emitc.call_opaque "__Runtime_wait"(%395) : (!emitc.opaque<"ioevent">) -> ()
    emitc.verbatim "/* AieRt debug snapshot */"
    emitc.verbatim "{"
    emitc.verbatim "  uint8_t _dbg_io_cols[] = {2, 0, 0, 0, 0, 2, 1, 1, 1, 1, 3, 2, 2, 2, 2, 3, 3, 3, 3, 3, 6, 0, 1, 2, 3, 3, 0, 1, 2, 3, 6, 0, 1, 2, 3, 3, 0, 1, 2, 3, 7, 0, 1, 2, 3, 2, 0, 1, 2, 3, 7, 0, 1, 2, 3, 2, 0, 1, 2, 3};"
    emitc.verbatim "  uint8_t _dbg_io_rows[] = {0, 3, 4, 5, 6, 0, 3, 4, 5, 6, 0, 3, 4, 5, 6, 0, 3, 4, 5, 6, 0, 3, 3, 3, 3, 0, 3, 3, 3, 3, 0, 4, 4, 4, 4, 0, 4, 4, 4, 4, 0, 5, 5, 5, 5, 0, 5, 5, 5, 5, 0, 6, 6, 6, 6, 0, 6, 6, 6, 6};"
    emitc.verbatim "  uint8_t _dbg_io_chs[] = {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0};"
    emitc.verbatim "  uint8_t _dbg_io_bds[] = {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 2, 2, 2, 3, 4, 4, 4, 4, 1, 2, 2, 2, 2, 8, 4, 4, 4, 4, 0, 2, 2, 2, 2, 3, 4, 4, 4, 4, 1, 2, 2, 2, 2, 8, 4, 4, 4, 4};"
    emitc.verbatim "  int _dbg_io_dirs[] = {DMA_MM2S, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_MM2S, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_MM2S, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_MM2S, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_MM2S, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_S2MM, DMA_MM2S, DMA_MM2S, DMA_MM2S, DMA_MM2S};"
    emitc.verbatim "  uint8_t _dbg_t_cols[] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};"
    emitc.verbatim "  uint8_t _dbg_t_rows[] = {3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6};"
    emitc.verbatim "  AieRt_DebugSnapshotFromCoords(g_DevInst,\0A      _dbg_io_cols, _dbg_io_rows, _dbg_io_chs, _dbg_io_bds, _dbg_io_dirs, 60,\0A      _dbg_t_cols, _dbg_t_rows, 16);"
    emitc.verbatim "}"
    emitc.return
  }
  emitc.func @dskernel_receiver(%arg0: index) attributes {specifiers = ["__global__"]} {
    emitc.verbatim "// the real kernel will be emitted separately\0A"
    emitc.return
  }
}
