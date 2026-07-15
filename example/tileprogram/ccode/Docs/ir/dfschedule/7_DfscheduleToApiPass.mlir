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
    %0 = "emitc.constant"() <{value = #emitc.opaque<"g_DevInst">}> : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1 = "emitc.constant"() <{value = #emitc.opaque<"XAIE_MEM_CACHEABLE">}> : () -> i32
    %2 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %3 = "emitc.constant"() <{value = 9 : i32}> : () -> i32
    %4 = "emitc.constant"() <{value = 10 : i32}> : () -> i32
    %5 = "emitc.constant"() <{value = 11 : i32}> : () -> i32
    %6 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %7 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %8 = "emitc.constant"() <{value = 6 : i32}> : () -> i32
    %9 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %10 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %11 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %12 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %13 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %14 = emitc.call_opaque "__runtime_buffer_offset"(%arg1, %13) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %15 = "emitc.constant"() <{value = 2 : i8}> : () -> i8
    %16 = "emitc.constant"() <{value = 0 : i8}> : () -> i8
    %17 = emitc.call_opaque "XAie_TileLoc"(%15, %16) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=16384, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %18 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %19 = "emitc.constant"() <{value = 16384 : i32}> : () -> i32
    %20 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %21 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %22 = emitc.call_opaque "__runtime_buffer_arg"(%14) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %23 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %24 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %25 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %26 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %27 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %28 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %29 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %17, %22, %11, %19, %20, %28, %21, %23, %24, %25, %26, %27) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %30 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %31 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %32 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=0, tile=(2,0), direction=MM2S */"
    %33 = emitc.call_opaque "__Runtime_dma_createio_4"(%17, %29, %30, %31, %32) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    %34 = "emitc.constant"() <{value = 0 : i8}> : () -> i8
    %35 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %36 = emitc.call_opaque "XAie_TileLoc"(%34, %35) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %37 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %38 = "emitc.constant"() <{value = #emitc.opaque<"(void*)36864">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %39 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %40 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %41 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %42 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %43 = emitc.call_opaque "__runtime_buffer_arg"(%38) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %44 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %45 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %46 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %47 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %48 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %49 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %50 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %36, %43, %12, %40, %41, %49, %42, %44, %45, %46, %47, %48) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(0,3) lock=2 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(0, 3), XAie_LockInit(2, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %51 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %52 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %53 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %54 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %55 = emitc.call_opaque "__runtime_buffer_arg"(%37) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %56 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %57 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %58 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %59 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %60 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %61 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %62 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %36, %55, %11, %52, %53, %61, %54, %56, %57, %58, %59, %60) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %63 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %64 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %65 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(0,3), direction=S2MM */"
    %66 = emitc.call_opaque "__Runtime_dma_createio_4"(%36, %62, %63, %64, %65) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 0 for tile (0,3) */"
    %67 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %68 = "emitc.constant"() <{value = 0 : i8}> : () -> i8
    %69 = "emitc.constant"() <{value = 4 : i8}> : () -> i8
    %70 = emitc.call_opaque "XAie_TileLoc"(%68, %69) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %71 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %72 = "emitc.constant"() <{value = #emitc.opaque<"(void*)36864">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %73 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %74 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %75 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %76 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %77 = emitc.call_opaque "__runtime_buffer_arg"(%72) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %78 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %79 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %80 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %81 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %82 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %83 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %84 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %70, %77, %12, %74, %75, %83, %76, %78, %79, %80, %81, %82) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(0,4) lock=2 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(0, 4), XAie_LockInit(2, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %85 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %86 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %87 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %88 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %89 = emitc.call_opaque "__runtime_buffer_arg"(%71) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %90 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %91 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %92 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %93 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %94 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %95 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %96 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %70, %89, %11, %86, %87, %95, %88, %90, %91, %92, %93, %94) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %97 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %98 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %99 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(0,4), direction=S2MM */"
    %100 = emitc.call_opaque "__Runtime_dma_createio_4"(%70, %96, %97, %98, %99) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 0 for tile (0,4) */"
    %101 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %102 = "emitc.constant"() <{value = 0 : i8}> : () -> i8
    %103 = "emitc.constant"() <{value = 5 : i8}> : () -> i8
    %104 = emitc.call_opaque "XAie_TileLoc"(%102, %103) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %105 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %106 = "emitc.constant"() <{value = #emitc.opaque<"(void*)36864">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %107 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %108 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %109 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %110 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %111 = emitc.call_opaque "__runtime_buffer_arg"(%106) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %112 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %113 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %114 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %115 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %116 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %117 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %118 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %104, %111, %12, %108, %109, %117, %110, %112, %113, %114, %115, %116) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(0,5) lock=2 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(0, 5), XAie_LockInit(2, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %119 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %120 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %121 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %122 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %123 = emitc.call_opaque "__runtime_buffer_arg"(%105) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %124 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %125 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %126 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %127 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %128 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %129 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %130 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %104, %123, %11, %120, %121, %129, %122, %124, %125, %126, %127, %128) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %131 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %132 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %133 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(0,5), direction=S2MM */"
    %134 = emitc.call_opaque "__Runtime_dma_createio_4"(%104, %130, %131, %132, %133) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 0 for tile (0,5) */"
    %135 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %136 = "emitc.constant"() <{value = 0 : i8}> : () -> i8
    %137 = "emitc.constant"() <{value = 6 : i8}> : () -> i8
    %138 = emitc.call_opaque "XAie_TileLoc"(%136, %137) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %139 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %140 = "emitc.constant"() <{value = #emitc.opaque<"(void*)36864">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %141 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %142 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %143 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %144 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %145 = emitc.call_opaque "__runtime_buffer_arg"(%140) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %146 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %147 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %148 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %149 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %150 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %151 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %152 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %138, %145, %12, %142, %143, %151, %144, %146, %147, %148, %149, %150) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(0,6) lock=2 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(0, 6), XAie_LockInit(2, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %153 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %154 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %155 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %156 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %157 = emitc.call_opaque "__runtime_buffer_arg"(%139) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %158 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %159 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %160 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %161 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %162 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %163 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %164 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %138, %157, %11, %154, %155, %163, %156, %158, %159, %160, %161, %162) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %165 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %166 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %167 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(0,6), direction=S2MM */"
    %168 = emitc.call_opaque "__Runtime_dma_createio_4"(%138, %164, %165, %166, %167) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 0 for tile (0,6) */"
    %169 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    emitc.verbatim "/* Allocated BD ID 0 for tile (2,0) */"
    %170 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %171 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %172 = emitc.call_opaque "__Runtime_startio"(%33, %170, %171) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %173 = "emitc.constant"() <{value = 16384 : i64}> : () -> i64
    %174 = emitc.call_opaque "__runtime_buffer_offset"(%arg1, %173) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=16384, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %175 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %176 = "emitc.constant"() <{value = 16384 : i32}> : () -> i32
    %177 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %178 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %179 = emitc.call_opaque "__runtime_buffer_arg"(%174) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %180 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %181 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %182 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %183 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %184 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %185 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %186 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %17, %179, %12, %176, %177, %185, %178, %180, %181, %182, %183, %184) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %187 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %188 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %189 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=1, tile=(2,0), direction=MM2S */"
    %190 = emitc.call_opaque "__Runtime_dma_createio_4"(%17, %186, %187, %188, %189) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    %191 = "emitc.constant"() <{value = 1 : i8}> : () -> i8
    %192 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %193 = emitc.call_opaque "XAie_TileLoc"(%191, %192) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %194 = "emitc.constant"() <{value = 16384 : i64}> : () -> i64
    %195 = emitc.call_opaque "__runtime_buffer_offset"(%174, %194) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %196 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %197 = "emitc.constant"() <{value = #emitc.opaque<"(void*)36864">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %198 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %199 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %200 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %201 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %202 = emitc.call_opaque "__runtime_buffer_arg"(%197) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %203 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %204 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %205 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %206 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %207 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %208 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %209 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %193, %202, %12, %199, %200, %208, %201, %203, %204, %205, %206, %207) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(1,3) lock=2 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(1, 3), XAie_LockInit(2, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %210 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %211 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %212 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %213 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %214 = emitc.call_opaque "__runtime_buffer_arg"(%196) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %215 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %216 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %217 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %218 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %219 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %220 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %221 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %193, %214, %11, %211, %212, %220, %213, %215, %216, %217, %218, %219) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %222 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %223 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %224 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(1,3), direction=S2MM */"
    %225 = emitc.call_opaque "__Runtime_dma_createio_4"(%193, %221, %222, %223, %224) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 0 for tile (1,3) */"
    %226 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %227 = "emitc.constant"() <{value = 1 : i8}> : () -> i8
    %228 = "emitc.constant"() <{value = 4 : i8}> : () -> i8
    %229 = emitc.call_opaque "XAie_TileLoc"(%227, %228) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %230 = "emitc.constant"() <{value = 16384 : i64}> : () -> i64
    %231 = emitc.call_opaque "__runtime_buffer_offset"(%174, %230) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %232 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %233 = "emitc.constant"() <{value = #emitc.opaque<"(void*)36864">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %234 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %235 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %236 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %237 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %238 = emitc.call_opaque "__runtime_buffer_arg"(%233) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %239 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %240 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %241 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %242 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %243 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %244 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %245 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %229, %238, %12, %235, %236, %244, %237, %239, %240, %241, %242, %243) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(1,4) lock=2 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(1, 4), XAie_LockInit(2, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %246 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %247 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %248 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %249 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %250 = emitc.call_opaque "__runtime_buffer_arg"(%232) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %251 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %252 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %253 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %254 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %255 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %256 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %257 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %229, %250, %11, %247, %248, %256, %249, %251, %252, %253, %254, %255) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %258 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %259 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %260 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(1,4), direction=S2MM */"
    %261 = emitc.call_opaque "__Runtime_dma_createio_4"(%229, %257, %258, %259, %260) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 0 for tile (1,4) */"
    %262 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %263 = "emitc.constant"() <{value = 1 : i8}> : () -> i8
    %264 = "emitc.constant"() <{value = 5 : i8}> : () -> i8
    %265 = emitc.call_opaque "XAie_TileLoc"(%263, %264) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %266 = "emitc.constant"() <{value = 16384 : i64}> : () -> i64
    %267 = emitc.call_opaque "__runtime_buffer_offset"(%174, %266) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %268 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %269 = "emitc.constant"() <{value = #emitc.opaque<"(void*)36864">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %270 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %271 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %272 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %273 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %274 = emitc.call_opaque "__runtime_buffer_arg"(%269) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %275 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %276 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %277 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %278 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %279 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %280 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %281 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %265, %274, %12, %271, %272, %280, %273, %275, %276, %277, %278, %279) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(1,5) lock=2 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(1, 5), XAie_LockInit(2, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %282 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %283 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %284 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %285 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %286 = emitc.call_opaque "__runtime_buffer_arg"(%268) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %287 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %288 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %289 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %290 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %291 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %292 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %293 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %265, %286, %11, %283, %284, %292, %285, %287, %288, %289, %290, %291) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %294 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %295 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %296 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(1,5), direction=S2MM */"
    %297 = emitc.call_opaque "__Runtime_dma_createio_4"(%265, %293, %294, %295, %296) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 0 for tile (1,5) */"
    %298 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %299 = "emitc.constant"() <{value = 1 : i8}> : () -> i8
    %300 = "emitc.constant"() <{value = 6 : i8}> : () -> i8
    %301 = emitc.call_opaque "XAie_TileLoc"(%299, %300) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %302 = "emitc.constant"() <{value = 16384 : i64}> : () -> i64
    %303 = emitc.call_opaque "__runtime_buffer_offset"(%174, %302) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %304 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %305 = "emitc.constant"() <{value = #emitc.opaque<"(void*)36864">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %306 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %307 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %308 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %309 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %310 = emitc.call_opaque "__runtime_buffer_arg"(%305) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %311 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %312 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %313 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %314 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %315 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %316 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %317 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %301, %310, %12, %307, %308, %316, %309, %311, %312, %313, %314, %315) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(1,6) lock=2 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(1, 6), XAie_LockInit(2, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %318 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %319 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %320 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %321 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %322 = emitc.call_opaque "__runtime_buffer_arg"(%304) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %323 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %324 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %325 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %326 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %327 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %328 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %329 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %301, %322, %11, %319, %320, %328, %321, %323, %324, %325, %326, %327) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %330 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %331 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %332 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(1,6), direction=S2MM */"
    %333 = emitc.call_opaque "__Runtime_dma_createio_4"(%301, %329, %330, %331, %332) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 0 for tile (1,6) */"
    %334 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    emitc.verbatim "/* Allocated BD ID 1 for tile (2,0) */"
    %335 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %336 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %337 = emitc.call_opaque "__Runtime_startio"(%190, %335, %336) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %338 = "emitc.constant"() <{value = 32768 : i64}> : () -> i64
    %339 = emitc.call_opaque "__runtime_buffer_offset"(%arg1, %338) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %340 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %341 = "emitc.constant"() <{value = 0 : i8}> : () -> i8
    %342 = emitc.call_opaque "XAie_TileLoc"(%340, %341) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=16384, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %343 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %344 = "emitc.constant"() <{value = 16384 : i32}> : () -> i32
    %345 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %346 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %347 = emitc.call_opaque "__runtime_buffer_arg"(%339) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %348 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %349 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %350 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %351 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %352 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %353 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %354 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %342, %347, %11, %344, %345, %353, %346, %348, %349, %350, %351, %352) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %355 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %356 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %357 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=0, tile=(3,0), direction=MM2S */"
    %358 = emitc.call_opaque "__Runtime_dma_createio_4"(%342, %354, %355, %356, %357) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    %359 = "emitc.constant"() <{value = 2 : i8}> : () -> i8
    %360 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %361 = emitc.call_opaque "XAie_TileLoc"(%359, %360) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %362 = "emitc.constant"() <{value = 32768 : i64}> : () -> i64
    %363 = emitc.call_opaque "__runtime_buffer_offset"(%339, %362) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %364 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %365 = "emitc.constant"() <{value = #emitc.opaque<"(void*)36864">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %366 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %367 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %368 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %369 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %370 = emitc.call_opaque "__runtime_buffer_arg"(%365) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %371 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %372 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %373 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %374 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %375 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %376 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %377 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %361, %370, %12, %367, %368, %376, %369, %371, %372, %373, %374, %375) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(2,3) lock=2 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(2, 3), XAie_LockInit(2, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %378 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %379 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %380 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %381 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %382 = emitc.call_opaque "__runtime_buffer_arg"(%364) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %383 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %384 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %385 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %386 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %387 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %388 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %389 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %361, %382, %11, %379, %380, %388, %381, %383, %384, %385, %386, %387) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %390 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %391 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %392 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(2,3), direction=S2MM */"
    %393 = emitc.call_opaque "__Runtime_dma_createio_4"(%361, %389, %390, %391, %392) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 0 for tile (2,3) */"
    %394 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %395 = "emitc.constant"() <{value = 2 : i8}> : () -> i8
    %396 = "emitc.constant"() <{value = 4 : i8}> : () -> i8
    %397 = emitc.call_opaque "XAie_TileLoc"(%395, %396) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %398 = "emitc.constant"() <{value = 32768 : i64}> : () -> i64
    %399 = emitc.call_opaque "__runtime_buffer_offset"(%339, %398) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %400 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %401 = "emitc.constant"() <{value = #emitc.opaque<"(void*)36864">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %402 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %403 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %404 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %405 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %406 = emitc.call_opaque "__runtime_buffer_arg"(%401) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %407 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %408 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %409 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %410 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %411 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %412 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %413 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %397, %406, %12, %403, %404, %412, %405, %407, %408, %409, %410, %411) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(2,4) lock=2 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(2, 4), XAie_LockInit(2, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %414 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %415 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %416 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %417 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %418 = emitc.call_opaque "__runtime_buffer_arg"(%400) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %419 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %420 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %421 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %422 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %423 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %424 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %425 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %397, %418, %11, %415, %416, %424, %417, %419, %420, %421, %422, %423) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %426 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %427 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %428 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(2,4), direction=S2MM */"
    %429 = emitc.call_opaque "__Runtime_dma_createio_4"(%397, %425, %426, %427, %428) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 0 for tile (2,4) */"
    %430 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %431 = "emitc.constant"() <{value = 2 : i8}> : () -> i8
    %432 = "emitc.constant"() <{value = 5 : i8}> : () -> i8
    %433 = emitc.call_opaque "XAie_TileLoc"(%431, %432) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %434 = "emitc.constant"() <{value = 32768 : i64}> : () -> i64
    %435 = emitc.call_opaque "__runtime_buffer_offset"(%339, %434) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %436 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %437 = "emitc.constant"() <{value = #emitc.opaque<"(void*)36864">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %438 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %439 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %440 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %441 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %442 = emitc.call_opaque "__runtime_buffer_arg"(%437) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %443 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %444 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %445 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %446 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %447 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %448 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %449 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %433, %442, %12, %439, %440, %448, %441, %443, %444, %445, %446, %447) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(2,5) lock=2 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(2, 5), XAie_LockInit(2, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %450 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %451 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %452 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %453 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %454 = emitc.call_opaque "__runtime_buffer_arg"(%436) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %455 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %456 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %457 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %458 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %459 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %460 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %461 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %433, %454, %11, %451, %452, %460, %453, %455, %456, %457, %458, %459) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %462 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %463 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %464 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(2,5), direction=S2MM */"
    %465 = emitc.call_opaque "__Runtime_dma_createio_4"(%433, %461, %462, %463, %464) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 0 for tile (2,5) */"
    %466 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %467 = "emitc.constant"() <{value = 2 : i8}> : () -> i8
    %468 = "emitc.constant"() <{value = 6 : i8}> : () -> i8
    %469 = emitc.call_opaque "XAie_TileLoc"(%467, %468) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %470 = "emitc.constant"() <{value = 32768 : i64}> : () -> i64
    %471 = emitc.call_opaque "__runtime_buffer_offset"(%339, %470) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %472 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %473 = "emitc.constant"() <{value = #emitc.opaque<"(void*)36864">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %474 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %475 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %476 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %477 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %478 = emitc.call_opaque "__runtime_buffer_arg"(%473) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %479 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %480 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %481 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %482 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %483 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %484 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %485 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %469, %478, %12, %475, %476, %484, %477, %479, %480, %481, %482, %483) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(2,6) lock=2 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(2, 6), XAie_LockInit(2, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %486 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %487 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %488 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %489 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %490 = emitc.call_opaque "__runtime_buffer_arg"(%472) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %491 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %492 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %493 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %494 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %495 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %496 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %497 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %469, %490, %11, %487, %488, %496, %489, %491, %492, %493, %494, %495) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %498 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %499 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %500 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(2,6), direction=S2MM */"
    %501 = emitc.call_opaque "__Runtime_dma_createio_4"(%469, %497, %498, %499, %500) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 0 for tile (2,6) */"
    %502 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    emitc.verbatim "/* Allocated BD ID 0 for tile (3,0) */"
    %503 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %504 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %505 = emitc.call_opaque "__Runtime_startio"(%358, %503, %504) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %506 = "emitc.constant"() <{value = 49152 : i64}> : () -> i64
    %507 = emitc.call_opaque "__runtime_buffer_offset"(%arg1, %506) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=16384, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %508 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %509 = "emitc.constant"() <{value = 16384 : i32}> : () -> i32
    %510 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %511 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %512 = emitc.call_opaque "__runtime_buffer_arg"(%507) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %513 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %514 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %515 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %516 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %517 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %518 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %519 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %342, %512, %12, %509, %510, %518, %511, %513, %514, %515, %516, %517) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %520 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %521 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %522 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=1, tile=(3,0), direction=MM2S */"
    %523 = emitc.call_opaque "__Runtime_dma_createio_4"(%342, %519, %520, %521, %522) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    %524 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %525 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %526 = emitc.call_opaque "XAie_TileLoc"(%524, %525) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %527 = "emitc.constant"() <{value = 49152 : i64}> : () -> i64
    %528 = emitc.call_opaque "__runtime_buffer_offset"(%507, %527) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %529 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %530 = "emitc.constant"() <{value = #emitc.opaque<"(void*)36864">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %531 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %532 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %533 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %534 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %535 = emitc.call_opaque "__runtime_buffer_arg"(%530) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %536 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %537 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %538 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %539 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %540 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %541 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %542 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %526, %535, %12, %532, %533, %541, %534, %536, %537, %538, %539, %540) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(3,3) lock=2 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(3, 3), XAie_LockInit(2, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %543 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %544 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %545 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %546 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %547 = emitc.call_opaque "__runtime_buffer_arg"(%529) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %548 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %549 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %550 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %551 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %552 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %553 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %554 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %526, %547, %11, %544, %545, %553, %546, %548, %549, %550, %551, %552) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %555 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %556 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %557 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(3,3), direction=S2MM */"
    %558 = emitc.call_opaque "__Runtime_dma_createio_4"(%526, %554, %555, %556, %557) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 0 for tile (3,3) */"
    %559 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %560 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %561 = "emitc.constant"() <{value = 4 : i8}> : () -> i8
    %562 = emitc.call_opaque "XAie_TileLoc"(%560, %561) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %563 = "emitc.constant"() <{value = 49152 : i64}> : () -> i64
    %564 = emitc.call_opaque "__runtime_buffer_offset"(%507, %563) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %565 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %566 = "emitc.constant"() <{value = #emitc.opaque<"(void*)36864">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %567 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %568 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %569 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %570 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %571 = emitc.call_opaque "__runtime_buffer_arg"(%566) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %572 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %573 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %574 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %575 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %576 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %577 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %578 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %562, %571, %12, %568, %569, %577, %570, %572, %573, %574, %575, %576) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(3,4) lock=2 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(3, 4), XAie_LockInit(2, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %579 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %580 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %581 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %582 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %583 = emitc.call_opaque "__runtime_buffer_arg"(%565) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %584 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %585 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %586 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %587 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %588 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %589 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %590 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %562, %583, %11, %580, %581, %589, %582, %584, %585, %586, %587, %588) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %591 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %592 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %593 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(3,4), direction=S2MM */"
    %594 = emitc.call_opaque "__Runtime_dma_createio_4"(%562, %590, %591, %592, %593) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 0 for tile (3,4) */"
    %595 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %596 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %597 = "emitc.constant"() <{value = 5 : i8}> : () -> i8
    %598 = emitc.call_opaque "XAie_TileLoc"(%596, %597) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %599 = "emitc.constant"() <{value = 49152 : i64}> : () -> i64
    %600 = emitc.call_opaque "__runtime_buffer_offset"(%507, %599) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %601 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %602 = "emitc.constant"() <{value = #emitc.opaque<"(void*)36864">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %603 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %604 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %605 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %606 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %607 = emitc.call_opaque "__runtime_buffer_arg"(%602) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %608 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %609 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %610 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %611 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %612 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %613 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %614 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %598, %607, %12, %604, %605, %613, %606, %608, %609, %610, %611, %612) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(3,5) lock=2 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(3, 5), XAie_LockInit(2, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %615 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %616 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %617 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %618 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %619 = emitc.call_opaque "__runtime_buffer_arg"(%601) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %620 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %621 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %622 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %623 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %624 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %625 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %626 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %598, %619, %11, %616, %617, %625, %618, %620, %621, %622, %623, %624) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %627 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %628 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %629 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(3,5), direction=S2MM */"
    %630 = emitc.call_opaque "__Runtime_dma_createio_4"(%598, %626, %627, %628, %629) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 0 for tile (3,5) */"
    %631 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %632 = "emitc.constant"() <{value = 3 : i8}> : () -> i8
    %633 = "emitc.constant"() <{value = 6 : i8}> : () -> i8
    %634 = emitc.call_opaque "XAie_TileLoc"(%632, %633) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    %635 = "emitc.constant"() <{value = 49152 : i64}> : () -> i64
    %636 = emitc.call_opaque "__runtime_buffer_offset"(%507, %635) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %637 = "emitc.constant"() <{value = #emitc.opaque<"(void*)32768">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %638 = "emitc.constant"() <{value = #emitc.opaque<"(void*)36864">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=0, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %639 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %640 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %641 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %642 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %643 = emitc.call_opaque "__runtime_buffer_arg"(%638) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %644 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %645 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %646 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %647 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %648 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %649 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %650 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %634, %643, %12, %640, %641, %649, %642, %644, %645, %646, %647, %648) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(3,6) lock=2 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(3, 6), XAie_LockInit(2, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=1, acquire_lock_id=2, acquire_lock_val=-1, release_lock_id=3, release_lock_val=1, ooo_bd_id=-1 */"
    %651 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %652 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %653 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %654 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %655 = emitc.call_opaque "__runtime_buffer_arg"(%637) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %656 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %657 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %658 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %659 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %660 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %661 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %662 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %634, %655, %11, %652, %653, %661, %654, %656, %657, %658, %659, %660) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %663 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %664 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %665 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=0, tile=(3,6), direction=S2MM */"
    %666 = emitc.call_opaque "__Runtime_dma_createio_4"(%634, %662, %663, %664, %665) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 0 for tile (3,6) */"
    %667 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    emitc.verbatim "/* Allocated BD ID 1 for tile (3,0) */"
    %668 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %669 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %670 = emitc.call_opaque "__Runtime_startio"(%523, %668, %669) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %671 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %672 = emitc.call_opaque "__runtime_buffer_offset"(%arg0, %671) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %673 = "emitc.constant"() <{value = 6 : i8}> : () -> i8
    %674 = "emitc.constant"() <{value = 0 : i8}> : () -> i8
    %675 = emitc.call_opaque "XAie_TileLoc"(%673, %674) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=16384, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %676 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %677 = "emitc.constant"() <{value = 16384 : i32}> : () -> i32
    %678 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %679 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %680 = emitc.call_opaque "__runtime_buffer_arg"(%672) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %681 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %682 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %683 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %684 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %685 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %686 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %687 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %675, %680, %11, %677, %678, %686, %679, %681, %682, %683, %684, %685) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %688 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %689 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %690 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=0, tile=(6,0), direction=MM2S */"
    %691 = emitc.call_opaque "__Runtime_dma_createio_4"(%675, %687, %688, %689, %690) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    %692 = "emitc.constant"() <{value = #emitc.opaque<"(void*)40960">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %693 = "emitc.constant"() <{value = #emitc.opaque<"(void*)45056">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=3, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %694 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %695 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %696 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %697 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %698 = emitc.call_opaque "__runtime_buffer_arg"(%693) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %699 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %700 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %701 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %702 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %703 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %704 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %705 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %36, %698, %9, %695, %696, %704, %697, %699, %700, %701, %702, %703) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(0,3) lock=0 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(0, 3), XAie_LockInit(0, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=2, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %706 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %707 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %708 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %709 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %710 = emitc.call_opaque "__runtime_buffer_arg"(%692) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %711 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %712 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %713 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %714 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %715 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %716 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %717 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %36, %710, %10, %707, %708, %716, %709, %711, %712, %713, %714, %715) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %718 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %719 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %720 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(0,3), direction=S2MM */"
    %721 = emitc.call_opaque "__Runtime_dma_createio_4"(%36, %717, %718, %719, %720) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 1 for tile (0,3) */"
    %722 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %723 = "emitc.constant"() <{value = #emitc.opaque<"(void*)40960">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %724 = "emitc.constant"() <{value = #emitc.opaque<"(void*)45056">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=3, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %725 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %726 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %727 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %728 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %729 = emitc.call_opaque "__runtime_buffer_arg"(%724) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %730 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %731 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %732 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %733 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %734 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %735 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %736 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %193, %729, %9, %726, %727, %735, %728, %730, %731, %732, %733, %734) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(1,3) lock=0 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(1, 3), XAie_LockInit(0, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=2, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %737 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %738 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %739 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %740 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %741 = emitc.call_opaque "__runtime_buffer_arg"(%723) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %742 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %743 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %744 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %745 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %746 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %747 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %748 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %193, %741, %10, %738, %739, %747, %740, %742, %743, %744, %745, %746) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %749 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %750 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %751 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(1,3), direction=S2MM */"
    %752 = emitc.call_opaque "__Runtime_dma_createio_4"(%193, %748, %749, %750, %751) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 1 for tile (1,3) */"
    %753 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %754 = "emitc.constant"() <{value = #emitc.opaque<"(void*)40960">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %755 = "emitc.constant"() <{value = #emitc.opaque<"(void*)45056">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=3, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %756 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %757 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %758 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %759 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %760 = emitc.call_opaque "__runtime_buffer_arg"(%755) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %761 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %762 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %763 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %764 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %765 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %766 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %767 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %361, %760, %9, %757, %758, %766, %759, %761, %762, %763, %764, %765) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(2,3) lock=0 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(2, 3), XAie_LockInit(0, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=2, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %768 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %769 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %770 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %771 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %772 = emitc.call_opaque "__runtime_buffer_arg"(%754) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %773 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %774 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %775 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %776 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %777 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %778 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %779 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %361, %772, %10, %769, %770, %778, %771, %773, %774, %775, %776, %777) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %780 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %781 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %782 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(2,3), direction=S2MM */"
    %783 = emitc.call_opaque "__Runtime_dma_createio_4"(%361, %779, %780, %781, %782) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 1 for tile (2,3) */"
    %784 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %785 = "emitc.constant"() <{value = #emitc.opaque<"(void*)40960">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %786 = "emitc.constant"() <{value = #emitc.opaque<"(void*)45056">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=3, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %787 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %788 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %789 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %790 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %791 = emitc.call_opaque "__runtime_buffer_arg"(%786) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %792 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %793 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %794 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %795 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %796 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %797 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %798 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %526, %791, %9, %788, %789, %797, %790, %792, %793, %794, %795, %796) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(3,3) lock=0 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(3, 3), XAie_LockInit(0, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=2, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %799 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %800 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %801 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %802 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %803 = emitc.call_opaque "__runtime_buffer_arg"(%785) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %804 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %805 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %806 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %807 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %808 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %809 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %810 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %526, %803, %10, %800, %801, %809, %802, %804, %805, %806, %807, %808) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %811 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %812 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %813 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(3,3), direction=S2MM */"
    %814 = emitc.call_opaque "__Runtime_dma_createio_4"(%526, %810, %811, %812, %813) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 1 for tile (3,3) */"
    %815 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    emitc.verbatim "/* Allocated BD ID 0 for tile (6,0) */"
    %816 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %817 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %818 = emitc.call_opaque "__Runtime_startio"(%691, %816, %817) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %819 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %820 = emitc.call_opaque "__runtime_buffer_offset"(%arg2, %819) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=6, offset=192, len=2048, enable_packet=false, packet_id=4, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %821 = "emitc.constant"() <{value = 192 : i32}> : () -> i32
    %822 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %823 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %824 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %825 = emitc.call_opaque "__runtime_buffer_arg"(%820) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %826 = "emitc.constant"() <{value = 192 : i64}> : () -> i64
    %827 = emitc.call_opaque "__runtime_buffer_offset"(%825, %826) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %828 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %829 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %830 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %831 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %832 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %833 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %834 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %835 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %836 = "emitc.constant"() <{value = 256 : i32}> : () -> i32
    %837 = "emitc.constant"() <{value = 32 : i32}> : () -> i32
    %838 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %839 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %840 = "emitc.constant"() <{value = 8192 : i32}> : () -> i32
    %841 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %842 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %843 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%0, %342, %827, %8, %822, %823, %842, %824, %828, %829, %830, %831, %832, %833, %834, %835, %836, %837, %838, %839, %840, %841) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=5, offset=128, len=2048, enable_packet=false, packet_id=3, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %844 = "emitc.constant"() <{value = 128 : i32}> : () -> i32
    %845 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %846 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %847 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %848 = emitc.call_opaque "__runtime_buffer_arg"(%820) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %849 = "emitc.constant"() <{value = 128 : i64}> : () -> i64
    %850 = emitc.call_opaque "__runtime_buffer_offset"(%848, %849) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %851 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %852 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %853 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %854 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %855 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %856 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %857 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %858 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %859 = "emitc.constant"() <{value = 256 : i32}> : () -> i32
    %860 = "emitc.constant"() <{value = 32 : i32}> : () -> i32
    %861 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %862 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %863 = "emitc.constant"() <{value = 8192 : i32}> : () -> i32
    %864 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %865 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %866 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%0, %342, %850, %7, %845, %846, %865, %847, %851, %852, %853, %854, %855, %856, %857, %858, %859, %860, %861, %862, %863, %864) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, offset=64, len=2048, enable_packet=false, packet_id=2, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %867 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
    %868 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %869 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %870 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %871 = emitc.call_opaque "__runtime_buffer_arg"(%820) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %872 = "emitc.constant"() <{value = 64 : i64}> : () -> i64
    %873 = emitc.call_opaque "__runtime_buffer_offset"(%871, %872) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %874 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %875 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %876 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %877 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %878 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %879 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %880 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %881 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %882 = "emitc.constant"() <{value = 256 : i32}> : () -> i32
    %883 = "emitc.constant"() <{value = 32 : i32}> : () -> i32
    %884 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %885 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %886 = "emitc.constant"() <{value = 8192 : i32}> : () -> i32
    %887 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %888 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %889 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%0, %342, %873, %6, %868, %869, %888, %870, %874, %875, %876, %877, %878, %879, %880, %881, %882, %883, %884, %885, %886, %887) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=3, offset=0, len=2048, enable_packet=false, packet_id=1, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %890 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %891 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %892 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %893 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %894 = emitc.call_opaque "__runtime_buffer_arg"(%820) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %895 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %896 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %897 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %898 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %899 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %900 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %901 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %902 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %903 = "emitc.constant"() <{value = 256 : i32}> : () -> i32
    %904 = "emitc.constant"() <{value = 32 : i32}> : () -> i32
    %905 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %906 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %907 = "emitc.constant"() <{value = 8192 : i32}> : () -> i32
    %908 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %909 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %910 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%0, %342, %894, %9, %891, %892, %909, %893, %895, %896, %897, %898, %899, %900, %901, %902, %903, %904, %905, %906, %907, %908) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %911 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %912 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %913 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=3, tile=(3,0), direction=S2MM */"
    emitc.verbatim "/* Enable out-of-order BD on tile(3,0) ch=0 dir=S2MM */"
    emitc.call_opaque "__Runtime_dma_channel_enable_ooo"(%0, %342, %911, %913) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, !emitc.opaque<"XAie_DmaDirection">) -> ()
    %914 = emitc.call_opaque "__Runtime_dma_createio_4"(%342, %910, %911, %912, %913) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    %915 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %916 = emitc.call_opaque "__runtime_buffer_offset"(%820, %915) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %917 = "emitc.constant"() <{value = #emitc.opaque<"(void*)49152">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %918 = "emitc.constant"() <{value = #emitc.opaque<"(void*)53248">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=5, offset=0, len=2048, enable_packet=true, packet_id=1, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=3 */"
    %919 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %920 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %921 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %922 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %923 = emitc.call_opaque "__runtime_buffer_arg"(%918) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %924 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %925 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %926 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %927 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %928 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %929 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %930 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %36, %923, %7, %920, %921, %929, %922, %924, %925, %926, %927, %928) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(0,3) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(0, 3), XAie_LockInit(4, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=4, offset=0, len=2048, enable_packet=true, packet_id=1, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=3 */"
    %931 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %932 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %933 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %934 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %935 = emitc.call_opaque "__runtime_buffer_arg"(%917) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %936 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %937 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %938 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %939 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %940 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %941 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %942 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %36, %935, %6, %932, %933, %941, %934, %936, %937, %938, %939, %940) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %943 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %944 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %945 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(0,3), direction=MM2S */"
    %946 = emitc.call_opaque "__Runtime_dma_createio_4"(%36, %942, %943, %944, %945) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 2 for tile (0,3) */"
    %947 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %948 = "emitc.constant"() <{value = 4096 : i64}> : () -> i64
    %949 = emitc.call_opaque "__runtime_buffer_offset"(%820, %948) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %950 = "emitc.constant"() <{value = #emitc.opaque<"(void*)49152">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %951 = "emitc.constant"() <{value = #emitc.opaque<"(void*)53248">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=5, offset=0, len=2048, enable_packet=true, packet_id=2, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=4 */"
    %952 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %953 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %954 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %955 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %956 = emitc.call_opaque "__runtime_buffer_arg"(%951) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %957 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %958 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %959 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %960 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %961 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %962 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %963 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %193, %956, %7, %953, %954, %962, %955, %957, %958, %959, %960, %961) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(1,3) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(1, 3), XAie_LockInit(4, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=4, offset=0, len=2048, enable_packet=true, packet_id=2, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=4 */"
    %964 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %965 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %966 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %967 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %968 = emitc.call_opaque "__runtime_buffer_arg"(%950) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %969 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %970 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %971 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %972 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %973 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %974 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %975 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %193, %968, %6, %965, %966, %974, %967, %969, %970, %971, %972, %973) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %976 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %977 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %978 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(1,3), direction=MM2S */"
    %979 = emitc.call_opaque "__Runtime_dma_createio_4"(%193, %975, %976, %977, %978) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 2 for tile (1,3) */"
    %980 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %981 = "emitc.constant"() <{value = 8192 : i64}> : () -> i64
    %982 = emitc.call_opaque "__runtime_buffer_offset"(%820, %981) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %983 = "emitc.constant"() <{value = #emitc.opaque<"(void*)49152">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %984 = "emitc.constant"() <{value = #emitc.opaque<"(void*)53248">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=5, offset=0, len=2048, enable_packet=true, packet_id=3, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=5 */"
    %985 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %986 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %987 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %988 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %989 = emitc.call_opaque "__runtime_buffer_arg"(%984) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %990 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %991 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %992 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %993 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %994 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %995 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %996 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %361, %989, %7, %986, %987, %995, %988, %990, %991, %992, %993, %994) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(2,3) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(2, 3), XAie_LockInit(4, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=4, offset=0, len=2048, enable_packet=true, packet_id=3, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=5 */"
    %997 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %998 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %999 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1000 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1001 = emitc.call_opaque "__runtime_buffer_arg"(%983) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1002 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1003 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1004 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1005 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1006 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1007 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1008 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %361, %1001, %6, %998, %999, %1007, %1000, %1002, %1003, %1004, %1005, %1006) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1009 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1010 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1011 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(2,3), direction=MM2S */"
    %1012 = emitc.call_opaque "__Runtime_dma_createio_4"(%361, %1008, %1009, %1010, %1011) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 2 for tile (2,3) */"
    %1013 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1014 = "emitc.constant"() <{value = 12288 : i64}> : () -> i64
    %1015 = emitc.call_opaque "__runtime_buffer_offset"(%820, %1014) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1016 = "emitc.constant"() <{value = #emitc.opaque<"(void*)49152">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %1017 = "emitc.constant"() <{value = #emitc.opaque<"(void*)53248">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=5, offset=0, len=2048, enable_packet=true, packet_id=4, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=6 */"
    %1018 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1019 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %1020 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1021 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1022 = emitc.call_opaque "__runtime_buffer_arg"(%1017) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1023 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1024 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1025 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1026 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1027 = "emitc.constant"() <{value = 6 : i32}> : () -> i32
    %1028 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1029 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %526, %1022, %7, %1019, %1020, %1028, %1021, %1023, %1024, %1025, %1026, %1027) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(3,3) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(3, 3), XAie_LockInit(4, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=4, offset=0, len=2048, enable_packet=true, packet_id=4, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=6 */"
    %1030 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1031 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %1032 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1033 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1034 = emitc.call_opaque "__runtime_buffer_arg"(%1016) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1035 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1036 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1037 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1038 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1039 = "emitc.constant"() <{value = 6 : i32}> : () -> i32
    %1040 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1041 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %526, %1034, %6, %1031, %1032, %1040, %1033, %1035, %1036, %1037, %1038, %1039) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1042 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1043 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1044 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(3,3), direction=MM2S */"
    %1045 = emitc.call_opaque "__Runtime_dma_createio_4"(%526, %1041, %1042, %1043, %1044) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 2 for tile (3,3) */"
    %1046 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    emitc.verbatim "/* Allocated BD ID 2 for tile (3,0) */"
    %1047 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1048 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %1049 = emitc.call_opaque "__Runtime_startio"(%914, %1047, %1048) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1050 = "emitc.constant"() <{value = 16384 : i64}> : () -> i64
    %1051 = emitc.call_opaque "__runtime_buffer_offset"(%arg0, %1050) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=16384, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %1052 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1053 = "emitc.constant"() <{value = 16384 : i32}> : () -> i32
    %1054 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1055 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1056 = emitc.call_opaque "__runtime_buffer_arg"(%1051) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1057 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1058 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1059 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1060 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1061 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1062 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1063 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %675, %1056, %12, %1053, %1054, %1062, %1055, %1057, %1058, %1059, %1060, %1061) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1064 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1065 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1066 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=1, tile=(6,0), direction=MM2S */"
    %1067 = emitc.call_opaque "__Runtime_dma_createio_4"(%675, %1063, %1064, %1065, %1066) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    %1068 = "emitc.constant"() <{value = 16384 : i64}> : () -> i64
    %1069 = emitc.call_opaque "__runtime_buffer_offset"(%1051, %1068) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1070 = "emitc.constant"() <{value = #emitc.opaque<"(void*)40960">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %1071 = "emitc.constant"() <{value = #emitc.opaque<"(void*)45056">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=3, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1072 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1073 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %1074 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1075 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1076 = emitc.call_opaque "__runtime_buffer_arg"(%1071) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1077 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1078 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1079 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1080 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1081 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1082 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1083 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %70, %1076, %9, %1073, %1074, %1082, %1075, %1077, %1078, %1079, %1080, %1081) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(0,4) lock=0 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(0, 4), XAie_LockInit(0, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=2, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1084 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1085 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %1086 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1087 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1088 = emitc.call_opaque "__runtime_buffer_arg"(%1070) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1089 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1090 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1091 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1092 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1093 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1094 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1095 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %70, %1088, %10, %1085, %1086, %1094, %1087, %1089, %1090, %1091, %1092, %1093) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1096 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1097 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1098 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(0,4), direction=S2MM */"
    %1099 = emitc.call_opaque "__Runtime_dma_createio_4"(%70, %1095, %1096, %1097, %1098) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 1 for tile (0,4) */"
    %1100 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1101 = "emitc.constant"() <{value = 16384 : i64}> : () -> i64
    %1102 = emitc.call_opaque "__runtime_buffer_offset"(%1051, %1101) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1103 = "emitc.constant"() <{value = #emitc.opaque<"(void*)40960">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %1104 = "emitc.constant"() <{value = #emitc.opaque<"(void*)45056">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=3, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1105 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1106 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %1107 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1108 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1109 = emitc.call_opaque "__runtime_buffer_arg"(%1104) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1110 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1111 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1112 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1113 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1114 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1115 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1116 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %229, %1109, %9, %1106, %1107, %1115, %1108, %1110, %1111, %1112, %1113, %1114) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(1,4) lock=0 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(1, 4), XAie_LockInit(0, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=2, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1117 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1118 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %1119 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1120 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1121 = emitc.call_opaque "__runtime_buffer_arg"(%1103) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1122 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1123 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1124 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1125 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1126 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1127 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1128 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %229, %1121, %10, %1118, %1119, %1127, %1120, %1122, %1123, %1124, %1125, %1126) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1129 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1130 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1131 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(1,4), direction=S2MM */"
    %1132 = emitc.call_opaque "__Runtime_dma_createio_4"(%229, %1128, %1129, %1130, %1131) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 1 for tile (1,4) */"
    %1133 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1134 = "emitc.constant"() <{value = 16384 : i64}> : () -> i64
    %1135 = emitc.call_opaque "__runtime_buffer_offset"(%1051, %1134) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1136 = "emitc.constant"() <{value = #emitc.opaque<"(void*)40960">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %1137 = "emitc.constant"() <{value = #emitc.opaque<"(void*)45056">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=3, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1138 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1139 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %1140 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1141 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1142 = emitc.call_opaque "__runtime_buffer_arg"(%1137) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1143 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1144 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1145 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1146 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1147 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1148 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1149 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %397, %1142, %9, %1139, %1140, %1148, %1141, %1143, %1144, %1145, %1146, %1147) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(2,4) lock=0 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(2, 4), XAie_LockInit(0, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=2, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1150 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1151 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %1152 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1153 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1154 = emitc.call_opaque "__runtime_buffer_arg"(%1136) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1155 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1156 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1157 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1158 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1159 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1160 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1161 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %397, %1154, %10, %1151, %1152, %1160, %1153, %1155, %1156, %1157, %1158, %1159) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1162 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1163 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1164 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(2,4), direction=S2MM */"
    %1165 = emitc.call_opaque "__Runtime_dma_createio_4"(%397, %1161, %1162, %1163, %1164) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 1 for tile (2,4) */"
    %1166 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1167 = "emitc.constant"() <{value = 16384 : i64}> : () -> i64
    %1168 = emitc.call_opaque "__runtime_buffer_offset"(%1051, %1167) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1169 = "emitc.constant"() <{value = #emitc.opaque<"(void*)40960">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %1170 = "emitc.constant"() <{value = #emitc.opaque<"(void*)45056">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=3, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1171 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1172 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %1173 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1174 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1175 = emitc.call_opaque "__runtime_buffer_arg"(%1170) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1176 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1177 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1178 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1179 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1180 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1181 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1182 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %562, %1175, %9, %1172, %1173, %1181, %1174, %1176, %1177, %1178, %1179, %1180) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(3,4) lock=0 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(3, 4), XAie_LockInit(0, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=2, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1183 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1184 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %1185 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1186 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1187 = emitc.call_opaque "__runtime_buffer_arg"(%1169) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1188 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1189 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1190 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1191 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1192 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1193 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1194 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %562, %1187, %10, %1184, %1185, %1193, %1186, %1188, %1189, %1190, %1191, %1192) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1195 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1196 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1197 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(3,4), direction=S2MM */"
    %1198 = emitc.call_opaque "__Runtime_dma_createio_4"(%562, %1194, %1195, %1196, %1197) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 1 for tile (3,4) */"
    %1199 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    emitc.verbatim "/* Allocated BD ID 1 for tile (6,0) */"
    %1200 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1201 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1202 = emitc.call_opaque "__Runtime_startio"(%1067, %1200, %1201) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1203 = "emitc.constant"() <{value = 16384 : i64}> : () -> i64
    %1204 = emitc.call_opaque "__runtime_buffer_offset"(%arg2, %1203) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=11, offset=192, len=2048, enable_packet=false, packet_id=8, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %1205 = "emitc.constant"() <{value = 192 : i32}> : () -> i32
    %1206 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %1207 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1208 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %1209 = emitc.call_opaque "__runtime_buffer_arg"(%1204) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1210 = "emitc.constant"() <{value = 192 : i64}> : () -> i64
    %1211 = emitc.call_opaque "__runtime_buffer_offset"(%1209, %1210) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1212 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1213 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1214 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1215 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1216 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1217 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1218 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1219 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %1220 = "emitc.constant"() <{value = 256 : i32}> : () -> i32
    %1221 = "emitc.constant"() <{value = 32 : i32}> : () -> i32
    %1222 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1223 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1224 = "emitc.constant"() <{value = 8192 : i32}> : () -> i32
    %1225 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1226 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1227 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%0, %342, %1211, %5, %1206, %1207, %1226, %1208, %1212, %1213, %1214, %1215, %1216, %1217, %1218, %1219, %1220, %1221, %1222, %1223, %1224, %1225) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=10, offset=128, len=2048, enable_packet=false, packet_id=7, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %1228 = "emitc.constant"() <{value = 128 : i32}> : () -> i32
    %1229 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %1230 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1231 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
    %1232 = emitc.call_opaque "__runtime_buffer_arg"(%1204) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1233 = "emitc.constant"() <{value = 128 : i64}> : () -> i64
    %1234 = emitc.call_opaque "__runtime_buffer_offset"(%1232, %1233) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1235 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1236 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1237 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1238 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1239 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1240 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1241 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1242 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %1243 = "emitc.constant"() <{value = 256 : i32}> : () -> i32
    %1244 = "emitc.constant"() <{value = 32 : i32}> : () -> i32
    %1245 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1246 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1247 = "emitc.constant"() <{value = 8192 : i32}> : () -> i32
    %1248 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1249 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1250 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%0, %342, %1234, %4, %1229, %1230, %1249, %1231, %1235, %1236, %1237, %1238, %1239, %1240, %1241, %1242, %1243, %1244, %1245, %1246, %1247, %1248) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=9, offset=64, len=2048, enable_packet=false, packet_id=6, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %1251 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
    %1252 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %1253 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1254 = "emitc.constant"() <{value = 6 : i32}> : () -> i32
    %1255 = emitc.call_opaque "__runtime_buffer_arg"(%1204) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1256 = "emitc.constant"() <{value = 64 : i64}> : () -> i64
    %1257 = emitc.call_opaque "__runtime_buffer_offset"(%1255, %1256) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1258 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1259 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1260 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1261 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1262 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1263 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1264 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1265 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %1266 = "emitc.constant"() <{value = 256 : i32}> : () -> i32
    %1267 = "emitc.constant"() <{value = 32 : i32}> : () -> i32
    %1268 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1269 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1270 = "emitc.constant"() <{value = 8192 : i32}> : () -> i32
    %1271 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1272 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1273 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%0, %342, %1257, %3, %1252, %1253, %1272, %1254, %1258, %1259, %1260, %1261, %1262, %1263, %1264, %1265, %1266, %1267, %1268, %1269, %1270, %1271) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=8, offset=0, len=2048, enable_packet=false, packet_id=5, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %1274 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1275 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %1276 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1277 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1278 = emitc.call_opaque "__runtime_buffer_arg"(%1204) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1279 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1280 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1281 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1282 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1283 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1284 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1285 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1286 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %1287 = "emitc.constant"() <{value = 256 : i32}> : () -> i32
    %1288 = "emitc.constant"() <{value = 32 : i32}> : () -> i32
    %1289 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1290 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1291 = "emitc.constant"() <{value = 8192 : i32}> : () -> i32
    %1292 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1293 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1294 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%0, %342, %1278, %2, %1275, %1276, %1293, %1277, %1279, %1280, %1281, %1282, %1283, %1284, %1285, %1286, %1287, %1288, %1289, %1290, %1291, %1292) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1295 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1296 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %1297 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=8, tile=(3,0), direction=S2MM */"
    emitc.verbatim "/* Enable out-of-order BD on tile(3,0) ch=1 dir=S2MM */"
    emitc.call_opaque "__Runtime_dma_channel_enable_ooo"(%0, %342, %1295, %1297) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, !emitc.opaque<"XAie_DmaDirection">) -> ()
    %1298 = emitc.call_opaque "__Runtime_dma_createio_4"(%342, %1294, %1295, %1296, %1297) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    %1299 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %1300 = emitc.call_opaque "__runtime_buffer_offset"(%1204, %1299) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1301 = "emitc.constant"() <{value = #emitc.opaque<"(void*)49152">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %1302 = "emitc.constant"() <{value = #emitc.opaque<"(void*)53248">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=5, offset=0, len=2048, enable_packet=true, packet_id=5, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=8 */"
    %1303 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1304 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %1305 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1306 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1307 = emitc.call_opaque "__runtime_buffer_arg"(%1302) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1308 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1309 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1310 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1311 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1312 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %1313 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1314 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %70, %1307, %7, %1304, %1305, %1313, %1306, %1308, %1309, %1310, %1311, %1312) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(0,4) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(0, 4), XAie_LockInit(4, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=4, offset=0, len=2048, enable_packet=true, packet_id=5, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=8 */"
    %1315 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1316 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %1317 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1318 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1319 = emitc.call_opaque "__runtime_buffer_arg"(%1301) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1320 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1321 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1322 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1323 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1324 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %1325 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1326 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %70, %1319, %6, %1316, %1317, %1325, %1318, %1320, %1321, %1322, %1323, %1324) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1327 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1328 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1329 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(0,4), direction=MM2S */"
    %1330 = emitc.call_opaque "__Runtime_dma_createio_4"(%70, %1326, %1327, %1328, %1329) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 2 for tile (0,4) */"
    %1331 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1332 = "emitc.constant"() <{value = 4096 : i64}> : () -> i64
    %1333 = emitc.call_opaque "__runtime_buffer_offset"(%1204, %1332) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1334 = "emitc.constant"() <{value = #emitc.opaque<"(void*)49152">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %1335 = "emitc.constant"() <{value = #emitc.opaque<"(void*)53248">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=5, offset=0, len=2048, enable_packet=true, packet_id=6, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=9 */"
    %1336 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1337 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %1338 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1339 = "emitc.constant"() <{value = 6 : i32}> : () -> i32
    %1340 = emitc.call_opaque "__runtime_buffer_arg"(%1335) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1341 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1342 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1343 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1344 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1345 = "emitc.constant"() <{value = 9 : i32}> : () -> i32
    %1346 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1347 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %229, %1340, %7, %1337, %1338, %1346, %1339, %1341, %1342, %1343, %1344, %1345) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(1,4) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(1, 4), XAie_LockInit(4, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=4, offset=0, len=2048, enable_packet=true, packet_id=6, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=9 */"
    %1348 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1349 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %1350 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1351 = "emitc.constant"() <{value = 6 : i32}> : () -> i32
    %1352 = emitc.call_opaque "__runtime_buffer_arg"(%1334) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1353 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1354 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1355 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1356 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1357 = "emitc.constant"() <{value = 9 : i32}> : () -> i32
    %1358 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1359 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %229, %1352, %6, %1349, %1350, %1358, %1351, %1353, %1354, %1355, %1356, %1357) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1360 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1361 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1362 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(1,4), direction=MM2S */"
    %1363 = emitc.call_opaque "__Runtime_dma_createio_4"(%229, %1359, %1360, %1361, %1362) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 2 for tile (1,4) */"
    %1364 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1365 = "emitc.constant"() <{value = 8192 : i64}> : () -> i64
    %1366 = emitc.call_opaque "__runtime_buffer_offset"(%1204, %1365) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1367 = "emitc.constant"() <{value = #emitc.opaque<"(void*)49152">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %1368 = "emitc.constant"() <{value = #emitc.opaque<"(void*)53248">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=5, offset=0, len=2048, enable_packet=true, packet_id=7, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=10 */"
    %1369 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1370 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %1371 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1372 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
    %1373 = emitc.call_opaque "__runtime_buffer_arg"(%1368) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1374 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1375 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1376 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1377 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1378 = "emitc.constant"() <{value = 10 : i32}> : () -> i32
    %1379 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1380 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %397, %1373, %7, %1370, %1371, %1379, %1372, %1374, %1375, %1376, %1377, %1378) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(2,4) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(2, 4), XAie_LockInit(4, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=4, offset=0, len=2048, enable_packet=true, packet_id=7, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=10 */"
    %1381 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1382 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %1383 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1384 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
    %1385 = emitc.call_opaque "__runtime_buffer_arg"(%1367) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1386 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1387 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1388 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1389 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1390 = "emitc.constant"() <{value = 10 : i32}> : () -> i32
    %1391 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1392 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %397, %1385, %6, %1382, %1383, %1391, %1384, %1386, %1387, %1388, %1389, %1390) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1393 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1394 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1395 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(2,4), direction=MM2S */"
    %1396 = emitc.call_opaque "__Runtime_dma_createio_4"(%397, %1392, %1393, %1394, %1395) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 2 for tile (2,4) */"
    %1397 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1398 = "emitc.constant"() <{value = 12288 : i64}> : () -> i64
    %1399 = emitc.call_opaque "__runtime_buffer_offset"(%1204, %1398) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1400 = "emitc.constant"() <{value = #emitc.opaque<"(void*)49152">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %1401 = "emitc.constant"() <{value = #emitc.opaque<"(void*)53248">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=5, offset=0, len=2048, enable_packet=true, packet_id=8, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=11 */"
    %1402 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1403 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %1404 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1405 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %1406 = emitc.call_opaque "__runtime_buffer_arg"(%1401) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1407 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1408 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1409 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1410 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1411 = "emitc.constant"() <{value = 11 : i32}> : () -> i32
    %1412 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1413 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %562, %1406, %7, %1403, %1404, %1412, %1405, %1407, %1408, %1409, %1410, %1411) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(3,4) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(3, 4), XAie_LockInit(4, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=4, offset=0, len=2048, enable_packet=true, packet_id=8, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=11 */"
    %1414 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1415 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %1416 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1417 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %1418 = emitc.call_opaque "__runtime_buffer_arg"(%1400) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1419 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1420 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1421 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1422 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1423 = "emitc.constant"() <{value = 11 : i32}> : () -> i32
    %1424 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1425 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %562, %1418, %6, %1415, %1416, %1424, %1417, %1419, %1420, %1421, %1422, %1423) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1426 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1427 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1428 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(3,4), direction=MM2S */"
    %1429 = emitc.call_opaque "__Runtime_dma_createio_4"(%562, %1425, %1426, %1427, %1428) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 2 for tile (3,4) */"
    %1430 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    emitc.verbatim "/* Allocated BD ID 3 for tile (3,0) */"
    %1431 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1432 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %1433 = emitc.call_opaque "__Runtime_startio"(%1298, %1431, %1432) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1434 = "emitc.constant"() <{value = 32768 : i64}> : () -> i64
    %1435 = emitc.call_opaque "__runtime_buffer_offset"(%arg0, %1434) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1436 = "emitc.constant"() <{value = 7 : i8}> : () -> i8
    %1437 = "emitc.constant"() <{value = 0 : i8}> : () -> i8
    %1438 = emitc.call_opaque "XAie_TileLoc"(%1436, %1437) : (i8, i8) -> !emitc.opaque<"XAie_LocType">
    emitc.verbatim "/* DMA BD Config: bd_id=0, offset=0, len=16384, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %1439 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1440 = "emitc.constant"() <{value = 16384 : i32}> : () -> i32
    %1441 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1442 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1443 = emitc.call_opaque "__runtime_buffer_arg"(%1435) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1444 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1445 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1446 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1447 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1448 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1449 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1450 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %1438, %1443, %11, %1440, %1441, %1449, %1442, %1444, %1445, %1446, %1447, %1448) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1451 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1452 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1453 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=0, tile=(7,0), direction=MM2S */"
    %1454 = emitc.call_opaque "__Runtime_dma_createio_4"(%1438, %1450, %1451, %1452, %1453) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    %1455 = "emitc.constant"() <{value = 32768 : i64}> : () -> i64
    %1456 = emitc.call_opaque "__runtime_buffer_offset"(%1435, %1455) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1457 = "emitc.constant"() <{value = #emitc.opaque<"(void*)40960">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %1458 = "emitc.constant"() <{value = #emitc.opaque<"(void*)45056">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=3, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1459 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1460 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %1461 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1462 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1463 = emitc.call_opaque "__runtime_buffer_arg"(%1458) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1464 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1465 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1466 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1467 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1468 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1469 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1470 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %104, %1463, %9, %1460, %1461, %1469, %1462, %1464, %1465, %1466, %1467, %1468) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(0,5) lock=0 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(0, 5), XAie_LockInit(0, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=2, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1471 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1472 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %1473 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1474 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1475 = emitc.call_opaque "__runtime_buffer_arg"(%1457) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1476 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1477 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1478 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1479 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1480 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1481 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1482 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %104, %1475, %10, %1472, %1473, %1481, %1474, %1476, %1477, %1478, %1479, %1480) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1483 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1484 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1485 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(0,5), direction=S2MM */"
    %1486 = emitc.call_opaque "__Runtime_dma_createio_4"(%104, %1482, %1483, %1484, %1485) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 1 for tile (0,5) */"
    %1487 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1488 = "emitc.constant"() <{value = 32768 : i64}> : () -> i64
    %1489 = emitc.call_opaque "__runtime_buffer_offset"(%1435, %1488) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1490 = "emitc.constant"() <{value = #emitc.opaque<"(void*)40960">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %1491 = "emitc.constant"() <{value = #emitc.opaque<"(void*)45056">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=3, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1492 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1493 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %1494 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1495 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1496 = emitc.call_opaque "__runtime_buffer_arg"(%1491) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1497 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1498 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1499 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1500 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1501 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1502 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1503 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %265, %1496, %9, %1493, %1494, %1502, %1495, %1497, %1498, %1499, %1500, %1501) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(1,5) lock=0 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(1, 5), XAie_LockInit(0, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=2, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1504 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1505 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %1506 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1507 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1508 = emitc.call_opaque "__runtime_buffer_arg"(%1490) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1509 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1510 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1511 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1512 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1513 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1514 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1515 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %265, %1508, %10, %1505, %1506, %1514, %1507, %1509, %1510, %1511, %1512, %1513) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1516 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1517 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1518 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(1,5), direction=S2MM */"
    %1519 = emitc.call_opaque "__Runtime_dma_createio_4"(%265, %1515, %1516, %1517, %1518) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 1 for tile (1,5) */"
    %1520 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1521 = "emitc.constant"() <{value = 32768 : i64}> : () -> i64
    %1522 = emitc.call_opaque "__runtime_buffer_offset"(%1435, %1521) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1523 = "emitc.constant"() <{value = #emitc.opaque<"(void*)40960">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %1524 = "emitc.constant"() <{value = #emitc.opaque<"(void*)45056">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=3, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1525 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1526 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %1527 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1528 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1529 = emitc.call_opaque "__runtime_buffer_arg"(%1524) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1530 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1531 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1532 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1533 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1534 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1535 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1536 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %433, %1529, %9, %1526, %1527, %1535, %1528, %1530, %1531, %1532, %1533, %1534) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(2,5) lock=0 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(2, 5), XAie_LockInit(0, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=2, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1537 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1538 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %1539 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1540 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1541 = emitc.call_opaque "__runtime_buffer_arg"(%1523) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1542 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1543 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1544 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1545 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1546 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1547 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1548 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %433, %1541, %10, %1538, %1539, %1547, %1540, %1542, %1543, %1544, %1545, %1546) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1549 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1550 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1551 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(2,5), direction=S2MM */"
    %1552 = emitc.call_opaque "__Runtime_dma_createio_4"(%433, %1548, %1549, %1550, %1551) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 1 for tile (2,5) */"
    %1553 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1554 = "emitc.constant"() <{value = 32768 : i64}> : () -> i64
    %1555 = emitc.call_opaque "__runtime_buffer_offset"(%1435, %1554) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1556 = "emitc.constant"() <{value = #emitc.opaque<"(void*)40960">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %1557 = "emitc.constant"() <{value = #emitc.opaque<"(void*)45056">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=3, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1558 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1559 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %1560 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1561 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1562 = emitc.call_opaque "__runtime_buffer_arg"(%1557) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1563 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1564 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1565 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1566 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1567 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1568 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1569 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %598, %1562, %9, %1559, %1560, %1568, %1561, %1563, %1564, %1565, %1566, %1567) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(3,5) lock=0 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(3, 5), XAie_LockInit(0, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=2, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1570 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1571 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %1572 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1573 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1574 = emitc.call_opaque "__runtime_buffer_arg"(%1556) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1575 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1576 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1577 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1578 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1579 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1580 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1581 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %598, %1574, %10, %1571, %1572, %1580, %1573, %1575, %1576, %1577, %1578, %1579) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1582 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1583 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1584 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(3,5), direction=S2MM */"
    %1585 = emitc.call_opaque "__Runtime_dma_createio_4"(%598, %1581, %1582, %1583, %1584) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 1 for tile (3,5) */"
    %1586 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    emitc.verbatim "/* Allocated BD ID 0 for tile (7,0) */"
    %1587 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1588 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1589 = emitc.call_opaque "__Runtime_startio"(%1454, %1587, %1588) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1590 = "emitc.constant"() <{value = 32768 : i64}> : () -> i64
    %1591 = emitc.call_opaque "__runtime_buffer_offset"(%arg2, %1590) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=6, offset=192, len=2048, enable_packet=false, packet_id=12, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %1592 = "emitc.constant"() <{value = 192 : i32}> : () -> i32
    %1593 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %1594 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1595 = "emitc.constant"() <{value = 12 : i32}> : () -> i32
    %1596 = emitc.call_opaque "__runtime_buffer_arg"(%1591) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1597 = "emitc.constant"() <{value = 192 : i64}> : () -> i64
    %1598 = emitc.call_opaque "__runtime_buffer_offset"(%1596, %1597) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1599 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1600 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1601 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1602 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1603 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1604 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1605 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1606 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %1607 = "emitc.constant"() <{value = 256 : i32}> : () -> i32
    %1608 = "emitc.constant"() <{value = 32 : i32}> : () -> i32
    %1609 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1610 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1611 = "emitc.constant"() <{value = 8192 : i32}> : () -> i32
    %1612 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1613 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1614 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%0, %17, %1598, %8, %1593, %1594, %1613, %1595, %1599, %1600, %1601, %1602, %1603, %1604, %1605, %1606, %1607, %1608, %1609, %1610, %1611, %1612) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=5, offset=128, len=2048, enable_packet=false, packet_id=11, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %1615 = "emitc.constant"() <{value = 128 : i32}> : () -> i32
    %1616 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %1617 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1618 = "emitc.constant"() <{value = 11 : i32}> : () -> i32
    %1619 = emitc.call_opaque "__runtime_buffer_arg"(%1591) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1620 = "emitc.constant"() <{value = 128 : i64}> : () -> i64
    %1621 = emitc.call_opaque "__runtime_buffer_offset"(%1619, %1620) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1622 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1623 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1624 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1625 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1626 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1627 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1628 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1629 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %1630 = "emitc.constant"() <{value = 256 : i32}> : () -> i32
    %1631 = "emitc.constant"() <{value = 32 : i32}> : () -> i32
    %1632 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1633 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1634 = "emitc.constant"() <{value = 8192 : i32}> : () -> i32
    %1635 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1636 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1637 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%0, %17, %1621, %7, %1616, %1617, %1636, %1618, %1622, %1623, %1624, %1625, %1626, %1627, %1628, %1629, %1630, %1631, %1632, %1633, %1634, %1635) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=4, offset=64, len=2048, enable_packet=false, packet_id=10, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %1638 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
    %1639 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %1640 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1641 = "emitc.constant"() <{value = 10 : i32}> : () -> i32
    %1642 = emitc.call_opaque "__runtime_buffer_arg"(%1591) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1643 = "emitc.constant"() <{value = 64 : i64}> : () -> i64
    %1644 = emitc.call_opaque "__runtime_buffer_offset"(%1642, %1643) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1645 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1646 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1647 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1648 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1649 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1650 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1651 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1652 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %1653 = "emitc.constant"() <{value = 256 : i32}> : () -> i32
    %1654 = "emitc.constant"() <{value = 32 : i32}> : () -> i32
    %1655 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1656 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1657 = "emitc.constant"() <{value = 8192 : i32}> : () -> i32
    %1658 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1659 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1660 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%0, %17, %1644, %6, %1639, %1640, %1659, %1641, %1645, %1646, %1647, %1648, %1649, %1650, %1651, %1652, %1653, %1654, %1655, %1656, %1657, %1658) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=3, offset=0, len=2048, enable_packet=false, packet_id=9, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %1661 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1662 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %1663 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1664 = "emitc.constant"() <{value = 9 : i32}> : () -> i32
    %1665 = emitc.call_opaque "__runtime_buffer_arg"(%1591) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1666 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1667 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1668 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1669 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1670 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1671 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1672 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1673 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %1674 = "emitc.constant"() <{value = 256 : i32}> : () -> i32
    %1675 = "emitc.constant"() <{value = 32 : i32}> : () -> i32
    %1676 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1677 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1678 = "emitc.constant"() <{value = 8192 : i32}> : () -> i32
    %1679 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1680 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1681 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%0, %17, %1665, %9, %1662, %1663, %1680, %1664, %1666, %1667, %1668, %1669, %1670, %1671, %1672, %1673, %1674, %1675, %1676, %1677, %1678, %1679) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1682 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1683 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1684 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=3, tile=(2,0), direction=S2MM */"
    emitc.verbatim "/* Enable out-of-order BD on tile(2,0) ch=0 dir=S2MM */"
    emitc.call_opaque "__Runtime_dma_channel_enable_ooo"(%0, %17, %1682, %1684) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, !emitc.opaque<"XAie_DmaDirection">) -> ()
    %1685 = emitc.call_opaque "__Runtime_dma_createio_4"(%17, %1681, %1682, %1683, %1684) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    %1686 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %1687 = emitc.call_opaque "__runtime_buffer_offset"(%1591, %1686) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1688 = "emitc.constant"() <{value = #emitc.opaque<"(void*)49152">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %1689 = "emitc.constant"() <{value = #emitc.opaque<"(void*)53248">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=5, offset=0, len=2048, enable_packet=true, packet_id=9, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=3 */"
    %1690 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1691 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %1692 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1693 = "emitc.constant"() <{value = 9 : i32}> : () -> i32
    %1694 = emitc.call_opaque "__runtime_buffer_arg"(%1689) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1695 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1696 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1697 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1698 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1699 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1700 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1701 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %104, %1694, %7, %1691, %1692, %1700, %1693, %1695, %1696, %1697, %1698, %1699) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(0,5) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(0, 5), XAie_LockInit(4, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=4, offset=0, len=2048, enable_packet=true, packet_id=9, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=3 */"
    %1702 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1703 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %1704 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1705 = "emitc.constant"() <{value = 9 : i32}> : () -> i32
    %1706 = emitc.call_opaque "__runtime_buffer_arg"(%1688) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1707 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1708 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1709 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1710 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1711 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1712 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1713 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %104, %1706, %6, %1703, %1704, %1712, %1705, %1707, %1708, %1709, %1710, %1711) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1714 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1715 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1716 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(0,5), direction=MM2S */"
    %1717 = emitc.call_opaque "__Runtime_dma_createio_4"(%104, %1713, %1714, %1715, %1716) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 2 for tile (0,5) */"
    %1718 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1719 = "emitc.constant"() <{value = 4096 : i64}> : () -> i64
    %1720 = emitc.call_opaque "__runtime_buffer_offset"(%1591, %1719) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1721 = "emitc.constant"() <{value = #emitc.opaque<"(void*)49152">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %1722 = "emitc.constant"() <{value = #emitc.opaque<"(void*)53248">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=5, offset=0, len=2048, enable_packet=true, packet_id=10, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=4 */"
    %1723 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1724 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %1725 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1726 = "emitc.constant"() <{value = 10 : i32}> : () -> i32
    %1727 = emitc.call_opaque "__runtime_buffer_arg"(%1722) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1728 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1729 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1730 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1731 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1732 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1733 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1734 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %265, %1727, %7, %1724, %1725, %1733, %1726, %1728, %1729, %1730, %1731, %1732) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(1,5) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(1, 5), XAie_LockInit(4, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=4, offset=0, len=2048, enable_packet=true, packet_id=10, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=4 */"
    %1735 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1736 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %1737 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1738 = "emitc.constant"() <{value = 10 : i32}> : () -> i32
    %1739 = emitc.call_opaque "__runtime_buffer_arg"(%1721) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1740 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1741 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1742 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1743 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1744 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1745 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1746 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %265, %1739, %6, %1736, %1737, %1745, %1738, %1740, %1741, %1742, %1743, %1744) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1747 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1748 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1749 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(1,5), direction=MM2S */"
    %1750 = emitc.call_opaque "__Runtime_dma_createio_4"(%265, %1746, %1747, %1748, %1749) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 2 for tile (1,5) */"
    %1751 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1752 = "emitc.constant"() <{value = 8192 : i64}> : () -> i64
    %1753 = emitc.call_opaque "__runtime_buffer_offset"(%1591, %1752) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1754 = "emitc.constant"() <{value = #emitc.opaque<"(void*)49152">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %1755 = "emitc.constant"() <{value = #emitc.opaque<"(void*)53248">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=5, offset=0, len=2048, enable_packet=true, packet_id=11, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=5 */"
    %1756 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1757 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %1758 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1759 = "emitc.constant"() <{value = 11 : i32}> : () -> i32
    %1760 = emitc.call_opaque "__runtime_buffer_arg"(%1755) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1761 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1762 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1763 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1764 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1765 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1766 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1767 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %433, %1760, %7, %1757, %1758, %1766, %1759, %1761, %1762, %1763, %1764, %1765) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(2,5) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(2, 5), XAie_LockInit(4, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=4, offset=0, len=2048, enable_packet=true, packet_id=11, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=5 */"
    %1768 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1769 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %1770 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1771 = "emitc.constant"() <{value = 11 : i32}> : () -> i32
    %1772 = emitc.call_opaque "__runtime_buffer_arg"(%1754) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1773 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1774 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1775 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1776 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1777 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1778 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1779 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %433, %1772, %6, %1769, %1770, %1778, %1771, %1773, %1774, %1775, %1776, %1777) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1780 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1781 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1782 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(2,5), direction=MM2S */"
    %1783 = emitc.call_opaque "__Runtime_dma_createio_4"(%433, %1779, %1780, %1781, %1782) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 2 for tile (2,5) */"
    %1784 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1785 = "emitc.constant"() <{value = 12288 : i64}> : () -> i64
    %1786 = emitc.call_opaque "__runtime_buffer_offset"(%1591, %1785) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1787 = "emitc.constant"() <{value = #emitc.opaque<"(void*)49152">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %1788 = "emitc.constant"() <{value = #emitc.opaque<"(void*)53248">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=5, offset=0, len=2048, enable_packet=true, packet_id=12, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=6 */"
    %1789 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1790 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %1791 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1792 = "emitc.constant"() <{value = 12 : i32}> : () -> i32
    %1793 = emitc.call_opaque "__runtime_buffer_arg"(%1788) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1794 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1795 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1796 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1797 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1798 = "emitc.constant"() <{value = 6 : i32}> : () -> i32
    %1799 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1800 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %598, %1793, %7, %1790, %1791, %1799, %1792, %1794, %1795, %1796, %1797, %1798) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(3,5) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(3, 5), XAie_LockInit(4, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=4, offset=0, len=2048, enable_packet=true, packet_id=12, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=6 */"
    %1801 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1802 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %1803 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1804 = "emitc.constant"() <{value = 12 : i32}> : () -> i32
    %1805 = emitc.call_opaque "__runtime_buffer_arg"(%1787) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1806 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1807 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1808 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1809 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1810 = "emitc.constant"() <{value = 6 : i32}> : () -> i32
    %1811 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1812 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %598, %1805, %6, %1802, %1803, %1811, %1804, %1806, %1807, %1808, %1809, %1810) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1813 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1814 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1815 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(3,5), direction=MM2S */"
    %1816 = emitc.call_opaque "__Runtime_dma_createio_4"(%598, %1812, %1813, %1814, %1815) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 2 for tile (3,5) */"
    %1817 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    emitc.verbatim "/* Allocated BD ID 2 for tile (2,0) */"
    %1818 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1819 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %1820 = emitc.call_opaque "__Runtime_startio"(%1685, %1818, %1819) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1821 = "emitc.constant"() <{value = 49152 : i64}> : () -> i64
    %1822 = emitc.call_opaque "__runtime_buffer_offset"(%arg0, %1821) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=1, offset=0, len=16384, enable_packet=false, packet_id=0, next_bd=-1, acquire_lock_id=0, acquire_lock_val=0, release_lock_id=0, release_lock_val=0, ooo_bd_id=-1 */"
    %1823 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1824 = "emitc.constant"() <{value = 16384 : i32}> : () -> i32
    %1825 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1826 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1827 = emitc.call_opaque "__runtime_buffer_arg"(%1822) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1828 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1829 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1830 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1831 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1832 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1833 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1834 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %1438, %1827, %12, %1824, %1825, %1833, %1826, %1828, %1829, %1830, %1831, %1832) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1835 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1836 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1837 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=1, tile=(7,0), direction=MM2S */"
    %1838 = emitc.call_opaque "__Runtime_dma_createio_4"(%1438, %1834, %1835, %1836, %1837) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    %1839 = "emitc.constant"() <{value = 49152 : i64}> : () -> i64
    %1840 = emitc.call_opaque "__runtime_buffer_offset"(%1822, %1839) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1841 = "emitc.constant"() <{value = #emitc.opaque<"(void*)40960">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %1842 = "emitc.constant"() <{value = #emitc.opaque<"(void*)45056">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=3, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1843 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1844 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %1845 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1846 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1847 = emitc.call_opaque "__runtime_buffer_arg"(%1842) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1848 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1849 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1850 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1851 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1852 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1853 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1854 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %138, %1847, %9, %1844, %1845, %1853, %1846, %1848, %1849, %1850, %1851, %1852) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(0,6) lock=0 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(0, 6), XAie_LockInit(0, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=2, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1855 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1856 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %1857 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1858 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1859 = emitc.call_opaque "__runtime_buffer_arg"(%1841) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1860 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1861 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1862 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1863 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1864 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1865 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1866 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %138, %1859, %10, %1856, %1857, %1865, %1858, %1860, %1861, %1862, %1863, %1864) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1867 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1868 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1869 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(0,6), direction=S2MM */"
    %1870 = emitc.call_opaque "__Runtime_dma_createio_4"(%138, %1866, %1867, %1868, %1869) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 1 for tile (0,6) */"
    %1871 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1872 = "emitc.constant"() <{value = 49152 : i64}> : () -> i64
    %1873 = emitc.call_opaque "__runtime_buffer_offset"(%1822, %1872) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1874 = "emitc.constant"() <{value = #emitc.opaque<"(void*)40960">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %1875 = "emitc.constant"() <{value = #emitc.opaque<"(void*)45056">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=3, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1876 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1877 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %1878 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1879 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1880 = emitc.call_opaque "__runtime_buffer_arg"(%1875) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1881 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1882 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1883 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1884 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1885 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1886 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1887 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %301, %1880, %9, %1877, %1878, %1886, %1879, %1881, %1882, %1883, %1884, %1885) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(1,6) lock=0 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(1, 6), XAie_LockInit(0, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=2, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1888 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1889 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %1890 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1891 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1892 = emitc.call_opaque "__runtime_buffer_arg"(%1874) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1893 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1894 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1895 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1896 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1897 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1898 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1899 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %301, %1892, %10, %1889, %1890, %1898, %1891, %1893, %1894, %1895, %1896, %1897) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1900 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1901 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1902 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(1,6), direction=S2MM */"
    %1903 = emitc.call_opaque "__Runtime_dma_createio_4"(%301, %1899, %1900, %1901, %1902) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 1 for tile (1,6) */"
    %1904 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1905 = "emitc.constant"() <{value = 49152 : i64}> : () -> i64
    %1906 = emitc.call_opaque "__runtime_buffer_offset"(%1822, %1905) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1907 = "emitc.constant"() <{value = #emitc.opaque<"(void*)40960">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %1908 = "emitc.constant"() <{value = #emitc.opaque<"(void*)45056">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=3, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1909 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1910 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %1911 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1912 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1913 = emitc.call_opaque "__runtime_buffer_arg"(%1908) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1914 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1915 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1916 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1917 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1918 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1919 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1920 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %469, %1913, %9, %1910, %1911, %1919, %1912, %1914, %1915, %1916, %1917, %1918) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(2,6) lock=0 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(2, 6), XAie_LockInit(0, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=2, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1921 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1922 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %1923 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1924 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1925 = emitc.call_opaque "__runtime_buffer_arg"(%1907) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1926 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1927 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1928 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1929 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1930 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1931 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1932 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %469, %1925, %10, %1922, %1923, %1931, %1924, %1926, %1927, %1928, %1929, %1930) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1933 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1934 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1935 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(2,6), direction=S2MM */"
    %1936 = emitc.call_opaque "__Runtime_dma_createio_4"(%469, %1932, %1933, %1934, %1935) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 1 for tile (2,6) */"
    %1937 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1938 = "emitc.constant"() <{value = 49152 : i64}> : () -> i64
    %1939 = emitc.call_opaque "__runtime_buffer_offset"(%1822, %1938) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1940 = "emitc.constant"() <{value = #emitc.opaque<"(void*)40960">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %1941 = "emitc.constant"() <{value = #emitc.opaque<"(void*)45056">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=3, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=2, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1942 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1943 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %1944 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1945 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1946 = emitc.call_opaque "__runtime_buffer_arg"(%1941) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1947 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1948 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1949 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1950 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1951 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1952 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1953 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %634, %1946, %9, %1943, %1944, %1952, %1945, %1947, %1948, %1949, %1950, %1951) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(3,6) lock=0 init_value=2 */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(3, 6), XAie_LockInit(0, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=2, offset=0, len=4096, enable_packet=false, packet_id=0, next_bd=3, acquire_lock_id=0, acquire_lock_val=-1, release_lock_id=1, release_lock_val=1, ooo_bd_id=-1 */"
    %1954 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1955 = "emitc.constant"() <{value = 4096 : i32}> : () -> i32
    %1956 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1957 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1958 = emitc.call_opaque "__runtime_buffer_arg"(%1940) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1959 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1960 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1961 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1962 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1963 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1964 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1965 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %634, %1958, %10, %1955, %1956, %1964, %1957, %1959, %1960, %1961, %1962, %1963) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %1966 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1967 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1968 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=2, tile=(3,6), direction=S2MM */"
    %1969 = emitc.call_opaque "__Runtime_dma_createio_4"(%634, %1965, %1966, %1967, %1968) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 1 for tile (3,6) */"
    %1970 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    emitc.verbatim "/* Allocated BD ID 1 for tile (7,0) */"
    %1971 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1972 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1973 = emitc.call_opaque "__Runtime_startio"(%1838, %1971, %1972) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %1974 = "emitc.constant"() <{value = 49152 : i64}> : () -> i64
    %1975 = emitc.call_opaque "__runtime_buffer_offset"(%arg2, %1974) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=11, offset=192, len=2048, enable_packet=false, packet_id=16, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %1976 = "emitc.constant"() <{value = 192 : i32}> : () -> i32
    %1977 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %1978 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1979 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %1980 = emitc.call_opaque "__runtime_buffer_arg"(%1975) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %1981 = "emitc.constant"() <{value = 192 : i64}> : () -> i64
    %1982 = emitc.call_opaque "__runtime_buffer_offset"(%1980, %1981) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %1983 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1984 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1985 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1986 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1987 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %1988 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1989 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1990 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %1991 = "emitc.constant"() <{value = 256 : i32}> : () -> i32
    %1992 = "emitc.constant"() <{value = 32 : i32}> : () -> i32
    %1993 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1994 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1995 = "emitc.constant"() <{value = 8192 : i32}> : () -> i32
    %1996 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1997 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1998 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%0, %17, %1982, %5, %1977, %1978, %1997, %1979, %1983, %1984, %1985, %1986, %1987, %1988, %1989, %1990, %1991, %1992, %1993, %1994, %1995, %1996) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=10, offset=128, len=2048, enable_packet=false, packet_id=15, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %1999 = "emitc.constant"() <{value = 128 : i32}> : () -> i32
    %2000 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %2001 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %2002 = "emitc.constant"() <{value = 15 : i32}> : () -> i32
    %2003 = emitc.call_opaque "__runtime_buffer_arg"(%1975) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %2004 = "emitc.constant"() <{value = 128 : i64}> : () -> i64
    %2005 = emitc.call_opaque "__runtime_buffer_offset"(%2003, %2004) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %2006 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %2007 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %2008 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %2009 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %2010 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %2011 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %2012 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %2013 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %2014 = "emitc.constant"() <{value = 256 : i32}> : () -> i32
    %2015 = "emitc.constant"() <{value = 32 : i32}> : () -> i32
    %2016 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %2017 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %2018 = "emitc.constant"() <{value = 8192 : i32}> : () -> i32
    %2019 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %2020 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %2021 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%0, %17, %2005, %4, %2000, %2001, %2020, %2002, %2006, %2007, %2008, %2009, %2010, %2011, %2012, %2013, %2014, %2015, %2016, %2017, %2018, %2019) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=9, offset=64, len=2048, enable_packet=false, packet_id=14, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %2022 = "emitc.constant"() <{value = 64 : i32}> : () -> i32
    %2023 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %2024 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %2025 = "emitc.constant"() <{value = 14 : i32}> : () -> i32
    %2026 = emitc.call_opaque "__runtime_buffer_arg"(%1975) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %2027 = "emitc.constant"() <{value = 64 : i64}> : () -> i64
    %2028 = emitc.call_opaque "__runtime_buffer_offset"(%2026, %2027) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %2029 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %2030 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %2031 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %2032 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %2033 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %2034 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %2035 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %2036 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %2037 = "emitc.constant"() <{value = 256 : i32}> : () -> i32
    %2038 = "emitc.constant"() <{value = 32 : i32}> : () -> i32
    %2039 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %2040 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %2041 = "emitc.constant"() <{value = 8192 : i32}> : () -> i32
    %2042 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %2043 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %2044 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%0, %17, %2028, %3, %2023, %2024, %2043, %2025, %2029, %2030, %2031, %2032, %2033, %2034, %2035, %2036, %2037, %2038, %2039, %2040, %2041, %2042) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* DMA BD Config: bd_id=8, offset=0, len=2048, enable_packet=false, packet_id=13, next_bd=-1, acquire_lock_id=-1, acquire_lock_val=0, release_lock_id=-1, release_lock_val=0, ooo_bd_id=-1 */"
    %2045 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %2046 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %2047 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %2048 = "emitc.constant"() <{value = 13 : i32}> : () -> i32
    %2049 = emitc.call_opaque "__runtime_buffer_arg"(%1975) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %2050 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %2051 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %2052 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %2053 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %2054 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %2055 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %2056 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %2057 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %2058 = "emitc.constant"() <{value = 256 : i32}> : () -> i32
    %2059 = "emitc.constant"() <{value = 32 : i32}> : () -> i32
    %2060 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %2061 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %2062 = "emitc.constant"() <{value = 8192 : i32}> : () -> i32
    %2063 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %2064 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %2065 = emitc.call_opaque "__Runtime_dma_bd_config_multidim_ooo"(%0, %17, %2049, %2, %2046, %2047, %2064, %2048, %2050, %2051, %2052, %2053, %2054, %2055, %2056, %2057, %2058, %2059, %2060, %2061, %2062, %2063) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %2066 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2067 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %2068 = "emitc.constant"() <{value = #emitc.opaque<"DMA_S2MM">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=1, bd_id=8, tile=(2,0), direction=S2MM */"
    emitc.verbatim "/* Enable out-of-order BD on tile(2,0) ch=1 dir=S2MM */"
    emitc.call_opaque "__Runtime_dma_channel_enable_ooo"(%0, %17, %2066, %2068) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, !emitc.opaque<"XAie_DmaDirection">) -> ()
    %2069 = emitc.call_opaque "__Runtime_dma_createio_4"(%17, %2065, %2066, %2067, %2068) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    %2070 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
    %2071 = emitc.call_opaque "__runtime_buffer_offset"(%1975, %2070) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %2072 = "emitc.constant"() <{value = #emitc.opaque<"(void*)49152">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %2073 = "emitc.constant"() <{value = #emitc.opaque<"(void*)53248">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=5, offset=0, len=2048, enable_packet=true, packet_id=13, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=8 */"
    %2074 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %2075 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %2076 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %2077 = "emitc.constant"() <{value = 13 : i32}> : () -> i32
    %2078 = emitc.call_opaque "__runtime_buffer_arg"(%2073) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %2079 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %2080 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %2081 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %2082 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2083 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %2084 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2085 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %138, %2078, %7, %2075, %2076, %2084, %2077, %2079, %2080, %2081, %2082, %2083) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(0,6) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(0, 6), XAie_LockInit(4, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=4, offset=0, len=2048, enable_packet=true, packet_id=13, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=8 */"
    %2086 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %2087 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %2088 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %2089 = "emitc.constant"() <{value = 13 : i32}> : () -> i32
    %2090 = emitc.call_opaque "__runtime_buffer_arg"(%2072) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %2091 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %2092 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %2093 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %2094 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2095 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %2096 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2097 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %138, %2090, %6, %2087, %2088, %2096, %2089, %2091, %2092, %2093, %2094, %2095) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %2098 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %2099 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %2100 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(0,6), direction=MM2S */"
    %2101 = emitc.call_opaque "__Runtime_dma_createio_4"(%138, %2097, %2098, %2099, %2100) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 2 for tile (0,6) */"
    %2102 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %2103 = "emitc.constant"() <{value = 4096 : i64}> : () -> i64
    %2104 = emitc.call_opaque "__runtime_buffer_offset"(%1975, %2103) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %2105 = "emitc.constant"() <{value = #emitc.opaque<"(void*)49152">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %2106 = "emitc.constant"() <{value = #emitc.opaque<"(void*)53248">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=5, offset=0, len=2048, enable_packet=true, packet_id=14, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=9 */"
    %2107 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %2108 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %2109 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %2110 = "emitc.constant"() <{value = 14 : i32}> : () -> i32
    %2111 = emitc.call_opaque "__runtime_buffer_arg"(%2106) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %2112 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %2113 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %2114 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %2115 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2116 = "emitc.constant"() <{value = 9 : i32}> : () -> i32
    %2117 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2118 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %301, %2111, %7, %2108, %2109, %2117, %2110, %2112, %2113, %2114, %2115, %2116) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(1,6) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(1, 6), XAie_LockInit(4, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=4, offset=0, len=2048, enable_packet=true, packet_id=14, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=9 */"
    %2119 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %2120 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %2121 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %2122 = "emitc.constant"() <{value = 14 : i32}> : () -> i32
    %2123 = emitc.call_opaque "__runtime_buffer_arg"(%2105) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %2124 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %2125 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %2126 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %2127 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2128 = "emitc.constant"() <{value = 9 : i32}> : () -> i32
    %2129 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2130 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %301, %2123, %6, %2120, %2121, %2129, %2122, %2124, %2125, %2126, %2127, %2128) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %2131 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %2132 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %2133 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(1,6), direction=MM2S */"
    %2134 = emitc.call_opaque "__Runtime_dma_createio_4"(%301, %2130, %2131, %2132, %2133) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 2 for tile (1,6) */"
    %2135 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %2136 = "emitc.constant"() <{value = 8192 : i64}> : () -> i64
    %2137 = emitc.call_opaque "__runtime_buffer_offset"(%1975, %2136) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %2138 = "emitc.constant"() <{value = #emitc.opaque<"(void*)49152">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %2139 = "emitc.constant"() <{value = #emitc.opaque<"(void*)53248">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=5, offset=0, len=2048, enable_packet=true, packet_id=15, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=10 */"
    %2140 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %2141 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %2142 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %2143 = "emitc.constant"() <{value = 15 : i32}> : () -> i32
    %2144 = emitc.call_opaque "__runtime_buffer_arg"(%2139) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %2145 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %2146 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %2147 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %2148 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2149 = "emitc.constant"() <{value = 10 : i32}> : () -> i32
    %2150 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2151 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %469, %2144, %7, %2141, %2142, %2150, %2143, %2145, %2146, %2147, %2148, %2149) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(2,6) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(2, 6), XAie_LockInit(4, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=4, offset=0, len=2048, enable_packet=true, packet_id=15, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=10 */"
    %2152 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %2153 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %2154 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %2155 = "emitc.constant"() <{value = 15 : i32}> : () -> i32
    %2156 = emitc.call_opaque "__runtime_buffer_arg"(%2138) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %2157 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %2158 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %2159 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %2160 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2161 = "emitc.constant"() <{value = 10 : i32}> : () -> i32
    %2162 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2163 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %469, %2156, %6, %2153, %2154, %2162, %2155, %2157, %2158, %2159, %2160, %2161) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %2164 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %2165 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %2166 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(2,6), direction=MM2S */"
    %2167 = emitc.call_opaque "__Runtime_dma_createio_4"(%469, %2163, %2164, %2165, %2166) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 2 for tile (2,6) */"
    %2168 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %2169 = "emitc.constant"() <{value = 12288 : i64}> : () -> i64
    %2170 = emitc.call_opaque "__runtime_buffer_offset"(%1975, %2169) : (!emitc.ptr<!emitc.opaque<"void">>, i64) -> !emitc.ptr<!emitc.opaque<"void">>
    %2171 = "emitc.constant"() <{value = #emitc.opaque<"(void*)49152">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    %2172 = "emitc.constant"() <{value = #emitc.opaque<"(void*)53248">}> : () -> !emitc.ptr<!emitc.opaque<"void">>
    emitc.verbatim "/* DMA BD Config: bd_id=5, offset=0, len=2048, enable_packet=true, packet_id=16, next_bd=4, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=11 */"
    %2173 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %2174 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %2175 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %2176 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %2177 = emitc.call_opaque "__runtime_buffer_arg"(%2172) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %2178 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %2179 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %2180 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %2181 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2182 = "emitc.constant"() <{value = 11 : i32}> : () -> i32
    %2183 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2184 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %634, %2177, %7, %2174, %2175, %2183, %2176, %2178, %2179, %2180, %2181, %2182) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    emitc.verbatim "/* Lock init: tile(3,6) lock=4 init_value=2 (kernel output acquire) */"
    emitc.verbatim "XAie_LockSetValue(g_DevInst, XAie_TileLoc(3, 6), XAie_LockInit(4, 2));"
    emitc.verbatim "/* DMA BD Config: bd_id=4, offset=0, len=2048, enable_packet=true, packet_id=16, next_bd=5, acquire_lock_id=5, acquire_lock_val=-1, release_lock_id=4, release_lock_val=1, ooo_bd_id=11 */"
    %2185 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %2186 = "emitc.constant"() <{value = 2048 : i32}> : () -> i32
    %2187 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %2188 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %2189 = emitc.call_opaque "__runtime_buffer_arg"(%2171) : (!emitc.ptr<!emitc.opaque<"void">>) -> !emitc.ptr<!emitc.opaque<"void">>
    %2190 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %2191 = "emitc.constant"() <{value = -1 : i32}> : () -> i32
    %2192 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %2193 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2194 = "emitc.constant"() <{value = 11 : i32}> : () -> i32
    %2195 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2196 = emitc.call_opaque "__Runtime_dma_bd_config"(%0, %634, %2189, %6, %2186, %2187, %2195, %2188, %2190, %2191, %2192, %2193, %2194) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<!emitc.opaque<"void">>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) -> !emitc.opaque<"XAie_DmaDesc">
    %2197 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %2198 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %2199 = "emitc.constant"() <{value = #emitc.opaque<"DMA_MM2S">}> : () -> !emitc.opaque<"XAie_DmaDirection">
    emitc.verbatim "/* Create IO: channel_id=0, bd_id=4, tile=(3,6), direction=MM2S */"
    %2200 = emitc.call_opaque "__Runtime_dma_createio_4"(%634, %2196, %2197, %2198, %2199) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_DmaDesc">, i32, i32, !emitc.opaque<"XAie_DmaDirection">) -> !emitc.opaque<"io">
    emitc.verbatim "/* Allocated BD ID 2 for tile (3,6) */"
    %2201 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    emitc.verbatim "/* Allocated BD ID 3 for tile (2,0) */"
    %2202 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %2203 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %2204 = emitc.call_opaque "__Runtime_startio"(%2069, %2202, %2203) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* Load Kernel Group: 16 tile(s) */"
    %2205 = "emitc.constant"() <{value = 16 : i32}> : () -> i32
    %2206 = emitc.call_opaque "__Runtime_load_kernel_group_16t"(%36, %70, %104, %138, %193, %229, %265, %301, %361, %397, %433, %469, %526, %562, %598, %634, %2205) : (!emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, !emitc.opaque<"XAie_LocType">, i32) -> !emitc.opaque<"kernel_group">
    emitc.verbatim "/* Launch Kernel Group */"
    %2207 = emitc.call_opaque "__Runtime_launch_kernel_group"(%2206) : (!emitc.opaque<"kernel_group">) -> !emitc.opaque<"event">
    %2208 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2209 = emitc.call_opaque "__Runtime_startio"(%66, %67, %2208) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2210 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2211 = emitc.call_opaque "__Runtime_startio"(%100, %101, %2210) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2212 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2213 = emitc.call_opaque "__Runtime_startio"(%134, %135, %2212) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2214 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2215 = emitc.call_opaque "__Runtime_startio"(%168, %169, %2214) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2216 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2217 = emitc.call_opaque "__Runtime_startio"(%225, %226, %2216) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2218 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2219 = emitc.call_opaque "__Runtime_startio"(%261, %262, %2218) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2220 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2221 = emitc.call_opaque "__Runtime_startio"(%297, %298, %2220) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2222 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2223 = emitc.call_opaque "__Runtime_startio"(%333, %334, %2222) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2224 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2225 = emitc.call_opaque "__Runtime_startio"(%393, %394, %2224) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2226 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2227 = emitc.call_opaque "__Runtime_startio"(%429, %430, %2226) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2228 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2229 = emitc.call_opaque "__Runtime_startio"(%465, %466, %2228) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2230 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2231 = emitc.call_opaque "__Runtime_startio"(%501, %502, %2230) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2232 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2233 = emitc.call_opaque "__Runtime_startio"(%558, %559, %2232) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2234 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2235 = emitc.call_opaque "__Runtime_startio"(%594, %595, %2234) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2236 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2237 = emitc.call_opaque "__Runtime_startio"(%630, %631, %2236) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2238 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2239 = emitc.call_opaque "__Runtime_startio"(%666, %667, %2238) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2240 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2241 = emitc.call_opaque "__Runtime_startio"(%721, %722, %2240) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2242 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2243 = emitc.call_opaque "__Runtime_startio"(%752, %753, %2242) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2244 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2245 = emitc.call_opaque "__Runtime_startio"(%783, %784, %2244) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2246 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2247 = emitc.call_opaque "__Runtime_startio"(%814, %815, %2246) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2248 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2249 = emitc.call_opaque "__Runtime_startio"(%946, %947, %2248) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2250 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2251 = emitc.call_opaque "__Runtime_startio"(%979, %980, %2250) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2252 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2253 = emitc.call_opaque "__Runtime_startio"(%1012, %1013, %2252) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2254 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2255 = emitc.call_opaque "__Runtime_startio"(%1045, %1046, %2254) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2256 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2257 = emitc.call_opaque "__Runtime_startio"(%1099, %1100, %2256) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2258 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2259 = emitc.call_opaque "__Runtime_startio"(%1132, %1133, %2258) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2260 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2261 = emitc.call_opaque "__Runtime_startio"(%1165, %1166, %2260) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2262 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2263 = emitc.call_opaque "__Runtime_startio"(%1198, %1199, %2262) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2264 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2265 = emitc.call_opaque "__Runtime_startio"(%1330, %1331, %2264) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2266 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2267 = emitc.call_opaque "__Runtime_startio"(%1363, %1364, %2266) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2268 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2269 = emitc.call_opaque "__Runtime_startio"(%1396, %1397, %2268) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2270 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2271 = emitc.call_opaque "__Runtime_startio"(%1429, %1430, %2270) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2272 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2273 = emitc.call_opaque "__Runtime_startio"(%1486, %1487, %2272) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2274 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2275 = emitc.call_opaque "__Runtime_startio"(%1519, %1520, %2274) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2276 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2277 = emitc.call_opaque "__Runtime_startio"(%1552, %1553, %2276) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2278 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2279 = emitc.call_opaque "__Runtime_startio"(%1585, %1586, %2278) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2280 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2281 = emitc.call_opaque "__Runtime_startio"(%1717, %1718, %2280) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2282 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2283 = emitc.call_opaque "__Runtime_startio"(%1750, %1751, %2282) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2284 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2285 = emitc.call_opaque "__Runtime_startio"(%1783, %1784, %2284) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2286 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2287 = emitc.call_opaque "__Runtime_startio"(%1816, %1817, %2286) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2288 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2289 = emitc.call_opaque "__Runtime_startio"(%1870, %1871, %2288) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2290 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2291 = emitc.call_opaque "__Runtime_startio"(%1903, %1904, %2290) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2292 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2293 = emitc.call_opaque "__Runtime_startio"(%1936, %1937, %2292) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2294 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2295 = emitc.call_opaque "__Runtime_startio"(%1969, %1970, %2294) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2296 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2297 = emitc.call_opaque "__Runtime_startio"(%2101, %2102, %2296) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2298 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2299 = emitc.call_opaque "__Runtime_startio"(%2134, %2135, %2298) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2300 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2301 = emitc.call_opaque "__Runtime_startio"(%2167, %2168, %2300) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    %2302 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %2303 = emitc.call_opaque "__Runtime_startio"(%2200, %2201, %2302) : (!emitc.opaque<"io">, i32, i32) -> !emitc.opaque<"ioevent">
    emitc.verbatim "/* Wait for 13 event(s) */"
    emitc.call_opaque "__Runtime_wait"(%2207) : (!emitc.opaque<"event">) -> ()
    emitc.call_opaque "__Runtime_wait"(%172) : (!emitc.opaque<"ioevent">) -> ()
    emitc.call_opaque "__Runtime_wait"(%337) : (!emitc.opaque<"ioevent">) -> ()
    emitc.call_opaque "__Runtime_wait"(%505) : (!emitc.opaque<"ioevent">) -> ()
    emitc.call_opaque "__Runtime_wait"(%670) : (!emitc.opaque<"ioevent">) -> ()
    emitc.call_opaque "__Runtime_wait"(%818) : (!emitc.opaque<"ioevent">) -> ()
    emitc.call_opaque "__Runtime_wait"(%1049) : (!emitc.opaque<"ioevent">) -> ()
    emitc.call_opaque "__Runtime_wait"(%1202) : (!emitc.opaque<"ioevent">) -> ()
    emitc.call_opaque "__Runtime_wait"(%1433) : (!emitc.opaque<"ioevent">) -> ()
    emitc.call_opaque "__Runtime_wait"(%1589) : (!emitc.opaque<"ioevent">) -> ()
    emitc.call_opaque "__Runtime_wait"(%1820) : (!emitc.opaque<"ioevent">) -> ()
    emitc.call_opaque "__Runtime_wait"(%1973) : (!emitc.opaque<"ioevent">) -> ()
    emitc.call_opaque "__Runtime_wait"(%2204) : (!emitc.opaque<"ioevent">) -> ()
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
