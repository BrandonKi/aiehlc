// *****************************************************************************
// * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
// * SPDX-License-Identifier: Apache-2.0
// *****************************************************************************
// *****************************************************************************
// * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
// * SPDX-License-Identifier: Apache-2.0
// *****************************************************************************
module attributes {codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"]} {
  func.func @main() {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<"0x0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F202122232425262728292A2B2C2D2E2F303132333435363738393A3B3C3D3E3F404142434445464748494A4B4C4D4E4F505152535455565758595A5B5C5D5E5F606162636465666768696A6B6C6D6E6F707172737475767778797A7B7C7D7E7F808182838485868788898A8B8C8D8E8F909192939495969798999A9B9C9D9E9FA0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBFC0C1C2C3C4C5C6C7C8C9CACBCCCDCECFD0D1D2D3D4D5D6D7D8D9DADBDCDDDEDFE0E1E2E3E4E5E6E7E8E9EAEBECEDEEEFF0F1F2F3F4F5F6F7F8F9FAFBFCFDFEFF00"> : tensor<16x16xi8>
    %0 = bufferization.to_memref %cst : memref<16x16xi8>
    %alloc = memref.alloc() : memref<16x16xi8>
    memref.copy %0, %alloc : memref<16x16xi8> to memref<16x16xi8>
    scf.execute_region {
      %1 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c0_i32 : i32) -> i32{
      ^bb0(%arg0: i32):
        %subview = memref.subview %alloc[0, 0] [8, 16] [1, 1] : memref<16x16xi8> to memref<8x16xi8, strided<[16, 1]>>
        %3 = dfschedule.declaretile {col = 2 : i32, row = 0 : i32} : !dfschedule.tile
        %c0_i32_0 = arith.constant 0 : i32
        %4 = dfschedule.buffer_view %subview {len = 128 : i64, offset = 0 : i64} : memref<8x16xi8, strided<[16, 1]>> -> memref<8x16xi8, strided<[16, 1]>>
        %5 = dfschedule.config.dma_bd(%4, %3, %c0_i32_0) {
          offset = 0,
          len = 128,
          enable_packet = true,
          packet_id = 1,
          next_bd = 4294967295,
          acquire_lock_id = 0,
          acquire_lock_val = 0,
          release_lock_id = 0,
          release_lock_val = 0,
          data_id = 0
        } : (memref<8x16xi8, strided<[16, 1]>>, !dfschedule.tile, i32) -> !dfschedule.bd_handle
        %6 = dfschedule.config.create_io(%5, %3) {
          channel = 0,
          direction = "S2MM",
          io_operation = "RECV"
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %7 = dfschedule.declaretile {col = 0 : i32, row = 3 : i32} : !dfschedule.tile
        %subview_1 = memref.subview %subview[0, 0] [4, 16] [1, 1] : memref<8x16xi8, strided<[16, 1]>> to memref<4x16xi8, strided<[16, 1]>>
        %8 = dfschedule.memref_mapping %subview_1 : (memref<4x16xi8, strided<[16, 1]>>) -> memref<4x16xi8>
        %9 = dfschedule.bind_core_buffer(%8, %7) {offset = 0 : i64} : (memref<4x16xi8>, !dfschedule.tile) -> memref<4x16xi8>
        %10 = dfschedule.bind_core_buffer(%8, %7) {offset = 64 : i64} : (memref<4x16xi8>, !dfschedule.tile) -> memref<4x16xi8>
        %c1_i32_2 = arith.constant 1 : i32
        %11 = dfschedule.config.dma_bd(%10, %7, %c1_i32_2) {
          offset = 0,
          len = 64,
          enable_packet = true,
          packet_id = 1,
          next_bd = 0,
          acquire_lock_id = 0,
          acquire_lock_val = -1,
          release_lock_id = 1,
          release_lock_val = 1,
          data_id = -1
        } : (memref<4x16xi8>, !dfschedule.tile, i32) -> !dfschedule.bd_handle
        %c0_i32_3 = arith.constant 0 : i32
        %12 = dfschedule.config.dma_bd(%9, %7, %c0_i32_3, %11) {
          offset = 0,
          len = 64,
          enable_packet = true,
          packet_id = 1,
          next_bd = 1,
          acquire_lock_id = 0,
          acquire_lock_val = -1,
          release_lock_id = 1,
          release_lock_val = 1,
          data_id = -1
        } : (memref<4x16xi8>, !dfschedule.tile, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %13 = dfschedule.config.create_io(%12, %7) {
          channel = 0,
          direction = "MM2S",
          io_operation = "SEND"
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %14 = dfschedule.schedule.getbdid(%7) : (!dfschedule.tile) -> i32
        %15 = dfschedule.schedule.start_io(%13, %14) {flow_index = 0 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %16 = dfschedule.declaretile {col = 1 : i32, row = 3 : i32} : !dfschedule.tile
        %subview_4 = memref.subview %subview[4, 0] [4, 16] [1, 1] : memref<8x16xi8, strided<[16, 1]>> to memref<4x16xi8, strided<[16, 1], offset: 64>>
        %17 = dfschedule.memref_mapping %subview_4 : (memref<4x16xi8, strided<[16, 1], offset: 64>>) -> memref<4x16xi8>
        %18 = dfschedule.bind_core_buffer(%17, %16) {offset = 0 : i64} : (memref<4x16xi8>, !dfschedule.tile) -> memref<4x16xi8>
        %19 = dfschedule.bind_core_buffer(%17, %16) {offset = 64 : i64} : (memref<4x16xi8>, !dfschedule.tile) -> memref<4x16xi8>
        %c1_i32_5 = arith.constant 1 : i32
        %20 = dfschedule.config.dma_bd(%19, %16, %c1_i32_5) {
          offset = 0,
          len = 64,
          enable_packet = true,
          packet_id = 2,
          next_bd = 0,
          acquire_lock_id = 0,
          acquire_lock_val = -1,
          release_lock_id = 1,
          release_lock_val = 1,
          data_id = -1
        } : (memref<4x16xi8>, !dfschedule.tile, i32) -> !dfschedule.bd_handle
        %c0_i32_6 = arith.constant 0 : i32
        %21 = dfschedule.config.dma_bd(%18, %16, %c0_i32_6, %20) {
          offset = 0,
          len = 64,
          enable_packet = true,
          packet_id = 2,
          next_bd = 1,
          acquire_lock_id = 0,
          acquire_lock_val = -1,
          release_lock_id = 1,
          release_lock_val = 1,
          data_id = -1
        } : (memref<4x16xi8>, !dfschedule.tile, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %22 = dfschedule.config.create_io(%21, %16) {
          channel = 0,
          direction = "MM2S",
          io_operation = "SEND"
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %23 = dfschedule.schedule.getbdid(%16) : (!dfschedule.tile) -> i32
        %24 = dfschedule.schedule.start_io(%22, %23) {flow_index = 0 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %25 = dfschedule.declare_kernel_config @kernelconfig0 {tile_configs = [{acquire_lock_id = 0 : i32, buffer_mode = 1 : i32, buffer_offset = 0 : i32, buffer_size = 64 : i32, dma_channel = 0 : i32, element_size = 1 : i32, flow_index = 0 : i32, num_buffers = 2 : i32, packet_id = 1 : i32, release_lock_id = 1 : i32, tile_index = 0 : i32}]}
        %26 = dfschedule.declare_kernel_config @kernelconfig1 {tile_configs = [{acquire_lock_id = 0 : i32, buffer_mode = 1 : i32, buffer_offset = 64 : i32, buffer_size = 64 : i32, dma_channel = 0 : i32, element_size = 1 : i32, flow_index = 0 : i32, num_buffers = 2 : i32, packet_id = 2 : i32, release_lock_id = 1 : i32, tile_index = 1 : i32}]}
        %27 = dfschedule.config.load_kernel_group(%7, %16) {
          callee = [@dskernel_receiver],
          distributed_compute_kernel_args = [@compute0, @compute0],
          distributed_args = [@kernelconfig0, @kernelconfig1]
        } : (!dfschedule.tile, !dfschedule.tile) -> !dfschedule.kernelgroup
        %28 = dfschedule.schedule.launch_kernel_group(%27) : (!dfschedule.kernelgroup) -> !dfschedule.event
        %29 = dfschedule.schedule.getbdid(%3) : (!dfschedule.tile) -> i32
        %30 = dfschedule.schedule.start_io(%6, %29) {flow_index = 0 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        dfschedule.schedule.wait(%28, %30) : (!dfschedule.event, !dfschedule.event)
        dfschedule.free_device_mem %subview : memref<8x16xi8, strided<[16, 1]>>
      }
      %2 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c1_i32 : i32) -> i32{
      ^bb0(%arg0: i32):
        %subview = memref.subview %alloc[8, 0] [8, 16] [1, 1] : memref<16x16xi8> to memref<8x16xi8, strided<[16, 1], offset: 128>>
        %3 = dfschedule.declaretile {col = 2 : i32, row = 0 : i32} : !dfschedule.tile
        %c1_i32_0 = arith.constant 1 : i32
        %4 = dfschedule.buffer_view %subview {len = 128 : i64, offset = 0 : i64} : memref<8x16xi8, strided<[16, 1], offset: 128>> -> memref<8x16xi8, strided<[16, 1], offset: 128>>
        %5 = dfschedule.config.dma_bd(%4, %3, %c1_i32_0) {
          offset = 0,
          len = 128,
          enable_packet = true,
          packet_id = 3,
          next_bd = 4294967295,
          acquire_lock_id = 0,
          acquire_lock_val = 0,
          release_lock_id = 0,
          release_lock_val = 0,
          data_id = 0
        } : (memref<8x16xi8, strided<[16, 1], offset: 128>>, !dfschedule.tile, i32) -> !dfschedule.bd_handle
        %6 = dfschedule.config.create_io(%5, %3) {
          channel = 1,
          direction = "S2MM",
          io_operation = "RECV"
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %7 = dfschedule.declaretile {col = 0 : i32, row = 4 : i32} : !dfschedule.tile
        %subview_1 = memref.subview %subview[0, 0] [4, 16] [1, 1] : memref<8x16xi8, strided<[16, 1], offset: 128>> to memref<4x16xi8, strided<[16, 1], offset: 128>>
        %8 = dfschedule.memref_mapping %subview_1 : (memref<4x16xi8, strided<[16, 1], offset: 128>>) -> memref<4x16xi8>
        %9 = dfschedule.bind_core_buffer(%8, %7) {offset = 0 : i64} : (memref<4x16xi8>, !dfschedule.tile) -> memref<4x16xi8>
        %10 = dfschedule.bind_core_buffer(%8, %7) {offset = 64 : i64} : (memref<4x16xi8>, !dfschedule.tile) -> memref<4x16xi8>
        %c1_i32_2 = arith.constant 1 : i32
        %11 = dfschedule.config.dma_bd(%10, %7, %c1_i32_2) {
          offset = 0,
          len = 64,
          enable_packet = true,
          packet_id = 3,
          next_bd = 0,
          acquire_lock_id = 0,
          acquire_lock_val = -1,
          release_lock_id = 1,
          release_lock_val = 1,
          data_id = -1
        } : (memref<4x16xi8>, !dfschedule.tile, i32) -> !dfschedule.bd_handle
        %c0_i32_3 = arith.constant 0 : i32
        %12 = dfschedule.config.dma_bd(%9, %7, %c0_i32_3, %11) {
          offset = 0,
          len = 64,
          enable_packet = true,
          packet_id = 3,
          next_bd = 1,
          acquire_lock_id = 0,
          acquire_lock_val = -1,
          release_lock_id = 1,
          release_lock_val = 1,
          data_id = -1
        } : (memref<4x16xi8>, !dfschedule.tile, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %13 = dfschedule.config.create_io(%12, %7) {
          channel = 0,
          direction = "MM2S",
          io_operation = "SEND"
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %14 = dfschedule.schedule.getbdid(%7) : (!dfschedule.tile) -> i32
        %15 = dfschedule.schedule.start_io(%13, %14) {flow_index = 1 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %16 = dfschedule.declaretile {col = 1 : i32, row = 4 : i32} : !dfschedule.tile
        %subview_4 = memref.subview %subview[4, 0] [4, 16] [1, 1] : memref<8x16xi8, strided<[16, 1], offset: 128>> to memref<4x16xi8, strided<[16, 1], offset: 192>>
        %17 = dfschedule.memref_mapping %subview_4 : (memref<4x16xi8, strided<[16, 1], offset: 192>>) -> memref<4x16xi8>
        %18 = dfschedule.bind_core_buffer(%17, %16) {offset = 0 : i64} : (memref<4x16xi8>, !dfschedule.tile) -> memref<4x16xi8>
        %19 = dfschedule.bind_core_buffer(%17, %16) {offset = 64 : i64} : (memref<4x16xi8>, !dfschedule.tile) -> memref<4x16xi8>
        %c1_i32_5 = arith.constant 1 : i32
        %20 = dfschedule.config.dma_bd(%19, %16, %c1_i32_5) {
          offset = 0,
          len = 64,
          enable_packet = true,
          packet_id = 4,
          next_bd = 0,
          acquire_lock_id = 0,
          acquire_lock_val = -1,
          release_lock_id = 1,
          release_lock_val = 1,
          data_id = -1
        } : (memref<4x16xi8>, !dfschedule.tile, i32) -> !dfschedule.bd_handle
        %c0_i32_6 = arith.constant 0 : i32
        %21 = dfschedule.config.dma_bd(%18, %16, %c0_i32_6, %20) {
          offset = 0,
          len = 64,
          enable_packet = true,
          packet_id = 4,
          next_bd = 1,
          acquire_lock_id = 0,
          acquire_lock_val = -1,
          release_lock_id = 1,
          release_lock_val = 1,
          data_id = -1
        } : (memref<4x16xi8>, !dfschedule.tile, i32, !dfschedule.bd_handle) -> !dfschedule.bd_handle
        %22 = dfschedule.config.create_io(%21, %16) {
          channel = 0,
          direction = "MM2S",
          io_operation = "SEND"
        } : (!dfschedule.bd_handle, !dfschedule.tile) -> !dfschedule.io_handle
        %23 = dfschedule.schedule.getbdid(%16) : (!dfschedule.tile) -> i32
        %24 = dfschedule.schedule.start_io(%22, %23) {flow_index = 1 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        %25 = dfschedule.declare_kernel_config @kernelconfig0 {tile_configs = [{acquire_lock_id = 0 : i32, buffer_mode = 1 : i32, buffer_offset = 0 : i32, buffer_size = 64 : i32, dma_channel = 0 : i32, element_size = 1 : i32, flow_index = 1 : i32, num_buffers = 2 : i32, packet_id = 3 : i32, release_lock_id = 1 : i32, tile_index = 0 : i32}]}
        %26 = dfschedule.declare_kernel_config @kernelconfig1 {tile_configs = [{acquire_lock_id = 0 : i32, buffer_mode = 1 : i32, buffer_offset = 64 : i32, buffer_size = 64 : i32, dma_channel = 0 : i32, element_size = 1 : i32, flow_index = 1 : i32, num_buffers = 2 : i32, packet_id = 4 : i32, release_lock_id = 1 : i32, tile_index = 1 : i32}]}
        %27 = dfschedule.config.load_kernel_group(%7, %16) {
          callee = [@dskernel_receiver],
          distributed_compute_kernel_args = [@compute0, @compute0],
          distributed_args = [@kernelconfig0, @kernelconfig1]
        } : (!dfschedule.tile, !dfschedule.tile) -> !dfschedule.kernelgroup
        %28 = dfschedule.schedule.launch_kernel_group(%27) : (!dfschedule.kernelgroup) -> !dfschedule.event
        %29 = dfschedule.schedule.getbdid(%3) : (!dfschedule.tile) -> i32
        %30 = dfschedule.schedule.start_io(%6, %29) {flow_index = 1 : i32} : (!dfschedule.io_handle, i32) -> !dfschedule.event
        dfschedule.schedule.wait(%28, %30) : (!dfschedule.event, !dfschedule.event)
        dfschedule.free_device_mem %subview : memref<8x16xi8, strided<[16, 1], offset: 128>>
      }
      scf.yield
    } {routing_memo = "row"}
    memref.dealloc %alloc : memref<16x16xi8>
    return
  }
  dfschedule.dskernel_receiver @dskernel_receiver {
  }
}
