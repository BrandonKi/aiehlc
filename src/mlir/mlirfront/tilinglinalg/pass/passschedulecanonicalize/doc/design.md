<\!-- Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->
S A MLIR EXPERT please advise what is the best solution to cannolize my IR "module attributes {codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"]} {

  memref.global "private" constant @my_constant : memref<16x16xi8> = dense<"0x0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F202122232425262728292A2B2C2D2E2F303132333435363738393A3B3C3D3E3F404142434445464748494A4B4C4D4E4F505152535455565758595A5B5C5D5E5F606162636465666768696A6B6C6D6E6F707172737475767778797A7B7C7D7E7F808182838485868788898A8B8C8D8E8F909192939495969798999A9B9C9D9E9FA0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBFC0C1C2C3C4C5C6C7C8C9CACBCCCDCECFD0D1D2D3D4D5D6D7D8D9DADBDCDDDEDFE0E1E2E3E4E5E6E7E8E9EAEBECEDEEEFF0F1F2F3F4F5F6F7F8F9FAFBFCFDFEFF00">

  func.func @main() {

    %c1_i32 = arith.constant 1 : i32

    %c0_i32 = arith.constant 0 : i32

    %0 = memref.get_global @my_constant : memref<16x16xi8>

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

          packet_id = 0,

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

          packet_id = 0,

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

          packet_id = 0,

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

          packet_id = 1,

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

          packet_id = 1,

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

        %25 = dfschedule.declare_kernel_config @kernelconfig0 {tile_configs = [{acquire_lock_id = 0 : i32, buffer_mode = 1 : i32, buffer_offset = 0 : i32, buffer_size = 64 : i32, dma_channel = 0 : i32, element_size = 1 : i32, flow_index = 0 : i32, num_buffers = 2 : i32, packet_id = 0 : i32, release_lock_id = 1 : i32, tile_index = 0 : i32}]}

        %26 = dfschedule.declare_kernel_config @kernelconfig1 {tile_configs = [{acquire_lock_id = 0 : i32, buffer_mode = 1 : i32, buffer_offset = 64 : i32, buffer_size = 64 : i32, dma_channel = 0 : i32, element_size = 1 : i32, flow_index = 0 : i32, num_buffers = 2 : i32, packet_id = 1 : i32, release_lock_id = 1 : i32, tile_index = 1 : i32}]}

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

        %c0_i32_0 = arith.constant 0 : i32

        %4 = dfschedule.buffer_view %subview {len = 128 : i64, offset = 0 : i64} : memref<8x16xi8, strided<[16, 1], offset: 128>> -> memref<8x16xi8, strided<[16, 1], offset: 128>>

        %5 = dfschedule.config.dma_bd(%4, %3, %c0_i32_0) {

          offset = 0,

          len = 128,

          enable_packet = true,

          packet_id = 0,

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

          packet_id = 0,

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

          packet_id = 0,

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

          packet_id = 1,

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

          packet_id = 1,

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

        %25 = dfschedule.declare_kernel_config @kernelconfig0 {tile_configs = [{acquire_lock_id = 0 : i32, buffer_mode = 1 : i32, buffer_offset = 0 : i32, buffer_size = 64 : i32, dma_channel = 0 : i32, element_size = 1 : i32, flow_index = 1 : i32, num_buffers = 2 : i32, packet_id = 0 : i32, release_lock_id = 1 : i32, tile_index = 0 : i32}]}

        %26 = dfschedule.declare_kernel_config @kernelconfig1 {tile_configs = [{acquire_lock_id = 0 : i32, buffer_mode = 1 : i32, buffer_offset = 64 : i32, buffer_size = 64 : i32, dma_channel = 0 : i32, element_size = 1 : i32, flow_index = 1 : i32, num_buffers = 2 : i32, packet_id = 1 : i32, release_lock_id = 1 : i32, tile_index = 1 : i32}]}

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

" , what I need is to remove the routingcreate and merge the shim receiving logic to merge the multiple shim recieve into one

Gemini said
To canonicalize this MLIR and merge multiple shim receiving operations into a single logical block, you need to lift the repetitive DMA configuration logic out of the routing.RoutingCreate regions.

In MLIR, the standard approach for this kind of "de-nesting" and "merging" involves a combination of Region Hoisting and Op Folding. Here is the expert advice on the best solution:

1. The Strategy: Hoisting and Merging
The goal is to move the dfschedule.config.dma_bd and dfschedule.config.create_io ops that target the same dfschedule.tile (the Shim tile at col = 2, row = 0) into a single initialization or configuration sequence.

Step A: Hoist Invariant Tile Declarations
Currently, %3 = dfschedule.declaretile {col = 2, row = 0} is declared inside every region.

Action: Move the Shim tile declaration to the top of the func.func. This allows all subsequent DMA ops to reference a single SSA value for that physical tile.

Step B: Merge Shim IO Configurations
Your Shim tile (Tile 2,0) is receiving data on multiple channels (Channel 0 in the first block, Channel 1 in the second).

Action: Instead of calling create_io inside separate routing regions, create a single sequence where you define the BDs (Buffer Descriptors) for both channels.

Optimization: If the memory being transferred is contiguous in the source %alloc, you can potentially merge the two 128-byte transfers into one larger transfer if the hardware/dialect allows, or simply chain the BDs using the next_bd attribute.

2. Recommended Canonical Form
To eliminate the routing.RoutingCreate and merge the logic, your IR should look like this:

MLIR
func.func @main() {
  %shim_tile = dfschedule.declaretile {col = 2 : i32, row = 0 : i32} : !dfschedule.tile
  %0 = memref.get_global @my_constant : memref<16x16xi8>
  %alloc = memref.alloc() : memref<16x16xi8>
  memref.copy %0, %alloc : memref<16x16xi8> to memref<16x16xi8>

  // 1. Unified Shim Configuration
  // Map the whole buffer or specific views once
  %full_view = dfschedule.buffer_view %alloc {len = 256 : i64, offset = 0 : i64} : memref<16x16xi8> -> memref<16x16xi8>
  
  // Configure BD for Channel 0
  %bd0 = dfschedule.config.dma_bd(%full_view, %shim_tile, %c0) { len = 128, ... }
  %io0 = dfschedule.config.create_io(%bd0, %shim_tile) { channel = 0, direction = "S2MM" }

  // Configure BD for Channel 1
  %bd1 = dfschedule.config.dma_bd(%full_view, %shim_tile, %c1) { len = 128, offset = 128, ... }
  %io1 = dfschedule.config.create_io(%bd1, %shim_tile) { channel = 1, direction = "S2MM" }

  // 2. Compute/Core Configurations
  // (Place your kernel group loading and core BD configs here)

  // 3. Unified Execution
  %ev0 = dfschedule.schedule.start_io(%io0, %id0)
  %ev1 = dfschedule.schedule.start_io(%io1, %id1)
  %ev_kernel = dfschedule.schedule.launch_kernel_group(...)
  
  dfschedule.schedule.wait(%ev0, %ev1, %ev_kernel)
}
3. Implementation Steps
To automate this, I recommend writing a MLIR Transformation Pass rather than just a simple rewrite pattern:

Analyze Dependencies: Identify all dfschedule.config.create_io ops that share the same !dfschedule.tile.

Topological Hoisting: Move declaretile ops to the entry block of the function.

Buffer View Normalization: If multiple subviews are used for the same alloc, try to compose them or reference the base memref with offsets in the dma_bd op to simplify the SSA graph.

Inline Region logic: Use the mlir::PatternRewriter::inlineRegionBefore method to move the ops out of routing.RoutingCreate and then delete the routing op.

Why this is the "Best" solution
You stopped this response
You said
AS A MLIR EXPERT please advise what is the best solution to cannolize my IR "module attributes {codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"]} {

  memref.global "private" constant @my_constant : memref<16x16xi8> = dense<"0x0102030405060708090A0B0C0D0E0F101112131415161718191A1B1C1D1E1F202122232425262728292A2B2C2D2E2F303132333435363738393A3B3C3D3E3F404142434445464748494A4B4C4D4E4F505152535455565758595A5B5C5D5E5F606162636465666768696A6B6C6D6E6F707172737475767778797A7B7C7D7E7F808182838485868788898A8B8C8D8E8F909192939495969798999A9B9C9D9E9FA0A1A2A3A4A5A6A7A8A9AAABACADAEAFB0B1B2B3B4B5B6B7B8B9BABBBCBDBEBFC0C1C2C3C4C5C6C7C8C9CACBCCCDCECFD0D1D2D3D4D5D6D7D8D9DADBDCDDDEDFE0E1E2E3E4E5E6E7E8E9EAEBECEDEEEFF0F1F2F3F4F5F6F7F8F9FAFBFCFDFEFF00">

  func.func @main() {

    %c1_i32 = arith.constant 1 : i32

    %c0_i32 = arith.constant 0 : i32

    %0 = memref.get_global @my_constant : memref<16x16xi8>

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

          packet_id = 0,

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

          packet_id = 0,

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

          packet_id = 0,

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

          packet_id = 1,

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

          packet_id = 1,

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

        %25 = dfschedule.declare_kernel_config @kernelconfig0 {tile_configs = [{acquire_lock_id = 0 : i32, buffer_mode = 1 : i32, buffer_offset = 0 : i32, buffer_size = 64 : i32, dma_channel = 0 : i32, element_size = 1 : i32, flow_index = 0 : i32, num_buffers = 2 : i32, packet_id = 0 : i32, release_lock_id = 1 : i32, tile_index = 0 : i32}]}

        %26 = dfschedule.declare_kernel_config @kernelconfig1 {tile_configs = [{acquire_lock_id = 0 : i32, buffer_mode = 1 : i32, buffer_offset = 64 : i32, buffer_size = 64 : i32, dma_channel = 0 : i32, element_size = 1 : i32, flow_index = 0 : i32, num_buffers = 2 : i32, packet_id = 1 : i32, release_lock_id = 1 : i32, tile_index = 1 : i32}]}

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

        %c0_i32_0 = arith.constant 0 : i32

        %4 = dfschedule.buffer_view %subview {len = 128 : i64, offset = 0 : i64} : memref<8x16xi8, strided<[16, 1], offset: 128>> -> memref<8x16xi8, strided<[16, 1], offset: 128>>

        %5 = dfschedule.config.dma_bd(%4, %3, %c0_i32_0) {

          offset = 0,

          len = 128,

          enable_packet = true,

          packet_id = 0,

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

          packet_id = 0,

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

          packet_id = 0,

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

          packet_id = 1,

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

          packet_id = 1,

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

        %25 = dfschedule.declare_kernel_config @kernelconfig0 {tile_configs = [{acquire_lock_id = 0 : i32, buffer_mode = 1 : i32, buffer_offset = 0 : i32, buffer_size = 64 : i32, dma_channel = 0 : i32, element_size = 1 : i32, flow_index = 1 : i32, num_buffers = 2 : i32, packet_id = 0 : i32, release_lock_id = 1 : i32, tile_index = 0 : i32}]}

        %26 = dfschedule.declare_kernel_config @kernelconfig1 {tile_configs = [{acquire_lock_id = 0 : i32, buffer_mode = 1 : i32, buffer_offset = 64 : i32, buffer_size = 64 : i32, dma_channel = 0 : i32, element_size = 1 : i32, flow_index = 1 : i32, num_buffers = 2 : i32, packet_id = 1 : i32, release_lock_id = 1 : i32, tile_index = 1 : i32}]}

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

" , what I need is to remove the routingcreate and merge the shim receiving logic to merge the multiple shim recieve into one

Gemini said
To canonicalize this IR effectively and merge the shim receiving logic, you should focus on Region Hoisting and SSA Value Normalization.

Currently, your IR is "fragmented" because each routing.RoutingCreate op essentially creates a local scope that hides redundant operations from global optimization passes. By flattening these regions and merging the redundant Shim tile configurations, you can significantly reduce the IR's complexity and the overhead of the final generated code.

The Canonical Solution: Global Hoisting and Unified IO
The best solution is to lift the static resource declarations (like tiles and constant descriptors) to the function entry block and consolidate the DMA configurations.

1. Hoist Resource Declarations
Move %3 = dfschedule.declaretile {col = 2, row = 0} out of the sub-regions. Since both routing blocks reference the same physical Shim tile, they should reference the same SSA value. This allows the compiler to see that the configuration is happening on a single hardware resource.

2. Consolidate Shim BDs and Channels
Instead of separate create_io calls spread across regions, you can define your Shim's behavior in a single block. Your current IR uses Channel 0 and Channel 1 on the same tile. You can merge these into a sequence of configurations that act on the base %alloc memref using different offsets.

Recommended "Canonical" IR Structure
Here is how the IR should look after removing the routing.RoutingCreate and merging the shim logic:

MLIR
func.func @main() {
  // --- 1. Global Resource Declarations ---
  %shim_tile = dfschedule.declaretile {col = 2 : i32, row = 0 : i32} : !dfschedule.tile
  %tile_r3 = dfschedule.declaretile {col = 0 : i32, row = 3 : i32} : !dfschedule.tile
  %tile_r4 = dfschedule.declaretile {col = 0 : i32, row = 4 : i32} : !dfschedule.tile
  // ... other tiles ...

  %0 = memref.get_global @my_constant : memref<16x16xi8>
  %alloc = memref.alloc() : memref<16x16xi8>
  memref.copy %0, %alloc : memref<16x16xi8> to memref<16x16xi8>

  // --- 2. Unified Shim Configuration ---
  // Create a single view or use offsets on the base memref
  %view = dfschedule.buffer_view %alloc {len = 256 : i64, offset = 0 : i64} : memref<16x16xi8> -> memref<16x16xi8>

  // Receive for Group 1 (Channel 0)
  %bd0 = dfschedule.config.dma_bd(%view, %shim_tile, %c0) { len = 128, offset = 0, ... }
  %io0 = dfschedule.config.create_io(%bd0, %shim_tile) { channel = 0, direction = "S2MM" }

  // Receive for Group 2 (Channel 1)
  %bd1 = dfschedule.config.dma_bd(%view, %shim_tile, %c1) { len = 128, offset = 128, ... }
  %io1 = dfschedule.config.create_io(%bd1, %shim_tile) { channel = 1, direction = "S2MM" }

  // --- 3. Merged Execution Logic ---
  // Instead of separate regions, we now have a linear schedule or a single scf.execute_region
  %ev0 = dfschedule.schedule.start_io(%io0, %shim_id)
  %ev1 = dfschedule.schedule.start_io(%io1, %shim_id)
  
  // Launch your kernel groups...
  %ev_k = dfschedule.schedule.launch_kernel_group(...)

  dfschedule.schedule.wait(%ev0, %ev1, %ev_k)
  
  memref.dealloc %alloc : memref<16x16xi8>
  return
}
Best Implementation Path
To achieve this programmatically, I recommend writing a Cleanup Pass using the following logic:

Tile Unification: Use a SymbolTable or a simple Map to track {col, row} pairs. Replace all duplicate declaretile ops with the first instance found in the function.

Region Inlining: Use mlir::PatternRewriter::inlineRegionBefore. This will "pull" the ops from inside routing.RoutingCreate into the main func.func body.

Dead Code Elimination (DCE): Once the ops are moved out, the routing.RoutingCreate ops will have no useful results and can be deleted.

Buffer Normalization: Identify memref.subview ops that point to the same base buffer. You can often merge these into a single larger view if the DMA engine supports strided access or simply adjust the offset attribute in the dma_bd ops.

Why this works
By merging the shim receives into one logical block, you allow the backend to potentially optimize the DMA descriptors (e.g., using a single multi-BD chain rather than two separate triggers). This significantly reduces the shim-to-core latency.

Would you like me to draft a C++ snippet for the MLIR rewrite pattern that performs this hoisting and merging?

You said
mlir::PatternRewriter::inlineRegionBefore此方法会将内部操作“拉入”routing.RoutingCreate主体中func.func。 啥意思

Gemini said
在 MLIR 的世界里，你可以把 routing.RoutingCreate 想象成一个**“嵌套的盒子”**，而 func.func 是你工作的大房间。

当你调用 inlineRegionBefore 时，本质上是在做**“拆盒取物”**的操作。

1. 核心含义：作用域扁平化 (Scope Flattening)
在 MLIR 中，Operation（操作）是层级嵌套的。你的原始 IR 结构如下：

func.func (大房间)

arith.constant

routing.RoutingCreate (盒子)

Region (盒子内部的空间)

Block

dfschedule.config.dma_bd (盒子里的工具 A)

dfschedule.config.create_io (盒子里的工具 B)

“拉入” (Inlining) 的意思就是：
你把“工具 A”和“工具 B”从盒子里拿出来，直接摆在“大房间”的地面上。操作执行完后，这些工具的“上级”不再是 routing.RoutingCreate，而是变成了 func.func。

2. 代码直观对比
Inlining 之前 (嵌套状态)
MLIR
func.func @main() {
  %c0 = arith.constant 0 : i32
  // 此时 dma_bd 在 RoutingCreate 的隔离区里
  %result = routing.RoutingCreate (scf_idx = %c0) {
    %bd = dfschedule.config.dma_bd ... 
    routing.yield %bd
  }
  return
}
Inlining 之后 (扁平状态)
MLIR
func.func @main() {
  %c0 = arith.constant 0 : i32
  // RoutingCreate 被“拆开”了，dma_bd 现在直接属于 func.func
  %bd = dfschedule.config.dma_bd ... 
  // 盒子被扔掉了，剩下的逻辑直接连在主体里
  return
}
3. C++ 层面发生了什么？
当你写 rewriter.inlineRegionBefore(sourceRegion, targetBlock, insertionPoint) 时：

移动 Block：它把 sourceRegion 里的所有 Block（块）直接移动到 targetBlock 的指定位置。

改变 Parent：原本这些 Op 的 getParentOp() 会返回 routing.RoutingCreate，现在会返回 func.func。

解除封装：这就解决了你之前提到的“想要移除 routing.RoutingCreate”的需求。一旦内部操作被拉出来，这个 routing 操作就变成了没有实际意义的空壳，你可以放心地用 rewriter.eraseOp(op) 把它删掉。

4. 为什么要这么做？
作为专家，我告诉你这样做最大的好处是打破优化壁垒：

全局可见性：在盒子里时，MLIR 的常规优化（如 CSE 消除重复操作、死代码删除）很难跨越区域（Region）边界去观察。

资源合并：正如你之前想合并多个 Shim 接收逻辑，只有把它们拉到同一个层级（同一个 Block 里），编译器才能发现：“嘿，这两个 DMA 配置的是同一个 Tile，我可以把它们合并！”

你想看一个具体的 PatternRewrite 类实现，演示如何写这段 C++ 代码吗？

You said
rewriter.inlineRegionBefore 举个例子

Gemini said
好的，我们直接看代码。假设我们要写一个 Canonicalization 模式，专门用来拆掉 routing.RoutingCreate 这个“盒子”。

在 MLIR 中，inlineRegionBefore 的操作通常伴随着结果替换（把盒子里 yield 出来的值，接头到盒子外面的 SSA 变量上）。

C++ 示例代码
这是一个典型的 OpRewritePattern 实现：

C++
#include "mlir/IR/PatternMatch.h"
#include "your/dialect/RoutingOps.h" // 假设你的 Dialect 定义在此

struct FlattenRoutingPattern : public mlir::OpRewritePattern<routing::RoutingCreateOp> {
    using OpRewritePattern<routing::RoutingCreateOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(routing::RoutingCreateOp op,
                                       mlir::PatternRewriter &rewriter) const override {
        // 1. 获取盒子内部的那个“空间” (Region)
        mlir::Region &sourceRegion = op.getRegion();

        // 2. 找到盒子所在的“房间位置” (Block)
        mlir::Block *currentBlock = op->getBlock();

        // 3. 执行“拆盒取物”：把盒子里的所有 Block 搬到盒子前面
        // 这个方法会把 sourceRegion 里的所有东西“拉入”到 currentBlock 所在的层级
        rewriter.inlineRegionBefore(sourceRegion, op->getBlock());

        // 4. 接头工作 (Handling Yield)
        // 盒子里通常有个 routing.yield 把值传出来。
        // 我们需要把原本指向 op 结果的地方，改为指向 yield 的操作数。
        mlir::Block &inlinedBlock = currentBlock->getPredecessor()->getNextNode(); // 简化处理
        auto yieldOp = mlir::cast<routing::RoutingYieldOp>(inlinedBlock.getTerminator());
        
        // 用 yield 的内容替换掉整个 RoutingCreateOp 的返回值
        rewriter.replaceOp(op, yieldOp.getOperands());

        // 5. 最后清理：把那个已经没用的 yield 操作删掉（因为它现在直接暴露在 func 里了）
        rewriter.eraseOp(yieldOp);

        return mlir::success();
    }
};
这段代码执行前后的 IR 变化
操作前 (Before)
你的 IR 像是在“套娃”：

MLIR
func.func @main() {
  %0 = routing.RoutingCreate ... {
    ^bb0:
      %1 = dfschedule.config.dma_bd ... // 此时这行代码的“父亲”是 RoutingCreate
      routing.yield %1
  }
  // 其他代码...
}
执行 inlineRegionBefore 后 (After)
结构变扁平了，dma_bd 被“提拔”了：

MLIR
func.func @main() {
  // 盒子被拆开了，bb0 变成了 func.func 的一部分
  %1 = dfschedule.config.dma_bd ... // 现在这行代码的“父亲”直接就是 func.func
  
  // 原本的 %0 被 %1 直接替换掉了
  // routing.yield 被删除了
  // 其他代码...
}
关键点拨
Block 的移动：inlineRegionBefore 并不是一行行复制 Op，而是直接移动整个 Block 的指针。这非常高效。

控制流转换：如果你的盒子内部有多个 Block，拉出来后它们会直接并入外层的控制流。

SSA 连续性：在拉出来之后，你需要确保原本依赖 RoutingCreate 返回值的后续操作，现在能正确地拿到盒子里算出来的那个值。这就是 rewriter.replaceOp 的作用。

专家提示
当你把这些操作都拉到 func.func 这一层后，你就可以编写另一个 Merge Pattern。那个 Pattern 会扫描同一个 Block 里的所有 dfschedule.config.create_io，如果发现它们的 tile 属性相同，就利用 rewriter 把它们合并成一个。

下一步，你需要我帮你写那个“合并两个相同 Tile 的 Shim 接收逻辑”的 C++ 逻辑吗？

You said
怎么用 FlattenRoutingPattern

Gemini said
要使用 FlattenRoutingPattern，你不能直接“调用”它。在 MLIR 中，模式（Pattern）需要被放入一个**“集合 (RewritePatternSet)”中，然后通过一个“通行证 (Pass)”**来驱动执行。

你可以按照以下三个步骤来部署它：

第一步：创建一个 Pass 来承载模式
在 MLIR 中，Pass 是执行变换的最小单位。你需要创建一个 OperationPass（通常针对 func::FuncOp 或 ModuleOp）。

C++
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// 定义你的 Pass
struct FlattenRoutingPass : public mlir::PassWrapper<FlattenRoutingPass, OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FlattenRoutingPass)

  void runOnOperation() override {
    auto func = getOperation();
    mlir::RewritePatternSet patterns(&getContext());
    
    // 将我们之前定义的模式添加到集合中
    patterns.add<FlattenRoutingPattern>(&getContext());

    // 使用 Greedy 驱动器应用这些模式
    // 它会不断扫描 IR，直到没有可以被 Flatten 的 RoutingCreate 为止
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }

  llvm::StringRef getArgument() const final { return "flatten-routing"; }
  llvm::StringRef getDescription() const final { return "Remove routing.RoutingCreate and hoist ops."; }
};
第二步：将 Pass 注册到流水线
有了 Pass 之后，你需要告诉编译器什么时候运行它。通常在你的 main 函数或者 PassPipeline 中注册：

C++
void registerMyPasses() {
    mlir::PassRegistration<FlattenRoutingPass>();
}

// 或者直接添加到 PassManager
mlir::PassManager pm(&context);
pm.addPass(std::make_unique<FlattenRoutingPass>());
第三步：在命令行调用
如果你已经把 Pass 注册到了 mlir-opt 工具中，你就可以直接对你的 IR 文件运行它：

Bash
# 运行你定义的 pass
my-mlir-opt --flatten-routing input.mlir -o output.mlir
专家建议：如何实现“合并”逻辑？
当你用 FlattenRoutingPattern 把所有操作“拉”到 func.func 之后，你会发现 IR 中存在多个针对同一 Tile 的 dfschedule.config.create_io。

为了实现你最初想要的**“合并多个 Shim 接收逻辑”**，你应该再写一个模式：MergeShimIOOpsPattern。

它的逻辑应该是：

匹配 dfschedule.config.create_io。

检查它的 tile 是否是 Shim Tile（例如 2,0）。

查找同一 Block 中是否已有另一个配置了相同 Tile 的 create_io。

如果存在，将两者的 BD 链（通过 next_bd 属性）连接起来，并删除其中一个冗余的 create_io。

总结
Flatten (拆盒)：用 inlineRegionBefore 把逻辑从 RoutingCreate 拿出来。

DCE (清理)：删除空的 RoutingCreate。

Merge (合并)：在同一层级下，通过另一个 Pattern 识别并合并相同资源的配置。

你想让我帮你写出那个“识别并合并相同 Tile 配置”的 Merge Pattern 伪代码吗？

You said
GREEDY PATTERN MATCH 在百万行代码境况下会出问题，会有什么问题 怎么解决

Gemini said
在处理百万行级别的 MLIR 代码时，applyPatternsAndFoldGreedily 确实会从“编译器的好帮手”变成“编译效率的噩梦”。作为专家，我必须提醒你，贪婪模式在超大规模 IR 面前主要面临时间复杂度、内存抖动和收敛稳定性三大挑战。

1. 百万行代码下的核心问题
🌀 时间复杂度爆炸 (Non-linear Complexity)
贪婪驱动器（Greedy Driver）的工作原理是维护一个工作列表 (Worklist)。每当一个 Op 被修改，驱动器会将受影响的邻近 Op 重新放回列表。

重复扫描：在百万行规模下，如果 Pattern 编写不当（例如频繁触发局部修改），一个 Op 可能被重复访问数千次。

收敛成本：贪婪匹配必须运行到不动点 (Fixed Point) 才能停止。对于超大 IR，达到这个稳定状态的步数可能呈指数级增长。

💾 内存与缓存压力
工作列表膨胀：百万行代码意味着工作列表可能瞬间塞入数十万个 Op 指针，消耗大量内存。

缓存失效 (Cache Miss)：贪婪驱动器在不同 Block 甚至不同 Region 之间来回跳跃，会导致 CPU 缓存命中率极低。

🔄 振荡与死循环 (Infinite Loops)
在小规模 IR 中，Pattern A 变 Pattern B，Pattern B 变 Pattern A 的逻辑错误很容易发现。但在百万行 IR 中，这种“循环引用”可能潜伏在复杂的逻辑分支里，导致编译器直接挂起。

2. 专家级解决方案
针对大规模 IR，我们通常会放弃或限制“贪婪模式”，转而采用更具确定性的方法。

🚀 方案 A：使用显式的 walk 代替贪婪驱动器
这是处理超大代码量时的首选。与其让驱动器“乱撞”，不如你自己控制遍历路径。

逻辑：使用 op->walk<WalkOrder::PreOrder>([](OpType op) { ... })。

优势：每个 Op 只被访问一次，复杂度严格控制在 O(N)。对于你之前的 FlattenRoutingPattern，由于它只需要单次提升，完全没必要用贪婪模式。

🛠️ 方案 B：精细化配置 GreedyRewriteConfig
如果你必须使用贪婪模式，可以通过配置项来给它戴上“枷锁”：

C++
GreedyRewriteConfig config;
config.useTopDownTraversal = true; // 从上往下走，通常更快收敛
config.maxIterations = 10;          // 强制限制迭代次数，防止死循环
config.strictMode = GreedyRewriteStrictness::ExistingOps; // 仅处理现有 Op，不处理新生成的
📦 方案 C：并行化处理 (Pass Manager)
不要试图在一个 ModulePass 里贪婪处理所有东西。

策略：将变换下沉到 FunctionPass 级别。

原理：MLIR 的 PassManager 会在多个线程上并行运行函数级的 Pass。这样每个线程处理的 IR 规模被限制在万行级别，且能充分利用多核 CPU。

🔍 方案 D：基于符号表 (SymbolTable) 的定向跳转
既然你想合并针对同一 Tile 的 Shim 逻辑，不要让驱动器去遍历所有 Op。

优化：在 Pass 开始时构建一个 Map<Tile, SmallVector<Op>> 的索引。

操作：直接在这个索引上操作，合并逻辑的时间复杂度将由 O(N 
2
 ) 降至接近 O(N)。

3. 性能对比参考
维度	贪婪模式 (Default)	手写 Walk / 定向优化
百万行耗时	可能几分钟甚至超时	秒级完成
内存占用	随 Worklist 波动剧烈	稳定且线性
收敛性	依赖 Pattern 质量	100% 确定
适用场景	复杂的代数化简 (Arith)	结构化变换 (Hoisting/Merge)
建议
对于你目前的 RoutingCreate 扁平化和 Shim 合并任务，强烈建议使用方案 A (Walk) 或方案 D (索引化合并)。这能保证你的编译器在处理大规模神经网络拓扑时依然健步如飞。

你想让我演示一下如何把之前的 Pattern 改造成高效的 walk 形式吗？

You said
这种情况下 (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns)))) {

signalPassFailure();

} 应当怎么改

Gemini said
针对百万行级别的代码，直接替换 applyPatternsAndFoldGreedily 的最佳方案是**“由面转点”**。

与其让编译器在大海捞针，不如进行**“外科手术式”的定向遍历**。以下是两种改进方案，按推荐程度排序：

方案一：使用 walk 进行 O(N) 定向遍历（最推荐）
对于像“提升操作”或“删除特定盒子”这种结构化变换，完全不需要模式匹配。通过 walk，你可以保证每个 Op 只被访问一次，复杂度是严格线性的。

修改后的代码示例：
C++
void runOnOperation() override {
    auto func = getOperation();
    mlir::IRRewriter rewriter(&getContext());

    // 1. 定向寻找所有的 RoutingCreateOp，使用 PreOrder 保证从外向里拆
    func.walk<mlir::WalkOrder::PreOrder>([&](routing::RoutingCreateOp op) {
        // 执行你之前的 Flatten 逻辑
        mlir::Region &sourceRegion = op.getRegion();
        
        // 获取内部的 yield 信息（假设你定义了 RoutingYieldOp）
        auto yieldOp = mlir::cast<routing::RoutingYieldOp>(sourceRegion.front().getTerminator());
        
        // 记录 yield 的值
        mlir::ValueRange yieldedValues = yieldOp.getOperands();

        // [核心修改]：直接手动 Inline，不再依赖 Greedy Driver
        rewriter.inlineBlockBefore(&sourceRegion.front(), op);

        // 替换原本 RoutingCreate 的返回值
        rewriter.replaceOp(op, yieldedValues);
        
        // 删除已经出来的 yield 占位符
        rewriter.eraseOp(yieldOp);
    });
}
为什么这能解决问题？

零重复扫描：它不会因为一个 Op 的改变就去重新检查周围的一万个 Op。

确定性：代码执行流程清晰，不会出现不收敛的情况。

方案二：如果你必须用 Pattern，请限制“贪婪”程度
如果你有一些非常复杂的折叠逻辑必须用 Pattern，请务必配置 GreedyRewriteConfig。

修改后的代码示例：
C++
void runOnOperation() override {
    auto func = getOperation();
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<FlattenRoutingPattern>(&getContext());

    mlir::GreedyRewriteConfig config;
    config.useTopDownTraversal = true; // 从上往下走，减少重新入队的次数
    config.maxIterations = 2;          // 强制限制迭代次数。如果 2 次都没搞定，说明逻辑有问题
    config.strictMode = mlir::GreedyRewriteStrictness::ExistingOps; // 只处理已存在的，不处理新生成的

    // 显式传入 config
    (void)mlir::applyPatternsAndFoldGreedily(func, std::move(patterns), config);
}
针对你“合并 Shim 接收”目标的终极建议
当你处理百万行 IR 且需要合并相同 Tile 的配置时，最高效的做法是**“索引化 (Indexing)”**：

第一步 (Flatten)：用上面的“方案一 (Walk)”把所有 RoutingCreate 拆掉。

第二步 (Collect)：在同一个 walk 中，用一个 DenseMap<TileAttr, SmallVector<CreateIOOp>> 把属于同一个 Tile 的配置存起来。

第三步 (Merge)：只对这个 Map 里的 SmallVector 进行循环合并。

这样你处理百万行 IR 的速度会从**“分钟级”直接降到“毫秒级”**。

💡 避坑指南
在大规模 IR 变换中，绝对不要在 applyPatternsAndFoldGreedily 运行过程中去修改 SymbolTable（比如重命名全局变量），这会导致驱动器频繁失效。

你现在的代码逻辑里，除了拆掉盒子，还有需要对内部数据做复杂代数计算（如 1+1=2）的部分吗？如果没有，就果断放弃 Greedy 模式吧。

You said
缺点是什么是不是有点HARDCODE

Gemini said
你的感觉非常敏锐。这确实更“硬编码”（Hardcoded），或者用编译器的话说，它从“声明式（Declarative）”退化到了“过程式（Procedural）”。

在 MLIR 专家眼中，这种做法是用灵活性换取极致的性能。以下是这种“手术刀”式处理方案的显著缺点，以及为什么你会觉得它“硬”：

1. 为什么说它有点“硬” (Disadvantages)
🌀 缺乏“连锁反应” (Missing Recursive Folding)
Greedy Driver 最强大的地方在于它能处理“多米诺骨牌”效应。

例子：如果你拆掉了一个盒子，导致里面的 %1 + %2 变成了 1 + 1，贪婪模式会自动帮你把 1 + 1 变成 2。

Walk 模式：它只管拆盒子。拆完之后，里面的 1 + 1 还是 1 + 1，除非你再写一个 walk 去处理加法。你必须手动预测并规划所有的后续动作。

🧩 组合性极差 (Poor Composability)
模式匹配：你可以通过 patterns.add<A, B, C>() 轻松地把三个人的优化逻辑组合在一起。

Walk 模式：如果你想在拆盒子的同时做其他优化，你必须把所有逻辑塞进那个巨大的 Lambda 表达式里。代码会迅速变得像“意大利面条”一样混乱，难以维护。

🛡️ 脆弱性与安全性 (Fragility)
模式匹配：由 PatternRewriter 托管，它会帮你处理很多底层的指针更新和合法性检查。

Walk 模式：你是在直接操作内存块（Block）。如果你不小心删掉了一个还有人引用的 Op，或者忘记处理 Terminator（结束符），编译器会直接崩溃（Segfault），而不会给你友好的错误提示。

2. 深度对比：优雅 vs. 效率
特性	贪婪模式 (Patterns)	手动遍历 (Walk)
思维模型	“看到 A 就变成 B”	“我要从第一行走到最后一行，遇到盒就拆”
自动折叠	强 (自动处理关联修改)	无 (需手动触发)
百万行表现	容易卡死或极慢	极快 (线性时间)
代码量	简洁、高度抽象	冗长、包含大量底层操作
硬编码感	低 (规则引擎)	高 (过程脚本)
3. 专家给出的中庸之道：Dialect Conversion
如果你觉得 Greedy 太慢，又觉得 Walk 太硬，MLIR 还有一个“终极武器”：Dialect Conversion (合法化框架)。

它是为了解决大规模转换而设计的。你可以设置一个 ConversionTarget，告诉它哪些 Op 是“非法”的（比如 RoutingCreate）。框架会扫一遍 IR，只针对非法 Op 进行转换。

优点：比贪婪模式快，因为它目标明确；比 Walk 优雅，因为它依然使用 Pattern 编写。

适用场景：当你需要大规模改变 IR 结构，但又不想写那种极其硬核的 inlineBlock 逻辑时。

总结建议
如果你现在处理的是百万行级别的 IR：

初期开发：先用 Greedy 验证逻辑是否正确，哪怕它运行要一分钟。

生产环境优化：如果速度不可接受，针对那些结构简单但数量巨大的操作（如你的 RoutingCreate 提升），改用 Walk；而对于复杂的计算优化，保留 Greedy 但限制其作用范围。

对于你那个合并 Shim 逻辑的需求，目前你是打算在同一个 C++ Pass 里完成，还是想分几个步骤（Passes）流水线作业？

You said
mlir::WalkOrder::PreOrder 啥意思

Gemini said
简单来说，mlir::WalkOrder::PreOrder 决定了你在遍历 MLIR 这种嵌套树状结构时的“访问顺序”。

在 MLIR 中，IR 是层层嵌套的：Module -> Function -> Block -> Operation -> Region -> Block -> Operation。

1. 核心定义：先父后子
PreOrder (先序遍历)：先访问父操作 (Parent Op)，再进入它内部的 Region/Block 访问里面的子操作 (Child Ops)。

PostOrder (后序遍历 - 默认值)：先访问完内部所有的子操作，最后才回到父操作本身。

2. 为什么在“拆盒子”时必须用 PreOrder？
在你之前的场景（删除 routing.RoutingCreate）中，PreOrder 是唯一的正确选择。

场景逻辑：你想把 RoutingCreate 里的东西拿出来，然后把这个“壳子”删掉。

使用 PreOrder 的流程：

遍历器走到 RoutingCreate。

触发你的函数：你发现这是一个盒子，于是把里面的内容“提拔”到外层。

此时内部的东西已经处理完了。

如果使用 PostOrder (默认)：

遍历器会先钻进盒子深处。

去优化盒子里的每一个小操作（比如里面的 arith.constant）。

全部处理完后，最后才跳出来处理 RoutingCreate。

风险：如果盒子里有嵌套的盒子，PostOrder 会让你从最深处开始拆，这在逻辑上非常低效且容易出错。

3. 代码行为对比
假设你有这样的 IR：

MLIR
%0 = "outer_op"() ({
  %1 = "inner_op"() : () -> i32
}) : () -> i32
遍历顺序	访问序列	你的逻辑触发点
PreOrder	1. outer_op -> 2. inner_op	你先看到大盒子，直接把它拆了，省心。
PostOrder	1. inner_op -> 2. outer_op	你先盯着零件看，最后才看到大盒子。
4. 关键 API 差异
在 C++ 中，你可以显式指定这个顺序：

C++
// 默认是 PostOrder，先处理子节点
op->walk([](MyOp op) { ... }); 

// 显式指定 PreOrder，先处理父节点
op->walk<mlir::WalkOrder::PreOrder>([](MyOp op) { ... }); 
5. 潜在陷阱
使用 PreOrder 时要特别注意：不要在遍历时随意删除当前节点以外的其他节点。
因为 PreOrder 接下来还要去访问子节点，如果你在处理父节点时直接把整个 Region 删了，而遍历器还打算往里走，就会导致空指针崩溃。

专家提示：如果你在 PreOrder 步中删除了当前 Op 或其 Region，必须返回 WalkResult::skip()，告诉遍历器：“我已经把这里拆了，别往里走了”。

既然你已经理解了遍历顺序，需要我帮你写一个结合 WalkOrder::PreOrder 和 WalkResult::skip() 的安全拆盒逻辑吗？

You said
什么情况下 用POSTORDER 什么时候用 PREORDER

Gemini said
在编译器开发中，选择遍历顺序就像是**“拆房子”和“组装零件”**的区别。选错了顺序，代码不仅效率低，还容易崩溃。

作为 MLIR 专家，我帮你总结了一套生存法则：

1. 什么时候用 PREORDER (Top-Down)？
核心逻辑：父操作决定了子操作的命运。

如果你要做的事情是**“结构化”的修改，或者你关心的是“上下文”**，请选 PreOrder。

提升 (Hoisting) / 扁平化 (Flattening)：比如你之前的任务，把 routing.RoutingCreate 里的东西拿出来。你必须先看到“盒子”，才能决定把里面的东西提出来。

作用域初始化 (Scope Initialization)：如果你在处理嵌套的作用域，且子节点依赖父节点定义的符号或属性。

删除整个区域 (Deletion)：如果你想根据父节点的某个属性直接删掉整个 Region，先看父节点可以让你直接跳过所有子节点的无用遍历。

⚠️ 注意：在 PreOrder 中删除当前节点后，一定要返回 WalkResult::skip()，否则遍历器会尝试进入一个已经不存在的区域，直接导致 Segfault。

2. 什么时候用 POSTORDER (Bottom-Up)？
核心逻辑：父操作的结果取决于子操作。

这是 MLIR 的默认顺序，也是大多数局部优化（Canonicalization）的首选。

算术折叠 (Arithmetic Folding)：

例子：add(mul(1, 2), 3)。

你必须先处理内部的 mul(1, 2) -> 2，父节点 add 才能变成 add(2, 3) -> 5。

死代码消除 (DCE)：只有当你知道一个操作的所有“用户”（Users，即子节点或后续节点）都不再需要它时，你才能安全地删除它。

类型推导 (Type Inference)：父操作的返回类型通常是由操作数（子节点提供的值）决定的。

3. 专家级对比表
特性	PREORDER (先序)	POSTORDER (后序)
遍历路径	根 → 叶	叶 → 根
直观比喻	拆除施工：先拆外墙，再管装修。	流水线：先做零件，再总装。
主要用途	结构变换、Inline、提升、权限下发。	计算优化、常量折叠、DCE、类型推导。
对 IR 修改的影响	适合删掉整个 Region 或移动 Op。	适合将复杂的 Op 树简化成一个常量。
性能优势	在大规模 IR 中，可以通过 skip() 避开大量节点。	能够保证在处理当前 Op 时，其输入已是最简形式。
4. 总结：该选哪个？
一个简单的判断标准：“我需不需要先看到大背景？”

如果你的逻辑是“因为这个盒子是 Routing 类型，所以我要把里面的逻辑……” → PREORDER。

如果你的逻辑是“如果两个加数都是常量，我就求和……” → POSTORDER。

💡 进阶技巧：并行与跳过
在处理百万行代码时，如果你用 PreOrder 配合索引，你甚至可以完全不走某些 Block。比如你的 Shim Tile 逻辑，你可以先搜寻所有的 Shim Tile 声明（PreOrder），然后直接“跳”到它们的关联操作上，这比漫无目的地走一遍全 IR 要快得多。

既然你已经掌握了顺序逻辑，想让我帮你写一个带 WalkResult::skip() 的安全拆盒（Flatten）函数模版吗？它可以防止你在百万行代码遍历中意外崩溃。

You said
THIS IS MY CUrrent canonicalize logic , please let me know the problem "/******************************************************************************

* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

* SPDX-License-Identifier: Apache-2.0

******************************************************************************/



#include "passschedulecanonicalize.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mlir/IR/Builders.h"

#include "mlir/IR/BuiltinDialect.h"

#include "mlir/IR/IRMapping.h"

#include "llvm/ADT/DenseMap.h"

#include "llvm/ADT/DenseSet.h"

#include "llvm/ADT/SmallVector.h"

#include "llvm/Support/raw_ostream.h"

#include "dfschedulemanager.h"

#include <map>

#include <set>

#include <vector>

#include <iostream>



using namespace mlir;

using namespace dfschedule;



namespace {



// Key for tile identification: (col, row)

using TileKey = std::pair<int64_t, int64_t>;



// Structure to hold collected schedule information per tile

struct TileScheduleInfo {

    TileKey key;

    SmallVector<Value> tileValues;           // All tile values for this (col, row)

    SmallVector<Operation*> packetOps;       // Packet ops targeting this tile

    SmallVector<SymbolRefAttr> packetSymbols; // Packet symbols for this tile (OLD)

    SmallVector<DictionaryAttr> configDicts;  // Config dictionaries for this tile (NEW)

    SmallVector<SymbolRefAttr> computeKernelArgs; // Compute args for this tile

    bool isShimTile = false;

    bool isCoreTile = false;

};



// Structure to hold shim tile DMA info

struct ShimDmaInfo {

    TileKey key;

    Value tileValue;

    SmallVector<Operation*> dmaBdOps;

    SmallVector<Operation*> createIoOps;

    SmallVector<Value> bdHandles;

    SmallVector<Value> ioHandles;

};



// Structure to hold tensor slice parameters

struct SliceParams {

    size_t index;  // Unique index for this slice

    RankedTensorType sourceType;

    SmallVector<int64_t, 4> offsets;

    SmallVector<int64_t, 4> sizes;

    SmallVector<int64_t, 4> strides;

    RankedTensorType resultType;

    size_t partitionIndex;  // Index of the partitiontensor this slice ultimately comes from

    int64_t parentSliceIndex;  // Index of parent slice if nested, -1 if directly from partition

    bool isFromPartition;  // true if immediate source is partitiontensor, false if from another slice

};



// Structure to hold partition tensor parameters

struct PartitionParams {

    size_t index;  // Unique index to distinguish partitions with same params but different flows

    RankedTensorType tensorType;

    int64_t splitnum;

    int64_t splitdim;

    std::string hw_axis_owner;

    std::string replicate_on;

    std::string single_tile_owner;

};



// Structure to hold operation with parent information

struct OpWithParent {

    Operation* op;

    Operation* parent;  // Parent operation (can be nullptr if top-level)

    bool isInDSKernelReceiver;  // true if parent is dfschedule.dskernel_receiver

    

    OpWithParent(Operation* operation, Operation* parentOp = nullptr)

        : op(operation), parent(parentOp) {

        // Check if parent is dskernel_receiver

        isInDSKernelReceiver = false;

        if (parentOp && parentOp->getName().getStringRef() == "dfschedule.dskernel_receiver") {

            isInDSKernelReceiver = true;

        }

    }

};



// Collected module-level schedule info

struct ModuleScheduleInfo {

    // Map from (col, row) to tile info

    std::map<TileKey, TileScheduleInfo> coreTiles;

    std::map<TileKey, ShimDmaInfo> shimTiles;

    

    // All collected operations with parent information

    SmallVector<OpWithParent> declareTensorOps;

    SmallVector<OpWithParent> declareTileOps;

    SmallVector<OpWithParent> configDmaBdOps;

    SmallVector<OpWithParent> configCreateIoOps;

    SmallVector<OpWithParent> packetOps;

    SmallVector<OpWithParent> loadKernelGroupOps;

    SmallVector<OpWithParent> launchKernelGroupOps;

    SmallVector<OpWithParent> getBdIdOps;

    SmallVector<OpWithParent> startIoOps;

    SmallVector<OpWithParent> scheduleWaitOps;

    SmallVector<OpWithParent> dskernelReceiverOps;

    

    // Operations to move from func.func main (with parent information)

    SmallVector<OpWithParent> tensorEmptyOps;

    SmallVector<OpWithParent> extractSliceOps;

    SmallVector<OpWithParent> partitionTensorOps;

    SmallVector<OpWithParent> executeRegionOps;

    SmallVector<OpWithParent> routingCreateOps;

    SmallVector<OpWithParent> declareDataOps;        // dfscheblueprint.declare_data

    SmallVector<OpWithParent> topLevelConstantOps;   // arith.constant at top level of main

    

    // Unique source tensor types (deduplicated)

    SmallVector<RankedTensorType> sourceTensorTypes;

    

    // Map from tensor type string key to its init tensor Value

    std::map<std::string, Value> initTensorMap;

    

    // Unique partition params (deduplicated)

    SmallVector<PartitionParams> uniquePartitionParams;

    

    // Unique extract_slice params (deduplicated)

    SmallVector<SliceParams> uniqueSliceParams;

    

    // Events to wait for

    SmallVector<Value> allEvents;

    

    // Kernel info

    StringRef kernelName = "dskernel_receiver";

    RankedTensorType kernelTensorType;

    int64_t bufferLen = 0;

    uint32_t basePacketId = 0;

    

    // Track how many packet streams each tile needs

    std::map<TileKey, int> tilePacketCount;

};



// Helper: Extract (col, row) from DeclareTileOp

static TileKey getTileKey(dfschedule::DeclareTileOp op) {

    return {op.getCol(), op.getRow()};

}



// Helper: Check if a tile is a shim tile (row == 0)

static bool isShimTile(TileKey key) {

    return key.second == 0;

}



// Helper: Get the parent operation (walks up the region hierarchy)

static Operation* getParentOp(Operation *op) {

    if (!op) return nullptr;

    Region *region = op->getParentRegion();

    if (!region) return nullptr;

    return region->getParentOp();

}



// Collect all dfschedule operations from the module

static void collectScheduleOps(ModuleOp moduleOp, ModuleScheduleInfo &info) {

    moduleOp.walk([&](Operation *op) {

        // Get parent operation

        Operation *parentOp = getParentOp(op);



        // DeclareTensorOp removed from dialect (tensor path deprecated)

        if (auto declareTile = dyn_cast<dfschedule::DeclareTileOp>(op)) {

            info.declareTileOps.push_back(OpWithParent(op, parentOp));

            TileKey key = getTileKey(declareTile);

            

            if (isShimTile(key)) {

                if (info.shimTiles.find(key) == info.shimTiles.end()) {

                    info.shimTiles[key] = ShimDmaInfo{key, declareTile.getTile(), {}, {}, {}, {}};

                }

            } else {

                if (info.coreTiles.find(key) == info.coreTiles.end()) {

                    info.coreTiles[key] = TileScheduleInfo{key, {}, {}, {}, {}, {}, false, true};

                }

                info.coreTiles[key].tileValues.push_back(declareTile.getTile());

            }

        } else if (auto configDmaBd = dyn_cast<dfschedule::ConfigDmaBdOp>(op)) {

            info.configDmaBdOps.push_back(OpWithParent(op, parentOp));

        } else if (auto createIo = dyn_cast<dfschedule::ConfigCreateIoOp>(op)) {

            info.configCreateIoOps.push_back(OpWithParent(op, parentOp));

        } else if (auto packet = dyn_cast<dfschedule::PacketOp>(op)) {

            info.packetOps.push_back(OpWithParent(op, parentOp));

        } else if (auto loadKernel = dyn_cast<dfschedule::LoadKernelGroupOp>(op)) {

            info.loadKernelGroupOps.push_back(OpWithParent(op, parentOp));

        } else if (auto launchKernel = dyn_cast<dfschedule::LaunchKernelGroupOp>(op)) {

            info.launchKernelGroupOps.push_back(OpWithParent(op, parentOp));

            info.allEvents.push_back(launchKernel.getEvent());

        } else if (auto getBdId = dyn_cast<dfschedule::GetBdIdOp>(op)) {

            info.getBdIdOps.push_back(OpWithParent(op, parentOp));

        } else if (auto startIo = dyn_cast<dfschedule::StartIoOp>(op)) {

            info.startIoOps.push_back(OpWithParent(op, parentOp));

            info.allEvents.push_back(startIo.getEvent());

        } else if (auto wait = dyn_cast<dfschedule::ScheduleWaitOp>(op)) {

            info.scheduleWaitOps.push_back(OpWithParent(op, parentOp));

        } else if (auto receiver = dyn_cast<dfschedule::DSKernelReceiverOp>(op)) {

            info.dskernelReceiverOps.push_back(OpWithParent(op, parentOp));

        } else if (auto emptyOp = dyn_cast<tensor::EmptyOp>(op)) {

            // Collect tensor.empty ops

            info.tensorEmptyOps.push_back(OpWithParent(op, parentOp));

            auto tensorType = cast<RankedTensorType>(emptyOp.getType());

            // Track unique tensor types

            bool found = false;

            for (auto &existingType : info.sourceTensorTypes) {

                if (existingType == tensorType) {

                    found = true;

                    break;

                }

            }

            if (!found) {

                info.sourceTensorTypes.push_back(tensorType);

            }

        } else if (auto extractSlice = dyn_cast<tensor::ExtractSliceOp>(op)) {

            // Collect tensor.extract_slice ops - NO deduplication!

            // Each slice is unique even if it has same offsets/sizes

            info.extractSliceOps.push_back(OpWithParent(op, parentOp));

            

            // Create slice params

            SliceParams params;

            params.index = info.uniqueSliceParams.size();  // Unique index

            params.sourceType = cast<RankedTensorType>(extractSlice.getSource().getType());

            for (auto o : extractSlice.getStaticOffsets()) params.offsets.push_back(o);

            for (auto s : extractSlice.getStaticSizes()) params.sizes.push_back(s);

            for (auto s : extractSlice.getStaticStrides()) params.strides.push_back(s);

            params.resultType = cast<RankedTensorType>(extractSlice.getType());

            params.partitionIndex = 0;

            params.parentSliceIndex = -1;  // Default: no parent slice

            params.isFromPartition = false;

            

            // Check immediate source

            Value source = extractSlice.getSource();

            if (Operation *defOp = source.getDefiningOp()) {

                if (defOp->getName().getStringRef() == "routing.partitiontensor") {

                    // Direct source is partitiontensor

                    params.isFromPartition = true;

                    for (size_t i = 0; i < info.partitionTensorOps.size(); ++i) {

                        if (info.partitionTensorOps[i].op == defOp) {

                            params.partitionIndex = i;

                            break;

                        }

                    }

                } else if (auto parentSlice = dyn_cast<tensor::ExtractSliceOp>(defOp)) {

                    // Direct source is another slice - find its index

                    params.isFromPartition = false;

                    for (size_t i = 0; i < info.extractSliceOps.size(); ++i) {

                        if (info.extractSliceOps[i].op == defOp) {

                            params.parentSliceIndex = static_cast<int64_t>(i);

                            break;

                        }

                    }

                    

                    // Walk up to find the ultimate partitiontensor

                    Value walkSource = parentSlice.getSource();

                    while (Operation *walkDefOp = walkSource.getDefiningOp()) {

                        if (walkDefOp->getName().getStringRef() == "routing.partitiontensor") {

                            for (size_t i = 0; i < info.partitionTensorOps.size(); ++i) {

                                if (info.partitionTensorOps[i].op == walkDefOp) {

                                    params.partitionIndex = i;

                                    break;

                                }

                            }

                            break;

                        } else if (auto grandParentSlice = dyn_cast<tensor::ExtractSliceOp>(walkDefOp)) {

                            walkSource = grandParentSlice.getSource();

                        } else {

                            break;

                        }

                    }

                }

            }

            

            // Always add - no deduplication for slices

            info.uniqueSliceParams.push_back(params);

        } else if (auto execRegion = dyn_cast<scf::ExecuteRegionOp>(op)) {

            // Collect scf.execute_region ops

            info.executeRegionOps.push_back(OpWithParent(op, parentOp));

        }



        // Check by operation name for routing dialect ops

        if (op->getName().getStringRef() == "routing.partitiontensor") {

            info.partitionTensorOps.push_back(OpWithParent(op, parentOp));



            // Extract partition parameters - NO deduplication here!

            // Each execute_region has its own partitiontensor representing different data flows

            // (e.g., producer flow vs consumer flow may have same partition params but different purposes)

            PartitionParams params;

            params.index = info.uniquePartitionParams.size();  // Unique index for each partition

            if (op->getNumResults() > 0) {

                params.tensorType = cast<RankedTensorType>(op->getResult(0).getType());

            }

            if (auto attr = op->getAttrOfType<IntegerAttr>("splitnum")) {

                params.splitnum = attr.getInt();

            }

            if (auto attr = op->getAttrOfType<IntegerAttr>("splitdim")) {

                params.splitdim = attr.getInt();

            }

            if (auto attr = op->getAttrOfType<StringAttr>("hw_axis_owner")) {

                params.hw_axis_owner = attr.getValue().str();

            }

            if (auto attr = op->getAttrOfType<StringAttr>("replicate_on")) {

                params.replicate_on = attr.getValue().str();

            }

            if (auto attr = op->getAttrOfType<StringAttr>("single_tile_owner")) {

                params.single_tile_owner = attr.getValue().str();

            }



            // Populate initTensorMap from the partitiontensor's source operand.

            // In new IR (after DeclareDataOpConversion), there is no declare_data wrapper —

            // the arith.constant tensor feeds partitiontensor directly.

            if (op->getNumOperands() > 0 && op->getNumResults() > 0) {

                auto tensorType = cast<RankedTensorType>(op->getResult(0).getType());

                std::string typeKey;

                llvm::raw_string_ostream os(typeKey);

                os << tensorType;

                if (info.initTensorMap.find(typeKey) == info.initTensorMap.end()) {

                    info.initTensorMap[typeKey] = op->getOperand(0);

                }

                // Also track the tensor type in sourceTensorTypes

                bool found = false;

                for (auto &existingType : info.sourceTensorTypes) {

                    if (existingType == tensorType) {

                        found = true;

                        break;

                    }

                }

                if (!found) {

                    info.sourceTensorTypes.push_back(tensorType);

                }

            }



            // Always add - each partitiontensor represents a unique data flow

            info.uniquePartitionParams.push_back(params);

        } else if (op->getName().getStringRef().starts_with("routing.RoutingCreate")) {

            info.routingCreateOps.push_back(OpWithParent(op, parentOp));

        }

        

        // Collect dfscheblueprint.declare_data

        if (op->getName().getStringRef() == "dfscheblueprint.declare_data") {

            info.declareDataOps.push_back(OpWithParent(op, parentOp));

            // Extract tensor type and init tensor for source tensor creation

            if (op->getNumResults() > 0) {

                if (auto tensorType = dyn_cast<RankedTensorType>(op->getResult(0).getType())) {

                    // Create a key for this tensor type

                    std::string typeKey;

                    llvm::raw_string_ostream os(typeKey);

                    os << tensorType;

                    

                    bool found = false;

                    for (auto &existingType : info.sourceTensorTypes) {

                        if (existingType == tensorType) {

                            found = true;

                            break;

                        }

                    }

                    if (!found) {

                        info.sourceTensorTypes.push_back(tensorType);

                    }

                    

                    // Capture the init tensor from the operand (new DeclareDataOp takes init_tensor as input)

                    if (op->getNumOperands() > 0 && info.initTensorMap.find(typeKey) == info.initTensorMap.end()) {

                        info.initTensorMap[typeKey] = op->getOperand(0);

                    }

                }

            }

        }

        

        // Collect top-level arith.constant in main function

        if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {

            // Only collect if direct child of func.func (not nested in regions)

            if (auto funcOp = dyn_cast<func::FuncOp>(op->getParentOp())) {

                if (funcOp.getName() == "main") {

                    info.topLevelConstantOps.push_back(OpWithParent(op, parentOp));

                }

            }

        }

    });

}



// Debug: Print parent information for all collected operations

static void printParentInfo(const ModuleScheduleInfo &info) {

    llvm::errs() << "\n=== Operation Parent Information ===\n";

    

    auto printOpList = [](const SmallVector<OpWithParent> &ops, StringRef name) {

        if (ops.empty()) return;

        llvm::errs() << "\n" << name << " (" << ops.size() << " operations):\n";

        for (size_t i = 0; i < ops.size(); ++i) {

            llvm::errs() << "  [" << i << "] " << ops[i].op->getName() << "\n";

            if (ops[i].parent) {

                llvm::errs() << "      Parent: " << ops[i].parent->getName();

                if (ops[i].isInDSKernelReceiver) {

                    llvm::errs() << " (IS dfschedule.dskernel_receiver)";

                } else {

                    llvm::errs() << " (NOT dfschedule.dskernel_receiver)";

                }

                llvm::errs() << "\n";

            } else {

                llvm::errs() << "      Parent: <none> (top-level)\n";

            }

        }

    };

    

    printOpList(info.declareTensorOps, "DeclareTensor");

    printOpList(info.declareTileOps, "DeclareTile");

    printOpList(info.configDmaBdOps, "ConfigDmaBd");

    printOpList(info.configCreateIoOps, "ConfigCreateIo");

    printOpList(info.packetOps, "Packet");

    printOpList(info.loadKernelGroupOps, "LoadKernelGroup");

    printOpList(info.launchKernelGroupOps, "LaunchKernelGroup");

    printOpList(info.getBdIdOps, "GetBdId");

    printOpList(info.startIoOps, "StartIo");

    printOpList(info.scheduleWaitOps, "ScheduleWait");

    printOpList(info.dskernelReceiverOps, "DSKernelReceiver");

    printOpList(info.tensorEmptyOps, "TensorEmpty");

    printOpList(info.extractSliceOps, "ExtractSlice");

    printOpList(info.partitionTensorOps, "PartitionTensor");

    printOpList(info.executeRegionOps, "ExecuteRegion");

    printOpList(info.routingCreateOps, "RoutingCreate");

    printOpList(info.declareDataOps, "DeclareData");

    printOpList(info.topLevelConstantOps, "TopLevelConstant");

    

    llvm::errs() << "\n====================================\n\n";

}



// Associate packets with their target tiles

static void associatePacketsWithTiles(ModuleScheduleInfo &info) {

    // For each LoadKernelGroupOp, extract tile-to-packet/config mapping

    for (auto &opWithParent : info.loadKernelGroupOps) {

        auto loadKernel = cast<dfschedule::LoadKernelGroupOp>(opWithParent.op);

        

        auto tiles = loadKernel.getTiles();

        auto computeArgs = loadKernel.getDistributedComputeKernelArgs();

        

        // Check for distributed_args (which now contains kernel config symbols)

        auto distArgs = loadKernel.getDistributedArgs();

        

        if (distArgs) {

            // Handle distributed_args pointing to individual kernel config symbols

            auto distArgsArray = *distArgs;  // Dereference the optional

            auto moduleOp = opWithParent.parent->getParentOfType<ModuleOp>();

            

            for (size_t i = 0; i < tiles.size(); ++i) {

                Value tileVal = tiles[i];

                if (auto declareTile = tileVal.getDefiningOp<dfschedule::DeclareTileOp>()) {

                    TileKey key = getTileKey(declareTile);

                    

                    if (!isShimTile(key)) {

                        auto &tileInfo = info.coreTiles[key];

                        

                        // Look up the kernel config symbol for this tile

                        if (i < distArgsArray.size()) {

                            if (auto symRef = dyn_cast<SymbolRefAttr>(distArgsArray[i])) {

                                // Lookup the DeclareKernelConfigOp

                                auto configOp = moduleOp.lookupSymbol<dfschedule::DeclareKernelConfigOp>(

                                    symRef.getRootReference());

                                

                                if (configOp) {

                                    auto tileConfigsAttr = configOp.getTileConfigs();

                                    // Each config op should have exactly one tile config

                                    if (tileConfigsAttr.size() > 0) {

                                        auto configDict = dyn_cast<DictionaryAttr>(tileConfigsAttr[0]);

                                        if (configDict) {

                                            tileInfo.configDicts.push_back(configDict);

                                        }

                                    }

                                }

                                

                                // Also store the symbol reference for backward compatibility

                                tileInfo.packetSymbols.push_back(symRef);

                            }

                        }

                        

                        // Add compute kernel arg

                        if (i < computeArgs.size()) {

                            if (auto symRef = dyn_cast<SymbolRefAttr>(computeArgs[i])) {

                                tileInfo.computeKernelArgs.push_back(symRef);

                            }

                        }

                        

                        // Track packet count per tile

                        info.tilePacketCount[key]++;

                    }

                }

            }

        }

    }

}



// Associate DMA configs with shim tiles

static void associateDmaWithShimTiles(ModuleScheduleInfo &info) {

    for (auto &opWithParent : info.configDmaBdOps) {

        auto dmaBd = cast<dfschedule::ConfigDmaBdOp>(opWithParent.op);

        Value tileVal = dmaBd.getTile();

        

        // Skip DMA BD operations inside dskernel_receiver (these are kernel-side, not host-side shim operations)

        if (opWithParent.isInDSKernelReceiver) {

            continue;

        }

        

        // Only include operations that are in dfschedule.host (the canonicalized region)

        if (!opWithParent.parent || opWithParent.parent->getName().getStringRef() != "dfschedule.host") {

            continue;

        }

        

        if (auto declareTile = tileVal.getDefiningOp<dfschedule::DeclareTileOp>()) {

            TileKey key = getTileKey(declareTile);

            

            if (isShimTile(key)) {

                info.shimTiles[key].dmaBdOps.push_back(opWithParent.op);

                info.shimTiles[key].bdHandles.push_back(dmaBd.getBdHandle());

            }

        }

    }

    ///*

    for (auto &opWithParent : info.configCreateIoOps) {

        auto createIo = cast<dfschedule::ConfigCreateIoOp>(opWithParent.op);

        Value tileVal = createIo.getTile();

        

        if (auto declareTile = tileVal.getDefiningOp<dfschedule::DeclareTileOp>()) {

            TileKey key = getTileKey(declareTile);

            if (isShimTile(key)) {

                info.shimTiles[key].createIoOps.push_back(opWithParent.op);

                info.shimTiles[key].ioHandles.push_back(createIo.getIoHandle());

            }

        }

    }

    //*/

    // Print all shim tiles and their associated DMA operations

    llvm::errs() << "\n=== Shim Tiles DMA Association ===\n";

    if (info.shimTiles.empty()) {

        llvm::errs() << "  No shim tiles found.\n";

    } else {

        for (auto &[key, shimInfo] : info.shimTiles) {

            llvm::errs() << "\nShim Tile (col=" << key.first << ", row=" << key.second << "):\n";

            

            // Print DMA BD operations

            llvm::errs() << "  DMA BD Operations: " << shimInfo.dmaBdOps.size() << "\n";

            for (size_t i = 0; i < shimInfo.dmaBdOps.size(); ++i) {

                auto dmaBd = cast<dfschedule::ConfigDmaBdOp>(shimInfo.dmaBdOps[i]);

                llvm::errs() << "    [" << i << "] dfschedule.config.dma_bd\n";

                llvm::errs() << "        Offset: " << dmaBd.getOffset() << "\n";

                llvm::errs() << "        Length: " << dmaBd.getLen() << "\n";

                llvm::errs() << "        Enable Packet: " << dmaBd.getEnablePacket() << "\n";

                if (dmaBd.getEnablePacket()) {

                    llvm::errs() << "        Packet ID: " << dmaBd.getPacketId() << "\n";

                }

                llvm::errs() << "        BD Handle: " << shimInfo.bdHandles[i] << "\n";

            }

            

            // Print IO Config operations

            llvm::errs() << "  IO Config Operations: " << shimInfo.createIoOps.size() << "\n";

            for (size_t i = 0; i < shimInfo.createIoOps.size(); ++i) {

                auto createIo = cast<dfschedule::ConfigCreateIoOp>(shimInfo.createIoOps[i]);

                llvm::errs() << "    [" << i << "] dfschedule.config.create_io\n";

                llvm::errs() << "        Channel: " << createIo.getChannel() << "\n";

                llvm::errs() << "        Direction: " << createIo.getDirection() << "\n";

                llvm::errs() << "        IO Operation: " << createIo.getIoOperation() << "\n";

                llvm::errs() << "        IO Handle: " << shimInfo.ioHandles[i] << "\n";

            }

        }

    }

    llvm::errs() << "\n==================================\n\n";

}



// Structure to hold DMA BD parameters extracted from original ops

struct DmaBdParams {

    TileKey shimKey;

    Type bufferType;

    int64_t offset;

    int64_t len;

    bool enablePacket;  // Changed to bool for BoolAttr

    int64_t packetId;

    int64_t nextBd;

    int bdIndex;

    int64_t sliceIndex;  // Index into uniqueSliceParams/declaredMemrefs for the buffer

    int64_t acquireLockId = -1;

    int64_t acquireLockVal = -1;

    int64_t releaseLockId = -1;

    int64_t releaseLockVal = -1;

    int32_t data_id = -1; // Root-tensor identity; set by BlueprintToSchedulePass

    bool isBufferView = false;         // True when buffer came from BufferViewOp (new IR path)

    int64_t shimSourceSliceIndex = -1; // Index into sliceMap for the intermediate tensor slice

};



// Structure to hold IO config parameters

struct IoConfigParams {

    TileKey shimKey;

    int64_t channel;

    std::string direction;    // StringAttr

    std::string ioOperation;  // StringAttr

    int bdIndex; // Which BD handle to use

};



// Structure to hold core tile DMA BD parameters (with lock IDs for ping-pong)

struct CoreDmaBdParams {

    TileKey coreKey;

    Type bufferType;

    int64_t offset;

    int64_t len;

    bool enablePacket;

    int64_t packetId;

    int64_t nextBd;

    int bdIndex;

    int64_t sliceIndex;

    int64_t acquireLockId;

    int64_t acquireLockVal;

    int64_t releaseLockId;

    int64_t releaseLockVal;

    int64_t originalBdId = -1;

    bool hasLinkedBd = false;

    bool isBindCoreBuffer = false; // True when buffer came from BindCoreBufferOp (sliceIndex = L1 byte offset)

};



// Structure to hold core tile IO config parameters

struct CoreIoConfigParams {

    TileKey coreKey;

    int64_t channel;

    std::string direction;

    std::string ioOperation;

    int64_t originalBdId = 0;

};



// Create canonicalized schedule in dfschedule.host at module level

// All operations use only constants/attributes, so IsolatedFromAbove is OK

static void createCanonicalizedSchedule(

    OpBuilder &builder,

    Location loc,

    ModuleScheduleInfo &info,

    ModuleOp moduleOp,

    func::FuncOp funcOp) {

    

    // Early exit if nothing to canonicalize

    if (info.coreTiles.empty() && info.shimTiles.empty()) {

        return;

    }

    

    // ==========================================================

    // Collect DMA BD and IO parameters from original operations

    // ==========================================================

    std::vector<DmaBdParams> allDmaBdParams;

    std::vector<IoConfigParams> allIoConfigParams;

    

    // Map to track BD index per shim tile

    std::map<TileKey, int> shimBdCounter;

    

    for (auto &opWithParent : info.configDmaBdOps) {

        auto dmaBd = cast<dfschedule::ConfigDmaBdOp>(opWithParent.op);

        Value tileVal = dmaBd.getTile();

        

        if (auto declareTile = tileVal.getDefiningOp<dfschedule::DeclareTileOp>()) {

            TileKey key = getTileKey(declareTile);

            if (isShimTile(key)) {

                DmaBdParams params;

                params.shimKey = key;

                params.bufferType = dmaBd.getBuffer().getType();

                params.offset = dmaBd.getOffset();

                params.len = dmaBd.getLen();

                params.enablePacket = dmaBd.getEnablePacket();

                params.packetId = dmaBd.getPacketId();

                params.nextBd = dmaBd.getNextBd();

                params.bdIndex = shimBdCounter[key]++;

                params.sliceIndex = -1;  // Default: no slice found

                params.acquireLockId = static_cast<int32_t>(dmaBd.getAcquireLockId());

                params.acquireLockVal = static_cast<int32_t>(dmaBd.getAcquireLockVal());

                params.releaseLockId = static_cast<int32_t>(dmaBd.getReleaseLockId());

                params.releaseLockVal = static_cast<int32_t>(dmaBd.getReleaseLockVal());

                // Read data_id propagated from DmaphopTodfscheblueprintPass via typed accessor

                params.data_id = static_cast<int32_t>(dmaBd.getDataId());



                // Fast-path: buffer_view carries offset directly — no slice chain tracing needed

                Value buffer = dmaBd.getBuffer();

                if (auto bvOp = buffer.getDefiningOp<dfschedule::BufferViewOp>()) {

                    // Encode buffer_view offset as sliceIndex for dedup (guaranteed unique per view)

                    params.sliceIndex = static_cast<int64_t>(bvOp.getOffset());

                    params.isBufferView = true;



                    // Trace back through alloc_device_mem -> unrealized_cast -> extract_slice

                    // to find the source tensor slice index for reconstruction

                    Value bvBase = bvOp.getBase();

                    if (auto allocDDR = bvBase.getDefiningOp<dfschedule::AllocDeviceMemOp>()) {

                        Value allocSrc = allocDDR.getSource();

                        // Unwrap memref_mapping if present (BlueprintToSchedulePass inserts it

                        // between unrealized_cast and alloc_device_mem)

                        if (auto mappingOp = allocSrc.getDefiningOp<dfschedule::MemRefMappingOp>()) {

                            allocSrc = mappingOp.getOperand();

                        }

                        // May be unrealized_conversion_cast tensor -> memref

                        if (auto castOp = allocSrc.getDefiningOp<UnrealizedConversionCastOp>()) {

                            if (!castOp.getInputs().empty()) {

                                Value castInput = castOp.getInputs()[0];

                                if (auto sliceOp = castInput.getDefiningOp<tensor::ExtractSliceOp>()) {

                                    for (size_t i = 0; i < info.extractSliceOps.size(); ++i) {

                                        if (info.extractSliceOps[i].op == sliceOp.getOperation()) {

                                            params.shimSourceSliceIndex = static_cast<int64_t>(i);

                                            break;

                                        }

                                    }

                                }

                            }

                        }

                    }



                    allDmaBdParams.push_back(params);

                    continue;

                }



                // Legacy path removed: DeclareTensorOp deprecated (tensor path)



                allDmaBdParams.push_back(params);

            }

        }

    }

    

    // Map to track IO index per shim tile

    std::map<TileKey, int> shimIoCounter;

    

    for (auto &opWithParent : info.configCreateIoOps) {

        auto createIo = cast<dfschedule::ConfigCreateIoOp>(opWithParent.op);

        Value tileVal = createIo.getTile();

        

        if (auto declareTile = tileVal.getDefiningOp<dfschedule::DeclareTileOp>()) {

            TileKey key = getTileKey(declareTile);

            if (isShimTile(key)) {

                IoConfigParams params;

                params.shimKey = key;

                params.channel = createIo.getChannel();

                params.direction = createIo.getDirection().str();

                params.ioOperation = createIo.getIoOperation().str();

                params.bdIndex = shimIoCounter[key]++;

                allIoConfigParams.push_back(params);

            }

        }

    }



    // ==========================================================

    // Deduplicate shim DMA BDs by (shimKey, data_id, sliceIndex).

    // BDs for the same root tensor AND same slice share a data_id and sliceIndex;

    // BDs for different slices (e.g. different partition rounds) must be kept

    // separate even if they share data_id, since they transfer different data.

    // Also deduplicate IO configs by (shimKey, channel, direction) for the same reason.

    // ==========================================================

    {

        using ShimMergeKey = std::tuple<TileKey, int32_t, int64_t>;

        std::set<ShimMergeKey> seenShimBdGroups;

        std::vector<DmaBdParams> deduped;

        for (auto &p : allDmaBdParams) {

            ShimMergeKey k = {p.shimKey, p.data_id, p.sliceIndex};

            if (seenShimBdGroups.find(k) == seenShimBdGroups.end()) {

                seenShimBdGroups.insert(k);

                deduped.push_back(p);

            }

        }

        allDmaBdParams = std::move(deduped);

    }

    {

        using IoMergeKey = std::tuple<TileKey, int64_t, std::string>;

        struct IoMergeKeyLess {

            bool operator()(const IoMergeKey &a, const IoMergeKey &b) const {

                if (std::get<0>(a) != std::get<0>(b))

                    return std::get<0>(a) < std::get<0>(b);

                if (std::get<1>(a) != std::get<1>(b))

                    return std::get<1>(a) < std::get<1>(b);

                return std::get<2>(a) < std::get<2>(b);

            }

        };

        std::set<IoMergeKey, IoMergeKeyLess> seenIoGroups;

        std::vector<IoConfigParams> deduped;

        for (auto &p : allIoConfigParams) {

            IoMergeKey k = {p.shimKey, p.channel, p.direction};

            if (seenIoGroups.find(k) == seenIoGroups.end()) {

                seenIoGroups.insert(k);

                deduped.push_back(p);

            }

        }

        allIoConfigParams = std::move(deduped);

    }



    // ==========================================================

    // Collect core tile DMA BD and IO parameters

    // ==========================================================

    std::vector<CoreDmaBdParams> allCoreDmaBdParams;

    std::vector<CoreIoConfigParams> allCoreIoConfigParams;



    std::map<TileKey, int> coreBdCounter;



    for (auto &opWithParent : info.configDmaBdOps) {

        auto dmaBd = cast<dfschedule::ConfigDmaBdOp>(opWithParent.op);

        Value tileVal = dmaBd.getTile();



        if (auto declareTile = tileVal.getDefiningOp<dfschedule::DeclareTileOp>()) {

            TileKey key = getTileKey(declareTile);

            if (!isShimTile(key)) {

                CoreDmaBdParams params;

                params.coreKey = key;

                params.bufferType = dmaBd.getBuffer().getType();

                params.offset = dmaBd.getOffset();

                params.len = dmaBd.getLen();

                params.enablePacket = dmaBd.getEnablePacket();

                params.packetId = dmaBd.getPacketId();

                params.nextBd = dmaBd.getNextBd();

                params.sliceIndex = -1;

                params.hasLinkedBd = (dmaBd.getLinkedBd() != Value());



                Value bdIdValue = dmaBd.getBdId();

                if (auto constOp = bdIdValue.getDefiningOp<arith::ConstantOp>()) {

                    params.originalBdId = cast<IntegerAttr>(constOp.getValue()).getInt();

                } else {

                    params.originalBdId = coreBdCounter[key];

                }

                params.bdIndex = params.originalBdId;

                coreBdCounter[key]++;



                // Read lock IDs and values directly from attributes

                params.acquireLockId = static_cast<int32_t>(dmaBd.getAcquireLockId());

                params.acquireLockVal = static_cast<int32_t>(dmaBd.getAcquireLockVal());

                params.releaseLockId = static_cast<int32_t>(dmaBd.getReleaseLockId());

                params.releaseLockVal = static_cast<int32_t>(dmaBd.getReleaseLockVal());



                // Fast-path: bind_core_buffer carries L1 offset directly

                Value buffer = dmaBd.getBuffer();

                if (auto bcbOp = buffer.getDefiningOp<dfschedule::BindCoreBufferOp>()) {

                    // Encode L1 offset as sliceIndex for buffer resolution in createCoreBd

                    params.sliceIndex = static_cast<int64_t>(bcbOp.getOffset());

                    params.isBindCoreBuffer = true;

                    allCoreDmaBdParams.push_back(params);

                    continue;

                }



                // Legacy path removed: DeclareTensorOp deprecated (tensor path)



                allCoreDmaBdParams.push_back(params);

            }

        }

    }



    for (auto &opWithParent : info.configCreateIoOps) {

        auto createIo = cast<dfschedule::ConfigCreateIoOp>(opWithParent.op);

        Value tileVal = createIo.getTile();



        if (auto declareTile = tileVal.getDefiningOp<dfschedule::DeclareTileOp>()) {

            TileKey key = getTileKey(declareTile);

            if (!isShimTile(key)) {

                CoreIoConfigParams params;

                params.coreKey = key;

                params.channel = createIo.getChannel();

                params.direction = createIo.getDirection().str();

                params.ioOperation = createIo.getIoOperation().str();



                // Trace the bd_config SSA operand to find original BD ID

                Value bdConfig = createIo.getBdConfig();

                if (auto dmaBd = bdConfig.getDefiningOp<dfschedule::ConfigDmaBdOp>()) {

                    Value bdIdVal = dmaBd.getBdId();

                    if (auto constOp = bdIdVal.getDefiningOp<arith::ConstantOp>()) {

                        params.originalBdId = cast<IntegerAttr>(constOp.getValue()).getInt();

                    }

                }



                allCoreIoConfigParams.push_back(params);

            }

        }

    }



    // ==========================================================

    // PART 1: Create dfschedule.host at MODULE level (after func.func)

    // ==========================================================

    builder.setInsertionPointAfter(funcOp);

    

    auto hostOp = builder.create<dfschedule::HostBlockOp>(

        loc,

        builder.getStringAttr("host_canonicalized"));

    

    Block *hostBody = &hostOp.getBody().emplaceBlock();

    builder.setInsertionPointToStart(hostBody);

    

    // All operations below are created INSIDE dfschedule.host

    // They only use constants and values defined within this block

    

    // Helper to create type key strings

    auto makeTensorTypeKey = [](RankedTensorType t) -> std::string {

        std::string key;

        llvm::raw_string_ostream os(key);

        os << t;

        return key;

    };

    

    // Helper to create partition key - include index for uniqueness

    auto makePartitionKey = [&](const PartitionParams &p) -> std::string {

        std::string key;

        llvm::raw_string_ostream os(key);

        os << p.index << "_" << p.tensorType << "_" << p.splitnum << "_" << p.splitdim << "_" << p.hw_axis_owner;

        return key;

    };



    // ==========================================================

    // DATA FLOW: (declare_data ->) partitiontensor -> extract_slice

    // In new IR: arith.constant -> partitiontensor -> extract_slice (no declare_data)

    // ==========================================================



    // 0a. Populate sourceTensorMap by cloning each unique source tensor's init op.

    // The init tensor is captured from routing.partitiontensor operands (new IR path)

    // or from dfscheblueprint.declare_data operands (old IR path) — both stored in initTensorMap.

    std::map<std::string, Value> sourceTensorMap;

    for (auto tensorType : info.sourceTensorTypes) {

        std::string key = makeTensorTypeKey(tensorType);

        if (sourceTensorMap.find(key) == sourceTensorMap.end()) {

            // Look up the captured init tensor from the original IR

            auto initTensorIt = info.initTensorMap.find(key);

            if (initTensorIt == info.initTensorMap.end()) {

                llvm::errs() << "Warning: No init tensor found for type " << key << "\n";

                continue;

            }



            Value originalInitTensor = initTensorIt->second;



            // Clone the init tensor (arith.constant) into the new block

            Operation *initDefOp = originalInitTensor.getDefiningOp();

            if (!initDefOp) {

                llvm::errs() << "Warning: Init tensor is not defined by an operation\n";

                continue;

            }

            Operation *clonedOp = builder.clone(*initDefOp);

            // Store the cloned constant directly — routing.partitiontensor will use it.

            // (No dfscheblueprint.declare_data wrapper needed in new IR.)

            sourceTensorMap[key] = clonedOp->getResult(0);

        }

    }

    

    // 0b. Create routing.partitiontensor for each unique partition config

    std::map<std::string, Value> partitionedTensorMap;

    for (const auto &params : info.uniquePartitionParams) {

        std::string partKey = makePartitionKey(params);

        if (partitionedTensorMap.find(partKey) == partitionedTensorMap.end()) {

            // Find the source tensor

            std::string srcKey = makeTensorTypeKey(params.tensorType);

            Value sourceTensor;

            if (sourceTensorMap.find(srcKey) != sourceTensorMap.end()) {

                sourceTensor = sourceTensorMap[srcKey];

            } else {

                // Fallback: clone the init tensor directly (no declare_data wrapper)

                auto initTensorIt = info.initTensorMap.find(srcKey);

                if (initTensorIt != info.initTensorMap.end()) {

                    Value originalInitTensor = initTensorIt->second;

                    Operation *initDefOp = originalInitTensor.getDefiningOp();

                    if (initDefOp) {

                        Operation *clonedOp = builder.clone(*initDefOp);

                        sourceTensorMap[srcKey] = clonedOp->getResult(0);

                        sourceTensor = clonedOp->getResult(0);

                    }

                }

            }



            if (!sourceTensor) {

                llvm::errs() << "Warning: No source tensor for partitiontensor — skipping\n";

                continue;

            }



            // Create routing.partitiontensor

            OperationState partState(loc, "routing.partitiontensor");

            partState.addOperands({sourceTensor});

            partState.addTypes({params.tensorType});

            partState.addAttribute("splitnum", builder.getI32IntegerAttr(params.splitnum));

            partState.addAttribute("splitdim", builder.getI32IntegerAttr(params.splitdim));

            partState.addAttribute("hw_axis_owner", builder.getStringAttr(params.hw_axis_owner));

            partState.addAttribute("replicate_on", builder.getStringAttr(params.replicate_on));

            partState.addAttribute("single_tile_owner", builder.getStringAttr(params.single_tile_owner));

            Operation *partitionOp = builder.create(partState);

            partitionedTensorMap[partKey] = partitionOp->getResult(0);

        }

    }

    

    // 0c. Create extract_slice operations - maintain proper chain

    // Each slice gets a unique entry, keyed by its index

    // We process in order: first slices from partitions, then nested slices

    

    std::map<size_t, Value> sliceMap;  // Map from slice index to created Value

    

    // First pass: create slices that come directly from partitiontensors

    for (const auto &params : info.uniqueSliceParams) {

        if (params.isFromPartition) {

            Value sourceTensor;

            

            // Find the partitioned tensor by index

            for (auto &[partKey, partValue] : partitionedTensorMap) {

                std::string indexPrefix = std::to_string(params.partitionIndex) + "_";

                if (partKey.find(indexPrefix) == 0) {

                    sourceTensor = partValue;

                    break;

                }

            }

            

            if (sourceTensor) {

                SmallVector<int64_t, 4> defaultStrides(params.offsets.size(), 1);

                auto newSlice = builder.create<tensor::ExtractSliceOp>(

                    loc,

                    params.resultType,

                    sourceTensor,

                    ValueRange{}, ValueRange{}, ValueRange{},

                    params.offsets,

                    params.sizes,

                    params.strides.empty() ? defaultStrides : params.strides);

                sliceMap[params.index] = newSlice.getResult();

            }

        }

    }

    

    // Second pass: create nested slices (from other slices)

    // May need multiple passes if there are deeply nested slices

    bool progress = true;

    while (progress) {

        progress = false;

        for (const auto &params : info.uniqueSliceParams) {

            // Skip if already created or if it's directly from partition

            if (sliceMap.find(params.index) != sliceMap.end() || params.isFromPartition) {

                continue;

            }

            

            // Check if parent slice is ready

            if (params.parentSliceIndex >= 0) {

                auto parentIt = sliceMap.find(static_cast<size_t>(params.parentSliceIndex));

                if (parentIt != sliceMap.end()) {

                    // Parent is ready, create this slice

                    Value sourceTensor = parentIt->second;

                    SmallVector<int64_t, 4> defaultStrides(params.offsets.size(), 1);

                    auto newSlice = builder.create<tensor::ExtractSliceOp>(

                        loc,

                        params.resultType,

                        sourceTensor,

                        ValueRange{}, ValueRange{}, ValueRange{},

                        params.offsets,

                        params.sizes,

                        params.strides.empty() ? defaultStrides : params.strides);

                    sliceMap[params.index] = newSlice.getResult();

                    progress = true;

                }

            }

        }

    }

    

    

    // Identify leaf slices (slices with no children)

    std::set<size_t> slicesWithChildren;

    for (const auto &params : info.uniqueSliceParams) {

        if (params.parentSliceIndex >= 0) {

            slicesWithChildren.insert(static_cast<size_t>(params.parentSliceIndex));

        }

    }



    // Collect which intermediate slice indices are actually referenced by Shim BDs

    // so we only create declaretensor ops for those that are needed.

    std::set<size_t> shimReferencedIntermediates;

    for (const auto &p : allDmaBdParams) {

        if (p.sliceIndex >= 0) {

            size_t idx = static_cast<size_t>(p.sliceIndex);

            if (slicesWithChildren.count(idx) > 0)

                shimReferencedIntermediates.insert(idx);

        }

    }



    // 0d. Create dfschedule.declaretensor for leaf slices (core tile BDs)

    //     and for intermediate slices actually referenced by Shim BDs.

    std::map<size_t, Value> declaredMemrefs;     // leaf slice index -> memref

    std::map<size_t, Value> intermediateMemrefs; // intermediate slice index -> memref

    for (const auto &params : info.uniqueSliceParams) {

        auto it = sliceMap.find(params.index);

        if (it == sliceMap.end())

            continue;



        bool isIntermediate = slicesWithChildren.count(params.index) > 0;

        if (isIntermediate && shimReferencedIntermediates.count(params.index) == 0)

            continue; // intermediate slice not used by any Shim BD -- skip



        Value sliceValue = it->second;

        auto sliceType = cast<RankedTensorType>(sliceValue.getType());

        auto memrefType = MemRefType::get(sliceType.getShape(), sliceType.getElementType());



        // DeclareTensorOp removed: use UnrealizedConversionCastOp (tensor path deprecated)

        auto castOp = builder.create<mlir::UnrealizedConversionCastOp>(loc, mlir::TypeRange{memrefType},

                                                                       mlir::ValueRange{sliceValue});

        mlir::Value memrefValue = castOp.getOutputs()[0];



        if (isIntermediate)

            intermediateMemrefs[params.index] = memrefValue;

        else

            declaredMemrefs[params.index] = memrefValue;

    }



    // 1. Create NEW deduplicated shim tile declarations

    std::map<TileKey, Value> shimTileMap;

    for (auto &[key, shimInfo] : info.shimTiles) {

        auto shimTile = builder.create<dfschedule::DeclareTileOp>(

            loc,

            dfschedule::TileType::get(builder.getContext()),

            builder.getI32IntegerAttr(key.first),

            builder.getI32IntegerAttr(key.second));

        shimTileMap[key] = shimTile.getTile();

    }

    

    // 2. Create NEW deduplicated core tile declarations

    std::map<TileKey, Value> coreTileMap;

    for (auto &[key, tileInfo] : info.coreTiles) {

        auto coreTile = builder.create<dfschedule::DeclareTileOp>(

            loc,

            dfschedule::TileType::get(builder.getContext()),

            builder.getI32IntegerAttr(key.first),

            builder.getI32IntegerAttr(key.second));

        coreTileMap[key] = coreTile.getTile();

    }

    

    // 3. Create external memory references for DMA buffers (using memref.alloc)

    // Group buffers by type to deduplicate

    std::map<std::string, Value> bufferMap;

    auto makeBufferKey = [](Type t) -> std::string {

        std::string key;

        llvm::raw_string_ostream os(key);

        os << t;

        return key;

    };

    

    // Map to store BD handles per shim tile

    std::map<TileKey, SmallVector<Value>> shimBdHandles;



    // 4. Create DMA BD configurations for shim tiles

    // Shim BDs keep the intermediate-slice buffer (the Shim receives the combined

    // data from DDR; packet routing distributes it to individual AIE core tiles).

    int bdIndexCounter = 0;

    for (const auto &params : allDmaBdParams) {

        if (shimTileMap.find(params.shimKey) == shimTileMap.end()) continue;

        Value shimTile = shimTileMap[params.shimKey];



        // Helper lambda to create one BD with a given buffer value

        auto createOneShimBd = [&](Value buffer) {

            auto bdIdConst = builder.create<arith::ConstantOp>(loc, builder.getI32Type(),

                                                               builder.getI32IntegerAttr(bdIndexCounter++));

            auto dmaBdOp = builder.create<dfschedule::ConfigDmaBdOp>(

                loc, dfschedule::BdHandleType::get(builder.getContext()), buffer, shimTile, bdIdConst.getResult(),

                builder.getI32IntegerAttr(params.offset), builder.getI32IntegerAttr(params.len),

                builder.getBoolAttr(params.enablePacket), builder.getI32IntegerAttr(params.packetId),

                builder.getI32IntegerAttr(params.nextBd), builder.getI32IntegerAttr(params.acquireLockId),

                builder.getI32IntegerAttr(params.acquireLockVal), builder.getI32IntegerAttr(params.releaseLockId),

                builder.getI32IntegerAttr(params.releaseLockVal),

                builder.getI32IntegerAttr(params.data_id), // propagate data_id

                Value());                                  // linked_bd (none for shim)

            shimBdHandles[params.shimKey].push_back(dmaBdOp.getBdHandle());

        };



        // Fast-path: isBufferView means buffer came from BufferViewOp (new IR path)

        // Reconstruct: unrealized_cast(tensor_slice) -> alloc_device_mem -> buffer_view

        if (params.isBufferView) {

            if (auto mrType = dyn_cast<MemRefType>(params.bufferType)) {

                // Find the intermediate tensor slice that feeds this DDR buffer

                Value srcTensor;

                if (params.shimSourceSliceIndex >= 0) {

                    auto it = sliceMap.find(static_cast<size_t>(params.shimSourceSliceIndex));

                    if (it != sliceMap.end()) {

                        srcTensor = it->second;

                    }

                }



                Value srcForAlloc;

                if (srcTensor) {

                    // Bridge: tensor -> flat memref (same shape, no memspace)

                    auto tensorType = cast<RankedTensorType>(srcTensor.getType());

                    auto flatMemrefType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());

                    auto castOp = builder.create<UnrealizedConversionCastOp>(loc, TypeRange{flatMemrefType},

                                                                             ValueRange{srcTensor});

                    srcForAlloc = castOp.getResult(0);

                } else {

                    // Fallback if we couldn't trace the source slice: use memref.alloc

                    // (DfscheduleToApiPass will emit a broken memcpy — better than crashing)

                    llvm::errs() << "Warning: shimSourceSliceIndex not found for BufferView BD; "

                                    "using alloc placeholder\n";

                    auto srcType = MemRefType::get(mrType.getShape(), mrType.getElementType());

                    srcForAlloc = builder.create<memref::AllocOp>(loc, srcType);

                }



                auto allocDDR = builder.create<dfschedule::AllocDeviceMemOp>(loc, mrType, srcForAlloc);

                Value ddrBase = allocDDR.getMemref();

                auto bvOp = builder.create<dfschedule::BufferViewOp>(

                    loc, mrType, ddrBase,

                    builder.getI64IntegerAttr(params.sliceIndex), // offset (buffer_view offset)

                    builder.getI64IntegerAttr(params.len));       // len

                createOneShimBd(bvOp.getView());

                continue;

            }

        }



        if (params.sliceIndex >= 0) {

            size_t idx = static_cast<size_t>(params.sliceIndex);



            // If this slice is an intermediate slice (has children), the Shim BD

            // should use the intermediate-slice buffer directly -- do NOT expand

            // to leaf descendants.

            auto intIt = intermediateMemrefs.find(idx);

            if (intIt != intermediateMemrefs.end()) {

                createOneShimBd(intIt->second);

                continue;

            }



            // Otherwise it is a leaf slice -- use the leaf memref.

            auto leafIt = declaredMemrefs.find(idx);

            if (leafIt != declaredMemrefs.end()) {

                createOneShimBd(leafIt->second);

                continue;

            }

        }



        // Fallback: no slice chain (sliceIndex == -1) or lookup failed.

        // Create exactly one BD using a fallback buffer (memref.alloc of full tensor).

        std::string bufKey = makeBufferKey(params.bufferType);

        Value buffer;

        if (bufferMap.find(bufKey) == bufferMap.end()) {

            auto memrefType = cast<MemRefType>(params.bufferType);

            buffer = builder.create<memref::AllocOp>(loc, memrefType);

            bufferMap[bufKey] = buffer;

        } else {

            buffer = bufferMap[bufKey];

        }

        createOneShimBd(buffer);

    }

    

    // 5. Create IO configurations for shim tiles

    std::map<TileKey, SmallVector<Value>> shimIoHandles;

    

    for (const auto &params : allIoConfigParams) {

        if (shimTileMap.find(params.shimKey) == shimTileMap.end()) continue;

        Value shimTile = shimTileMap[params.shimKey];

        

        // Get corresponding BD handle

        Value bdHandle;

        if (params.bdIndex < (int)shimBdHandles[params.shimKey].size()) {

            bdHandle = shimBdHandles[params.shimKey][params.bdIndex];

        } else if (!shimBdHandles[params.shimKey].empty()) {

            bdHandle = shimBdHandles[params.shimKey].back();

        } else {

            continue; // No BD handle available, skip this IO

        }

        

        // Create IO config

        auto createIoOp = builder.create<dfschedule::ConfigCreateIoOp>(

            loc,

            dfschedule::IoHandleType::get(builder.getContext()),

            bdHandle,

            shimTile,

            builder.getI32IntegerAttr(params.channel),

            builder.getStringAttr(params.direction),

            builder.getStringAttr(params.ioOperation));

        

        shimIoHandles[params.shimKey].push_back(createIoOp.getIoHandle());

    }



    // 5b. Create DMA BD configurations for core tiles (ping-pong)

    // Map from (tileKey, originalBdId) → new BD handle

    std::map<TileKey, std::map<int64_t, Value>> coreBdHandleMap;



    // Group core BD params by tile for ping-pong pair detection

    std::map<TileKey, SmallVector<size_t>> coreBdParamsByTile;

    for (size_t i = 0; i < allCoreDmaBdParams.size(); ++i) {

        coreBdParamsByTile[allCoreDmaBdParams[i].coreKey].push_back(i);

    }



    // Map from (tileKey, L1Offset) -> BindCoreBufferOp result (to avoid duplicate bind ops per tile/offset)

    std::map<std::pair<TileKey, int64_t>, Value> bindCoreBufferMap;



    auto resolveCoreBdBuffer = [&](const CoreDmaBdParams &params, Value coreTileValue) -> Value {

        Value buffer;



        // Fast-path: buffer came from BindCoreBufferOp; sliceIndex encodes L1 byte offset

        if (params.isBindCoreBuffer) {

            if (auto mrType = dyn_cast<MemRefType>(params.bufferType)) {

                // L1 buffer: recreate BindCoreBufferOp in the canonical host block.

                // Source type strips memspace (BindCoreBufferOp takes a plain memref input).

                auto srcType = MemRefType::get(mrType.getShape(), mrType.getElementType());

                auto key = std::make_pair(params.coreKey, params.sliceIndex);

                auto it = bindCoreBufferMap.find(key);

                if (it != bindCoreBufferMap.end()) {

                    buffer = it->second;

                } else {

                    // Alloc placeholder lowered to real L1 address by DfscheduleToApiPass

                    Value srcAlloc = builder.create<memref::AllocOp>(loc, srcType);

                    auto bindOp = builder.create<dfschedule::BindCoreBufferOp>(

                        loc, mrType, srcAlloc, coreTileValue, builder.getI64IntegerAttr(params.sliceIndex));

                    buffer = bindOp.getBuffer();

                    bindCoreBufferMap[key] = buffer;

                }

                return buffer;

            }

        }



        // Legacy path: sliceIndex is an index into declaredMemrefs

        if (params.sliceIndex >= 0) {

            auto memIt = declaredMemrefs.find(static_cast<size_t>(params.sliceIndex));

            if (memIt != declaredMemrefs.end())

                buffer = memIt->second;

        }

        if (!buffer) {

            std::string bufKey = makeBufferKey(params.bufferType);

            if (bufferMap.find(bufKey) == bufferMap.end()) {

                auto memrefType = cast<MemRefType>(params.bufferType);

                buffer = builder.create<memref::AllocOp>(loc, memrefType);

                bufferMap[bufKey] = buffer;

            } else {

                buffer = bufferMap[bufKey];

            }

        }

        return buffer;

    };



    auto createCoreBd = [&](const CoreDmaBdParams &params, Value coreTile, Value linkedBd) -> Value {

        Value buffer = resolveCoreBdBuffer(params, coreTile);

        auto bdIdConst =

            builder.create<arith::ConstantOp>(loc, builder.getI32Type(), builder.getI32IntegerAttr(params.bdIndex));

        auto dmaBdOp = builder.create<dfschedule::ConfigDmaBdOp>(

            loc, dfschedule::BdHandleType::get(builder.getContext()), buffer, coreTile, bdIdConst.getResult(),

            builder.getI32IntegerAttr(params.offset), builder.getI32IntegerAttr(params.len),

            builder.getBoolAttr(params.enablePacket), builder.getI32IntegerAttr(params.packetId),

            builder.getI32IntegerAttr(params.nextBd), builder.getI32IntegerAttr(params.acquireLockId),

            builder.getI32IntegerAttr(params.acquireLockVal), builder.getI32IntegerAttr(params.releaseLockId),

            builder.getI32IntegerAttr(params.releaseLockVal),

            builder.getI32IntegerAttr(-1), // data_id

            linkedBd);

        return dmaBdOp.getBdHandle();

    };



    for (auto &[tileKey, paramIndices] : coreBdParamsByTile) {

        if (coreTileMap.find(tileKey) == coreTileMap.end())

            continue;

        Value coreTile = coreTileMap[tileKey];



        if (paramIndices.size() == 2) {

            // Detect ping/pong by checking hasLinkedBd (SSA-based, not positional)

            size_t pingIdx, pongIdx;

            if (allCoreDmaBdParams[paramIndices[0]].hasLinkedBd) {

                pingIdx = paramIndices[0];

                pongIdx = paramIndices[1];

            } else {

                pongIdx = paramIndices[0];

                pingIdx = paramIndices[1];

            }

            Value pongHandle = createCoreBd(allCoreDmaBdParams[pongIdx], coreTile, Value());

            Value pingHandle = createCoreBd(allCoreDmaBdParams[pingIdx], coreTile, pongHandle);

            coreBdHandleMap[tileKey][allCoreDmaBdParams[pingIdx].originalBdId] = pingHandle;

            coreBdHandleMap[tileKey][allCoreDmaBdParams[pongIdx].originalBdId] = pongHandle;

        } else {

            for (size_t idx : paramIndices) {

                Value handle = createCoreBd(allCoreDmaBdParams[idx], coreTile, Value());

                coreBdHandleMap[tileKey][allCoreDmaBdParams[idx].originalBdId] = handle;

            }

        }

    }



    // 5c. Create IO configurations for core tiles (create_io + start_io immediately)

    for (const auto &params : allCoreIoConfigParams) {

        if (coreTileMap.find(params.coreKey) == coreTileMap.end())

            continue;

        Value coreTile = coreTileMap[params.coreKey];



        // Look up BD handle by originalBdId traced from SSA operand

        auto tileIt = coreBdHandleMap.find(params.coreKey);

        if (tileIt == coreBdHandleMap.end())

            continue;

        auto bdIt = tileIt->second.find(params.originalBdId);

        if (bdIt == tileIt->second.end()) {

            if (tileIt->second.empty())

                continue;

            bdIt = tileIt->second.begin();

        }

        Value bdHandle = bdIt->second;



        auto coreCreateIoOp = builder.create<dfschedule::ConfigCreateIoOp>(

            loc, dfschedule::IoHandleType::get(builder.getContext()), bdHandle, coreTile,

            builder.getI32IntegerAttr(params.channel), builder.getStringAttr(params.direction),

            builder.getStringAttr(params.ioOperation));

        // start_io immediately after create_io for core tiles (fire-and-forget, not waited on)

        auto coreBdIdOp = builder.create<dfschedule::GetBdIdOp>(loc, builder.getI32Type(), coreTile);

        builder.create<dfschedule::StartIoOp>(loc, dfschedule::EventType::get(builder.getContext()),

                                              coreCreateIoOp.getIoHandle(), coreBdIdOp.getBdId(),

                                              builder.getI32IntegerAttr(0));

    }



    // 6. Build list of core tiles and their configs for merged kernel group

    SmallVector<Value> allCoreTiles;

    SmallVector<Attribute> allTileConfigs;

    SmallVector<Attribute> allComputeKernelArgs;

    

    int tileIdx = 0;

    for (auto &[key, tileInfo] : info.coreTiles) {

        if (coreTileMap.find(key) == coreTileMap.end()) continue;

        Value coreTile = coreTileMap[key];

        allCoreTiles.push_back(coreTile);

        

        // Create config dictionary for this tile

        if (!tileInfo.configDicts.empty()) {

            // Use existing config (should already have all fields including lock IDs)

            allTileConfigs.push_back(tileInfo.configDicts[0]);

        } else {

            // Create default config (for backward compatibility)

            NamedAttrList configAttrs;

            configAttrs.append("tile_index", builder.getI32IntegerAttr(tileIdx));

            configAttrs.append("packet_id", builder.getI32IntegerAttr(tileIdx));

            configAttrs.append("dma_channel", builder.getI32IntegerAttr(0)); // default channel

            configAttrs.append("buffer_mode", builder.getI32IntegerAttr(1)); // 1 = ping-pong

            configAttrs.append("num_buffers", builder.getI32IntegerAttr(2)); // 2 buffers

            configAttrs.append("buffer_size", builder.getI32IntegerAttr(256)); // default 256 bytes

            // Add default lock IDs

            configAttrs.append("ping_acquire_lock_id", builder.getI32IntegerAttr(tileIdx * 4 + 0));

            configAttrs.append("pong_acquire_lock_id", builder.getI32IntegerAttr(tileIdx * 4 + 1));

            configAttrs.append("ping_release_lock_id", builder.getI32IntegerAttr(tileIdx * 4 + 2));

            configAttrs.append("pong_release_lock_id", builder.getI32IntegerAttr(tileIdx * 4 + 3));

            allTileConfigs.push_back(builder.getDictionaryAttr(configAttrs));

        }

        

        // Use the first compute kernel arg for this tile (or default)

        if (!tileInfo.computeKernelArgs.empty()) {

            allComputeKernelArgs.push_back(tileInfo.computeKernelArgs[0]);

        } else {

            allComputeKernelArgs.push_back(SymbolRefAttr::get(builder.getContext(), "compute0"));

        }

        

        tileIdx++;

    }

    

    // 7. Create SINGLE merged load_kernel_group (if core tiles exist)

    Value launchEvent;

    if (!allCoreTiles.empty()) {

        SmallVector<Attribute> calleeAttrs;

        calleeAttrs.push_back(SymbolRefAttr::get(builder.getContext(), info.kernelName));

        

        // Create individual kernel_config ops for each tile (e.g., @kernelconfig_merged0, @kernelconfig_merged1)

        SmallVector<Attribute> kernelConfigSymbols;

        for (size_t i = 0; i < allTileConfigs.size(); ++i) {

            std::string configName = "kernelconfig_merged" + std::to_string(i);

            

            // Create a kernel_config op with a single tile's config

            SmallVector<Attribute> singleTileConfig;

            singleTileConfig.push_back(allTileConfigs[i]);

            

            auto kernelConfigOp = builder.create<dfschedule::DeclareKernelConfigOp>(

                loc,

                dfschedule::KernelConfigType::get(builder.getContext()),

                builder.getStringAttr(configName),

                builder.getArrayAttr(singleTileConfig));

            

            // Store symbol reference

            kernelConfigSymbols.push_back(SymbolRefAttr::get(builder.getContext(), configName));

        }

        

        auto loadKernelGroupOp = builder.create<dfschedule::LoadKernelGroupOp>(

            loc,

            dfschedule::KernelGroupType::get(builder.getContext()),

            allCoreTiles,

            builder.getArrayAttr(calleeAttrs),

            builder.getArrayAttr(allComputeKernelArgs),

            nullptr,  // kernel_config = nullptr (not used)

            builder.getArrayAttr(kernelConfigSymbols));  // distributed_args = [@kernelconfig_merged0, @kernelconfig_merged1, ...]

        

        // 8. Create SINGLE launch_kernel_group

        auto launchKernelGroupOp = builder.create<dfschedule::LaunchKernelGroupOp>(

            loc,

            dfschedule::EventType::get(builder.getContext()),

            loadKernelGroupOp.getKernelGroup());

        

        launchEvent = launchKernelGroupOp.getEvent();

    }

    

    // 9. Create getBdId and start_io for each shim tile

    SmallVector<Value> allEvents;

    if (launchEvent) {

        allEvents.push_back(launchEvent);

    }

    

    for (auto &[key, ioHandles] : shimIoHandles) {

        if (shimTileMap.find(key) == shimTileMap.end()) continue;

        Value shimTile = shimTileMap[key];

        

        // Create getBdId

        auto getBdIdOp = builder.create<dfschedule::GetBdIdOp>(

            loc,

            builder.getI32Type(),

            shimTile);

        

            // Create start_io for each IO handle

        for (Value ioHandle : ioHandles) {

            auto startIoOp = builder.create<dfschedule::StartIoOp>(

                loc,

                dfschedule::EventType::get(builder.getContext()),

                ioHandle,

                getBdIdOp.getBdId(),

                builder.getI32IntegerAttr(0)

            );

            allEvents.push_back(startIoOp.getEvent());

        }

    }

    

    // 10. Create SINGLE merged schedule.wait with ALL events

    if (!allEvents.empty()) {

        builder.create<dfschedule::ScheduleWaitOp>(loc, allEvents);

    }

    

    // ==========================================================

    // PART 2: Add a call to host_canonicalized inside func.func @main()

    // ==========================================================

    Block &mainBlock = funcOp.getBody().front();

    Operation *terminator = mainBlock.getTerminator();

    if (terminator) {

        builder.setInsertionPoint(terminator);

    } else {

        builder.setInsertionPointToEnd(&mainBlock);

    }

    

    // Use dfschedule.launchhost to invoke the host schedule block

    auto hostSymbol = SymbolRefAttr::get(builder.getContext(), "host_canonicalized");

    builder.create<dfschedule::LaunchHostOp>(loc, hostSymbol);

}



// Remove old distributed schedule operations

static void removeOldScheduleOps(ModuleScheduleInfo &info) {

    // Mark operations for removal (in reverse order to handle dependencies)

    SmallVector<Operation*> opsToRemove;

    

    // Remove wait ops first

    for (auto &opWithParent : info.scheduleWaitOps) {

        opsToRemove.push_back(opWithParent.op);

    }

    

    // Remove start_io ops

    for (auto &opWithParent : info.startIoOps) {

        opsToRemove.push_back(opWithParent.op);

    }

    

    // Remove getBdId ops

    for (auto &opWithParent : info.getBdIdOps) {

        opsToRemove.push_back(opWithParent.op);

    }

    

    // Remove launch ops

    for (auto &opWithParent : info.launchKernelGroupOps) {

        opsToRemove.push_back(opWithParent.op);

    }

    

    // Remove load_kernel_group ops

    for (auto &opWithParent : info.loadKernelGroupOps) {

        opsToRemove.push_back(opWithParent.op);

    }

    

    // Remove packet ops

    for (auto &opWithParent : info.packetOps) {

        opsToRemove.push_back(opWithParent.op);

    }



    auto isInHostBlock = [](const OpWithParent &owp) -> bool {

        if (owp.parent && owp.parent->getName().getStringRef() == "dfschedule.host")

            return true;

        return false;

    };



    // Remove createIo ops (but NOT those inside dskernel_receiver or dfschedule.host)

    for (auto &opWithParent : info.configCreateIoOps) {

        if (!opWithParent.isInDSKernelReceiver && !isInHostBlock(opWithParent)) {

            opsToRemove.push_back(opWithParent.op);

        }

    }



    // Remove dmaBd ops (but NOT those inside dskernel_receiver or dfschedule.host)

    for (auto &opWithParent : info.configDmaBdOps) {

        if (!opWithParent.isInDSKernelReceiver && !isInHostBlock(opWithParent)) {

            opsToRemove.push_back(opWithParent.op);

        }

    }

    

    // Remove declareTile ops

    for (auto &opWithParent : info.declareTileOps) {

        opsToRemove.push_back(opWithParent.op);

    }

    

    // Remove declareTensor ops

    for (auto &opWithParent : info.declareTensorOps) {

        opsToRemove.push_back(opWithParent.op);

    }

    

    // Erase dfschedule operations (safe, no nested structure issues)

    for (auto *op : opsToRemove) {

        if (op->use_empty()) {

            op->erase();

        }

    }

}



// Remove scf.execute_region blocks, tensor.empty, declare_data, and constants from func.func main

// This is done separately to avoid memory corruption from nested op pointer invalidation

static void removeExecuteRegionsFromMain(func::FuncOp mainFunc) {

    if (!mainFunc) return;

    

    // Collect ops to erase (fresh collection, not using old pointers)

    SmallVector<Operation*> regionsToErase;

    SmallVector<Operation*> declareDataOps;

    SmallVector<Operation*> constantOps;

    SmallVector<Operation*> emptyOps;

    

    mainFunc.walk([&](Operation *op) {

        if (isa<scf::ExecuteRegionOp>(op)) {

            regionsToErase.push_back(op);

        } else if (op->getParentOp() == mainFunc.getOperation()) {

            // Only collect ops that are direct children of main's block

            if (isa<tensor::EmptyOp>(op)) {

                emptyOps.push_back(op);

            } else if (isa<arith::ConstantOp>(op)) {

                constantOps.push_back(op);

            } else if (op->getName().getStringRef() == "dfscheblueprint.declare_data") {

                declareDataOps.push_back(op);

            }

        }

    });

    

    // Erase scf.execute_region ops first (this also erases all nested ops)

    for (auto *op : regionsToErase) {

        if (op->use_empty()) {

            op->erase();

        }

    }

    

    // Erase in dependency order:

    // 1. tensor.empty (no dependencies)

    for (auto *op : emptyOps) {

        if (op->use_empty()) {

            op->erase();

        }

    }

    

    // 2. declare_data (uses constants)

    for (auto *op : declareDataOps) {

        if (op->use_empty()) {

            op->erase();

        }

    }

    

    // 3. constants (used by declare_data, so remove after declare_data)

    for (auto *op : constantOps) {

        if (op->use_empty()) {

            op->erase();

        }

    }

}



} // namespace



namespace mlir {



void ScheduleCanonicalizePass::runOnOperation() {

    ModuleOp moduleOp = getOperation();

    

    ModuleScheduleInfo info;

    

    // Step 1: Collect all dfschedule operations

    collectScheduleOps(moduleOp, info);

    

    // Debug: Print parent information

    printParentInfo(info);

    

    // Early exit if no schedule ops found

    if (info.declareTileOps.empty() && info.loadKernelGroupOps.empty()) {

        return;

    }

    

    // Step 2: Associate packets with tiles

    associatePacketsWithTiles(info);

    

    // Step 3: Find the main function to insert canonicalized host block

    func::FuncOp mainFunc = nullptr;

    moduleOp.walk([&](func::FuncOp funcOp) {

        if (funcOp.getName() == "main") {

            mainFunc = funcOp;

        }

    });

    

    if (!mainFunc) {

        // No main function, skip

        return;

    }

    

    // Step 5: Create canonicalized schedule inside func.func @main()

    // Operations are placed in an scf.execute_region block to group them

    OpBuilder builder(moduleOp.getContext());

    Location loc = mainFunc.getLoc();

    

    createCanonicalizedSchedule(builder, loc, info, moduleOp, mainFunc);

    

    // Step 5.5: Re-collect DMA BD operations after canonicalization to get correct parent info

    // Clear the old collection and re-collect from the canonicalized IR

    info.configDmaBdOps.clear();

    info.configCreateIoOps.clear();

    moduleOp.walk([&](Operation *op) {

        Operation *parentOp = getParentOp(op);

        if (auto configDmaBd = dyn_cast<dfschedule::ConfigDmaBdOp>(op)) {

            info.configDmaBdOps.push_back(OpWithParent(op, parentOp));

        } else if (auto createIo = dyn_cast<dfschedule::ConfigCreateIoOp>(op)) {

            info.configCreateIoOps.push_back(OpWithParent(op, parentOp));

        }

    });

    

    // Step 5.6: Associate DMA configs with shim tiles (after canonicalization)

    associateDmaWithShimTiles(info);

    

    // Step 6: Remove old distributed dfschedule operations

    removeOldScheduleOps(info);

    

    // Step 7: Remove scf.execute_region blocks (contains extract_slice, routing ops)

    // and tensor.empty from func.func main

    removeExecuteRegionsFromMain(mainFunc);

}



} // namespace mlir



"

Gemini said
这份代码实现了一个非常典型的“全局收集并重构”（Collect-and-Rebuild）样式的 Pass。虽然它能工作，但在 MLIR 专家眼中，它存在几个严重的工程隐患和性能瓶颈，尤其是在处理大规模 IR（百万行级别）时。

以下是该 Canonicalize 逻辑的主要问题及改进建议：

1. 致命的性能瓶颈：字符串作为 Map 的 Key
代码中大量使用了 makeTensorTypeKey 和 std::map<std::string, Value>。

问题：在 MLIR 中，Type 和 Attribute 本质上是唯一化的指针（Unique Pointers）。直接比较两个 Type 对象的指针速度极快。你将它们转成字符串再做字符串查找，性能下降了几个数量级。在百万行代码下，这会导致编译器在字符串拼接和哈希上耗费几分钟时间。

改进：直接使用 std::map<Type, Value> 或 llvm::DenseMap<Type, Value>。

2. 脆弱的手动 SSA 链重构（Slice Tracing）
代码尝试手动重建 partitiontensor -> extract_slice 的链条。

问题：你使用了 uniqueSliceParams 和 sliceMap 手动维护索引。这种“过程式”的逻辑非常硬编码。如果 IR 稍微发生变化（例如中间多了一个 unrealized_conversion_cast），你的追踪逻辑就会断掉，导致 sourceTensor 找不到，最后生成 memref.alloc 这种“错误但能跑”的占位符。

改进：利用 mlir::IRMapping (以前叫 BlockAndValueMapping)。当你克隆一个操作时，IRMapping 会自动帮你把旧的输入值映射到新的输出值，你不需要手动维护 index。

3. “先删除后检查”的风险
removeOldScheduleOps 函数通过循环删除操作，并依赖 op->use_empty()。

问题：在复杂的 IR 转换中，由于存在嵌套的 ExecuteRegion，简单的顺序删除极易导致空指针引用或悬挂指针。如果某个被删除的 Op 还有用户（User）没被清理，erase() 会直接触发崩溃。

改进：使用 rewriter.eraseOp(op) 或者在 Walk 结束时统一使用 PatternRewriter 托管删除逻辑。

4. 重复 Walk 导致的复杂度升高
你在 runOnOperation 里做了多次 moduleOp.walk。

问题：每一次 walk 都是 O(N)。在大型模型中，反复扫描整个 Module 是极其昂贵的。

改进：尽量在一次 walk 中完成所有信息的收集。或者使用索引化的数据结构（如 SymbolTable）直接跳转到目标 Op。

5. 逻辑不收敛（Non-Canonical Form）
你的 Shim DMA 合并逻辑是基于 std::set 去重的。

问题：如果你有两个 DMA 任务，虽然 sliceIndex 不同，但它们物理上是连续的（可以合并成一个长度更大的 DMA），你的逻辑无法识别这种“空间上的连续性”，因为它只是在做简单的属性对比。

改进：在合并逻辑中加入地址分析（Address Analysis），判断两个 offset + len 是否可以无缝拼接。

专家级重构建议 (The "Clean" Way)
与其写一个几百行的函数来手动重建所有东西，不如采用以下策略：

A. 引入 IRMapping 简化克隆
不要手动记录 sliceIndex，使用映射表：

C++
mlir::IRMapping mapper;
// 当你克隆基础 Tensor 时
auto newConst = builder.clone(*oldConst, mapper);
mapper.map(oldConst->getResult(0), newConst->getResult(0));

// 后续克隆 Slice 时，mapper 会自动处理映射
auto newSlice = builder.clone(*oldSlice, mapper); 
B. 结构化“拆盒”
不要手动收集 ExecuteRegion 再删除。利用我们之前讨论的 inlineRegionBefore：

第一步：Walk 所有的 routing.RoutingCreate，将其内容 inline 到 func.func。

第二步：此时所有的 Shim create_io 都在同一层级了。

第三步：跑一个局部匹配，找到相同 Tile 的 create_io 并合并。

C. 修正 Shim DMA 合并逻辑
目前的去重逻辑太“硬”：

C++
// 你的逻辑
using ShimMergeKey = std::tuple<TileKey, int32_t, int64_t>;
// ... 如果 sliceIndex 不同，就永远不会合并
建议：改用 基于 Tile 的 Bucket 收集。先按 Tile 把所有 DMA 塞进一个 List，然后对 List 按 offset 排序，检查是否可以 Merge。

总结
你现在的代码更像是一个 "IR Translator"（把一种方言翻译成另一种），而不是一个 "Canonicalizer"。

最大的风险在于： 你手动构建了一个极其复杂的 ModuleScheduleInfo 结构体来镜像 IR 的状态。在编译器开发中，真相永远在 IR 本身。尽量避免在 side-structure（辅助结构）中存储过多的状态信息，因为一旦辅助信息和 IR 同步出错，就会产生非常隐蔽的 Bug。

你想让我帮你针对 associateDmaWithShimTiles 这一块，写一个更健壮的、基于地址合并的逻辑示例吗？

You said
IRMapping 有什么用

Gemini said
简单来说，IRMapping（以前叫 BlockAndValueMapping）是 MLIR 里的**“SSA 值映射词典”**。

在编译器进行变换（特别是 Clone（克隆） 或 Inlining（内联））时，它解决了一个核心问题：“如何让新生成的代码正确地引用新生成的操作数，而不是指向旧的、甚至已经被删掉的代码？”

1. 为什么你需要它？（痛点分析）
在你之前的代码里，你手动维护了 sliceMap、partitionedTensorMap 等各种索引。这非常累，而且容易出错。

假设你有一段代码：

MLIR
%0 = arith.constant 1
%1 = arith.addi %0, %0
你想把它搬移到一个新的 dfschedule.host 块里。

你克隆了 %0，得到了 %0_new。

轮到克隆 %1 时，如果你直接复制，它会尝试去加 %0（旧的那个）。

后果：旧的 %0 很快就会被你删掉，你的新代码就会指向一个“死人”，导致编译器崩溃（Segfault）。

2. IRMapping 的工作原理
IRMapping 就像是一个自动更新的映射表。你告诉它：“从现在起，凡是引用 %0 的地方，都给我换成 %0_new。”

核心流程：
注册映射：当你克隆一个常量 %0 时，调用 mapper.map(old_val, new_val)。

自动查找：当你使用 builder.clone(op, mapper) 克隆 %1 时，MLIR 会自动检查这个 Op 的所有输入操作数。

它发现 %1 用了 %0。

它在 mapper 里查到了 %0 已经变成了 %0_new。

它自动把新生成的 addi 指向了 %0_new。

3. 代码对比：手动维护 vs. IRMapping
你的旧方法（HARDCODE 风险）：
C++
// 需要维护复杂的 map 和 index
std::map<size_t, Value> sliceMap;
// ... 各种逻辑来确保 index 正确 ...
auto newSlice = builder.create<tensor::ExtractSliceOp>(..., sliceMap[params.parentIndex]);
使用 IRMapping 的专家方法：
C++
mlir::IRMapping mapper;

// 1. 克隆基础操作并登记
Operation *newConst = builder.clone(*oldConst, mapper);
// mapper 内部现在记住了：old_val -> new_val

// 2. 批量克隆后续操作，完全不需要手动管谁是谁
for (auto &oldOp : oldOps) {
    // 这一行魔法会自动帮你把所有输入操作数替换成“新版”
    Operation *newOp = builder.clone(*oldOp, mapper);
}
4. 在你代码里的具体妙用
针对你那个超长的 passschedulecanonicalize.h，IRMapping 可以帮你干掉至少 200 行代码：

处理嵌套 Slice：你不再需要写 while(progress) 这种循环去一层层解嵌套。你只需要按顺序 clone。只要父 Slice 被 clone 过了并存在于 mapper 里，子 Slice 克隆时会自动连上。

清理 ModuleScheduleInfo：你不再需要存储 SliceParams 里的 index 和 partitionIndex。mapper 就是你的天然索引。

跨块合并：当你把多个 routing.RoutingCreate 的内容拉出来时，IRMapping 能保证即使多个块里都用了同一个 %tile，在合并后它们都能正确指向唯一的那个 DeclareTileOp。

5. 什么时候必须用？
Inline（内联）：把一个函数或 Region 的内容搬到另一个地方。

Unrolling（循环展开）：复制多次循环体。

Module 重组：像你现在这样，把原本散落在各处的 Op 集中到一个 dfschedule.host 里。

专家提示
使用 IRMapping 时，最常见的错误是忘记初始化映射。
如果你要搬移的逻辑依赖于外部已经定义好的值（比如 main 函数的参数），你必须在开始 clone 之前，手动把这些初始值 map 进去。

C++
mapper.map(oldFunc.getArgument(0), newFunc.getArgument(0));
你想让我帮你把那段复杂的 createCanonicalizedSchedule 逻辑中，关于 Slice 重建的部分用 IRMapping 重写一下试试吗？这能让代码变得非常丝滑。

You said
使用 %0 = arith.constant 1

%1 = arith.addi %0, %0 作为一个例子

Gemini said
为了让你彻底理解 IRMapping 的威力，我们用你给出的这两行代码做一个**“灵魂克隆”**实验。

想象一下，你现在要把这段逻辑从 FuncA 搬运到 FuncB。

1. 原始 IR (Source)
这是我们想要克隆的对象：

MLIR
// 原有的 Block
%0 = arith.constant 1 : i32
%1 = arith.addi %0, %0 : i32
2. C++ 操作演示
如果我们不用 IRMapping，你克隆出的 %1 会因为找不到它的操作数（即 %0 还留在旧函数里）而导致新函数变成“断头路”。

使用 IRMapping 的标准姿势如下：

C++
mlir::IRMapping mapper;
mlir::OpBuilder builder(context);

// --- 步骤 A: 克隆第一个 Op ---
// 假设 originalOp0 是指那个 arith.constant
Operation *clonedOp0 = builder.clone(*originalOp0, mapper);

// 此时，mapper 内部自动建立了一个“户口登记”：
// { %0 (旧) -> %0_new (新) }

// --- 步骤 B: 克隆第二个 Op ---
// 假设 originalOp1 是指那个 arith.addi
Operation *clonedOp1 = builder.clone(*originalOp1, mapper);

// 魔法就在这里！
// builder.clone 会自动去问 mapper：“嘿，addi 依赖的值 %0 在新世界里叫啥？”
// mapper 告诉它：“叫 %0_new”。
// 于是新生成的 addi 自动连接到了 %0_new 上。
3. 克隆后的 IR (Target)
最终生成的新代码看起来非常自然，SSA 链条是完美的：

MLIR
// 新的 Block
%0_new = arith.constant 1 : i32
%1_new = arith.addi %0_new, %0_new : i32  // 自动指向了新常量
4. 为什么这对你极其重要？
在你那个复杂的 ScheduleCanonicalizePass 中，你有很多嵌套的 extract_slice。如果没有 IRMapping，你的逻辑会变成这样：

找到 Slice_A 的父节点。

去你的 std::map 里查父节点对应的 Value 是否已经创建。

如果创建了，取出来作为参数传给 builder.create<Slice_A>(...)。

使用 IRMapping 后，逻辑简化为：
你只需要按照从根到叶 (PreOrder) 的顺序 clone 所有的 Op。IRMapping 会像影子一样跟着你，自动处理掉所有的指针替换。你根本不需要关心 Slice_A 的父节点到底是谁，只要它在之前被 clone 过，mapper 就能搞定一切。

专家级 Tips
多对一映射：如果你想把多个旧值合并为一个新值（例如你想把所有的 Shim Tile 引用都指向同一个新声明的 Tile），你可以手动调用 mapper.map(oldValue, sharedNewValue)。

清理：如果你的 Pass 要运行多次，记得清空 mapper.clear()。

通过这个例子，你是否发现你之前代码里那套手动维护的 index 逻辑其实可以被这个 mapper 完美替代？要不要我帮你把那部分核心的克隆逻辑重构一下？

You said
CLONE 不是必须指定MAPPER吗，有不用指定MAPPER的吗

Gemini said
是的，在 MLIR 的 C++ API 中，clone 确实有不带 Mapper 的版本，但它的用法非常讲究，如果不小心就会导致编译器崩溃。

作为专家，我帮你拆解一下这两种 clone 的本质区别：

1. 不带 Mapper 的 op->clone()
这是 mlir::Operation 类的一个普通成员函数。

它的行为：它会复制这个 Op 本身、它的属性、它的类型，以及它的 Region。

它的局限性：它会原封不动地复制操作数（Operands）。

例子：
如果你克隆 %1 = addi %0, %0，新生成的 Op 仍然会尝试去连接原始代码里的那个 %0。

什么时候可以用不带 Mapper 的版本？
局部复制：你在同一个 Block 里想复制一份指令，且希望新指令的输入和旧指令一模一样。

只读分析：你只是想临时克隆一份 IR 来做某种实验性分析，不打算把它放回真正的程序流中。

危险信号：如果你把这种“不带 Mapper”的克隆体移动到一个新的函数里，而旧函数随后被删除了，那么克隆体就会指向一个非法内存地址。

2. 带 Mapper 的 builder.clone(op, mapper)
这是 mlir::OpBuilder 或 mlir::IRRewriter 提供的工具函数。

它的行为：它在复制 Op 的同时，会去查这张“户口表”（Mapper）。

它的优势：如果发现操作数在表里有记录，它会自动把新 Op 的输入替换成“新世界”的值。

3. 为什么在你的代码里，Mapper 是“必须”的？
你正在做的是 “结构重组”（从 scf.execute_region 搬运到 dfschedule.host）。在这种情况下，不用 Mapper 会遇到以下两个灾难：

作用域违规 (Scope Violation)：
旧的 SSA 值（比如 %subview）定义在 scf.execute_region 内部。你把它克隆到外面去后，新 Op 还在引用那个“盒子里”的值。这违反了 MLIR 的支配关系 (Dominance)，验证器（Verifier）会报错。

悬挂指针 (Dangling Pointer)：
你的代码最后会调用 removeExecuteRegionsFromMain 把旧的 Region 删掉。删掉的那一刻，旧的 SSA 值也随之消失。如果不通过 Mapper 建立新旧连接，你的新 IR 就会直接“断连”。

4. 专家级对比总结
方式	语法	操作数处理	适用场景
裸克隆	op->clone()	指向旧值	同一 Block 内部复制，不涉及搬迁。
映射克隆	builder.clone(op, mapper)	指向映射后的新值	跨 Block、跨 Region 搬运代码（你的场景）。
💡 一个省力的小技巧
如果你觉得手动管理 IRMapping 还是麻烦，其实还有一个更高级的 API：
rewriter.inlineRegionBefore(sourceRegion, targetBlock, insertionPoint)。

这个函数内部会自动处理所有 Block 内部操作数的连贯性，它比一个一个 clone 要快得多，也更不容易出错。

既然我们讨论到这了，你想看一看如何用 rewriter.clone 配合 IRMapping 简洁地重构你那段 createOneShimBd 的逻辑吗？（能省掉你一大堆手动的 sliceIndex 查找）