// ******************************************************************************
// * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
// * SPDX-License-Identifier: Apache-2.0
// ******************************************************************************

module attributes {codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"]} {
  func.func @main(%arg0: memref<256x256xi8>, %arg1: memref<256x256xi8>, %arg2: memref<256x256xi8>) {
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = bufferization.to_tensor %arg0 : memref<256x256xi8>
    %1 = dfscheblueprint.declare_data %0 : tensor<256x256xi8> -> tensor<256x256xi8>
    %2 = bufferization.to_tensor %arg1 : memref<256x256xi8>
    %3 = dfscheblueprint.declare_data %2 : tensor<256x256xi8> -> tensor<256x256xi8>
    %4 = bufferization.to_tensor %arg2 : memref<256x256xi8>
    %5 = dfscheblueprint.declare_data %4 : tensor<256x256xi8> -> tensor<256x256xi8>
    scf.execute_region {
      %6 = routing.partitiontensor tensor = %3 : tensor<256x256xi8> {
          splitnum = 4,
          splitdim = 0,
          hw_axis_owner = "col",
          replicate_on = "row",
          single_tile_owner = ""
     } -> tensor<256x256xi8>
      %7 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c0_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        dfscheblueprint.tile_group @group_src_0{tiles = [[2, 0]]}
        dfscheblueprint.tile_group @group_dst_0{tiles = [[0, 3], [0, 4], [0, 5], [0, 6]]}
        %extracted_slice = tensor.extract_slice %6[0, 0] [64, 256] [1, 1] {tag = "partitionslice0"} : tensor<256x256xi8> to tensor<64x256xi8>
        %11 = dfscheblueprint.data_slice @consumer_slice_0_0 wrap %extracted_slice : tensor<64x256xi8>
        %12 = dfscheblueprint.data_slice @consumer_slice_0_1 wrap %extracted_slice : tensor<64x256xi8>
        %13 = dfscheblueprint.data_slice @consumer_slice_0_2 wrap %extracted_slice : tensor<64x256xi8>
        %14 = dfscheblueprint.data_slice @consumer_slice_0_3 wrap %extracted_slice : tensor<64x256xi8>
        dfscheblueprint.flowconfig @flow_src_0 {
          target = @group_src_0,
          view = %extracted_slice : tensor<64x256xi8>,
          distribution = "root",
          dma = #dfscheblueprint.DMA<channels = [0], direction = MM2S>
          ,type = "shim"
          ,data_id = 0 : i32
        }
        dfscheblueprint.flowconfig @flow_dst_0 {
          target = @group_dst_0,
          view = %extracted_slice : tensor<64x256xi8>,
          distribution = "linear",
          dma = #dfscheblueprint.DMA<channels = [1], direction = S2MM>
          ,slice_symbols = [@consumer_slice_0_0, @consumer_slice_0_1, @consumer_slice_0_2, @consumer_slice_0_3]
          ,type = "core"
        }
        dfscheblueprint.flow_transfer @transfer_0 {
          type = "one_to_many",
          from = @flow_src_0,
          to = @flow_dst_0
          ,ordering = "sequential"
          ,base_packet_id = 0 : i32,
          flow_index = 0 : i32
        }
        "routing.yield"() : () -> ()
      }
      %8 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c1_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        dfscheblueprint.tile_group @group_src_1{tiles = [[2, 0]]}
        dfscheblueprint.tile_group @group_dst_1{tiles = [[1, 3], [1, 4], [1, 5], [1, 6]]}
        %extracted_slice = tensor.extract_slice %6[64, 0] [64, 256] [1, 1] {tag = "partitionslice1"} : tensor<256x256xi8> to tensor<64x256xi8>
        %11 = dfscheblueprint.data_slice @consumer_slice_1_0 wrap %extracted_slice : tensor<64x256xi8>
        %12 = dfscheblueprint.data_slice @consumer_slice_1_1 wrap %extracted_slice : tensor<64x256xi8>
        %13 = dfscheblueprint.data_slice @consumer_slice_1_2 wrap %extracted_slice : tensor<64x256xi8>
        %14 = dfscheblueprint.data_slice @consumer_slice_1_3 wrap %extracted_slice : tensor<64x256xi8>
        dfscheblueprint.flowconfig @flow_src_1 {
          target = @group_src_1,
          view = %extracted_slice : tensor<64x256xi8>,
          distribution = "root",
          dma = #dfscheblueprint.DMA<channels = [1], direction = MM2S>
          ,type = "shim"
          ,data_id = 0 : i32
        }
        dfscheblueprint.flowconfig @flow_dst_1 {
          target = @group_dst_1,
          view = %extracted_slice : tensor<64x256xi8>,
          distribution = "linear",
          dma = #dfscheblueprint.DMA<channels = [1], direction = S2MM>
          ,slice_symbols = [@consumer_slice_1_0, @consumer_slice_1_1, @consumer_slice_1_2, @consumer_slice_1_3]
          ,type = "core"
        }
        dfscheblueprint.flow_transfer @transfer_1 {
          type = "one_to_many",
          from = @flow_src_1,
          to = @flow_dst_1
          ,ordering = "sequential"
          ,base_packet_id = 0 : i32,
          flow_index = 1 : i32
        }
        "routing.yield"() : () -> ()
      }
      %9 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c2_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        dfscheblueprint.tile_group @group_src_2{tiles = [[3, 0]]}
        dfscheblueprint.tile_group @group_dst_2{tiles = [[2, 3], [2, 4], [2, 5], [2, 6]]}
        %extracted_slice = tensor.extract_slice %6[128, 0] [64, 256] [1, 1] {tag = "partitionslice2"} : tensor<256x256xi8> to tensor<64x256xi8>
        %11 = dfscheblueprint.data_slice @consumer_slice_2_0 wrap %extracted_slice : tensor<64x256xi8>
        %12 = dfscheblueprint.data_slice @consumer_slice_2_1 wrap %extracted_slice : tensor<64x256xi8>
        %13 = dfscheblueprint.data_slice @consumer_slice_2_2 wrap %extracted_slice : tensor<64x256xi8>
        %14 = dfscheblueprint.data_slice @consumer_slice_2_3 wrap %extracted_slice : tensor<64x256xi8>
        dfscheblueprint.flowconfig @flow_src_2 {
          target = @group_src_2,
          view = %extracted_slice : tensor<64x256xi8>,
          distribution = "root",
          dma = #dfscheblueprint.DMA<channels = [0], direction = MM2S>
          ,type = "shim"
          ,data_id = 0 : i32
        }
        dfscheblueprint.flowconfig @flow_dst_2 {
          target = @group_dst_2,
          view = %extracted_slice : tensor<64x256xi8>,
          distribution = "linear",
          dma = #dfscheblueprint.DMA<channels = [1], direction = S2MM>
          ,slice_symbols = [@consumer_slice_2_0, @consumer_slice_2_1, @consumer_slice_2_2, @consumer_slice_2_3]
          ,type = "core"
        }
        dfscheblueprint.flow_transfer @transfer_2 {
          type = "one_to_many",
          from = @flow_src_2,
          to = @flow_dst_2
          ,ordering = "sequential"
          ,base_packet_id = 0 : i32,
          flow_index = 2 : i32
        }
        "routing.yield"() : () -> ()
      }
      %10 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c3_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        dfscheblueprint.tile_group @group_src_3{tiles = [[3, 0]]}
        dfscheblueprint.tile_group @group_dst_3{tiles = [[3, 3], [3, 4], [3, 5], [3, 6]]}
        %extracted_slice = tensor.extract_slice %6[192, 0] [64, 256] [1, 1] {tag = "partitionslice3"} : tensor<256x256xi8> to tensor<64x256xi8>
        %11 = dfscheblueprint.data_slice @consumer_slice_3_0 wrap %extracted_slice : tensor<64x256xi8>
        %12 = dfscheblueprint.data_slice @consumer_slice_3_1 wrap %extracted_slice : tensor<64x256xi8>
        %13 = dfscheblueprint.data_slice @consumer_slice_3_2 wrap %extracted_slice : tensor<64x256xi8>
        %14 = dfscheblueprint.data_slice @consumer_slice_3_3 wrap %extracted_slice : tensor<64x256xi8>
        dfscheblueprint.flowconfig @flow_src_3 {
          target = @group_src_3,
          view = %extracted_slice : tensor<64x256xi8>,
          distribution = "root",
          dma = #dfscheblueprint.DMA<channels = [1], direction = MM2S>
          ,type = "shim"
          ,data_id = 0 : i32
        }
        dfscheblueprint.flowconfig @flow_dst_3 {
          target = @group_dst_3,
          view = %extracted_slice : tensor<64x256xi8>,
          distribution = "linear",
          dma = #dfscheblueprint.DMA<channels = [1], direction = S2MM>
          ,slice_symbols = [@consumer_slice_3_0, @consumer_slice_3_1, @consumer_slice_3_2, @consumer_slice_3_3]
          ,type = "core"
        }
        dfscheblueprint.flow_transfer @transfer_3 {
          type = "one_to_many",
          from = @flow_src_3,
          to = @flow_dst_3
          ,ordering = "sequential"
          ,base_packet_id = 0 : i32,
          flow_index = 3 : i32
        }
        "routing.yield"() : () -> ()
      }
      scf.yield
    } {routing_memo = "col"}
    scf.execute_region {
      %6 = routing.partitiontensor tensor = %1 : tensor<256x256xi8> {
          splitnum = 4,
          splitdim = 0,
          hw_axis_owner = "row",
          replicate_on = "col",
          single_tile_owner = ""
     } -> tensor<256x256xi8>
      %7 = routing.partitiontensor tensor = %5 : tensor<256x256xi8> {
          splitnum = 4,
          splitdim = 0,
          hw_axis_owner = "row",
          replicate_on = "col",
          single_tile_owner = ""
     } -> tensor<256x256xi8>
      %8 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c0_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        dfscheblueprint.tile_group @group_src_5{tiles = [[0, 3], [1, 3], [2, 3], [3, 3]]}
        dfscheblueprint.tile_group @group_dst_5{tiles = [[3, 0]]}
        dfscheblueprint.tile_group @group_src_4{tiles = [[6, 0]]}
        dfscheblueprint.tile_group @group_dst_4{tiles = [[0, 3], [1, 3], [2, 3], [3, 3]]}
        %extracted_slice = tensor.extract_slice %6[0, 0] [64, 256] [1, 1] {tag = "partitionslice0"} : tensor<256x256xi8> to tensor<64x256xi8>
        %12 = dfscheblueprint.data_slice @consumer_slice_4_0 wrap %extracted_slice : tensor<64x256xi8>
        %13 = dfscheblueprint.data_slice @consumer_slice_4_1 wrap %extracted_slice : tensor<64x256xi8>
        %14 = dfscheblueprint.data_slice @consumer_slice_4_2 wrap %extracted_slice : tensor<64x256xi8>
        %15 = dfscheblueprint.data_slice @consumer_slice_4_3 wrap %extracted_slice : tensor<64x256xi8>
        dfscheblueprint.flowconfig @flow_src_4 {
          target = @group_src_4,
          view = %extracted_slice : tensor<64x256xi8>,
          distribution = "root",
          dma = #dfscheblueprint.DMA<channels = [0], direction = MM2S>
          ,type = "shim"
          ,data_id = 1 : i32
        }
        dfscheblueprint.flowconfig @flow_dst_4 {
          target = @group_dst_4,
          view = %extracted_slice : tensor<64x256xi8>,
          distribution = "linear",
          dma = #dfscheblueprint.DMA<channels = [0], direction = S2MM>
          ,slice_symbols = [@consumer_slice_4_0, @consumer_slice_4_1, @consumer_slice_4_2, @consumer_slice_4_3]
          ,type = "core"
        }
        dfscheblueprint.flow_transfer @transfer_4 {
          type = "one_to_many",
          from = @flow_src_4,
          to = @flow_dst_4
          ,ordering = "sequential"
          ,base_packet_id = 0 : i32,
          flow_index = 4 : i32
        }
        %extracted_slice_0 = tensor.extract_slice %7[0, 0] [64, 256] [1, 1] {tag = "partitionslice0"} : tensor<256x256xi8> to tensor<64x256xi8>
        %extracted_slice_1 = tensor.extract_slice %extracted_slice_0[0, 0] [16, 256] [1, 1] {tag = "producer0"} : tensor<64x256xi8> to tensor<16x256xi8>
        %extracted_slice_2 = tensor.extract_slice %extracted_slice_0[16, 0] [16, 256] [1, 1] {tag = "producer1"} : tensor<64x256xi8> to tensor<16x256xi8>
        %extracted_slice_3 = tensor.extract_slice %extracted_slice_0[32, 0] [16, 256] [1, 1] {tag = "producer2"} : tensor<64x256xi8> to tensor<16x256xi8>
        %extracted_slice_4 = tensor.extract_slice %extracted_slice_0[48, 0] [16, 256] [1, 1] {tag = "producer3"} : tensor<64x256xi8> to tensor<16x256xi8>
        %16 = dfscheblueprint.data_slice @producer_slice_5_0 wrap %extracted_slice_1 : tensor<16x256xi8>
        %17 = dfscheblueprint.data_slice @producer_slice_5_1 wrap %extracted_slice_2 : tensor<16x256xi8>
        %18 = dfscheblueprint.data_slice @producer_slice_5_2 wrap %extracted_slice_3 : tensor<16x256xi8>
        %19 = dfscheblueprint.data_slice @producer_slice_5_3 wrap %extracted_slice_4 : tensor<16x256xi8>
        dfscheblueprint.flowconfig @flow_src_5 {
          target = @group_src_5,
          view = %extracted_slice_0 : tensor<64x256xi8>,
          distribution = "linear",
          dma = #dfscheblueprint.DMA<channels = [0], direction = MM2S>
          ,slice_symbols = [@producer_slice_5_0, @producer_slice_5_1, @producer_slice_5_2, @producer_slice_5_3]
          ,type = "core"
        }
        dfscheblueprint.flowconfig @flow_dst_5 {
          target = @group_dst_5,
          view = %extracted_slice_0 : tensor<64x256xi8>,
          distribution = "root",
          dma = #dfscheblueprint.DMA<channels = [0], direction = S2MM>
          ,type = "shim"
          ,data_id = 2 : i32
          ,shim_dim_strides = [4 : i32, 256 : i32, 64 : i32]
          ,shim_dim_wraps = [16 : i32, 64 : i32, 4 : i32]
        }
        dfscheblueprint.flow_transfer @transfer_5 {
          type = "many_to_one",
          from = @flow_src_5,
          to = @flow_dst_5
          ,ordering = "sequential"
          ,base_packet_id = 1 : i32,
          flow_index = 5 : i32
        }
        "routing.yield"() : () -> ()
      }
      %9 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c1_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        dfscheblueprint.tile_group @group_src_7{tiles = [[0, 4], [1, 4], [2, 4], [3, 4]]}
        dfscheblueprint.tile_group @group_dst_7{tiles = [[3, 0]]}
        dfscheblueprint.tile_group @group_src_6{tiles = [[6, 0]]}
        dfscheblueprint.tile_group @group_dst_6{tiles = [[0, 4], [1, 4], [2, 4], [3, 4]]}
        %extracted_slice = tensor.extract_slice %6[64, 0] [64, 256] [1, 1] {tag = "partitionslice1"} : tensor<256x256xi8> to tensor<64x256xi8>
        %12 = dfscheblueprint.data_slice @consumer_slice_6_0 wrap %extracted_slice : tensor<64x256xi8>
        %13 = dfscheblueprint.data_slice @consumer_slice_6_1 wrap %extracted_slice : tensor<64x256xi8>
        %14 = dfscheblueprint.data_slice @consumer_slice_6_2 wrap %extracted_slice : tensor<64x256xi8>
        %15 = dfscheblueprint.data_slice @consumer_slice_6_3 wrap %extracted_slice : tensor<64x256xi8>
        dfscheblueprint.flowconfig @flow_src_6 {
          target = @group_src_6,
          view = %extracted_slice : tensor<64x256xi8>,
          distribution = "root",
          dma = #dfscheblueprint.DMA<channels = [1], direction = MM2S>
          ,type = "shim"
          ,data_id = 1 : i32
        }
        dfscheblueprint.flowconfig @flow_dst_6 {
          target = @group_dst_6,
          view = %extracted_slice : tensor<64x256xi8>,
          distribution = "linear",
          dma = #dfscheblueprint.DMA<channels = [0], direction = S2MM>
          ,slice_symbols = [@consumer_slice_6_0, @consumer_slice_6_1, @consumer_slice_6_2, @consumer_slice_6_3]
          ,type = "core"
        }
        dfscheblueprint.flow_transfer @transfer_6 {
          type = "one_to_many",
          from = @flow_src_6,
          to = @flow_dst_6
          ,ordering = "sequential"
          ,base_packet_id = 0 : i32,
          flow_index = 6 : i32
        }
        %extracted_slice_0 = tensor.extract_slice %7[64, 0] [64, 256] [1, 1] {tag = "partitionslice1"} : tensor<256x256xi8> to tensor<64x256xi8>
        %extracted_slice_1 = tensor.extract_slice %extracted_slice_0[0, 0] [16, 256] [1, 1] {tag = "producer0"} : tensor<64x256xi8> to tensor<16x256xi8>
        %extracted_slice_2 = tensor.extract_slice %extracted_slice_0[16, 0] [16, 256] [1, 1] {tag = "producer1"} : tensor<64x256xi8> to tensor<16x256xi8>
        %extracted_slice_3 = tensor.extract_slice %extracted_slice_0[32, 0] [16, 256] [1, 1] {tag = "producer2"} : tensor<64x256xi8> to tensor<16x256xi8>
        %extracted_slice_4 = tensor.extract_slice %extracted_slice_0[48, 0] [16, 256] [1, 1] {tag = "producer3"} : tensor<64x256xi8> to tensor<16x256xi8>
        %16 = dfscheblueprint.data_slice @producer_slice_7_0 wrap %extracted_slice_1 : tensor<16x256xi8>
        %17 = dfscheblueprint.data_slice @producer_slice_7_1 wrap %extracted_slice_2 : tensor<16x256xi8>
        %18 = dfscheblueprint.data_slice @producer_slice_7_2 wrap %extracted_slice_3 : tensor<16x256xi8>
        %19 = dfscheblueprint.data_slice @producer_slice_7_3 wrap %extracted_slice_4 : tensor<16x256xi8>
        dfscheblueprint.flowconfig @flow_src_7 {
          target = @group_src_7,
          view = %extracted_slice_0 : tensor<64x256xi8>,
          distribution = "linear",
          dma = #dfscheblueprint.DMA<channels = [0], direction = MM2S>
          ,slice_symbols = [@producer_slice_7_0, @producer_slice_7_1, @producer_slice_7_2, @producer_slice_7_3]
          ,type = "core"
        }
        dfscheblueprint.flowconfig @flow_dst_7 {
          target = @group_dst_7,
          view = %extracted_slice_0 : tensor<64x256xi8>,
          distribution = "root",
          dma = #dfscheblueprint.DMA<channels = [1], direction = S2MM>
          ,type = "shim"
          ,data_id = 2 : i32
          ,shim_dim_strides = [4 : i32, 256 : i32, 64 : i32]
          ,shim_dim_wraps = [16 : i32, 64 : i32, 4 : i32]
        }
        dfscheblueprint.flow_transfer @transfer_7 {
          type = "many_to_one",
          from = @flow_src_7,
          to = @flow_dst_7
          ,ordering = "sequential"
          ,base_packet_id = 5 : i32,
          flow_index = 7 : i32
        }
        "routing.yield"() : () -> ()
      }
      %10 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c2_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        dfscheblueprint.tile_group @group_src_9{tiles = [[0, 5], [1, 5], [2, 5], [3, 5]]}
        dfscheblueprint.tile_group @group_dst_9{tiles = [[2, 0]]}
        dfscheblueprint.tile_group @group_src_8{tiles = [[7, 0]]}
        dfscheblueprint.tile_group @group_dst_8{tiles = [[0, 5], [1, 5], [2, 5], [3, 5]]}
        %extracted_slice = tensor.extract_slice %6[128, 0] [64, 256] [1, 1] {tag = "partitionslice2"} : tensor<256x256xi8> to tensor<64x256xi8>
        %12 = dfscheblueprint.data_slice @consumer_slice_8_0 wrap %extracted_slice : tensor<64x256xi8>
        %13 = dfscheblueprint.data_slice @consumer_slice_8_1 wrap %extracted_slice : tensor<64x256xi8>
        %14 = dfscheblueprint.data_slice @consumer_slice_8_2 wrap %extracted_slice : tensor<64x256xi8>
        %15 = dfscheblueprint.data_slice @consumer_slice_8_3 wrap %extracted_slice : tensor<64x256xi8>
        dfscheblueprint.flowconfig @flow_src_8 {
          target = @group_src_8,
          view = %extracted_slice : tensor<64x256xi8>,
          distribution = "root",
          dma = #dfscheblueprint.DMA<channels = [0], direction = MM2S>
          ,type = "shim"
          ,data_id = 1 : i32
        }
        dfscheblueprint.flowconfig @flow_dst_8 {
          target = @group_dst_8,
          view = %extracted_slice : tensor<64x256xi8>,
          distribution = "linear",
          dma = #dfscheblueprint.DMA<channels = [0], direction = S2MM>
          ,slice_symbols = [@consumer_slice_8_0, @consumer_slice_8_1, @consumer_slice_8_2, @consumer_slice_8_3]
          ,type = "core"
        }
        dfscheblueprint.flow_transfer @transfer_8 {
          type = "one_to_many",
          from = @flow_src_8,
          to = @flow_dst_8
          ,ordering = "sequential"
          ,base_packet_id = 0 : i32,
          flow_index = 8 : i32
        }
        %extracted_slice_0 = tensor.extract_slice %7[128, 0] [64, 256] [1, 1] {tag = "partitionslice2"} : tensor<256x256xi8> to tensor<64x256xi8>
        %extracted_slice_1 = tensor.extract_slice %extracted_slice_0[0, 0] [16, 256] [1, 1] {tag = "producer0"} : tensor<64x256xi8> to tensor<16x256xi8>
        %extracted_slice_2 = tensor.extract_slice %extracted_slice_0[16, 0] [16, 256] [1, 1] {tag = "producer1"} : tensor<64x256xi8> to tensor<16x256xi8>
        %extracted_slice_3 = tensor.extract_slice %extracted_slice_0[32, 0] [16, 256] [1, 1] {tag = "producer2"} : tensor<64x256xi8> to tensor<16x256xi8>
        %extracted_slice_4 = tensor.extract_slice %extracted_slice_0[48, 0] [16, 256] [1, 1] {tag = "producer3"} : tensor<64x256xi8> to tensor<16x256xi8>
        %16 = dfscheblueprint.data_slice @producer_slice_9_0 wrap %extracted_slice_1 : tensor<16x256xi8>
        %17 = dfscheblueprint.data_slice @producer_slice_9_1 wrap %extracted_slice_2 : tensor<16x256xi8>
        %18 = dfscheblueprint.data_slice @producer_slice_9_2 wrap %extracted_slice_3 : tensor<16x256xi8>
        %19 = dfscheblueprint.data_slice @producer_slice_9_3 wrap %extracted_slice_4 : tensor<16x256xi8>
        dfscheblueprint.flowconfig @flow_src_9 {
          target = @group_src_9,
          view = %extracted_slice_0 : tensor<64x256xi8>,
          distribution = "linear",
          dma = #dfscheblueprint.DMA<channels = [0], direction = MM2S>
          ,slice_symbols = [@producer_slice_9_0, @producer_slice_9_1, @producer_slice_9_2, @producer_slice_9_3]
          ,type = "core"
        }
        dfscheblueprint.flowconfig @flow_dst_9 {
          target = @group_dst_9,
          view = %extracted_slice_0 : tensor<64x256xi8>,
          distribution = "root",
          dma = #dfscheblueprint.DMA<channels = [0], direction = S2MM>
          ,type = "shim"
          ,data_id = 2 : i32
          ,shim_dim_strides = [4 : i32, 256 : i32, 64 : i32]
          ,shim_dim_wraps = [16 : i32, 64 : i32, 4 : i32]
        }
        dfscheblueprint.flow_transfer @transfer_9 {
          type = "many_to_one",
          from = @flow_src_9,
          to = @flow_dst_9
          ,ordering = "sequential"
          ,base_packet_id = 9 : i32,
          flow_index = 9 : i32
        }
        "routing.yield"() : () -> ()
      }
      %11 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c3_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        dfscheblueprint.tile_group @group_src_11{tiles = [[0, 6], [1, 6], [2, 6], [3, 6]]}
        dfscheblueprint.tile_group @group_dst_11{tiles = [[2, 0]]}
        dfscheblueprint.tile_group @group_src_10{tiles = [[7, 0]]}
        dfscheblueprint.tile_group @group_dst_10{tiles = [[0, 6], [1, 6], [2, 6], [3, 6]]}
        %extracted_slice = tensor.extract_slice %6[192, 0] [64, 256] [1, 1] {tag = "partitionslice3"} : tensor<256x256xi8> to tensor<64x256xi8>
        %12 = dfscheblueprint.data_slice @consumer_slice_10_0 wrap %extracted_slice : tensor<64x256xi8>
        %13 = dfscheblueprint.data_slice @consumer_slice_10_1 wrap %extracted_slice : tensor<64x256xi8>
        %14 = dfscheblueprint.data_slice @consumer_slice_10_2 wrap %extracted_slice : tensor<64x256xi8>
        %15 = dfscheblueprint.data_slice @consumer_slice_10_3 wrap %extracted_slice : tensor<64x256xi8>
        dfscheblueprint.flowconfig @flow_src_10 {
          target = @group_src_10,
          view = %extracted_slice : tensor<64x256xi8>,
          distribution = "root",
          dma = #dfscheblueprint.DMA<channels = [1], direction = MM2S>
          ,type = "shim"
          ,data_id = 1 : i32
        }
        dfscheblueprint.flowconfig @flow_dst_10 {
          target = @group_dst_10,
          view = %extracted_slice : tensor<64x256xi8>,
          distribution = "linear",
          dma = #dfscheblueprint.DMA<channels = [0], direction = S2MM>
          ,slice_symbols = [@consumer_slice_10_0, @consumer_slice_10_1, @consumer_slice_10_2, @consumer_slice_10_3]
          ,type = "core"
        }
        dfscheblueprint.flow_transfer @transfer_10 {
          type = "one_to_many",
          from = @flow_src_10,
          to = @flow_dst_10
          ,ordering = "sequential"
          ,base_packet_id = 0 : i32,
          flow_index = 10 : i32
        }
        %extracted_slice_0 = tensor.extract_slice %7[192, 0] [64, 256] [1, 1] {tag = "partitionslice3"} : tensor<256x256xi8> to tensor<64x256xi8>
        %extracted_slice_1 = tensor.extract_slice %extracted_slice_0[0, 0] [16, 256] [1, 1] {tag = "producer0"} : tensor<64x256xi8> to tensor<16x256xi8>
        %extracted_slice_2 = tensor.extract_slice %extracted_slice_0[16, 0] [16, 256] [1, 1] {tag = "producer1"} : tensor<64x256xi8> to tensor<16x256xi8>
        %extracted_slice_3 = tensor.extract_slice %extracted_slice_0[32, 0] [16, 256] [1, 1] {tag = "producer2"} : tensor<64x256xi8> to tensor<16x256xi8>
        %extracted_slice_4 = tensor.extract_slice %extracted_slice_0[48, 0] [16, 256] [1, 1] {tag = "producer3"} : tensor<64x256xi8> to tensor<16x256xi8>
        %16 = dfscheblueprint.data_slice @producer_slice_11_0 wrap %extracted_slice_1 : tensor<16x256xi8>
        %17 = dfscheblueprint.data_slice @producer_slice_11_1 wrap %extracted_slice_2 : tensor<16x256xi8>
        %18 = dfscheblueprint.data_slice @producer_slice_11_2 wrap %extracted_slice_3 : tensor<16x256xi8>
        %19 = dfscheblueprint.data_slice @producer_slice_11_3 wrap %extracted_slice_4 : tensor<16x256xi8>
        dfscheblueprint.flowconfig @flow_src_11 {
          target = @group_src_11,
          view = %extracted_slice_0 : tensor<64x256xi8>,
          distribution = "linear",
          dma = #dfscheblueprint.DMA<channels = [0], direction = MM2S>
          ,slice_symbols = [@producer_slice_11_0, @producer_slice_11_1, @producer_slice_11_2, @producer_slice_11_3]
          ,type = "core"
        }
        dfscheblueprint.flowconfig @flow_dst_11 {
          target = @group_dst_11,
          view = %extracted_slice_0 : tensor<64x256xi8>,
          distribution = "root",
          dma = #dfscheblueprint.DMA<channels = [1], direction = S2MM>
          ,type = "shim"
          ,data_id = 2 : i32
          ,shim_dim_strides = [4 : i32, 256 : i32, 64 : i32]
          ,shim_dim_wraps = [16 : i32, 64 : i32, 4 : i32]
        }
        dfscheblueprint.flow_transfer @transfer_11 {
          type = "many_to_one",
          from = @flow_src_11,
          to = @flow_dst_11
          ,ordering = "sequential"
          ,base_packet_id = 13 : i32,
          flow_index = 11 : i32
        }
        "routing.yield"() : () -> ()
      }
      scf.yield
    } {routing_memo = "row"}
    return
  }
}
