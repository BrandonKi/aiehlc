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
    %0 = routing.routingcreatehwmesh row = 4, col = 4 -> i32
    %1 = bufferization.to_tensor %arg0 : memref<256x256xi8>
    %2 = routing.routingcreatescheduletensor %1 : tensor<256x256xi8> shape = [256, 256], dim = 2 -> tensor<256x256xi8>
    %3 = bufferization.to_tensor %arg1 : memref<256x256xi8>
    %4 = routing.routingcreatescheduletensor %3 : tensor<256x256xi8> shape = [256, 256], dim = 2 -> tensor<256x256xi8>
    %5 = bufferization.to_tensor %arg2 : memref<256x256xi8>
    %6 = routing.routingcreatescheduletensor %5 : tensor<256x256xi8> shape = [256, 256], dim = 2 -> tensor<256x256xi8>
    scf.execute_region {
      %7 = routing.partitionmesh mesh = %0, splitnum = 4, splitaxis = "col" : i32 -> i32
      %8 = routing.partitiontensor tensor = %4 : tensor<256x256xi8> {
          splitnum = 4,
          splitdim = 0,
          hw_axis_owner = "col",
          replicate_on = "row",
          single_tile_owner = ""
     } -> tensor<256x256xi8>
      %9 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c0_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %13 = routing.routingextract_tiles %7, %arg3 : i32, i32 -> i32
        %14 = routing.routingextract_data %8, %arg3 : tensor<256x256xi8>, i32 -> tensor<64x256xi8>
        %15 = routing.routingcreatehwiowithtarget targettilelist = %13 : i32 {direction = "input", iotype = "mem2"} -> i32
        %16 = routing.routingmovedatabyio tensordata = %14, hwiowithtarget = %15 : tensor<64x256xi8>, i32 -> i32
        "routing.yield"() : () -> ()
      }
      %10 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c1_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %13 = routing.routingextract_tiles %7, %arg3 : i32, i32 -> i32
        %14 = routing.routingextract_data %8, %arg3 : tensor<256x256xi8>, i32 -> tensor<64x256xi8>
        %15 = routing.routingcreatehwiowithtarget targettilelist = %13 : i32 {direction = "input", iotype = "mem2"} -> i32
        %16 = routing.routingmovedatabyio tensordata = %14, hwiowithtarget = %15 : tensor<64x256xi8>, i32 -> i32
        "routing.yield"() : () -> ()
      }
      %11 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c2_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %13 = routing.routingextract_tiles %7, %arg3 : i32, i32 -> i32
        %14 = routing.routingextract_data %8, %arg3 : tensor<256x256xi8>, i32 -> tensor<64x256xi8>
        %15 = routing.routingcreatehwiowithtarget targettilelist = %13 : i32 {direction = "input", iotype = "mem2"} -> i32
        %16 = routing.routingmovedatabyio tensordata = %14, hwiowithtarget = %15 : tensor<64x256xi8>, i32 -> i32
        "routing.yield"() : () -> ()
      }
      %12 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c3_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %13 = routing.routingextract_tiles %7, %arg3 : i32, i32 -> i32
        %14 = routing.routingextract_data %8, %arg3 : tensor<256x256xi8>, i32 -> tensor<64x256xi8>
        %15 = routing.routingcreatehwiowithtarget targettilelist = %13 : i32 {direction = "input", iotype = "mem2"} -> i32
        %16 = routing.routingmovedatabyio tensordata = %14, hwiowithtarget = %15 : tensor<64x256xi8>, i32 -> i32
        "routing.yield"() : () -> ()
      }
      scf.yield
    } {routing_memo = "col"}
    scf.execute_region {
      %7 = routing.partitionmesh mesh = %0, splitnum = 4, splitaxis = "row" : i32 -> i32
      %8 = routing.partitiontensor tensor = %2 : tensor<256x256xi8> {
          splitnum = 4,
          splitdim = 0,
          hw_axis_owner = "row",
          replicate_on = "col",
          single_tile_owner = ""
     } -> tensor<256x256xi8>
      %9 = routing.partitiontensor tensor = %6 : tensor<256x256xi8> {
          splitnum = 4,
          splitdim = 0,
          hw_axis_owner = "row",
          replicate_on = "col",
          single_tile_owner = ""
     } -> tensor<256x256xi8>
      %10 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c0_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %14 = routing.routingextract_tiles %7, %arg3 : i32, i32 -> i32
        %15 = routing.routingextract_data %8, %arg3 : tensor<256x256xi8>, i32 -> tensor<64x256xi8>
        %16 = routing.routingcreatehwiowithtarget targettilelist = %14 : i32 {direction = "input", iotype = "mem2"} -> i32
        %17 = routing.routingmovedatabyio tensordata = %15, hwiowithtarget = %16 : tensor<64x256xi8>, i32 -> i32
        %18 = routing.routingextract_data %9, %arg3 : tensor<256x256xi8>, i32 -> tensor<64x256xi8>
        %19 = routing.routingroutinggatherout tilelist = %14, tensordata = %18 : i32, tensor<64x256xi8> -> tensor<64x256xi8>
        %20 = routing.routingcreatehwiowithtarget targettilelist = %14 : i32 {direction = "output", iotype = "mem2"} -> i32
        %21 = routing.routingmovedatabyio tensordata = %19, hwiowithtarget = %20 : tensor<64x256xi8>, i32 -> i32
        "routing.yield"() : () -> ()
      }
      %11 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c1_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %14 = routing.routingextract_tiles %7, %arg3 : i32, i32 -> i32
        %15 = routing.routingextract_data %8, %arg3 : tensor<256x256xi8>, i32 -> tensor<64x256xi8>
        %16 = routing.routingcreatehwiowithtarget targettilelist = %14 : i32 {direction = "input", iotype = "mem2"} -> i32
        %17 = routing.routingmovedatabyio tensordata = %15, hwiowithtarget = %16 : tensor<64x256xi8>, i32 -> i32
        %18 = routing.routingextract_data %9, %arg3 : tensor<256x256xi8>, i32 -> tensor<64x256xi8>
        %19 = routing.routingroutinggatherout tilelist = %14, tensordata = %18 : i32, tensor<64x256xi8> -> tensor<64x256xi8>
        %20 = routing.routingcreatehwiowithtarget targettilelist = %14 : i32 {direction = "output", iotype = "mem2"} -> i32
        %21 = routing.routingmovedatabyio tensordata = %19, hwiowithtarget = %20 : tensor<64x256xi8>, i32 -> i32
        "routing.yield"() : () -> ()
      }
      %12 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c2_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %14 = routing.routingextract_tiles %7, %arg3 : i32, i32 -> i32
        %15 = routing.routingextract_data %8, %arg3 : tensor<256x256xi8>, i32 -> tensor<64x256xi8>
        %16 = routing.routingcreatehwiowithtarget targettilelist = %14 : i32 {direction = "input", iotype = "mem2"} -> i32
        %17 = routing.routingmovedatabyio tensordata = %15, hwiowithtarget = %16 : tensor<64x256xi8>, i32 -> i32
        %18 = routing.routingextract_data %9, %arg3 : tensor<256x256xi8>, i32 -> tensor<64x256xi8>
        %19 = routing.routingroutinggatherout tilelist = %14, tensordata = %18 : i32, tensor<64x256xi8> -> tensor<64x256xi8>
        %20 = routing.routingcreatehwiowithtarget targettilelist = %14 : i32 {direction = "output", iotype = "mem2"} -> i32
        %21 = routing.routingmovedatabyio tensordata = %19, hwiowithtarget = %20 : tensor<64x256xi8>, i32 -> i32
        "routing.yield"() : () -> ()
      }
      %13 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c3_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %14 = routing.routingextract_tiles %7, %arg3 : i32, i32 -> i32
        %15 = routing.routingextract_data %8, %arg3 : tensor<256x256xi8>, i32 -> tensor<64x256xi8>
        %16 = routing.routingcreatehwiowithtarget targettilelist = %14 : i32 {direction = "input", iotype = "mem2"} -> i32
        %17 = routing.routingmovedatabyio tensordata = %15, hwiowithtarget = %16 : tensor<64x256xi8>, i32 -> i32
        %18 = routing.routingextract_data %9, %arg3 : tensor<256x256xi8>, i32 -> tensor<64x256xi8>
        %19 = routing.routingroutinggatherout tilelist = %14, tensordata = %18 : i32, tensor<64x256xi8> -> tensor<64x256xi8>
        %20 = routing.routingcreatehwiowithtarget targettilelist = %14 : i32 {direction = "output", iotype = "mem2"} -> i32
        %21 = routing.routingmovedatabyio tensordata = %19, hwiowithtarget = %20 : tensor<64x256xi8>, i32 -> i32
        "routing.yield"() : () -> ()
      }
      scf.yield
    } {routing_memo = "row"}
    return
  }
}
