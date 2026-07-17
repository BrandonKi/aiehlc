module attributes {codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"], routing.pp_depth_map = {tensor_0 = 2 : i32, tensor_1 = 2 : i32, tensor_2 = 2 : i32}} {
  func.func @main(%arg0: memref<16x64xi32>, %arg1: memref<64x16xi32>, %arg2: memref<16x16xi32>) {
    %c0_i32 = arith.constant 0 : i32
    %0 = routing.routingcreatehwmesh row = 1, col = 1 partition = 0, 3, 0, 6 -> i32
    %1 = bufferization.to_tensor %arg0 : memref<16x64xi32>
    %2 = routing.routingcreatescheduletensor %1 : tensor<16x64xi32> shape = [16, 64], dim = 2 -> tensor<16x64xi32>
    %3 = bufferization.to_tensor %arg1 : memref<64x16xi32>
    %4 = routing.routingcreatescheduletensor %3 : tensor<64x16xi32> shape = [64, 16], dim = 2 -> tensor<64x16xi32>
    %5 = bufferization.to_tensor %arg2 : memref<16x16xi32>
    %6 = routing.routingcreatescheduletensor %5 : tensor<16x16xi32> shape = [16, 16], dim = 2 -> tensor<16x16xi32>
    scf.execute_region {
      %7 = routing.partitionmesh mesh = %0, splitnum = 1, splitaxis = "col" : i32 -> i32
      %8 = routing.partitiontensor tensor = %4 : tensor<64x16xi32> {
          splitnum = 1,
          splitdim = 0,
          hw_axis_owner = "col",
          replicate_on = "row",
          single_tile_owner = ""
     } -> tensor<64x16xi32>
      %9 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c0_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %10 = routing.routingextract_tiles %7, %arg3 : i32, i32 -> i32
        %11 = routing.routingextract_data %8, %arg3 : tensor<64x16xi32>, i32 -> tensor<64x16xi32>
        %12 = routing.routingcreatehwiowithtarget targettilelist = %10 : i32 {direction = "input", iotype = "mem2"} -> i32
        %13 = routing.routingmovedatabyio tensordata = %11, hwiowithtarget = %12 : tensor<64x16xi32>, i32 -> i32
        "routing.yield"() : () -> ()
      }
      scf.yield
    } {routing_memo = "col"}
    scf.execute_region {
      %7 = routing.partitionmesh mesh = %0, splitnum = 1, splitaxis = "row" : i32 -> i32
      %8 = routing.partitiontensor tensor = %2 : tensor<16x64xi32> {
          splitnum = 1,
          splitdim = 0,
          hw_axis_owner = "row",
          replicate_on = "col",
          single_tile_owner = ""
     } -> tensor<16x64xi32>
      %9 = routing.partitiontensor tensor = %6 : tensor<16x16xi32> {
          splitnum = 1,
          splitdim = 0,
          hw_axis_owner = "row",
          replicate_on = "col",
          single_tile_owner = ""
     } -> tensor<16x16xi32>
      %10 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c0_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %11 = routing.routingextract_tiles %7, %arg3 : i32, i32 -> i32
        %12 = routing.routingextract_data %8, %arg3 : tensor<16x64xi32>, i32 -> tensor<16x64xi32>
        %13 = routing.routingcreatehwiowithtarget targettilelist = %11 : i32 {direction = "input", iotype = "mem2"} -> i32
        %14 = routing.routingmovedatabyio tensordata = %12, hwiowithtarget = %13 : tensor<16x64xi32>, i32 -> i32
        %15 = routing.routingextract_data %9, %arg3 : tensor<16x16xi32>, i32 -> tensor<16x16xi32>
        %16 = routing.routingroutinggatherout tilelist = %11, tensordata = %15 : i32, tensor<16x16xi32> -> tensor<16x16xi32>
        %17 = routing.routingcreatehwiowithtarget targettilelist = %11 : i32 {direction = "output", iotype = "mem2"} -> i32
        %18 = routing.routingmovedatabyio tensordata = %16, hwiowithtarget = %17 : tensor<16x16xi32>, i32 -> i32
        "routing.yield"() : () -> ()
      }
      scf.yield
    } {routing_memo = "row"}
    return
  }
}
