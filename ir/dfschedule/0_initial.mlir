module attributes {routing.pp_depth_map = {tensor_0 = 2 : i32, tensor_1 = 2 : i32, tensor_2 = 2 : i32}} {
  func.func @main(%arg0: memref<16x64xi32>, %arg1: memref<64x16xi32>, %arg2: memref<16x16xi32>) {
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
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_0 = arith.constant 1 : index
      scf.for %arg3 = %c0 to %c1 step %c1_0 {
        %9 = arith.index_cast %arg3 : index to i32
        %10 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %9 : i32) -> i32{
        ^bb0(%arg4: i32):
          %11 = routing.routingextract_tiles %7, %arg4 : i32, i32 -> i32
          %12 = routing.routingextract_data %8, %arg4 : tensor<64x16xi32>, i32 -> tensor<64x16xi32>
          %13 = routing.routingcreatehwiowithtarget targettilelist = %11 : i32 {direction = "input", iotype = "mem2"} -> i32
          %14 = routing.routingmovedatabyio tensordata = %12, hwiowithtarget = %13 : tensor<64x16xi32>, i32 -> i32
          "routing.yield"() : () -> ()
        }
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
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_0 = arith.constant 1 : index
      scf.for %arg3 = %c0 to %c1 step %c1_0 {
        %10 = arith.index_cast %arg3 : index to i32
        %11 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %10 : i32) -> i32{
        ^bb0(%arg4: i32):
          %12 = routing.routingextract_tiles %7, %arg4 : i32, i32 -> i32
          %13 = routing.routingextract_data %8, %arg4 : tensor<16x64xi32>, i32 -> tensor<16x64xi32>
          %14 = routing.routingcreatehwiowithtarget targettilelist = %12 : i32 {direction = "input", iotype = "mem2"} -> i32
          %15 = routing.routingmovedatabyio tensordata = %13, hwiowithtarget = %14 : tensor<16x64xi32>, i32 -> i32
          %16 = routing.routingextract_data %9, %arg4 : tensor<16x16xi32>, i32 -> tensor<16x16xi32>
          %17 = routing.routingroutinggatherout tilelist = %12, tensordata = %16 : i32, tensor<16x16xi32> -> tensor<16x16xi32>
          %18 = routing.routingcreatehwiowithtarget targettilelist = %12 : i32 {direction = "output", iotype = "mem2"} -> i32
          %19 = routing.routingmovedatabyio tensordata = %17, hwiowithtarget = %18 : tensor<16x16xi32>, i32 -> i32
          "routing.yield"() : () -> ()
        }
      }
      scf.yield
    } {routing_memo = "row"}
    return
  }
}
