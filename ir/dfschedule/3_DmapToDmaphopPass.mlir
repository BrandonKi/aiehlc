module attributes {codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"], routing.pp_depth_map = {tensor_0 = 2 : i32, tensor_1 = 2 : i32, tensor_2 = 2 : i32}} {
  func.func @main(%arg0: memref<16x64xi32>, %arg1: memref<64x16xi32>, %arg2: memref<16x16xi32>) {
    %c0_i32 = arith.constant 0 : i32
    %0 = bufferization.to_tensor %arg0 : memref<16x64xi32>
    %1 = routing.routingcreatescheduletensor %0 : tensor<16x64xi32> shape = [16, 64], dim = 2 -> tensor<16x64xi32>
    %2 = bufferization.to_tensor %arg1 : memref<64x16xi32>
    %3 = routing.routingcreatescheduletensor %2 : tensor<64x16xi32> shape = [64, 16], dim = 2 -> tensor<64x16xi32>
    %4 = bufferization.to_tensor %arg2 : memref<16x16xi32>
    %5 = routing.routingcreatescheduletensor %4 : tensor<16x16xi32> shape = [16, 16], dim = 2 -> tensor<16x16xi32>
    scf.execute_region {
      %6 = routing.partitiontensor tensor = %3 : tensor<64x16xi32> {
          splitnum = 1,
          splitdim = 0,
          hw_axis_owner = "col",
          replicate_on = "row",
          single_tile_owner = ""
     } -> tensor<64x16xi32>
      %7 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c0_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %8 = routing.routingextract_data %6, %arg3 : tensor<64x16xi32>, i32 -> tensor<64x16xi32>
        %9 = dmaphop.tile{TILETYPE = "core", col = 0, row = 3} -> !dmaphop.tile
        %10 = dmaphop.port @f0_corePortIn0 on %9 { direction = "In", direction_channel = 1 } : !dmaphop.tile -> !dmaphop.port
        %11 = dmaphop.port @f0_corePortOut0 on %9 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %12 = dmaphop.consumer @f0_consumer0 {dma_port = 1 : i64, from = @f0_corePortIn0}
        %13 = dmaphop.tile{TILETYPE = "shim", col = 0, row = 0} -> !dmaphop.tile
        %14 = dmaphop.port @f0_shimPortOut on %13 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %15 = dmaphop.port @f0_shimPortIn on %13 { direction = "In", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %16 = dmaphop.create_hop %14 -> %10 -> !dmaphop.hop
        %17 = dmaphop.create_path[%16] {producers = [[@f0_shimPortIn]], consumers = [[@f0_consumer0]], tee_points = [[]]} -> !dmaphop.path
        dmaphop.push %8 into %17 consumer(%8 at %10) : tensor<64x16xi32> !dmaphop.path tensor<64x16xi32> !dmaphop.port
        dmaphop.sync %17
        "routing.yield"() : () -> ()
      }
      scf.yield
    } {routing_memo = "col"}
    scf.execute_region {
      %6 = routing.partitiontensor tensor = %1 : tensor<16x64xi32> {
          splitnum = 1,
          splitdim = 0,
          hw_axis_owner = "row",
          replicate_on = "col",
          single_tile_owner = ""
     } -> tensor<16x64xi32>
      %7 = routing.partitiontensor tensor = %5 : tensor<16x16xi32> {
          splitnum = 1,
          splitdim = 0,
          hw_axis_owner = "row",
          replicate_on = "col",
          single_tile_owner = ""
     } -> tensor<16x16xi32>
      %8 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c0_i32 : i32) -> i32{
      ^bb0(%arg3: i32):
        %9 = routing.routingextract_data %6, %arg3 : tensor<16x64xi32>, i32 -> tensor<16x64xi32>
        %10 = dmaphop.tile{TILETYPE = "core", col = 0, row = 3} -> !dmaphop.tile
        %11 = dmaphop.port @f1_corePortIn0 on %10 { direction = "In", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %12 = dmaphop.port @f1_corePortOut0 on %10 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %13 = dmaphop.consumer @f1_consumer0 {dma_port = 0 : i64, from = @f1_corePortIn0}
        %14 = dmaphop.tile{TILETYPE = "shim", col = 0, row = 0} -> !dmaphop.tile
        %15 = dmaphop.port @f1_shimPortOut on %14 { direction = "Out", direction_channel = 1 } : !dmaphop.tile -> !dmaphop.port
        %16 = dmaphop.port @f1_shimPortIn on %14 { direction = "In", direction_channel = 1 } : !dmaphop.tile -> !dmaphop.port
        %17 = dmaphop.create_hop %15 -> %11 -> !dmaphop.hop
        %18 = dmaphop.create_path[%17] {producers = [[@f1_shimPortIn]], consumers = [[@f1_consumer0]], tee_points = [[]]} -> !dmaphop.path
        dmaphop.push %9 into %18 consumer(%9 at %11) : tensor<16x64xi32> !dmaphop.path tensor<16x64xi32> !dmaphop.port
        dmaphop.sync %18
        %19 = routing.routingextract_data %7, %arg3 : tensor<16x16xi32>, i32 -> tensor<16x16xi32>
        %20 = dmaphop.tile{TILETYPE = "core", col = 0, row = 3} -> !dmaphop.tile
        %21 = dmaphop.port @f2_corePortIn0 on %20 { direction = "In", direction_channel = 2 } : !dmaphop.tile -> !dmaphop.port
        %22 = dmaphop.port @f2_corePortOut0 on %20 { direction = "Out", direction_channel = 0, dmapktid = 1 : i32 } : !dmaphop.tile -> !dmaphop.port
        %23 = dmaphop.producer @f2_producer0 {dma_port = 0 : i64, tp = @f2_corePortOut0}
        %24 = dmaphop.tile{TILETYPE = "shim", col = 0, row = 0} -> !dmaphop.tile
        %25 = dmaphop.port @f2_shimPortOut on %24 { direction = "Out", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %26 = dmaphop.port @f2_shimPortIn on %24 { direction = "In", direction_channel = 0 } : !dmaphop.tile -> !dmaphop.port
        %27 = dmaphop.create_hop %22 -> %26 -> !dmaphop.hop
        %28 = dmaphop.create_path[%27] {producers = [[@f2_producer0]], consumers = [[@f2_shimPortOut]], tee_points = [[]]} -> !dmaphop.path
        %extracted_slice = tensor.extract_slice %19[0, 0] [16, 16] [1, 1] {tag = "producer0"} : tensor<16x16xi32> to tensor<16x16xi32>
        dmaphop.pull %19 from %28 producer(%extracted_slice at %21) : tensor<16x16xi32> !dmaphop.path tensor<16x16xi32> !dmaphop.port
        dmaphop.sync %28
        "routing.yield"() : () -> ()
      }
      scf.yield
    } {routing_memo = "row"}
    return
  }
}
