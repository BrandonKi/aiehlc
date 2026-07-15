module attributes {routing.fullconnect_auto = 0 : i64, routing.pp_depth_map = {tensor_0 = 2 : i32, tensor_1 = 2 : i32, tensor_2 = 2 : i32}, routing.spatial_halo_buf_size = 4636 : i64, routing.spatial_out_rounds = 16 : i64, tensor_0.halo = {k_rounds = 4 : i32, k_slice = 244 : i32, k_step = 224 : i32, l2_rounds = 4 : i32, l2_slice = 19 : i32, l2_step = 14 : i32, ow_t = 28 : i32, row_pitch = 920 : i32, slice = 61 : i32, split_dim = 0 : i32, step = 56 : i32, w_rounds = 4 : i32, w_slice = 61 : i32, w_step = 56 : i32}, tensor_0.layout_transform = "dma_shuffle", tensor_1.layout_transform = "dma_shuffle"} {
  func.func @main(%arg0: memref<230x920xi8>, %arg1: memref<196x64xi8>, %arg2: memref<12544x64xi8>) {
    %0 = routing.routingcreatehwmesh row = 4, col = 4 partition = 3, 6, 0, 6 -> i32
    %1 = bufferization.to_tensor %arg0 : memref<230x920xi8>
    %2 = routing.routingcreatescheduletensor %1 : tensor<230x920xi8> shape = [230, 920], dim = 2 -> tensor<230x920xi8>
    %3 = bufferization.to_tensor %arg1 : memref<196x64xi8>
    %4 = routing.routingcreatescheduletensor %3 : tensor<196x64xi8> shape = [196, 64], dim = 2 -> tensor<196x64xi8>
    %5 = bufferization.to_tensor %arg2 : memref<12544x64xi8>
    %6 = routing.routingcreatescheduletensor %5 : tensor<12544x64xi8> shape = [12544, 64], dim = 2 -> tensor<12544x64xi8>
    scf.execute_region {
      %7 = routing.partitionmesh mesh = %0, splitnum = 4, splitaxis = "col" : i32 -> i32
      %8 = routing.partitiontensor %4 : tensor<196x64xi8> {
  partition = #routing.partition<splitnum = 4, splitdim = 0, hwAxisOwner = "col", replicateOn = "row", singleTileOwner = "">
} -> tensor<196x64xi8>
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %arg3 = %c0 to %c4 step %c1 {
        %9 = arith.index_cast %arg3 : index to i32
        %10 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %9 : i32) -> i32{
        ^bb0(%arg4: i32):
          %11 = routing.routingextract_tiles %7, %arg4 : i32, i32 -> i32
          %12 = routing.routingextract_data %8, %arg4 : tensor<196x64xi8>, i32 -> tensor<49x64xi8>
          %13 = routing.routingcreatehwiowithtarget targettilelist = %11 : i32 {direction = "input", iotype = "mem2"} -> i32
          %14 = routing.routingmovedatabyio tensordata = %12, hwiowithtarget = %13 : tensor<49x64xi8>, i32 -> i32
          "routing.yield"() : () -> ()
        }
      }
      scf.yield
    } {routing_memo = "col"}
    scf.execute_region {
      %7 = routing.partitionmesh mesh = %0, splitnum = 4, splitaxis = "row" : i32 -> i32
      %8 = routing.partitiontensor %2 : tensor<230x920xi8> {
  partition = #routing.partition<splitnum = 4, splitdim = 0, hwAxisOwner = "row", replicateOn = "col", singleTileOwner = "">,
  tiling = #routing.tiling<d0 = #routing.dim<outer = #routing.level<base = 230, total = 244, slice = 61, step = 56, rounds = 4, slice_tiling = #routing.level<base = 61, total = 76, slice = 19, step = 14, rounds = 4>>>, d1 = #routing.dim<outer = #routing.level<base = 920, total = 976, slice = 244, step = 224, rounds = 4>>>
} -> tensor<230x920xi8>
      %9 = routing.partitiontensor %6 : tensor<12544x64xi8> {
  partition = #routing.partition<splitnum = 4, splitdim = 0, hwAxisOwner = "row", replicateOn = "col", singleTileOwner = "">
} -> tensor<12544x64xi8>
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %arg3 = %c0 to %c4 step %c1 {
        %10 = arith.index_cast %arg3 : index to i32
        %11 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %10 : i32) -> i32{
        ^bb0(%arg4: i32):
          %12 = routing.routingextract_tiles %7, %arg4 : i32, i32 -> i32
          %13 = routing.routingextract_data %8, %arg4 : tensor<230x920xi8>, i32 -> tensor<61x920xi8>
          %14 = routing.routingcreatehwiowithtarget targettilelist = %12 : i32 {direction = "input", iotype = "mem2"} -> i32
          %15 = routing.routingmovedatabyio tensordata = %13, hwiowithtarget = %14 : tensor<61x920xi8>, i32 -> i32
          %16 = routing.routingextract_data %9, %arg4 : tensor<12544x64xi8>, i32 -> tensor<3136x64xi8>
          %17 = routing.routingroutinggatherout tilelist = %12, tensordata = %16 : i32, tensor<3136x64xi8> -> tensor<3136x64xi8>
          %18 = routing.routingcreatehwiowithtarget targettilelist = %12 : i32 {direction = "output", iotype = "mem2"} -> i32
          %19 = routing.routingmovedatabyio tensordata = %17, hwiowithtarget = %18 : tensor<3136x64xi8>, i32 -> i32
          "routing.yield"() : () -> ()
        }
      }
      scf.yield
    } {routing_memo = "row"}
    return
  }
}
