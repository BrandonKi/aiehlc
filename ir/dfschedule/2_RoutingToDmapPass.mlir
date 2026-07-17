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
        %9 = dmap.define_io_engine {io_id = 0 : i32, ioattr = "SHIM"} -> !dmap.dmapioenginetype
        %10 = dmap.define_core_group {core_count = 1 : i32, group_axis = "col", group_idx = 0 : i32} -> !dmap.dmacoreenginegroupType
        %11 = dmap.define_port_configure @receive_port_0 : {"RECEIVE", 16, 1, 1} -> !dmap.dmapportconfig
        %12 = dmap.create_io_engin_with_config %9 : !dmap.dmapioenginetype {accesspattern = #dmap<dataaccesspattern{"SEND", 16, 1, 1}>} -> !dmap.dmapioconfig
        %13 = dmap.create_core_group_with_config %10{[{0, @receive_port_0}], "row"} : !dmap.dmacoreenginegroupType -> !dmap.dmacoregroupconfig
        %14 = dmap.create_stream src = %12, dst = %13, !dmap.dmapioconfig !dmap.dmacoregroupconfig {streamType = #dmap.io<DMAP_SHIMIO>, stream_group_index = 0 : i32, stream_id = 1 : i32} -> !dmap.dmapportstream
        dmap.push %8 : tensor<64x16xi32> to %14 : !dmap.dmapportstream
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
        %10 = dmap.define_io_engine {io_id = 1 : i32, ioattr = "SHIM"} -> !dmap.dmapioenginetype
        %11 = dmap.define_core_group {core_count = 1 : i32, group_axis = "row", group_idx = 0 : i32} -> !dmap.dmacoreenginegroupType
        %12 = dmap.define_port_configure @receive_port_1 : {"RECEIVE", 16, 1, 1} -> !dmap.dmapportconfig
        %13 = dmap.create_io_engin_with_config %10 : !dmap.dmapioenginetype {accesspattern = #dmap<dataaccesspattern{"SEND", 16, 1, 1}>} -> !dmap.dmapioconfig
        %14 = dmap.create_core_group_with_config %11{[{0, @receive_port_1}], "row"} : !dmap.dmacoreenginegroupType -> !dmap.dmacoregroupconfig
        %15 = dmap.create_stream src = %13, dst = %14, !dmap.dmapioconfig !dmap.dmacoregroupconfig {streamType = #dmap.io<DMAP_SHIMIO>, stream_group_index = 0 : i32, stream_id = 1 : i32} -> !dmap.dmapportstream
        dmap.push %9 : tensor<16x64xi32> to %15 : !dmap.dmapportstream
        %16 = routing.routingextract_data %7, %arg3 : tensor<16x16xi32>, i32 -> tensor<16x16xi32>
        %17 = dmap.define_io_engine {io_id = 2 : i32, ioattr = "SHIM"} -> !dmap.dmapioenginetype
        %18 = dmap.define_core_group {core_count = 1 : i32, group_axis = "row", group_idx = 0 : i32} -> !dmap.dmacoreenginegroupType
        %19 = dmap.define_port_configure @send_port_2 : {"SEND", 16, 1, 1} -> !dmap.dmapportconfig
        %20 = dmap.create_io_engin_with_config %17 : !dmap.dmapioenginetype {accesspattern = #dmap<dataaccesspattern{"RECEIVE", 16, 1, 1}>} -> !dmap.dmapioconfig
        %21 = dmap.create_core_group_with_config %18{[{0, @send_port_2}], "row"} : !dmap.dmacoreenginegroupType -> !dmap.dmacoregroupconfig
        %22 = dmap.create_stream src = %21, dst = %20, !dmap.dmacoregroupconfig !dmap.dmapioconfig {streamType = #dmap.io<DMAP_SHIMIO>, stream_group_index = 0 : i32, stream_id = 1 : i32} -> !dmap.dmapportstream
        dmap.pull %16 : tensor<16x16xi32> from %22 : !dmap.dmapportstream
        "routing.yield"() : () -> ()
      }
      scf.yield
    } {routing_memo = "row"}
    return
  }
}
