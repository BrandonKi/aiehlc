module attributes {codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"], routing.pp_depth_map = {tensor_0 = 2 : i32, tensor_1 = 2 : i32, tensor_2 = 2 : i32}} {
  func.func @routing(%arg0: !emitc.ptr<!emitc.opaque<"XAie_DevInst">>, %arg1: memref<16x64xi32>, %arg2: memref<64x16xi32>, %arg3: memref<16x16xi32>) {
    %c0_i32 = arith.constant 0 : i32
    %0 = bufferization.to_tensor %arg1 : memref<16x64xi32>
    %1 = bufferization.to_tensor %arg2 : memref<64x16xi32>
    %2 = bufferization.to_tensor %arg3 : memref<16x16xi32>
    scf.execute_region {
      %3 = routing.RoutingCreate<Memo = "col"> ( scf_idx = %c0_i32 : i32) -> i32{
      ^bb0(%arg4: i32):
        %4 = routinghw.tilecreate {col = 0 : i32, comments = "core_tile", row = 3 : i32} -> i32
        %5 = routinghw.ioshimtilecreate {IOID = 4 : i32, channelused = 0 : i32, col = 0 : i32, comments = "shim_dma_4", dmadirection = 0 : i32, row = 0 : i32} -> i32
        %6 = routinghw.tilecreate {col = 0 : i32, comments = "tile in path", row = 0 : i32} -> i32
        %7 = routinghw.tilecreate {col = 0 : i32, comments = "tile in path", row = 1 : i32} -> i32
        %8 = routinghw.tilecreate {col = 0 : i32, comments = "tile in path", row = 2 : i32} -> i32
        %9 = routinghw.enableexttoaieshimport %5 : i32 {portdirection = "SOUTH", portidx = 3 : i32} -> i32
        %10 = routinghw.connectsinglestreamswitchport %5 : {slaveportdirection = "SOUTH", slaveportidx = 3, masterportdirection = "NORTH", masterportidx = 0} : i32
        %11 = routinghw.connectsinglestreamswitchport %7 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "NORTH", masterportidx = 0} : i32
        %12 = routinghw.connectsinglestreamswitchport %8 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "NORTH", masterportidx = 0} : i32
        %13 = routinghw.connectsinglestreamswitchport %4 : {slaveportdirection = "SOUTH", slaveportidx = 0, masterportdirection = "DMA", masterportidx = 1} : i32
        "routing.yield"() : () -> ()
      }
      scf.yield
    } {routing_memo = "col"}
    scf.execute_region {
      %3 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c0_i32 : i32) -> i32{
      ^bb0(%arg4: i32):
        %4 = routinghw.tilecreate {col = 0 : i32, comments = "core_tile", row = 3 : i32} -> i32
        %5 = routinghw.ioshimtilecreate {IOID = 5 : i32, channelused = 1 : i32, col = 0 : i32, comments = "shim_dma_5", dmadirection = 0 : i32, row = 0 : i32} -> i32
        %6 = routinghw.tilecreate {col = 0 : i32, comments = "tile in path", row = 0 : i32} -> i32
        %7 = routinghw.tilecreate {col = 0 : i32, comments = "tile in path", row = 1 : i32} -> i32
        %8 = routinghw.tilecreate {col = 0 : i32, comments = "tile in path", row = 2 : i32} -> i32
        %9 = routinghw.enableexttoaieshimport %5 : i32 {portdirection = "SOUTH", portidx = 7 : i32} -> i32
        %10 = routinghw.connectsinglestreamswitchport %5 : {slaveportdirection = "SOUTH", slaveportidx = 7, masterportdirection = "NORTH", masterportidx = 1} : i32
        %11 = routinghw.connectsinglestreamswitchport %7 : {slaveportdirection = "SOUTH", slaveportidx = 1, masterportdirection = "NORTH", masterportidx = 1} : i32
        %12 = routinghw.connectsinglestreamswitchport %8 : {slaveportdirection = "SOUTH", slaveportidx = 1, masterportdirection = "NORTH", masterportidx = 1} : i32
        %13 = routinghw.connectsinglestreamswitchport %4 : {slaveportdirection = "SOUTH", slaveportidx = 1, masterportdirection = "DMA", masterportidx = 0} : i32
        %14 = routinghw.tilecreate {col = 0 : i32, comments = "core_tile", row = 3 : i32} -> i32
        %15 = routinghw.ioshimtilecreate {IOID = 6 : i32, channelused = 0 : i32, col = 0 : i32, comments = "shim_dma_6", dmadirection = 1 : i32, row = 0 : i32} -> i32
        %16 = routinghw.connectpktstreamswitchport %14 : i32 {forwardmasterdirection = "NONE", forwardmasterportidx = 0 : i32, localdmadirection = "DMA", localdmapktid = 1 : i32, localdmapkttype = 0 : i32, localdmaportidx = 0 : i32, preserveheader = true, receiveslavedirection = "NONE", receiveslavepktid = 0 : i32, receiveslavepkttype = 0 : i32, receiveslaveportidx = 0 : i32} -> i32
        %17 = routinghw.tilecreate {col = 0 : i32, comments = "tile in path", row = 2 : i32} -> i32
        %18 = routinghw.tilecreate {col = 0 : i32, comments = "tile in path", row = 1 : i32} -> i32
        %19 = routinghw.tilecreate {col = 0 : i32, comments = "tile in path", row = 0 : i32} -> i32
        %20 = routinghw.enableaietoextshimport %15 : i32 {portdirection = "NORTH", portidx = 1 : i32} -> i32
        %21 = routinghw.connectpktstreamswitchport %14 : i32 {forwardmasterdirection = "SOUTH", forwardmasterportidx = 0 : i32, localdmadirection = "NONE", localdmapktid = 0 : i32, localdmapkttype = 0 : i32, localdmaportidx = 0 : i32, preserveheader = true, receiveslavedirection = "NONE", receiveslavepktid = 0 : i32, receiveslavepkttype = 0 : i32, receiveslaveportidx = 0 : i32} -> i32
        %22 = routinghw.connectsinglestreamswitchport %17 : {slaveportdirection = "NORTH", slaveportidx = 0, masterportdirection = "SOUTH", masterportidx = 0} : i32
        %23 = routinghw.connectsinglestreamswitchport %18 : {slaveportdirection = "NORTH", slaveportidx = 0, masterportdirection = "SOUTH", masterportidx = 0} : i32
        %24 = routinghw.connectsinglestreamswitchport %15 : {slaveportdirection = "NORTH", slaveportidx = 0, masterportdirection = "SOUTH", masterportidx = 1} : i32
        "routing.yield"() : () -> ()
      }
      scf.yield
    } {routing_memo = "row"}
    return
  }
}
