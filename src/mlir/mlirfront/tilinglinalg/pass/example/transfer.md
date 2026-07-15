/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/
//Layer 1
module {
  func.func @main() {
    %0 = routing.routingcreatehwmesh {col = 4 : i64, row = 4 : i64} -> i32
    %1 = routing.routingcreatedummytensor {dim = 2 : i64, shape = [10, 20]} -> i32
    scf.execute_region {
      %2 = routing.partitionmesh mesh = %0 : i32 {splitaxis = "row", splitnum = 4 : i32} -> i32
      %3 = routing.partitiontensor tensor = %1 : i32 {hw_axis_owner = "row", replicate_on = "col", single_tile_owner = "", splitdim = 0 : i32, splitnum = 4 : i32} -> i32
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %4 = arith.index_cast %arg0 : index to i32
        %5 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %4 : i32) -> i32{
        ^bb0(%arg1: i32):
          %6 = routing.routingextract_data %3, %arg1 : i32, i32 -> i32
          %7 = routing.routingextract_tiles %2, %arg1 : i32, i32 -> i32
          %8 = routing.routingcreatehwiowithtarget targettilelist = %7 : i32 {direction = "input", iotype = "mem2"} -> i32
          %9 = routing.routingmovedatabyio gatherout_tensorin = %6, hwiowithtarget = %8 : i32, i32 -> i32
        }
      }
      scf.yield
    } {routing_memo = "row"}
    return
  }
}
//Layer 2
"builtin.module"() ({
  "dmap.func"() ({
    %0 = "dmap.create_data"() {element_type = f32, shape = [16, 16]} : () -> !dmap.dmapdata
    %1 = "dmap.create_core_engine_group"() {core_count = 4 : i32, group_axis = "row", group_idx = 0 : i32} : () -> !dmap.dmacoreenginegroupType
    %2 = "dmap.create_io_engine"() {io_id = 0 : i32, ioattr = "SHIM"} : () -> !dmap.dmapioenginetype
    %3 = "dmap.create_io_engine"() {io_id = 0 : i32, ioattr = "MEM"} : () -> !dmap.dmapioenginetype
    %4 = "dmap.port_configure_create"() {accesspattern = #dmap<dataaccesspattern{"RECEIVE", 16, 1, 1}>, sym_name = "receive1"} : () -> !dmap.dmapportconfig
    %5 = "dmap.configure_io_engine"(%2) {accesspattern = #dmap<dataaccesspattern{"SEND", 16, 1, 1}>} : (!dmap.dmapioenginetype) -> !dmap.dmapioconfig
    %6 = "dmap.configure_coregroup"(%1) {comments = "row", map = #dmap<dataconfigmap[{0, @receive1}, {1, @receive1}, {2, @receive1}, {3, @receive1}]>} : (!dmap.dmacoreenginegroupType) -> !dmap.dmacoregroupconfig
    %7 = "dmap.configure_io_engine"(%3) {accesspattern = #dmap<dataaccesspattern{"RECEIVE", 16, 1, 1}>} : (!dmap.dmapioenginetype) -> !dmap.dmapioconfig
    %8 = "dmap.configure_io_engine"(%3) {accesspattern = #dmap<dataaccesspattern{"SEND", 16, 1, 1}>} : (!dmap.dmapioenginetype) -> !dmap.dmapioconfig
    %9 = "dmap.create_stream"(%5, %7) {streamType = #dmap.io<DMAP_SHIMIO>, stream_group_index = 0 : i32, stream_id = 1 : i32} : (!dmap.dmapioconfig, !dmap.dmapioconfig) -> !dmap.dmapportstream
    %10 = "dmap.create_stream"(%8, %6) {streamType = #dmap.io<DMAP_MEMTILEIO>, stream_group_index = 0 : i32, stream_id = 1 : i32} : (!dmap.dmapioconfig, !dmap.dmacoregroupconfig) -> !dmap.dmapportstream
    %11 = "dmap.create_chain_stream"(%9, %10) : (!dmap.dmapportstream, !dmap.dmapportstream) -> !dmap.dmapportchainstream
    "dmap.push"(%0, %11) : (!dmap.dmapdata, !dmap.dmapportchainstream) -> ()
    "func.return"() : () -> ()
  }) {funcType = () -> (), sym_name = "main"} : () -> ()
}) : () -> ()main

// layer 3

"builtin.module"() ({
  "dmaphop.func"() ({
    %0 = "dmaphop.tile"() {col = 0 : i64, row = 0 : i64, tiletype = "shim"} : () -> !dmaphop.tile
    %1 = "dmaphop.tile"() {col = 1 : i64, row = 2 : i64, tiletype = "core"} : () -> !dmaphop.tile
    %2 = "dmaphop.tile"() {col = 2 : i64, row = 2 : i64, tiletype = "core"} : () -> !dmaphop.tile
    %3 = "dmaphop.port"(%0) {direction = "Out", direction_channel = 0 : i64, sym_name = "portShimOut"} : (!dmaphop.tile) -> !dmaphop.port
    %4 = "dmaphop.port"(%1) {direction = "In", direction_channel = 0 : i64, sym_name = "portAIn"} : (!dmaphop.tile) -> !dmaphop.port
    %5 = "dmaphop.port"(%1) {direction = "Out", direction_channel = 0 : i64, sym_name = "portAOut"} : (!dmaphop.tile) -> !dmaphop.port
    %6 = "dmaphop.port"(%2) {direction = "In", direction_channel = 0 : i64, sym_name = "portBIn"} : (!dmaphop.tile) -> !dmaphop.port
    %7 = "dmaphop.create_hop"(%3, %4) : (!dmaphop.port, !dmaphop.port) -> !dmaphop.hop
    %8 = "dmaphop.create_hop"(%5, %6) : (!dmaphop.port, !dmaphop.port) -> !dmaphop.hop
    %9 = "dmaphop.create_path"(%7, %8) {consumers = [@portAIn, @portBIn], tee_points = [@portAIn]} : (!dmaphop.hop, !dmaphop.hop) -> !dmaphop.path
    %10 = "memref.alloc"() <{operandSegmentSizes = array<i32: 0, 0>}> : () -> memref<1024xf32>
    %11 = "dmaphop.alloc_buffer"(%1, %10) : (!dmaphop.tile, memref<1024xf32>) -> memref<1024xf32>
    %12 = "dmaphop.alloc_buffer"(%2, %10) : (!dmaphop.tile, memref<1024xf32>) -> memref<1024xf32>
    "dmaphop.push"(%10, %9, %11, %12, %4, %6) : (memref<1024xf32>, !dmaphop.path, memref<1024xf32>, memref<1024xf32>, !dmaphop.port, !dmaphop.port) -> ()
    "dmaphop.sync"(%9) : (!dmaphop.path) -> ()
    "dmaphop.dealloc_buffer"(%11) : (memref<1024xf32>) -> ()
    "dmaphop.dealloc_buffer"(%12) : (memref<1024xf32>) -> ()
    "memref.dealloc"(%10) : (memref<1024xf32>) -> ()
    "func.return"() : () -> ()
    "func.return"() : () -> ()
  }) {funcType = () -> (), sym_name = "main"} : () -> ()

//LAYER 4.a DMA config and Runtime schedule
module {
  func.func @dskernel_coretile_compute(%arg0: i32) {
    %alloca = memref.alloca() {buffer_type = "ping"} : memref<256xf32>
    %alloca_0 = memref.alloca() {buffer_type = "pong"} : memref<256xf32>
    
    // --- Lock Definitions (from your code) ---
    // %0 = ping_aquire (DMA releases this)
    // %1 = pong_aquire (DMA releases this)
    // %2 = ping_release (DMA waits for this)
    // %3 = pong_release (DMA waits for this)
    %0 = dskernel.lock_init(0, "ping_aquire_lock") -> !dskernel.lock
    %1 = dskernel.lock_init(0, "pong_aquire_lock") -> !dskernel.lock
    %2 = dskernel.lock_init(1, "ping_release_lock") -> !dskernel.lock
    %3 = dskernel.lock_init(0, "pong_release_lock") -> !dskernel.lock
    
    // --- Thread A (DMA): Runs in Parallel ---
    // (Your code was correct, but the initial lock values must 
    //  match the parallel compute loop's needs. 
    //  DMA waits for Ping-Release(1) and releases Ping-Acquire(0))
    dskernel.launch_dma_s2m_loop %alloca, %alloca_0, %arg0, %0, %1, %2, %3 : !{
    }
    
    // --- Thread B (Compute): Runs in Parallel ---
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index // <-- **NEW: Need '2' for modulo**

    scf.for %arg1 = %c0 to %c4 step %c1 {
      // --- **START: Ping-Pong Switch Logic** ---
      
      // 1. Is the iteration %arg1 even or odd?
      %is_even = arith.cmpi "eq", arith.remui(%arg1, %c2), %c0

      // 2. Select the *correct buffer* based on iteration
      %buffer = scf.if %is_even -> (memref<256xf32>) {
        scf.yield %alloca // Even: Use Ping
      } else {
        scf.yield %alloca_0 // Odd: Use Pong
      }

      // 3. Select the *correct acquire lock* (the one DMA releases)
      %acquire_lock = scf.if %is_even -> (!dskernel.lock) {
        scf.yield %0 // Even: Wait for Ping-Acquire
      } else {
        scf.yield %1 // Odd: Wait for Pong-Acquire
      }

      // 4. Select the *correct release lock* (the one DMA waits for)
      %release_lock = scf.if %is_even -> (!dskernel.lock) {
        scf.yield %2 // Even: Release Ping-Release
      } else {
        scf.yield %3 // Odd: Release Pong-Release
      }
      
      // --- **END: Ping-Pong Switch Logic** ---

      // 5. Get the value to acquire/release (e.g., 1, 2, 3, 4)
      //    (Your logic was correct)
      %c1_i32 = arith.constant 1 : i32
      %idx_i32 = arith.index_cast %arg1 : index to i32
      %lock_val = arith.addi %idx_i32, %c1_i32 : i32

      // 6. Wait for the *selected* buffer to be full
      dskernel.acquire_lock %acquire_lock, %lock_val

      // 7. Compute using the *selected* buffer (Smem reuse)
      %c0_inner = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1_inner = arith.constant 1 : index
      scf.for %arg2 = %c0_inner to %c10 step %c1_inner {
         // --- This is where your computation goes ---
         "core.compute"(%buffer) : (memref<256xf32>) -> ()
      }

      // 8. Release the *selected* buffer (for DMA to refill)
      dskernel.release_lock %release_lock, %lock_val
    }
    return
  }
}

// layer 4.b ROUTING CONFI
// -----// IR Dump After RoutingLowerPass: //----- //
module attributes {codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"]} {
  func.func @main() {
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    scf.execute_region {
      %0 = routing.RoutingCreate<Memo = "row"> ( scf_idx = %c0_i32 : i32) -> i32{
      ^bb0(%arg0: i32):
        %4 = routinghw.tilearrayhandlecreate {name = "array handle"} : i32
        %5 = routinghw.ioshimtilecreate {IOID = 1 : i32, channelused = 0 : i32, col = 3 : i32, comments = "dio0", dmadirection = 0 : i32, row = 0 : i32} -> i32
        %6 = routinghw.tilecreate %4 : i32 {col = 0 : i32, comments = "tile reserved", row = 3 : i32} -> i32
        %7 = routinghw.tilecreate %4 : i32 {col = 1 : i32, comments = "tile reserved", row = 3 : i32} -> i32
        %8 = routinghw.tilecreate %4 : i32 {col = 2 : i32, comments = "tile reserved", row = 3 : i32} -> i32
        %9 = routinghw.tilecreate %4 : i32 {col = 3 : i32, comments = "tile reserved", row = 3 : i32} -> i32
        %10 = routinghw.connectpktstreamswitchport %6 : i32 {forwardmasterdirection = "EAST", forwardmasterportidx = 0 : i32, localdmadirection = "DMA", localdmapktid = 1 : i32, localdmapkttype = 0 : i32, localdmaportidx = 0 : i32, receiveslavedirection = "NONE", receiveslavepktid = 0 : i32, receiveslavepkttype = 0 : i32, receiveslaveportidx = 0 : i32} -> i32
        %11 = routinghw.connectpktstreamswitchport %7 : i32 {forwardmasterdirection = "EAST", forwardmasterportidx = 0 : i32, localdmadirection = "DMA", localdmapktid = 2 : i32, localdmapkttype = 0 : i32, localdmaportidx = 0 : i32, receiveslavedirection = "WEST", receiveslavepktid = 0 : i32, receiveslavepkttype = 0 : i32, receiveslaveportidx = 0 : i32} -> i32
        %12 = routinghw.connectpktstreamswitchport %8 : i32 {forwardmasterdirection = "EAST", forwardmasterportidx = 0 : i32, localdmadirection = "DMA", localdmapktid = 3 : i32, localdmapkttype = 0 : i32, localdmaportidx = 0 : i32, receiveslavedirection = "WEST", receiveslavepktid = 0 : i32, receiveslavepkttype = 0 : i32, receiveslaveportidx = 0 : i32} -> i32
        %13 = routinghw.connectpktstreamswitchport %9 : i32 {forwardmasterdirection = "NONE", forwardmasterportidx = 0 : i32, localdmadirection = "DMA", localdmapktid = 4 : i32, localdmapkttype = 0 : i32, localdmaportidx = 0 : i32, receiveslavedirection = "WEST", receiveslavepktid = 0 : i32, receiveslavepkttype = 0 : i32, receiveslaveportidx = 0 : i32} -> i32
        %14 = routinghw.tilecreate %4 : i32 {col = 3 : i32, comments = "tile in path", row = 2 : i32} -> i32
        %15 = routinghw.tilecreate %4 : i32 {col = 3 : i32, comments = "tile in path", row = 1 : i32} -> i32
        %16 = routinghw.tilecreate %4 : i32 {col = 3 : i32, comments = "tile in path", row = 0 : i32} -> i32
        %17 = routinghw.connectpktstreamswitchport %9 : i32 {forwardmasterdirection = "SOUTH", forwardmasterportidx = 0 : i32, localdmadirection = "NONE", localdmapktid = 0 : i32, localdmapkttype = 0 : i32, localdmaportidx = 0 : i32, receiveslavedirection = "NONE", receiveslavepktid = 0 : i32, receiveslavepkttype = 0 : i32, receiveslaveportidx = 0 : i32} -> i32
        %18 = routinghw.connectsinglestreamswitchport %14 : i32 {masterportdirection = "NORTH", masterportidx = 0 : i32, slaveportdirection = "SOUTH", slaveportidx = 0 : i32} -> i32
        %19 = routinghw.connectsinglestreamswitchport %15 : i32 {masterportdirection = "NORTH", masterportidx = 0 : i32, slaveportdirection = "SOUTH", slaveportidx = 0 : i32} -> i32
        %20 = routinghw.enableaietoextshimport %16 : i32 {portdirection = "SOUTH", portidx = 2 : i32} -> i32
        %21 = routinghw.connectsinglestreamswitchport %16 : i32 {masterportdirection = "SOUTH", masterportidx = 2 : i32, slaveportdirection = "SOUTH", slaveportidx = 2 : i32} -> i32
      }
  }
}
//layer 5
// -----// IR Dump After RoutingHWLowerPass: //----- //
module attributes {codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"]} {
  emitc.include <"xaiengine.h">
  emitc.func private @XAie_TileLoc(i32, i32) -> !emitc.opaque<"XAie_LocType">
  emitc.func private @XAie_Packet(i32, i32) -> !emitc.opaque<"XAie_Packet">
  emitc.func private @getOrCreateDeviceInstance() -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
  emitc.func private @XAie_EnableShimDmaToAieStrmPort(!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
  emitc.func private @XAie_EnableAieToShimDmaStrmPort(!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
  emitc.func private @XAie_StrmConnCctEnable(!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
  emitc.func private @XAie_StrmPktSwSlaveSlotEnable(!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
  emitc.func private @XAie_StrmPktSwMstrPortEnable(!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
  func.func @main() {
    %0 = "emitc.constant"() <{value = 15 : i32}> : () -> i32
    %1 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %2 = "emitc.constant"() <{value = 9 : i32}> : () -> i32
    %3 = "emitc.constant"() <{value = 10 : i32}> : () -> i32
    %4 = "emitc.constant"() <{value = 11 : i32}> : () -> i32
    %5 = "emitc.constant"() <{value = 12 : i32}> : () -> i32
    %6 = "emitc.constant"() <{value = 13 : i32}> : () -> i32
    %7 = "emitc.constant"() <{value = 14 : i32}> : () -> i32
    %8 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
    %9 = "emitc.constant"() <{value = 6 : i32}> : () -> i32
    %10 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %11 = "emitc.constant"() <{value = #emitc.opaque<"NORTH">}> : () -> !emitc.ptr<i8>
    %12 = "emitc.constant"() <{value = #emitc.opaque<"SOUTH">}> : () -> !emitc.ptr<i8>
    %13 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %14 = "emitc.constant"() <{value = #emitc.opaque<"WEST">}> : () -> !emitc.ptr<i8>
    %15 = "emitc.constant"() <{value = #emitc.opaque<"XAIE_SS_PKT_DROP_HEADER">}> : () -> !emitc.ptr<i8>
    %16 = "emitc.constant"() <{value = #emitc.opaque<"EAST">}> : () -> !emitc.ptr<i8>
    %17 = "emitc.constant"() <{value = 31 : i32}> : () -> i32
    %18 = "emitc.constant"() <{value = #emitc.opaque<"DMA">}> : () -> !emitc.ptr<i8>
    %19 = "emitc.constant"() <{value = #emitc.opaque<"NONE">}> : () -> !emitc.ptr<i8>
    %20 = "emitc.constant"() <{value = true}> : () -> i1
    %21 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %22 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %23 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %24 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    emitc.verbatim "\0A{ //----routing creation in row ----start-------"
    emitc.verbatim "\0A//round is 0 hw split in : row -----------"
    emitc.if %20 {
      %25 = emitc.call @XAie_TileLoc(%21, %24) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %26 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %27 = emitc.call @XAie_Packet(%24, %24) : (i32, i32) -> !emitc.opaque<"XAie_Packet">
      %28 = emitc.call @XAie_Packet(%23, %24) : (i32, i32) -> !emitc.opaque<"XAie_Packet">
      %29 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%26, %25, %18, %24, %24, %28, %17, %24, %24) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %30 = emitc.call @XAie_StrmPktSwMstrPortEnable(%26, %25, %16, %24, %15, %24, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %31 = emitc.call @XAie_TileLoc(%21, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %32 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %33 = emitc.call @XAie_Packet(%24, %24) : (i32, i32) -> !emitc.opaque<"XAie_Packet">
      %34 = emitc.call @XAie_Packet(%22, %24) : (i32, i32) -> !emitc.opaque<"XAie_Packet">
      %35 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%32, %31, %14, %24, %24, %33, %24, %24, %24) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %36 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%32, %31, %18, %24, %24, %34, %17, %24, %24) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %37 = emitc.call @XAie_StrmPktSwMstrPortEnable(%32, %31, %16, %24, %15, %24, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %38 = emitc.call @XAie_TileLoc(%21, %22) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %39 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %40 = emitc.call @XAie_Packet(%24, %24) : (i32, i32) -> !emitc.opaque<"XAie_Packet">
      %41 = emitc.call @XAie_Packet(%21, %24) : (i32, i32) -> !emitc.opaque<"XAie_Packet">
      %42 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%39, %38, %14, %24, %24, %40, %24, %24, %24) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %43 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%39, %38, %18, %24, %24, %41, %17, %24, %24) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %44 = emitc.call @XAie_StrmPktSwMstrPortEnable(%39, %38, %16, %24, %15, %24, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %45 = emitc.call @XAie_TileLoc(%21, %21) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %46 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %47 = emitc.call @XAie_Packet(%24, %24) : (i32, i32) -> !emitc.opaque<"XAie_Packet">
      %48 = emitc.call @XAie_Packet(%13, %24) : (i32, i32) -> !emitc.opaque<"XAie_Packet">
      %49 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%46, %45, %14, %24, %24, %47, %24, %24, %24) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %50 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%46, %45, %18, %24, %24, %48, %17, %24, %24) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %51 = emitc.call @XAie_TileLoc(%21, %21) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %52 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %53 = emitc.call @XAie_Packet(%24, %24) : (i32, i32) -> !emitc.opaque<"XAie_Packet">
      %54 = emitc.call @XAie_Packet(%24, %24) : (i32, i32) -> !emitc.opaque<"XAie_Packet">
      %55 = emitc.call @XAie_StrmPktSwMstrPortEnable(%52, %51, %12, %24, %15, %24, %23) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %56 = emitc.call @XAie_TileLoc(%22, %21) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %57 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %58 = emitc.call @XAie_StrmConnCctEnable(%57, %56, %11, %24, %12, %24) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %59 = emitc.call @XAie_TileLoc(%23, %21) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %60 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %61 = emitc.call @XAie_StrmConnCctEnable(%60, %59, %11, %24, %12, %24) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %62 = emitc.call @XAie_TileLoc(%24, %21) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %63 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %64 = emitc.call @XAie_EnableAieToShimDmaStrmPort(%63, %62, %22) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
      %65 = emitc.call @XAie_TileLoc(%24, %21) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %66 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %67 = emitc.call @XAie_StrmConnCctEnable(%66, %65, %12, %22, %12, %22) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
    }
    emitc.verbatim "\0A//round is 1 hw split in : row -----------"
    emitc.verbatim "\0A} //----routing creation in col ----end-------\0A"
    return
  }
}
module attributes {codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"]} {
  emitc.include <"xaiengine.h">
  emitc.func private @getOrCreateDeviceInstance() -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
  func.func @main() {
    %0 = "emitc.constant"() <{value = true}> : () -> i1
    emitc.verbatim "\0A{ //----routing creation in row ----start-------"
    emitc.verbatim "\0A//round is 0 hw split in : row -----------"
    emitc.if %0 {
      %1 = emitc.call_opaque "XAie_StrmPktSwSlaveSlotEnable"() {args = [#emitc.opaque<"getOrCreateDeviceInstance()">, #emitc.opaque<"XAie_TileLoc(3,0)">, #emitc.opaque<"DMA">, 0 : i32, 0 : i32, #emitc.opaque<"XAie_Packet(1,0)">, 31 : i32, 0 : i32, 0 : i32]} : () -> i32
      %2 = emitc.call_opaque "XAie_StrmPktSwMstrPortEnable"() {args = [#emitc.opaque<"getOrCreateDeviceInstance()">, #emitc.opaque<"XAie_TileLoc(3,0)">, #emitc.opaque<"EAST">, 0 : i32, #emitc.opaque<"XAIE_SS_PKT_DROP_HEADER">, 0 : i32, 1 : i32]} : () -> i32
      %3 = emitc.call_opaque "XAie_StrmPktSwSlaveSlotEnable"() {args = [#emitc.opaque<"getOrCreateDeviceInstance()">, #emitc.opaque<"XAie_TileLoc(3,1)">, #emitc.opaque<"WEST">, 0 : i32, 0 : i32, #emitc.opaque<"XAie_Packet(0,0)">, 0 : i32, 0 : i32, 0 : i32]} : () -> i32
      %4 = emitc.call_opaque "XAie_StrmPktSwSlaveSlotEnable"() {args = [#emitc.opaque<"getOrCreateDeviceInstance()">, #emitc.opaque<"XAie_TileLoc(3,1)">, #emitc.opaque<"DMA">, 0 : i32, 0 : i32, #emitc.opaque<"XAie_Packet(2,0)">, 31 : i32, 0 : i32, 0 : i32]} : () -> i32
      %5 = emitc.call_opaque "XAie_StrmPktSwMstrPortEnable"() {args = [#emitc.opaque<"getOrCreateDeviceInstance()">, #emitc.opaque<"XAie_TileLoc(3,1)">, #emitc.opaque<"EAST">, 0 : i32, #emitc.opaque<"XAIE_SS_PKT_DROP_HEADER">, 0 : i32, 1 : i32]} : () -> i32
      %6 = emitc.call_opaque "XAie_StrmPktSwSlaveSlotEnable"() {args = [#emitc.opaque<"getOrCreateDeviceInstance()">, #emitc.opaque<"XAie_TileLoc(3,2)">, #emitc.opaque<"WEST">, 0 : i32, 0 : i32, #emitc.opaque<"XAie_Packet(0,0)">, 0 : i32, 0 : i32, 0 : i32]} : () -> i32
      %7 = emitc.call_opaque "XAie_StrmPktSwSlaveSlotEnable"() {args = [#emitc.opaque<"getOrCreateDeviceInstance()">, #emitc.opaque<"XAie_TileLoc(3,2)">, #emitc.opaque<"DMA">, 0 : i32, 0 : i32, #emitc.opaque<"XAie_Packet(3,0)">, 31 : i32, 0 : i32, 0 : i32]} : () -> i32
      %8 = emitc.call_opaque "XAie_StrmPktSwMstrPortEnable"() {args = [#emitc.opaque<"getOrCreateDeviceInstance()">, #emitc.opaque<"XAie_TileLoc(3,2)">, #emitc.opaque<"EAST">, 0 : i32, #emitc.opaque<"XAIE_SS_PKT_DROP_HEADER">, 0 : i32, 1 : i32]} : () -> i32
      %9 = emitc.call_opaque "XAie_StrmPktSwSlaveSlotEnable"() {args = [#emitc.opaque<"getOrCreateDeviceInstance()">, #emitc.opaque<"XAie_TileLoc(3,3)">, #emitc.opaque<"WEST">, 0 : i32, 0 : i32, #emitc.opaque<"XAie_Packet(0,0)">, 0 : i32, 0 : i32, 0 : i32]} : () -> i32
      %10 = emitc.call_opaque "XAie_StrmPktSwSlaveSlotEnable"() {args = [#emitc.opaque<"getOrCreateDeviceInstance()">, #emitc.opaque<"XAie_TileLoc(3,3)">, #emitc.opaque<"DMA">, 0 : i32, 0 : i32, #emitc.opaque<"XAie_Packet(4,0)">, 31 : i32, 0 : i32, 0 : i32]} : () -> i32
      %11 = emitc.call_opaque "XAie_StrmPktSwMstrPortEnable"() {args = [#emitc.opaque<"getOrCreateDeviceInstance()">, #emitc.opaque<"XAie_TileLoc(3,3)">, #emitc.opaque<"SOUTH">, 0 : i32, #emitc.opaque<"XAIE_SS_PKT_DROP_HEADER">, 0 : i32, 1 : i32]} : () -> i32
      %12 = emitc.call_opaque "XAie_StrmConnCctEnable"() {args = [#emitc.opaque<"getOrCreateDeviceInstance()">, #emitc.opaque<"XAie_TileLoc(2,3)">, #emitc.opaque<"NORTH">, 0 : i32, #emitc.opaque<"SOUTH">, 0 : i32]} : () -> i32
      %13 = emitc.call_opaque "XAie_StrmConnCctEnable"() {args = [#emitc.opaque<"getOrCreateDeviceInstance()">, #emitc.opaque<"XAie_TileLoc(1,3)">, #emitc.opaque<"NORTH">, 0 : i32, #emitc.opaque<"SOUTH">, 0 : i32]} : () -> i32
      %14 = emitc.call_opaque "XAie_EnableAieToShimDmaStrmPort"() {args = [#emitc.opaque<"getOrCreateDeviceInstance()">, #emitc.opaque<"XAie_TileLoc(0,3)">, 2 : i32]} : () -> i32
      %15 = emitc.call_opaque "XAie_StrmConnCctEnable"() {args = [#emitc.opaque<"getOrCreateDeviceInstance()">, #emitc.opaque<"XAie_TileLoc(0,3)">, #emitc.opaque<"SOUTH">, 2 : i32, #emitc.opaque<"SOUTH">, 2 : i32]} : () -> i32
    }
//layer 6 
void main() {
  bool v1 = true;

  { //----routing creation in row ----start-------

  //round is 0 hw split in : row -----------
  if (v1) {
    int32_t v2 = XAie_StrmPktSwSlaveSlotEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,0), DMA, 0, 0, XAie_Packet(1,0), 31, 0, 0);
    int32_t v3 = XAie_StrmPktSwMstrPortEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,0), EAST, 0, XAIE_SS_PKT_DROP_HEADER, 0, 1);
    int32_t v4 = XAie_StrmPktSwSlaveSlotEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,1), WEST, 0, 0, XAie_Packet(0,0), 0, 0, 0);
    int32_t v5 = XAie_StrmPktSwSlaveSlotEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,1), DMA, 0, 0, XAie_Packet(2,0), 31, 0, 0);
    int32_t v6 = XAie_StrmPktSwMstrPortEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,1), EAST, 0, XAIE_SS_PKT_DROP_HEADER, 0, 1);
    int32_t v7 = XAie_StrmPktSwSlaveSlotEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,2), WEST, 0, 0, XAie_Packet(0,0), 0, 0, 0);
    int32_t v8 = XAie_StrmPktSwSlaveSlotEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,2), DMA, 0, 0, XAie_Packet(3,0), 31, 0, 0);
    int32_t v9 = XAie_StrmPktSwMstrPortEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,2), EAST, 0, XAIE_SS_PKT_DROP_HEADER, 0, 1);
    int32_t v10 = XAie_StrmPktSwSlaveSlotEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,3), WEST, 0, 0, XAie_Packet(0,0), 0, 0, 0);
    int32_t v11 = XAie_StrmPktSwSlaveSlotEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,3), DMA, 0, 0, XAie_Packet(4,0), 31, 0, 0);
    int32_t v12 = XAie_StrmPktSwMstrPortEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,3), SOUTH, 0, XAIE_SS_PKT_DROP_HEADER, 0, 1);
    int32_t v13 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2,3), NORTH, 0, SOUTH, 0);
    int32_t v14 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(1,3), NORTH, 0, SOUTH, 0);
    int32_t v15 = XAie_EnableAieToShimDmaStrmPort(getOrCreateDeviceInstance(), XAie_TileLoc(0,3), 2);
    int32_t v16 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(0,3), SOUTH, 2, SOUTH, 2);
  }

  //round is 1 hw split in : row -----------
  if (v1) {
    int32_t v17 = XAie_StrmPktSwSlaveSlotEnable(getOrCreateDeviceInstance(), XAie_TileLoc(4,0), DMA, 0, 0, XAie_Packet(1,0), 31, 0, 0);
    int32_t v18 = XAie_StrmPktSwMstrPortEnable(getOrCreateDeviceInstance(), XAie_TileLoc(4,0), EAST, 0, XAIE_SS_PKT_DROP_HEADER, 0, 1);
    int32_t v19 = XAie_StrmPktSwSlaveSlotEnable(getOrCreateDeviceInstance(), XAie_TileLoc(4,1), WEST, 0, 0, XAie_Packet(0,0), 0, 0, 0);
    int32_t v20 = XAie_StrmPktSwSlaveSlotEnable(getOrCreateDeviceInstance(), XAie_TileLoc(4,1), DMA, 0, 0, XAie_Packet(2,0), 31, 0, 0);
    int32_t v21 = XAie_StrmPktSwMstrPortEnable(getOrCreateDeviceInstance(), XAie_TileLoc(4,1), EAST, 0, XAIE_SS_PKT_DROP_HEADER, 0, 1);
    int32_t v22 = XAie_StrmPktSwSlaveSlotEnable(getOrCreateDeviceInstance(), XAie_TileLoc(4,2), WEST, 0, 0, XAie_Packet(0,0), 0, 0, 0);
    int32_t v23 = XAie_StrmPktSwSlaveSlotEnable(getOrCreateDeviceInstance(), XAie_TileLoc(4,2), DMA, 0, 0, XAie_Packet(3,0), 31, 0, 0);
    int32_t v24 = XAie_StrmPktSwMstrPortEnable(getOrCreateDeviceInstance(), XAie_TileLoc(4,2), EAST, 0, XAIE_SS_PKT_DROP_HEADER, 0, 1);
    int32_t v25 = XAie_StrmPktSwSlaveSlotEnable(getOrCreateDeviceInstance(), XAie_TileLoc(4,3), WEST, 0, 0, XAie_Packet(0,0), 0, 0, 0);
    int32_t v26 = XAie_StrmPktSwSlaveSlotEnable(getOrCreateDeviceInstance(), XAie_TileLoc(4,3), DMA, 0, 0, XAie_Packet(4,0), 31, 0, 0);
    int32_t v27 = XAie_StrmPktSwMstrPortEnable(getOrCreateDeviceInstance(), XAie_TileLoc(4,3), SOUTH, 0, XAIE_SS_PKT_DROP_HEADER, 0, 1);
    int32_t v28 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,3), NORTH, 0, SOUTH, 1);
    int32_t v29 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2,3), NORTH, 1, SOUTH, 1);
    int32_t v30 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(1,3), NORTH, 1, SOUTH, 1);
    int32_t v31 = XAie_EnableAieToShimDmaStrmPort(getOrCreateDeviceInstance(), XAie_TileLoc(0,3), 3);
    int32_t v32 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(0,3), SOUTH, 3, SOUTH, 3);
  }

  //round is 2 hw split in : row -----------
  if (v1) {
    int32_t v33 = XAie_StrmPktSwSlaveSlotEnable(getOrCreateDeviceInstance(), XAie_TileLoc(5,0), DMA, 0, 0, XAie_Packet(1,0), 31, 0, 0);
    int32_t v34 = XAie_StrmPktSwMstrPortEnable(getOrCreateDeviceInstance(), XAie_TileLoc(5,0), EAST, 0, XAIE_SS_PKT_DROP_HEADER, 0, 1);
    int32_t v35 = XAie_StrmPktSwSlaveSlotEnable(getOrCreateDeviceInstance(), XAie_TileLoc(5,1), WEST, 0, 0, XAie_Packet(0,0), 0, 0, 0);
    int32_t v36 = XAie_StrmPktSwSlaveSlotEnable(getOrCreateDeviceInstance(), XAie_TileLoc(5,1), DMA, 0, 0, XAie_Packet(2,0), 31, 0, 0);
    int32_t v37 = XAie_StrmPktSwMstrPortEnable(getOrCreateDeviceInstance(), XAie_TileLoc(5,1), EAST, 0, XAIE_SS_PKT_DROP_HEADER, 0, 1);
    int32_t v38 = XAie_StrmPktSwSlaveSlotEnable(getOrCreateDeviceInstance(), XAie_TileLoc(5,2), WEST, 0, 0, XAie_Packet(0,0), 0, 0, 0);
    int32_t v39 = XAie_StrmPktSwSlaveSlotEnable(getOrCreateDeviceInstance(), XAie_TileLoc(5,2), DMA, 0, 0, XAie_Packet(3,0), 31, 0, 0);
    int32_t v40 = XAie_StrmPktSwMstrPortEnable(getOrCreateDeviceInstance(), XAie_TileLoc(5,2), EAST, 0, XAIE_SS_PKT_DROP_HEADER, 0, 1);
    int32_t v41 = XAie_StrmPktSwSlaveSlotEnable(getOrCreateDeviceInstance(), XAie_TileLoc(5,3), WEST, 0, 0, XAie_Packet(0,0), 0, 0, 0);
    int32_t v42 = XAie_StrmPktSwSlaveSlotEnable(getOrCreateDeviceInstance(), XAie_TileLoc(5,3), DMA, 0, 0, XAie_Packet(4,0), 31, 0, 0);
    int32_t v43 = XAie_StrmPktSwMstrPortEnable(getOrCreateDeviceInstance(), XAie_TileLoc(5,3), SOUTH, 0, XAIE_SS_PKT_DROP_HEADER, 0, 1);
    int32_t v44 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(4,3), NORTH, 0, SOUTH, 1);
    int32_t v45 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,3), NORTH, 1, SOUTH, 2);
    int32_t v46 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2,3), NORTH, 2, SOUTH, 2);
    int32_t v47 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(1,3), NORTH, 2, WEST, 0);
    int32_t v48 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(1,2), EAST, 0, SOUTH, 0);
    int32_t v49 = XAie_EnableAieToShimDmaStrmPort(getOrCreateDeviceInstance(), XAie_TileLoc(0,2), 2);
    int32_t v50 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(0,2), SOUTH, 2, SOUTH, 2);
  }

  //round is 3 hw split in : row -----------
  if (v1) {
    int32_t v51 = XAie_StrmPktSwSlaveSlotEnable(getOrCreateDeviceInstance(), XAie_TileLoc(6,0), DMA, 0, 0, XAie_Packet(1,0), 31, 0, 0);
    int32_t v52 = XAie_StrmPktSwMstrPortEnable(getOrCreateDeviceInstance(), XAie_TileLoc(6,0), EAST, 0, XAIE_SS_PKT_DROP_HEADER, 0, 1);
    int32_t v53 = XAie_StrmPktSwSlaveSlotEnable(getOrCreateDeviceInstance(), XAie_TileLoc(6,1), WEST, 0, 0, XAie_Packet(0,0), 0, 0, 0);
    int32_t v54 = XAie_StrmPktSwSlaveSlotEnable(getOrCreateDeviceInstance(), XAie_TileLoc(6,1), DMA, 0, 0, XAie_Packet(2,0), 31, 0, 0);
    int32_t v55 = XAie_StrmPktSwMstrPortEnable(getOrCreateDeviceInstance(), XAie_TileLoc(6,1), EAST, 0, XAIE_SS_PKT_DROP_HEADER, 0, 1);
    int32_t v56 = XAie_StrmPktSwSlaveSlotEnable(getOrCreateDeviceInstance(), XAie_TileLoc(6,2), WEST, 0, 0, XAie_Packet(0,0), 0, 0, 0);
    int32_t v57 = XAie_StrmPktSwSlaveSlotEnable(getOrCreateDeviceInstance(), XAie_TileLoc(6,2), DMA, 0, 0, XAie_Packet(3,0), 31, 0, 0);
    int32_t v58 = XAie_StrmPktSwMstrPortEnable(getOrCreateDeviceInstance(), XAie_TileLoc(6,2), EAST, 0, XAIE_SS_PKT_DROP_HEADER, 0, 1);
    int32_t v59 = XAie_StrmPktSwSlaveSlotEnable(getOrCreateDeviceInstance(), XAie_TileLoc(6,3), WEST, 0, 0, XAie_Packet(0,0), 0, 0, 0);
    int32_t v60 = XAie_StrmPktSwSlaveSlotEnable(getOrCreateDeviceInstance(), XAie_TileLoc(6,3), DMA, 0, 0, XAie_Packet(4,0), 31, 0, 0);
    int32_t v61 = XAie_StrmPktSwMstrPortEnable(getOrCreateDeviceInstance(), XAie_TileLoc(6,3), SOUTH, 0, XAIE_SS_PKT_DROP_HEADER, 0, 1);
    int32_t v62 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(5,3), NORTH, 0, SOUTH, 1);
    int32_t v63 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(4,3), NORTH, 1, SOUTH, 2);
    int32_t v64 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,3), NORTH, 2, SOUTH, 3);
    int32_t v65 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2,3), NORTH, 3, SOUTH, 3);
    int32_t v66 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(1,3), NORTH, 3, WEST, 1);
    int32_t v67 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(1,2), EAST, 1, SOUTH, 1);
    int32_t v68 = XAie_EnableAieToShimDmaStrmPort(getOrCreateDeviceInstance(), XAie_TileLoc(0,2), 3);
    int32_t v69 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(0,2), SOUTH, 3, SOUTH, 3);
  }

  } //----routing creation in row ----end-------


  { //----routing creation in row ----start-------

  //round is 0 hw split in : row -----------
  if (v1) {
    int32_t v70 = XAie_EnableShimDmaToAieStrmPort(getOrCreateDeviceInstance(), XAie_TileLoc(0,6), 3);
    int32_t v71 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(0,6), SOUTH, 3, NORTH, 0);
    int32_t v72 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(1,6), SOUTH, 0, NORTH, 0);
    int32_t v73 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2,6), SOUTH, 0, NORTH, 0);
    int32_t v74 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,6), SOUTH, 0, WEST, 0);
    int32_t v75 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,5), EAST, 0, WEST, 0);
    int32_t v76 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,4), EAST, 0, WEST, 0);
    int32_t v77 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,3), EAST, 0, WEST, 0);
    int32_t v78 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,3), EAST, 0, DMA, 1);
    int32_t v79 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,2), EAST, 0, WEST, 0);
    int32_t v80 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,2), EAST, 0, DMA, 1);
    int32_t v81 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,1), EAST, 0, WEST, 0);
    int32_t v82 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,1), EAST, 0, DMA, 1);
    int32_t v83 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,0), EAST, 0, DMA, 1);
  }

  //round is 1 hw split in : row -----------
  if (v1) {
    int32_t v84 = XAie_EnableShimDmaToAieStrmPort(getOrCreateDeviceInstance(), XAie_TileLoc(0,6), 7);
    int32_t v85 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(0,6), SOUTH, 7, NORTH, 1);
    int32_t v86 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(1,6), SOUTH, 1, NORTH, 1);
    int32_t v87 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2,6), SOUTH, 1, NORTH, 1);
    int32_t v88 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,6), SOUTH, 1, NORTH, 0);
    int32_t v89 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(4,6), SOUTH, 0, WEST, 0);
    int32_t v90 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(4,5), EAST, 0, WEST, 0);
    int32_t v91 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(4,4), EAST, 0, WEST, 0);
    int32_t v92 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(4,3), EAST, 0, WEST, 0);
    int32_t v93 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(4,3), EAST, 0, DMA, 1);
    int32_t v94 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(4,2), EAST, 0, WEST, 0);
    int32_t v95 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(4,2), EAST, 0, DMA, 1);
    int32_t v96 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(4,1), EAST, 0, WEST, 0);
    int32_t v97 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(4,1), EAST, 0, DMA, 1);
    int32_t v98 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(4,0), EAST, 0, DMA, 1);
  }

  //round is 2 hw split in : row -----------
  if (v1) {
    int32_t v99 = XAie_EnableShimDmaToAieStrmPort(getOrCreateDeviceInstance(), XAie_TileLoc(0,7), 3);
    int32_t v100 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(0,7), SOUTH, 3, NORTH, 0);
    int32_t v101 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(1,7), SOUTH, 0, NORTH, 0);
    int32_t v102 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2,7), SOUTH, 0, NORTH, 0);
    int32_t v103 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,7), SOUTH, 0, NORTH, 0);
    int32_t v104 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(4,7), SOUTH, 0, NORTH, 0);
    int32_t v105 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(5,7), SOUTH, 0, WEST, 0);
    int32_t v106 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(5,6), EAST, 0, WEST, 0);
    int32_t v107 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(5,5), EAST, 0, WEST, 0);
    int32_t v108 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(5,4), EAST, 0, WEST, 0);
    int32_t v109 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(5,3), EAST, 0, WEST, 0);
    int32_t v110 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(5,3), EAST, 0, DMA, 1);
    int32_t v111 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(5,2), EAST, 0, WEST, 0);
    int32_t v112 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(5,2), EAST, 0, DMA, 1);
    int32_t v113 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(5,1), EAST, 0, WEST, 0);
    int32_t v114 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(5,1), EAST, 0, DMA, 1);
    int32_t v115 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(5,0), EAST, 0, DMA, 1);
  }

  //round is 3 hw split in : row -----------
  if (v1) {
    int32_t v116 = XAie_EnableShimDmaToAieStrmPort(getOrCreateDeviceInstance(), XAie_TileLoc(0,7), 7);
    int32_t v117 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(0,7), SOUTH, 7, NORTH, 1);
    int32_t v118 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(1,7), SOUTH, 1, NORTH, 1);
    int32_t v119 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2,7), SOUTH, 1, NORTH, 1);
    int32_t v120 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,7), SOUTH, 1, NORTH, 1);
    int32_t v121 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(4,7), SOUTH, 1, NORTH, 1);
    int32_t v122 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(5,7), SOUTH, 1, NORTH, 0);
    int32_t v123 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(6,7), SOUTH, 0, WEST, 0);
    int32_t v124 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(6,6), EAST, 0, WEST, 0);
    int32_t v125 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(6,5), EAST, 0, WEST, 0);
    int32_t v126 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(6,4), EAST, 0, WEST, 0);
    int32_t v127 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(6,3), EAST, 0, WEST, 0);
    int32_t v128 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(6,3), EAST, 0, DMA, 1);
    int32_t v129 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(6,2), EAST, 0, WEST, 0);
    int32_t v130 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(6,2), EAST, 0, DMA, 1);
    int32_t v131 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(6,1), EAST, 0, WEST, 0);
    int32_t v132 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(6,1), EAST, 0, DMA, 1);
    int32_t v133 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(6,0), EAST, 0, DMA, 1);
  }

  } //----routing creation in row ----end-------


  { //----routing creation in col ----start-------

  //round is 0 hw split in : col -----------
  if (v1) {
    int32_t v134 = XAie_EnableShimDmaToAieStrmPort(getOrCreateDeviceInstance(), XAie_TileLoc(0,14), 3);
    int32_t v135 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(0,14), SOUTH, 3, NORTH, 0);
    int32_t v136 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(1,14), SOUTH, 0, NORTH, 0);
    int32_t v137 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2,14), SOUTH, 0, NORTH, 0);
    int32_t v138 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,14), SOUTH, 0, WEST, 0);
    int32_t v139 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,13), EAST, 0, WEST, 0);
    int32_t v140 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,12), EAST, 0, WEST, 0);
    int32_t v141 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,11), EAST, 0, WEST, 0);
    int32_t v142 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,10), EAST, 0, WEST, 0);
    int32_t v143 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,9), EAST, 0, WEST, 0);
    int32_t v144 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,8), EAST, 0, WEST, 0);
    int32_t v145 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,7), EAST, 0, WEST, 0);
    int32_t v146 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,6), EAST, 0, WEST, 1);
    int32_t v147 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,5), EAST, 1, WEST, 1);
    int32_t v148 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,4), EAST, 1, WEST, 1);
    int32_t v149 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,3), EAST, 1, WEST, 1);
    int32_t v150 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,2), EAST, 1, WEST, 1);
    int32_t v151 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,1), EAST, 1, WEST, 1);
    int32_t v152 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,0), EAST, 1, NORTH, 0);
    int32_t v153 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,0), EAST, 1, DMA, 2);
    int32_t v154 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(4,0), SOUTH, 0, NORTH, 0);
    int32_t v155 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(4,0), SOUTH, 0, DMA, 2);
    int32_t v156 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(5,0), SOUTH, 0, NORTH, 0);
    int32_t v157 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(5,0), SOUTH, 0, DMA, 2);
    int32_t v158 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(6,0), SOUTH, 0, DMA, 2);
  }

  //round is 1 hw split in : col -----------
  if (v1) {
    int32_t v159 = XAie_EnableShimDmaToAieStrmPort(getOrCreateDeviceInstance(), XAie_TileLoc(0,14), 7);
    int32_t v160 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(0,14), SOUTH, 7, NORTH, 1);
    int32_t v161 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(1,14), SOUTH, 1, NORTH, 1);
    int32_t v162 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2,14), SOUTH, 1, NORTH, 1);
    int32_t v163 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,14), SOUTH, 1, WEST, 1);
    int32_t v164 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,13), EAST, 1, WEST, 1);
    int32_t v165 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,12), EAST, 1, WEST, 1);
    int32_t v166 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,11), EAST, 1, WEST, 1);
    int32_t v167 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,10), EAST, 1, WEST, 1);
    int32_t v168 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,9), EAST, 1, WEST, 1);
    int32_t v169 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,8), EAST, 1, WEST, 1);
    int32_t v170 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,7), EAST, 1, WEST, 1);
    int32_t v171 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,6), EAST, 1, WEST, 2);
    int32_t v172 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,5), EAST, 2, WEST, 2);
    int32_t v173 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,4), EAST, 2, WEST, 2);
    int32_t v174 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,3), EAST, 2, WEST, 2);
    int32_t v175 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,2), EAST, 2, WEST, 2);
    int32_t v176 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,1), EAST, 2, NORTH, 0);
    int32_t v177 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,1), EAST, 2, DMA, 2);
    int32_t v178 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(4,1), SOUTH, 0, NORTH, 0);
    int32_t v179 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(4,1), SOUTH, 0, DMA, 2);
    int32_t v180 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(5,1), SOUTH, 0, NORTH, 0);
    int32_t v181 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(5,1), SOUTH, 0, DMA, 2);
    int32_t v182 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(6,1), SOUTH, 0, DMA, 2);
  }

  //round is 2 hw split in : col -----------
  if (v1) {
    int32_t v183 = XAie_EnableShimDmaToAieStrmPort(getOrCreateDeviceInstance(), XAie_TileLoc(0,15), 3);
    int32_t v184 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(0,15), SOUTH, 3, NORTH, 0);
    int32_t v185 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(1,15), SOUTH, 0, NORTH, 0);
    int32_t v186 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2,15), SOUTH, 0, NORTH, 0);
    int32_t v187 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,15), SOUTH, 0, WEST, 0);
    int32_t v188 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,14), EAST, 0, WEST, 2);
    int32_t v189 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,13), EAST, 2, WEST, 2);
    int32_t v190 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,12), EAST, 2, WEST, 2);
    int32_t v191 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,11), EAST, 2, WEST, 2);
    int32_t v192 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,10), EAST, 2, WEST, 2);
    int32_t v193 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,9), EAST, 2, WEST, 2);
    int32_t v194 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,8), EAST, 2, WEST, 2);
    int32_t v195 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,7), EAST, 2, WEST, 2);
    int32_t v196 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,6), EAST, 2, WEST, 3);
    int32_t v197 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,5), EAST, 3, WEST, 3);
    int32_t v198 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,4), EAST, 3, WEST, 3);
    int32_t v199 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,3), EAST, 3, WEST, 3);
    int32_t v200 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,2), EAST, 3, NORTH, 0);
    int32_t v201 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,2), EAST, 3, DMA, 2);
    int32_t v202 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(4,2), SOUTH, 0, NORTH, 0);
    int32_t v203 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(4,2), SOUTH, 0, DMA, 2);
    int32_t v204 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(5,2), SOUTH, 0, NORTH, 0);
    int32_t v205 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(5,2), SOUTH, 0, DMA, 2);
    int32_t v206 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(6,2), SOUTH, 0, DMA, 2);
  }

  //round is 3 hw split in : col -----------
  if (v1) {
    int32_t v207 = XAie_EnableShimDmaToAieStrmPort(getOrCreateDeviceInstance(), XAie_TileLoc(0,15), 7);
    int32_t v208 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(0,15), SOUTH, 7, NORTH, 1);
    int32_t v209 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(1,15), SOUTH, 1, NORTH, 1);
    int32_t v210 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2,15), SOUTH, 1, WEST, 0);
    int32_t v211 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2,14), EAST, 0, WEST, 0);
    int32_t v212 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2,13), EAST, 0, WEST, 0);
    int32_t v213 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2,12), EAST, 0, WEST, 0);
    int32_t v214 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2,11), EAST, 0, WEST, 0);
    int32_t v215 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2,10), EAST, 0, WEST, 0);
    int32_t v216 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2,9), EAST, 0, WEST, 0);
    int32_t v217 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2,8), EAST, 0, WEST, 0);
    int32_t v218 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2,7), EAST, 0, WEST, 0);
    int32_t v219 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2,6), EAST, 0, WEST, 0);
    int32_t v220 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2,5), EAST, 0, WEST, 0);
    int32_t v221 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2,4), EAST, 0, WEST, 0);
    int32_t v222 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(2,3), EAST, 0, NORTH, 0);
    int32_t v223 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,3), SOUTH, 0, NORTH, 0);
    int32_t v224 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(3,3), SOUTH, 0, DMA, 2);
    int32_t v225 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(4,3), SOUTH, 0, NORTH, 0);
    int32_t v226 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(4,3), SOUTH, 0, DMA, 2);
    int32_t v227 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(5,3), SOUTH, 0, NORTH, 0);
    int32_t v228 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(5,3), SOUTH, 0, DMA, 2);
    int32_t v229 = XAie_StrmConnCctEnable(getOrCreateDeviceInstance(), XAie_TileLoc(6,3), SOUTH, 0, DMA, 2);
  }

  } //----routing creation in col ----end-------

  return;
}



