module attributes {codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"], routing.pp_depth_map = {tensor_0 = 2 : i32, tensor_1 = 2 : i32, tensor_2 = 2 : i32}} {
  func.func @routing(%arg0: !emitc.ptr<!emitc.opaque<"XAie_DevInst">>) {
    %0 = "emitc.constant"() <{value = true}> : () -> i1
    emitc.verbatim "\0A//round is 0 hw split in : col -----------"
    emitc.if %0 {
      %1 = emitc.call_opaque "XAie_EnableShimDmaToAieStrmPort"(%arg0) {args = [0 : index, #emitc.opaque<"XAie_TileLoc(0,0)">, 3 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>) -> i32
      %2 = emitc.call_opaque "XAie_StrmConnCctEnable"(%arg0) {args = [0 : index, #emitc.opaque<"XAie_TileLoc(0,0)">, #emitc.opaque<"SOUTH">, 3 : i32, #emitc.opaque<"NORTH">, 0 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>) -> i32
      %3 = emitc.call_opaque "XAie_StrmConnCctEnable"(%arg0) {args = [0 : index, #emitc.opaque<"XAie_TileLoc(0,1)">, #emitc.opaque<"SOUTH">, 0 : i32, #emitc.opaque<"NORTH">, 0 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>) -> i32
      %4 = emitc.call_opaque "XAie_StrmConnCctEnable"(%arg0) {args = [0 : index, #emitc.opaque<"XAie_TileLoc(0,2)">, #emitc.opaque<"SOUTH">, 0 : i32, #emitc.opaque<"NORTH">, 0 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>) -> i32
      %5 = emitc.call_opaque "XAie_StrmConnCctEnable"(%arg0) {args = [0 : index, #emitc.opaque<"XAie_TileLoc(0,3)">, #emitc.opaque<"SOUTH">, 0 : i32, #emitc.opaque<"DMA">, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>) -> i32
    }
    emitc.verbatim "\0A//round is 0 hw split in : row -----------"
    emitc.if %0 {
      %1 = emitc.call_opaque "XAie_EnableShimDmaToAieStrmPort"(%arg0) {args = [0 : index, #emitc.opaque<"XAie_TileLoc(0,0)">, 7 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>) -> i32
      %2 = emitc.call_opaque "XAie_StrmConnCctEnable"(%arg0) {args = [0 : index, #emitc.opaque<"XAie_TileLoc(0,0)">, #emitc.opaque<"SOUTH">, 7 : i32, #emitc.opaque<"NORTH">, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>) -> i32
      %3 = emitc.call_opaque "XAie_StrmConnCctEnable"(%arg0) {args = [0 : index, #emitc.opaque<"XAie_TileLoc(0,1)">, #emitc.opaque<"SOUTH">, 1 : i32, #emitc.opaque<"NORTH">, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>) -> i32
      %4 = emitc.call_opaque "XAie_StrmConnCctEnable"(%arg0) {args = [0 : index, #emitc.opaque<"XAie_TileLoc(0,2)">, #emitc.opaque<"SOUTH">, 1 : i32, #emitc.opaque<"NORTH">, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>) -> i32
      %5 = emitc.call_opaque "XAie_StrmConnCctEnable"(%arg0) {args = [0 : index, #emitc.opaque<"XAie_TileLoc(0,3)">, #emitc.opaque<"SOUTH">, 1 : i32, #emitc.opaque<"DMA">, 0 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>) -> i32
      %6 = emitc.call_opaque "XAie_StrmPktSwSlaveSlotEnable"(%arg0) {args = [0 : index, #emitc.opaque<"XAie_TileLoc(0,3)">, #emitc.opaque<"DMA">, 0 : i32, 0 : i32, #emitc.opaque<"XAie_PacketInit(1, 0)">, 31 : i32, 0 : i32, 0 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>) -> i32
      %7 = emitc.call_opaque "XAie_StrmPktSwSlavePortEnable"(%arg0) {args = [0 : index, #emitc.opaque<"XAie_TileLoc(0,3)">, #emitc.opaque<"DMA">, 0 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>) -> i32
      %8 = emitc.call_opaque "XAie_EnableAieToShimDmaStrmPort"(%arg0) {args = [0 : index, #emitc.opaque<"XAie_TileLoc(0,0)">, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>) -> i32
      %9 = emitc.call_opaque "XAie_StrmPktSwMstrPortEnable"(%arg0) {args = [0 : index, #emitc.opaque<"XAie_TileLoc(0,3)">, #emitc.opaque<"SOUTH">, 0 : i32, #emitc.opaque<"XAIE_SS_PKT_DONOT_DROP_HEADER">, 0 : i32, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>) -> i32
      %10 = emitc.call_opaque "XAie_StrmConnCctEnable"(%arg0) {args = [0 : index, #emitc.opaque<"XAie_TileLoc(0,2)">, #emitc.opaque<"NORTH">, 0 : i32, #emitc.opaque<"SOUTH">, 0 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>) -> i32
      %11 = emitc.call_opaque "XAie_StrmConnCctEnable"(%arg0) {args = [0 : index, #emitc.opaque<"XAie_TileLoc(0,1)">, #emitc.opaque<"NORTH">, 0 : i32, #emitc.opaque<"SOUTH">, 0 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>) -> i32
      %12 = emitc.call_opaque "XAie_StrmConnCctEnable"(%arg0) {args = [0 : index, #emitc.opaque<"XAie_TileLoc(0,0)">, #emitc.opaque<"NORTH">, 0 : i32, #emitc.opaque<"SOUTH">, 1 : i32]} : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>) -> i32
    }
    return
  }
  emitc.include <"xaiengine.h">
}
