module attributes {codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"], routing.pp_depth_map = {tensor_0 = 2 : i32, tensor_1 = 2 : i32, tensor_2 = 2 : i32}} {
  func.func @routing(%arg0: !emitc.ptr<!emitc.opaque<"XAie_DevInst">>) {
    %0 = "emitc.constant"() <{value = #emitc.opaque<"XAIE_SS_PKT_DONOT_DROP_HEADER">}> : () -> !emitc.ptr<i8>
    %1 = "emitc.constant"() <{value = 31 : i32}> : () -> i32
    %2 = "emitc.constant"() <{value = #emitc.opaque<"NONE">}> : () -> !emitc.ptr<i8>
    %3 = "emitc.constant"() <{value = #emitc.opaque<"XAie_PacketInit(1, 0)">}> : () -> !emitc.opaque<"XAie_Packet">
    %4 = "emitc.constant"() <{value = #emitc.opaque<"XAie_PacketInit(0, 0)">}> : () -> !emitc.opaque<"XAie_Packet">
    %5 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
    %6 = "emitc.constant"() <{value = #emitc.opaque<"DMA">}> : () -> !emitc.ptr<i8>
    %7 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %8 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %9 = "emitc.constant"() <{value = #emitc.opaque<"SOUTH">}> : () -> !emitc.ptr<i8>
    %10 = "emitc.constant"() <{value = #emitc.opaque<"NORTH">}> : () -> !emitc.ptr<i8>
    %11 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %12 = "emitc.constant"() <{value = true}> : () -> i1
    %13 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    emitc.verbatim "\0A//round is 0 hw split in : col -----------"
    emitc.if %12 {
      %14 = emitc.call @XAie_TileLoc(%13, %13) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %15 = emitc.call @XAie_EnableShimDmaToAieStrmPort(%arg0, %14, %11) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
      %16 = emitc.call @XAie_TileLoc(%13, %13) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %17 = emitc.call @XAie_StrmConnCctEnable(%arg0, %16, %9, %11, %10, %13) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %18 = emitc.call @XAie_TileLoc(%13, %8) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %19 = emitc.call @XAie_StrmConnCctEnable(%arg0, %18, %9, %13, %10, %13) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %20 = emitc.call @XAie_TileLoc(%13, %7) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %21 = emitc.call @XAie_StrmConnCctEnable(%arg0, %20, %9, %13, %10, %13) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %22 = emitc.call @XAie_TileLoc(%13, %11) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %23 = emitc.call @XAie_StrmConnCctEnable(%arg0, %22, %9, %13, %6, %8) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
    }
    emitc.verbatim "\0A//round is 0 hw split in : row -----------"
    emitc.if %12 {
      %14 = emitc.call @XAie_TileLoc(%13, %13) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %15 = emitc.call @XAie_EnableShimDmaToAieStrmPort(%arg0, %14, %5) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
      %16 = emitc.call @XAie_TileLoc(%13, %13) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %17 = emitc.call @XAie_StrmConnCctEnable(%arg0, %16, %9, %5, %10, %8) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %18 = emitc.call @XAie_TileLoc(%13, %8) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %19 = emitc.call @XAie_StrmConnCctEnable(%arg0, %18, %9, %8, %10, %8) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %20 = emitc.call @XAie_TileLoc(%13, %7) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %21 = emitc.call @XAie_StrmConnCctEnable(%arg0, %20, %9, %8, %10, %8) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %22 = emitc.call @XAie_TileLoc(%13, %11) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %23 = emitc.call @XAie_StrmConnCctEnable(%arg0, %22, %9, %8, %6, %13) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %24 = emitc.call @XAie_TileLoc(%13, %11) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %25 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%arg0, %24, %6, %13, %13, %3, %1, %13, %13) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %26 = emitc.call @XAie_StrmPktSwSlavePortEnable(%arg0, %24, %6, %13) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %27 = emitc.call @XAie_TileLoc(%13, %13) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %28 = emitc.call @XAie_EnableAieToShimDmaStrmPort(%arg0, %27, %8) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
      %29 = emitc.call @XAie_TileLoc(%13, %11) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %30 = emitc.call @XAie_StrmPktSwMstrPortEnable(%arg0, %29, %9, %13, %0, %13, %8) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %31 = emitc.call @XAie_TileLoc(%13, %7) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %32 = emitc.call @XAie_StrmConnCctEnable(%arg0, %31, %10, %13, %9, %13) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %33 = emitc.call @XAie_TileLoc(%13, %8) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %34 = emitc.call @XAie_StrmConnCctEnable(%arg0, %33, %10, %13, %9, %13) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %35 = emitc.call @XAie_TileLoc(%13, %13) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %36 = emitc.call @XAie_StrmConnCctEnable(%arg0, %35, %10, %13, %9, %8) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
    }
    return
  }
  emitc.include <"xaiengine.h">
  emitc.func private @XAie_TileLoc(i32, i32) -> !emitc.opaque<"XAie_LocType">
  emitc.func private @XAie_EnableShimDmaToAieStrmPort(!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
  emitc.func private @XAie_EnableAieToShimDmaStrmPort(!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
  emitc.func private @XAie_StrmConnCctEnable(!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
  emitc.func private @XAie_StrmPktSwSlaveSlotEnable(!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
  emitc.func private @XAie_StrmPktSwMstrPortEnable(!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
  emitc.func private @XAie_StrmPktSwSlavePortEnable(!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
}
