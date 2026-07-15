// ******************************************************************************
// * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
// * SPDX-License-Identifier: Apache-2.0
// ******************************************************************************

module attributes {codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"]} {
  emitc.include <"xaiengine.h">
  emitc.func private @XAie_TileLoc(i32, i32) -> !emitc.opaque<"XAie_LocType">
  emitc.func private @getOrCreateDeviceInstance() -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
  emitc.func private @XAie_EnableShimDmaToAieStrmPort(!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
  emitc.func private @XAie_EnableAieToShimDmaStrmPort(!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
  emitc.func private @XAie_StrmConnCctEnable(!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
  emitc.func private @XAie_StrmPktSwSlaveSlotEnable(!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
  emitc.func private @XAie_StrmPktSwMstrPortEnable(!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
  emitc.func private @XAie_StrmPktSwSlavePortEnable(!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
  func.func @routing(%arg0: memref<256x256xi8>, %arg1: memref<256x256xi8>, %arg2: memref<256x256xi8>) {
    %0 = "emitc.constant"() <{value = #emitc.opaque<"XAie_PacketInit(16, 0)">}> : () -> !emitc.opaque<"XAie_Packet">
    %1 = "emitc.constant"() <{value = #emitc.opaque<"XAie_PacketInit(15, 0)">}> : () -> !emitc.opaque<"XAie_Packet">
    %2 = "emitc.constant"() <{value = #emitc.opaque<"XAie_PacketInit(14, 0)">}> : () -> !emitc.opaque<"XAie_Packet">
    %3 = "emitc.constant"() <{value = #emitc.opaque<"XAie_PacketInit(13, 0)">}> : () -> !emitc.opaque<"XAie_Packet">
    %4 = "emitc.constant"() <{value = #emitc.opaque<"XAie_PacketInit(12, 0)">}> : () -> !emitc.opaque<"XAie_Packet">
    %5 = "emitc.constant"() <{value = #emitc.opaque<"XAie_PacketInit(11, 0)">}> : () -> !emitc.opaque<"XAie_Packet">
    %6 = "emitc.constant"() <{value = #emitc.opaque<"XAie_PacketInit(10, 0)">}> : () -> !emitc.opaque<"XAie_Packet">
    %7 = "emitc.constant"() <{value = #emitc.opaque<"XAie_PacketInit(9, 0)">}> : () -> !emitc.opaque<"XAie_Packet">
    %8 = "emitc.constant"() <{value = #emitc.opaque<"XAie_PacketInit(8, 0)">}> : () -> !emitc.opaque<"XAie_Packet">
    %9 = "emitc.constant"() <{value = #emitc.opaque<"XAie_PacketInit(7, 0)">}> : () -> !emitc.opaque<"XAie_Packet">
    %10 = "emitc.constant"() <{value = #emitc.opaque<"XAie_PacketInit(6, 0)">}> : () -> !emitc.opaque<"XAie_Packet">
    %11 = "emitc.constant"() <{value = #emitc.opaque<"XAie_PacketInit(5, 0)">}> : () -> !emitc.opaque<"XAie_Packet">
    %12 = "emitc.constant"() <{value = #emitc.opaque<"XAie_PacketInit(4, 0)">}> : () -> !emitc.opaque<"XAie_Packet">
    %13 = "emitc.constant"() <{value = #emitc.opaque<"XAie_PacketInit(3, 0)">}> : () -> !emitc.opaque<"XAie_Packet">
    %14 = "emitc.constant"() <{value = #emitc.opaque<"XAie_PacketInit(2, 0)">}> : () -> !emitc.opaque<"XAie_Packet">
    %15 = "emitc.constant"() <{value = #emitc.opaque<"XAIE_SS_PKT_DONOT_DROP_HEADER">}> : () -> !emitc.ptr<i8>
    %16 = "emitc.constant"() <{value = 31 : i32}> : () -> i32
    %17 = "emitc.constant"() <{value = #emitc.opaque<"NONE">}> : () -> !emitc.ptr<i8>
    %18 = "emitc.constant"() <{value = #emitc.opaque<"XAie_PacketInit(1, 0)">}> : () -> !emitc.opaque<"XAie_Packet">
    %19 = "emitc.constant"() <{value = #emitc.opaque<"XAie_PacketInit(0, 0)">}> : () -> !emitc.opaque<"XAie_Packet">
    %20 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
    %21 = "emitc.constant"() <{value = 6 : i32}> : () -> i32
    %22 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %23 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %24 = "emitc.constant"() <{value = #emitc.opaque<"DMA">}> : () -> !emitc.ptr<i8>
    %25 = "emitc.constant"() <{value = #emitc.opaque<"EAST">}> : () -> !emitc.ptr<i8>
    %26 = "emitc.constant"() <{value = #emitc.opaque<"WEST">}> : () -> !emitc.ptr<i8>
    %27 = "emitc.constant"() <{value = #emitc.opaque<"SOUTH">}> : () -> !emitc.ptr<i8>
    %28 = "emitc.constant"() <{value = #emitc.opaque<"NORTH">}> : () -> !emitc.ptr<i8>
    %29 = "emitc.constant"() <{value = true}> : () -> i1
    %30 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %31 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %32 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %33 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    emitc.verbatim "\0A//round is 0 hw split in : col -----------"
    emitc.if %29 {
      %34 = emitc.call @XAie_TileLoc(%31, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %35 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %36 = emitc.call @XAie_EnableShimDmaToAieStrmPort(%35, %34, %30) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
      %37 = emitc.call @XAie_TileLoc(%31, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %38 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %39 = emitc.call @XAie_StrmConnCctEnable(%38, %37, %27, %30, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %40 = emitc.call @XAie_TileLoc(%31, %32) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %41 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %42 = emitc.call @XAie_StrmConnCctEnable(%41, %40, %27, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %43 = emitc.call @XAie_TileLoc(%31, %31) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %44 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %45 = emitc.call @XAie_StrmConnCctEnable(%44, %43, %27, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %46 = emitc.call @XAie_TileLoc(%31, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %47 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %48 = emitc.call @XAie_StrmConnCctEnable(%47, %46, %27, %33, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %49 = emitc.call @XAie_TileLoc(%32, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %50 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %51 = emitc.call @XAie_StrmConnCctEnable(%50, %49, %25, %33, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %52 = emitc.call @XAie_TileLoc(%33, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %53 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %54 = emitc.call @XAie_StrmConnCctEnable(%53, %52, %25, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %55 = emitc.call @XAie_TileLoc(%33, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %56 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %57 = emitc.call @XAie_StrmConnCctEnable(%56, %55, %25, %33, %24, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %58 = emitc.call @XAie_TileLoc(%33, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %59 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %60 = emitc.call @XAie_StrmConnCctEnable(%59, %58, %27, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %61 = emitc.call @XAie_TileLoc(%33, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %62 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %63 = emitc.call @XAie_StrmConnCctEnable(%62, %61, %27, %33, %24, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %64 = emitc.call @XAie_TileLoc(%33, %22) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %65 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %66 = emitc.call @XAie_StrmConnCctEnable(%65, %64, %27, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %67 = emitc.call @XAie_TileLoc(%33, %22) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %68 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %69 = emitc.call @XAie_StrmConnCctEnable(%68, %67, %27, %33, %24, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %70 = emitc.call @XAie_TileLoc(%33, %21) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %71 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %72 = emitc.call @XAie_StrmConnCctEnable(%71, %70, %27, %33, %24, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
    }
    emitc.verbatim "\0A//round is 1 hw split in : col -----------"
    emitc.if %29 {
      %34 = emitc.call @XAie_TileLoc(%31, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %35 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %36 = emitc.call @XAie_EnableShimDmaToAieStrmPort(%35, %34, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
      %37 = emitc.call @XAie_TileLoc(%31, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %38 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %39 = emitc.call @XAie_StrmConnCctEnable(%38, %37, %27, %20, %28, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %40 = emitc.call @XAie_TileLoc(%31, %32) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %41 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %42 = emitc.call @XAie_StrmConnCctEnable(%41, %40, %27, %32, %28, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %43 = emitc.call @XAie_TileLoc(%31, %31) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %44 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %45 = emitc.call @XAie_StrmConnCctEnable(%44, %43, %27, %32, %28, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %46 = emitc.call @XAie_TileLoc(%31, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %47 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %48 = emitc.call @XAie_StrmConnCctEnable(%47, %46, %27, %32, %26, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %49 = emitc.call @XAie_TileLoc(%32, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %50 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %51 = emitc.call @XAie_StrmConnCctEnable(%50, %49, %25, %32, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %52 = emitc.call @XAie_TileLoc(%32, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %53 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %54 = emitc.call @XAie_StrmConnCctEnable(%53, %52, %25, %32, %24, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %55 = emitc.call @XAie_TileLoc(%32, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %56 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %57 = emitc.call @XAie_StrmConnCctEnable(%56, %55, %27, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %58 = emitc.call @XAie_TileLoc(%32, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %59 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %60 = emitc.call @XAie_StrmConnCctEnable(%59, %58, %27, %33, %24, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %61 = emitc.call @XAie_TileLoc(%32, %22) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %62 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %63 = emitc.call @XAie_StrmConnCctEnable(%62, %61, %27, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %64 = emitc.call @XAie_TileLoc(%32, %22) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %65 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %66 = emitc.call @XAie_StrmConnCctEnable(%65, %64, %27, %33, %24, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %67 = emitc.call @XAie_TileLoc(%32, %21) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %68 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %69 = emitc.call @XAie_StrmConnCctEnable(%68, %67, %27, %33, %24, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
    }
    emitc.verbatim "\0A//round is 2 hw split in : col -----------"
    emitc.if %29 {
      %34 = emitc.call @XAie_TileLoc(%30, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %35 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %36 = emitc.call @XAie_EnableShimDmaToAieStrmPort(%35, %34, %30) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
      %37 = emitc.call @XAie_TileLoc(%30, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %38 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %39 = emitc.call @XAie_StrmConnCctEnable(%38, %37, %27, %30, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %40 = emitc.call @XAie_TileLoc(%30, %32) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %41 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %42 = emitc.call @XAie_StrmConnCctEnable(%41, %40, %27, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %43 = emitc.call @XAie_TileLoc(%30, %31) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %44 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %45 = emitc.call @XAie_StrmConnCctEnable(%44, %43, %27, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %46 = emitc.call @XAie_TileLoc(%30, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %47 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %48 = emitc.call @XAie_StrmConnCctEnable(%47, %46, %27, %33, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %49 = emitc.call @XAie_TileLoc(%31, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %50 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %51 = emitc.call @XAie_StrmConnCctEnable(%50, %49, %25, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %52 = emitc.call @XAie_TileLoc(%31, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %53 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %54 = emitc.call @XAie_StrmConnCctEnable(%53, %52, %25, %33, %24, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %55 = emitc.call @XAie_TileLoc(%31, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %56 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %57 = emitc.call @XAie_StrmConnCctEnable(%56, %55, %27, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %58 = emitc.call @XAie_TileLoc(%31, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %59 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %60 = emitc.call @XAie_StrmConnCctEnable(%59, %58, %27, %33, %24, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %61 = emitc.call @XAie_TileLoc(%31, %22) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %62 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %63 = emitc.call @XAie_StrmConnCctEnable(%62, %61, %27, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %64 = emitc.call @XAie_TileLoc(%31, %22) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %65 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %66 = emitc.call @XAie_StrmConnCctEnable(%65, %64, %27, %33, %24, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %67 = emitc.call @XAie_TileLoc(%31, %21) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %68 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %69 = emitc.call @XAie_StrmConnCctEnable(%68, %67, %27, %33, %24, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
    }
    emitc.verbatim "\0A//round is 3 hw split in : col -----------"
    emitc.if %29 {
      %34 = emitc.call @XAie_TileLoc(%30, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %35 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %36 = emitc.call @XAie_EnableShimDmaToAieStrmPort(%35, %34, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
      %37 = emitc.call @XAie_TileLoc(%30, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %38 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %39 = emitc.call @XAie_StrmConnCctEnable(%38, %37, %27, %20, %28, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %40 = emitc.call @XAie_TileLoc(%30, %32) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %41 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %42 = emitc.call @XAie_StrmConnCctEnable(%41, %40, %27, %32, %28, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %43 = emitc.call @XAie_TileLoc(%30, %31) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %44 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %45 = emitc.call @XAie_StrmConnCctEnable(%44, %43, %27, %32, %28, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %46 = emitc.call @XAie_TileLoc(%30, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %47 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %48 = emitc.call @XAie_StrmConnCctEnable(%47, %46, %27, %32, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %49 = emitc.call @XAie_TileLoc(%30, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %50 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %51 = emitc.call @XAie_StrmConnCctEnable(%50, %49, %27, %32, %24, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %52 = emitc.call @XAie_TileLoc(%30, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %53 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %54 = emitc.call @XAie_StrmConnCctEnable(%53, %52, %27, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %55 = emitc.call @XAie_TileLoc(%30, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %56 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %57 = emitc.call @XAie_StrmConnCctEnable(%56, %55, %27, %33, %24, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %58 = emitc.call @XAie_TileLoc(%30, %22) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %59 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %60 = emitc.call @XAie_StrmConnCctEnable(%59, %58, %27, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %61 = emitc.call @XAie_TileLoc(%30, %22) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %62 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %63 = emitc.call @XAie_StrmConnCctEnable(%62, %61, %27, %33, %24, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %64 = emitc.call @XAie_TileLoc(%30, %21) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %65 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %66 = emitc.call @XAie_StrmConnCctEnable(%65, %64, %27, %33, %24, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
    }
    emitc.verbatim "\0A//round is 0 hw split in : row -----------"
    emitc.if %29 {
      %34 = emitc.call @XAie_TileLoc(%21, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %35 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %36 = emitc.call @XAie_EnableShimDmaToAieStrmPort(%35, %34, %30) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
      %37 = emitc.call @XAie_TileLoc(%21, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %38 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %39 = emitc.call @XAie_StrmConnCctEnable(%38, %37, %27, %30, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %40 = emitc.call @XAie_TileLoc(%21, %32) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %41 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %42 = emitc.call @XAie_StrmConnCctEnable(%41, %40, %27, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %43 = emitc.call @XAie_TileLoc(%21, %31) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %44 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %45 = emitc.call @XAie_StrmConnCctEnable(%44, %43, %27, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %46 = emitc.call @XAie_TileLoc(%21, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %47 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %48 = emitc.call @XAie_StrmConnCctEnable(%47, %46, %27, %33, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %49 = emitc.call @XAie_TileLoc(%22, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %50 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %51 = emitc.call @XAie_StrmConnCctEnable(%50, %49, %25, %33, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %52 = emitc.call @XAie_TileLoc(%23, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %53 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %54 = emitc.call @XAie_StrmConnCctEnable(%53, %52, %25, %33, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %55 = emitc.call @XAie_TileLoc(%30, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %56 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %57 = emitc.call @XAie_StrmConnCctEnable(%56, %55, %25, %33, %26, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %58 = emitc.call @XAie_TileLoc(%30, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %59 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %60 = emitc.call @XAie_StrmConnCctEnable(%59, %58, %25, %33, %24, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %61 = emitc.call @XAie_TileLoc(%31, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %62 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %63 = emitc.call @XAie_StrmConnCctEnable(%62, %61, %25, %32, %26, %31) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %64 = emitc.call @XAie_TileLoc(%31, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %65 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %66 = emitc.call @XAie_StrmConnCctEnable(%65, %64, %25, %32, %24, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %67 = emitc.call @XAie_TileLoc(%32, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %68 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %69 = emitc.call @XAie_StrmConnCctEnable(%68, %67, %25, %31, %26, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %70 = emitc.call @XAie_TileLoc(%32, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %71 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %72 = emitc.call @XAie_StrmConnCctEnable(%71, %70, %25, %31, %24, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %73 = emitc.call @XAie_TileLoc(%33, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %74 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %75 = emitc.call @XAie_StrmConnCctEnable(%74, %73, %25, %32, %24, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %76 = emitc.call @XAie_TileLoc(%33, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %77 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %78 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%77, %76, %24, %33, %33, %18, %16, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %79 = emitc.call @XAie_StrmPktSwSlavePortEnable(%77, %76, %24, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %80 = emitc.call @XAie_StrmPktSwMstrPortEnable(%77, %76, %25, %33, %15, %33, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %81 = emitc.call @XAie_TileLoc(%32, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %82 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %83 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%82, %81, %26, %33, %33, %19, %33, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %84 = emitc.call @XAie_StrmPktSwSlavePortEnable(%82, %81, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %85 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%82, %81, %24, %33, %33, %14, %16, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %86 = emitc.call @XAie_StrmPktSwSlavePortEnable(%82, %81, %24, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %87 = emitc.call @XAie_StrmPktSwMstrPortEnable(%82, %81, %25, %33, %15, %33, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %88 = emitc.call @XAie_TileLoc(%31, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %89 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %90 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%89, %88, %26, %33, %33, %19, %33, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %91 = emitc.call @XAie_StrmPktSwSlavePortEnable(%89, %88, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %92 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%89, %88, %24, %33, %33, %13, %16, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %93 = emitc.call @XAie_StrmPktSwSlavePortEnable(%89, %88, %24, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %94 = emitc.call @XAie_StrmPktSwMstrPortEnable(%89, %88, %25, %33, %15, %33, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %95 = emitc.call @XAie_TileLoc(%30, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %96 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %97 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%96, %95, %26, %33, %33, %19, %33, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %98 = emitc.call @XAie_StrmPktSwSlavePortEnable(%96, %95, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %99 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%96, %95, %24, %33, %33, %12, %16, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %100 = emitc.call @XAie_StrmPktSwSlavePortEnable(%96, %95, %24, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %101 = emitc.call @XAie_TileLoc(%30, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %102 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %103 = emitc.call @XAie_EnableAieToShimDmaStrmPort(%102, %101, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
      %104 = emitc.call @XAie_TileLoc(%30, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %105 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %106 = emitc.call @XAie_StrmPktSwMstrPortEnable(%105, %104, %27, %33, %15, %33, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %107 = emitc.call @XAie_TileLoc(%30, %31) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %108 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %109 = emitc.call @XAie_StrmConnCctEnable(%108, %107, %28, %33, %27, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %110 = emitc.call @XAie_TileLoc(%30, %32) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %111 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %112 = emitc.call @XAie_StrmConnCctEnable(%111, %110, %28, %33, %27, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %113 = emitc.call @XAie_TileLoc(%30, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %114 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %115 = emitc.call @XAie_StrmConnCctEnable(%114, %113, %28, %33, %27, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
    }
    emitc.verbatim "\0A//round is 1 hw split in : row -----------"
    emitc.if %29 {
      %34 = emitc.call @XAie_TileLoc(%21, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %35 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %36 = emitc.call @XAie_EnableShimDmaToAieStrmPort(%35, %34, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
      %37 = emitc.call @XAie_TileLoc(%21, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %38 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %39 = emitc.call @XAie_StrmConnCctEnable(%38, %37, %27, %20, %28, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %40 = emitc.call @XAie_TileLoc(%21, %32) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %41 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %42 = emitc.call @XAie_StrmConnCctEnable(%41, %40, %27, %32, %28, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %43 = emitc.call @XAie_TileLoc(%21, %31) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %44 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %45 = emitc.call @XAie_StrmConnCctEnable(%44, %43, %27, %32, %28, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %46 = emitc.call @XAie_TileLoc(%21, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %47 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %48 = emitc.call @XAie_StrmConnCctEnable(%47, %46, %27, %32, %26, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %49 = emitc.call @XAie_TileLoc(%22, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %50 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %51 = emitc.call @XAie_StrmConnCctEnable(%50, %49, %25, %32, %26, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %52 = emitc.call @XAie_TileLoc(%23, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %53 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %54 = emitc.call @XAie_StrmConnCctEnable(%53, %52, %25, %32, %26, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %55 = emitc.call @XAie_TileLoc(%30, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %56 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %57 = emitc.call @XAie_StrmConnCctEnable(%56, %55, %25, %32, %26, %31) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %58 = emitc.call @XAie_TileLoc(%30, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %59 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %60 = emitc.call @XAie_StrmConnCctEnable(%59, %58, %25, %32, %28, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %61 = emitc.call @XAie_TileLoc(%31, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %62 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %63 = emitc.call @XAie_StrmConnCctEnable(%62, %61, %25, %31, %26, %30) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %64 = emitc.call @XAie_TileLoc(%31, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %65 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %66 = emitc.call @XAie_StrmConnCctEnable(%65, %64, %25, %31, %28, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %67 = emitc.call @XAie_TileLoc(%32, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %68 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %69 = emitc.call @XAie_StrmConnCctEnable(%68, %67, %25, %30, %26, %31) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %70 = emitc.call @XAie_TileLoc(%32, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %71 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %72 = emitc.call @XAie_StrmConnCctEnable(%71, %70, %25, %30, %28, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %73 = emitc.call @XAie_TileLoc(%33, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %74 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %75 = emitc.call @XAie_StrmConnCctEnable(%74, %73, %25, %31, %28, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %76 = emitc.call @XAie_TileLoc(%33, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %77 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %78 = emitc.call @XAie_StrmConnCctEnable(%77, %76, %27, %32, %24, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %79 = emitc.call @XAie_TileLoc(%32, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %80 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %81 = emitc.call @XAie_StrmConnCctEnable(%80, %79, %27, %32, %24, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %82 = emitc.call @XAie_TileLoc(%31, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %83 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %84 = emitc.call @XAie_StrmConnCctEnable(%83, %82, %27, %32, %24, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %85 = emitc.call @XAie_TileLoc(%30, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %86 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %87 = emitc.call @XAie_StrmConnCctEnable(%86, %85, %27, %32, %24, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %88 = emitc.call @XAie_TileLoc(%33, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %89 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %90 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%89, %88, %24, %33, %33, %11, %16, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %91 = emitc.call @XAie_StrmPktSwSlavePortEnable(%89, %88, %24, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %92 = emitc.call @XAie_StrmPktSwMstrPortEnable(%89, %88, %25, %33, %15, %33, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %93 = emitc.call @XAie_TileLoc(%32, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %94 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %95 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%94, %93, %26, %33, %33, %19, %33, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %96 = emitc.call @XAie_StrmPktSwSlavePortEnable(%94, %93, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %97 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%94, %93, %24, %33, %33, %10, %16, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %98 = emitc.call @XAie_StrmPktSwSlavePortEnable(%94, %93, %24, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %99 = emitc.call @XAie_StrmPktSwMstrPortEnable(%94, %93, %25, %33, %15, %33, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %100 = emitc.call @XAie_TileLoc(%31, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %101 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %102 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%101, %100, %26, %33, %33, %19, %33, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %103 = emitc.call @XAie_StrmPktSwSlavePortEnable(%101, %100, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %104 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%101, %100, %24, %33, %33, %9, %16, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %105 = emitc.call @XAie_StrmPktSwSlavePortEnable(%101, %100, %24, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %106 = emitc.call @XAie_StrmPktSwMstrPortEnable(%101, %100, %25, %33, %15, %33, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %107 = emitc.call @XAie_TileLoc(%30, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %108 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %109 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%108, %107, %26, %33, %33, %19, %33, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %110 = emitc.call @XAie_StrmPktSwSlavePortEnable(%108, %107, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %111 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%108, %107, %24, %33, %33, %8, %16, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %112 = emitc.call @XAie_StrmPktSwSlavePortEnable(%108, %107, %24, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %113 = emitc.call @XAie_TileLoc(%30, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %114 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %115 = emitc.call @XAie_EnableAieToShimDmaStrmPort(%114, %113, %30) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
      %116 = emitc.call @XAie_TileLoc(%30, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %117 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %118 = emitc.call @XAie_StrmPktSwMstrPortEnable(%117, %116, %27, %33, %15, %33, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %119 = emitc.call @XAie_TileLoc(%30, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %120 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %121 = emitc.call @XAie_StrmConnCctEnable(%120, %119, %28, %33, %27, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %122 = emitc.call @XAie_TileLoc(%30, %31) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %123 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %124 = emitc.call @XAie_StrmConnCctEnable(%123, %122, %28, %32, %27, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %125 = emitc.call @XAie_TileLoc(%30, %32) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %126 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %127 = emitc.call @XAie_StrmConnCctEnable(%126, %125, %28, %32, %27, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %128 = emitc.call @XAie_TileLoc(%30, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %129 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %130 = emitc.call @XAie_StrmConnCctEnable(%129, %128, %28, %32, %27, %30) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
    }
    emitc.verbatim "\0A//round is 2 hw split in : row -----------"
    emitc.if %29 {
      %34 = emitc.call @XAie_TileLoc(%20, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %35 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %36 = emitc.call @XAie_EnableShimDmaToAieStrmPort(%35, %34, %30) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
      %37 = emitc.call @XAie_TileLoc(%20, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %38 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %39 = emitc.call @XAie_StrmConnCctEnable(%38, %37, %27, %30, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %40 = emitc.call @XAie_TileLoc(%20, %32) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %41 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %42 = emitc.call @XAie_StrmConnCctEnable(%41, %40, %27, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %43 = emitc.call @XAie_TileLoc(%20, %31) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %44 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %45 = emitc.call @XAie_StrmConnCctEnable(%44, %43, %27, %33, %28, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %46 = emitc.call @XAie_TileLoc(%20, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %47 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %48 = emitc.call @XAie_StrmConnCctEnable(%47, %46, %27, %33, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %49 = emitc.call @XAie_TileLoc(%21, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %50 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %51 = emitc.call @XAie_StrmConnCctEnable(%50, %49, %25, %33, %26, %31) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %52 = emitc.call @XAie_TileLoc(%22, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %53 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %54 = emitc.call @XAie_StrmConnCctEnable(%53, %52, %25, %31, %26, %31) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %55 = emitc.call @XAie_TileLoc(%23, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %56 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %57 = emitc.call @XAie_StrmConnCctEnable(%56, %55, %25, %31, %26, %31) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %58 = emitc.call @XAie_TileLoc(%30, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %59 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %60 = emitc.call @XAie_StrmConnCctEnable(%59, %58, %25, %31, %26, %30) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %61 = emitc.call @XAie_TileLoc(%31, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %62 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %63 = emitc.call @XAie_StrmConnCctEnable(%62, %61, %25, %30, %28, %31) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %64 = emitc.call @XAie_TileLoc(%31, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %65 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %66 = emitc.call @XAie_StrmConnCctEnable(%65, %64, %27, %31, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %67 = emitc.call @XAie_TileLoc(%31, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %68 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %69 = emitc.call @XAie_StrmConnCctEnable(%68, %67, %27, %31, %28, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %70 = emitc.call @XAie_TileLoc(%32, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %71 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %72 = emitc.call @XAie_StrmConnCctEnable(%71, %70, %25, %33, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %73 = emitc.call @XAie_TileLoc(%32, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %74 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %75 = emitc.call @XAie_StrmConnCctEnable(%74, %73, %25, %33, %28, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %76 = emitc.call @XAie_TileLoc(%33, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %77 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %78 = emitc.call @XAie_StrmConnCctEnable(%77, %76, %25, %33, %28, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %79 = emitc.call @XAie_TileLoc(%33, %22) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %80 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %81 = emitc.call @XAie_StrmConnCctEnable(%80, %79, %27, %32, %24, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %82 = emitc.call @XAie_TileLoc(%32, %22) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %83 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %84 = emitc.call @XAie_StrmConnCctEnable(%83, %82, %27, %32, %24, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %85 = emitc.call @XAie_TileLoc(%31, %22) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %86 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %87 = emitc.call @XAie_StrmConnCctEnable(%86, %85, %27, %32, %25, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %88 = emitc.call @XAie_TileLoc(%31, %22) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %89 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %90 = emitc.call @XAie_StrmConnCctEnable(%89, %88, %27, %32, %24, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %91 = emitc.call @XAie_TileLoc(%30, %22) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %92 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %93 = emitc.call @XAie_StrmConnCctEnable(%92, %91, %26, %33, %24, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %94 = emitc.call @XAie_TileLoc(%33, %22) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %95 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %96 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%95, %94, %24, %33, %33, %7, %16, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %97 = emitc.call @XAie_StrmPktSwSlavePortEnable(%95, %94, %24, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %98 = emitc.call @XAie_StrmPktSwMstrPortEnable(%95, %94, %25, %33, %15, %33, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %99 = emitc.call @XAie_TileLoc(%32, %22) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %100 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %101 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%100, %99, %26, %33, %33, %19, %33, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %102 = emitc.call @XAie_StrmPktSwSlavePortEnable(%100, %99, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %103 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%100, %99, %24, %33, %33, %6, %16, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %104 = emitc.call @XAie_StrmPktSwSlavePortEnable(%100, %99, %24, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %105 = emitc.call @XAie_StrmPktSwMstrPortEnable(%100, %99, %25, %33, %15, %33, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %106 = emitc.call @XAie_TileLoc(%31, %22) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %107 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %108 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%107, %106, %26, %33, %33, %19, %33, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %109 = emitc.call @XAie_StrmPktSwSlavePortEnable(%107, %106, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %110 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%107, %106, %24, %33, %33, %5, %16, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %111 = emitc.call @XAie_StrmPktSwSlavePortEnable(%107, %106, %24, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %112 = emitc.call @XAie_StrmPktSwMstrPortEnable(%107, %106, %25, %32, %15, %33, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %113 = emitc.call @XAie_TileLoc(%30, %22) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %114 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %115 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%114, %113, %26, %32, %33, %19, %33, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %116 = emitc.call @XAie_StrmPktSwSlavePortEnable(%114, %113, %26, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %117 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%114, %113, %24, %33, %33, %4, %16, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %118 = emitc.call @XAie_StrmPktSwSlavePortEnable(%114, %113, %24, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %119 = emitc.call @XAie_TileLoc(%31, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %120 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %121 = emitc.call @XAie_EnableAieToShimDmaStrmPort(%120, %119, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
      %122 = emitc.call @XAie_TileLoc(%30, %22) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %123 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %124 = emitc.call @XAie_StrmPktSwMstrPortEnable(%123, %122, %27, %33, %15, %33, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %125 = emitc.call @XAie_TileLoc(%30, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %126 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %127 = emitc.call @XAie_StrmConnCctEnable(%126, %125, %28, %33, %26, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %128 = emitc.call @XAie_TileLoc(%31, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %129 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %130 = emitc.call @XAie_StrmConnCctEnable(%129, %128, %25, %33, %27, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %131 = emitc.call @XAie_TileLoc(%31, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %132 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %133 = emitc.call @XAie_StrmConnCctEnable(%132, %131, %28, %33, %27, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %134 = emitc.call @XAie_TileLoc(%31, %31) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %135 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %136 = emitc.call @XAie_StrmConnCctEnable(%135, %134, %28, %33, %27, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %137 = emitc.call @XAie_TileLoc(%31, %32) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %138 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %139 = emitc.call @XAie_StrmConnCctEnable(%138, %137, %28, %33, %27, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %140 = emitc.call @XAie_TileLoc(%31, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %141 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %142 = emitc.call @XAie_StrmConnCctEnable(%141, %140, %28, %33, %27, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
    }
    emitc.verbatim "\0A//round is 3 hw split in : row -----------"
    emitc.if %29 {
      %34 = emitc.call @XAie_TileLoc(%20, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %35 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %36 = emitc.call @XAie_EnableShimDmaToAieStrmPort(%35, %34, %20) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
      %37 = emitc.call @XAie_TileLoc(%20, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %38 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %39 = emitc.call @XAie_StrmConnCctEnable(%38, %37, %27, %20, %28, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %40 = emitc.call @XAie_TileLoc(%20, %32) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %41 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %42 = emitc.call @XAie_StrmConnCctEnable(%41, %40, %27, %32, %28, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %43 = emitc.call @XAie_TileLoc(%20, %31) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %44 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %45 = emitc.call @XAie_StrmConnCctEnable(%44, %43, %27, %32, %28, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %46 = emitc.call @XAie_TileLoc(%20, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %47 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %48 = emitc.call @XAie_StrmConnCctEnable(%47, %46, %27, %32, %26, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %49 = emitc.call @XAie_TileLoc(%21, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %50 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %51 = emitc.call @XAie_StrmConnCctEnable(%50, %49, %25, %32, %26, %30) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %52 = emitc.call @XAie_TileLoc(%22, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %53 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %54 = emitc.call @XAie_StrmConnCctEnable(%53, %52, %25, %30, %26, %30) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %55 = emitc.call @XAie_TileLoc(%23, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %56 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %57 = emitc.call @XAie_StrmConnCctEnable(%56, %55, %25, %30, %26, %30) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %58 = emitc.call @XAie_TileLoc(%30, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %59 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %60 = emitc.call @XAie_StrmConnCctEnable(%59, %58, %25, %30, %28, %31) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %61 = emitc.call @XAie_TileLoc(%30, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %62 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %63 = emitc.call @XAie_StrmConnCctEnable(%62, %61, %27, %31, %26, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %64 = emitc.call @XAie_TileLoc(%31, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %65 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %66 = emitc.call @XAie_StrmConnCctEnable(%65, %64, %25, %32, %26, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %67 = emitc.call @XAie_TileLoc(%32, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %68 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %69 = emitc.call @XAie_StrmConnCctEnable(%68, %67, %25, %32, %26, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %70 = emitc.call @XAie_TileLoc(%33, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %71 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %72 = emitc.call @XAie_StrmConnCctEnable(%71, %70, %25, %32, %28, %31) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %73 = emitc.call @XAie_TileLoc(%33, %22) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %74 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %75 = emitc.call @XAie_StrmConnCctEnable(%74, %73, %27, %31, %28, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %76 = emitc.call @XAie_TileLoc(%33, %21) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %77 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %78 = emitc.call @XAie_StrmConnCctEnable(%77, %76, %27, %32, %25, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %79 = emitc.call @XAie_TileLoc(%33, %21) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %80 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %81 = emitc.call @XAie_StrmConnCctEnable(%80, %79, %27, %32, %24, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %82 = emitc.call @XAie_TileLoc(%32, %21) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %83 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %84 = emitc.call @XAie_StrmConnCctEnable(%83, %82, %26, %33, %25, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %85 = emitc.call @XAie_TileLoc(%32, %21) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %86 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %87 = emitc.call @XAie_StrmConnCctEnable(%86, %85, %26, %33, %24, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %88 = emitc.call @XAie_TileLoc(%31, %21) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %89 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %90 = emitc.call @XAie_StrmConnCctEnable(%89, %88, %26, %33, %25, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %91 = emitc.call @XAie_TileLoc(%31, %21) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %92 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %93 = emitc.call @XAie_StrmConnCctEnable(%92, %91, %26, %33, %24, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %94 = emitc.call @XAie_TileLoc(%30, %21) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %95 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %96 = emitc.call @XAie_StrmConnCctEnable(%95, %94, %26, %33, %24, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %97 = emitc.call @XAie_TileLoc(%33, %21) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %98 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %99 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%98, %97, %24, %33, %33, %3, %16, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %100 = emitc.call @XAie_StrmPktSwSlavePortEnable(%98, %97, %24, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %101 = emitc.call @XAie_StrmPktSwMstrPortEnable(%98, %97, %25, %32, %15, %33, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %102 = emitc.call @XAie_TileLoc(%32, %21) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %103 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %104 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%103, %102, %26, %32, %33, %19, %33, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %105 = emitc.call @XAie_StrmPktSwSlavePortEnable(%103, %102, %26, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %106 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%103, %102, %24, %33, %33, %2, %16, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %107 = emitc.call @XAie_StrmPktSwSlavePortEnable(%103, %102, %24, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %108 = emitc.call @XAie_StrmPktSwMstrPortEnable(%103, %102, %25, %32, %15, %33, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %109 = emitc.call @XAie_TileLoc(%31, %21) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %110 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %111 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%110, %109, %26, %32, %33, %19, %33, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %112 = emitc.call @XAie_StrmPktSwSlavePortEnable(%110, %109, %26, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %113 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%110, %109, %24, %33, %33, %1, %16, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %114 = emitc.call @XAie_StrmPktSwSlavePortEnable(%110, %109, %24, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %115 = emitc.call @XAie_StrmPktSwMstrPortEnable(%110, %109, %25, %32, %15, %33, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %116 = emitc.call @XAie_TileLoc(%30, %21) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %117 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %118 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%117, %116, %26, %32, %33, %19, %33, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %119 = emitc.call @XAie_StrmPktSwSlavePortEnable(%117, %116, %26, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %120 = emitc.call @XAie_StrmPktSwSlaveSlotEnable(%117, %116, %24, %33, %33, %0, %16, %33, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, i32, !emitc.opaque<"XAie_Packet">, i32, i32, i32) -> i32
      %121 = emitc.call @XAie_StrmPktSwSlavePortEnable(%117, %116, %24, %33) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32) -> i32
      %122 = emitc.call @XAie_TileLoc(%31, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %123 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %124 = emitc.call @XAie_EnableAieToShimDmaStrmPort(%123, %122, %30) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
      %125 = emitc.call @XAie_TileLoc(%30, %21) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %126 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %127 = emitc.call @XAie_StrmPktSwMstrPortEnable(%126, %125, %27, %33, %15, %33, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32, i32) -> i32
      %128 = emitc.call @XAie_TileLoc(%30, %22) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %129 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %130 = emitc.call @XAie_StrmConnCctEnable(%129, %128, %28, %33, %27, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %131 = emitc.call @XAie_TileLoc(%30, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %132 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %133 = emitc.call @XAie_StrmConnCctEnable(%132, %131, %28, %32, %26, %31) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %134 = emitc.call @XAie_TileLoc(%31, %23) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %135 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %136 = emitc.call @XAie_StrmConnCctEnable(%135, %134, %25, %31, %27, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %137 = emitc.call @XAie_TileLoc(%31, %30) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %138 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %139 = emitc.call @XAie_StrmConnCctEnable(%138, %137, %28, %32, %27, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %140 = emitc.call @XAie_TileLoc(%31, %31) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %141 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %142 = emitc.call @XAie_StrmConnCctEnable(%141, %140, %28, %32, %27, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %143 = emitc.call @XAie_TileLoc(%31, %32) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %144 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %145 = emitc.call @XAie_StrmConnCctEnable(%144, %143, %28, %32, %27, %32) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
      %146 = emitc.call @XAie_TileLoc(%31, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
      %147 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
      %148 = emitc.call @XAie_StrmConnCctEnable(%147, %146, %28, %32, %27, %30) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, !emitc.ptr<i8>, i32, !emitc.ptr<i8>, i32) -> i32
    }
    return
  }
}
