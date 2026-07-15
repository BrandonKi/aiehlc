/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/
module attributes {codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"]} {
  func.func @createroute() {
    %0 = "emitc.constant"() <{value = 35 : i32}> : () -> i32
    %1 = "emitc.constant"() <{value = 9 : i32}> : () -> i32
    %2 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %3 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
    %4 = "emitc.constant"() <{value = 6 : i32}> : () -> i32
    %5 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %6 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %7 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %8 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %9 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %10 = "emitc.constant"() <{value = 34 : i32}> : () -> i32
    %11 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %12 = emitc.call @XAie_TileLoc(%11, %10) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %13 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %14 = emitc.call @XAie_EnableShimDmaToAieStrmPort(%13, %12, %11) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
    %15 = emitc.call @XAie_TileLoc(%11, %10) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %16 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %17 = emitc.call @XAie_StrmConnCctEnable(%16, %15, %9, %11, %11, %11) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %18 = emitc.call @XAie_TileLoc(%9, %10) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %19 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %20 = emitc.call @XAie_StrmConnCctEnable(%19, %18, %9, %11, %11, %11) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %21 = emitc.call @XAie_TileLoc(%8, %10) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %22 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %23 = emitc.call @XAie_StrmConnCctEnable(%22, %21, %9, %11, %11, %11) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %24 = emitc.call @XAie_TileLoc(%7, %10) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %25 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %26 = emitc.call @XAie_StrmConnCctEnable(%25, %24, %9, %11, %11, %11) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %27 = emitc.call @XAie_TileLoc(%6, %10) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %28 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %29 = emitc.call @XAie_StrmConnCctEnable(%28, %27, %9, %11, %11, %11) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %30 = emitc.call @XAie_TileLoc(%5, %10) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %31 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %32 = emitc.call @XAie_StrmConnCctEnable(%31, %30, %9, %11, %11, %11) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %33 = emitc.call @XAie_TileLoc(%4, %10) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %34 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %35 = emitc.call @XAie_StrmConnCctEnable(%34, %33, %9, %11, %11, %11) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %36 = emitc.call @XAie_TileLoc(%3, %10) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %37 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %38 = emitc.call @XAie_StrmConnCctEnable(%37, %36, %9, %11, %11, %11) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %39 = emitc.call @XAie_TileLoc(%2, %10) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %40 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %41 = emitc.call @XAie_StrmConnCctEnable(%40, %39, %9, %11, %11, %11) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %42 = emitc.call @XAie_TileLoc(%1, %10) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %43 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %44 = emitc.call @XAie_StrmConnCctEnable(%43, %42, %9, %11, %11, %11) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %45 = emitc.call @XAie_TileLoc(%11, %10) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %46 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %47 = emitc.call @XAie_EnableShimDmaToAieStrmPort(%46, %45, %9) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
    %48 = emitc.call @XAie_TileLoc(%11, %10) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %49 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %50 = emitc.call @XAie_StrmConnCctEnable(%49, %48, %9, %9, %11, %9) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %51 = emitc.call @XAie_TileLoc(%9, %10) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %52 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %53 = emitc.call @XAie_StrmConnCctEnable(%52, %51, %9, %9, %11, %9) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %54 = emitc.call @XAie_TileLoc(%8, %10) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %55 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %56 = emitc.call @XAie_StrmConnCctEnable(%55, %54, %9, %9, %11, %9) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %57 = emitc.call @XAie_TileLoc(%7, %10) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %58 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %59 = emitc.call @XAie_StrmConnCctEnable(%58, %57, %9, %9, %8, %11) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %60 = emitc.call @XAie_TileLoc(%7, %0) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %61 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %62 = emitc.call @XAie_StrmConnCctEnable(%61, %60, %7, %11, %11, %11) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %63 = emitc.call @XAie_TileLoc(%6, %0) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %64 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %65 = emitc.call @XAie_StrmConnCctEnable(%64, %63, %9, %11, %11, %11) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %66 = emitc.call @XAie_TileLoc(%5, %0) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %67 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %68 = emitc.call @XAie_StrmConnCctEnable(%67, %66, %9, %11, %11, %11) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %69 = emitc.call @XAie_TileLoc(%4, %0) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %70 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %71 = emitc.call @XAie_StrmConnCctEnable(%70, %69, %9, %11, %11, %11) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %72 = emitc.call @XAie_TileLoc(%3, %0) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %73 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %74 = emitc.call @XAie_StrmConnCctEnable(%73, %72, %9, %11, %11, %11) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %75 = emitc.call @XAie_TileLoc(%2, %0) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %76 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %77 = emitc.call @XAie_StrmConnCctEnable(%76, %75, %9, %11, %11, %11) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %78 = emitc.call @XAie_TileLoc(%1, %0) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %79 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %80 = emitc.call @XAie_StrmConnCctEnable(%79, %78, %9, %11, %11, %11) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    return
  }
  emitc.include <"xaiengine.h">
  emitc.func private @XAie_TileLoc(i32, i32) -> !emitc.opaque<"XAie_LocType">
  emitc.func private @getOrCreateDeviceInstance() -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
  emitc.func private @XAie_EnableShimDmaToAieStrmPort(!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
  emitc.func private @XAie_StrmConnCctEnable(!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
  func.func @main() {
    call @createroute() : () -> ()
    return
  }
}