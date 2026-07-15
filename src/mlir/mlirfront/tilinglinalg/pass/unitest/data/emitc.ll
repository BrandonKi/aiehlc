/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/
module attributes {codegen.headers = ["stdint.h", "stdio.h", "custom_lib.h"]} {
  emitc.include <"xaiengine.h">
  emitc.func private @XAie_TileLoc(i32, i32) -> !emitc.opaque<"XAie_LocType">
  emitc.func private @getOrCreateDeviceInstance() -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
  emitc.func private @XAie_EnableShimDmaToAieStrmPort(!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
  emitc.func private @XAie_StrmConnCctEnable(!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
  func.func @createroute(%arg0: index, %arg1: index) {
    %0 = "emitc.constant"() <{value = 34 : i32}> : () -> i32
    %1 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %2 = emitc.call @XAie_TileLoc(%1, %0) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %3 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %4 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %5 = emitc.call @XAie_EnableShimDmaToAieStrmPort(%3, %2, %4) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
    %6 = "emitc.constant"() <{value = 34 : i32}> : () -> i32
    %7 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %8 = emitc.call @XAie_TileLoc(%7, %6) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %9 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %10 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %11 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %12 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %13 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %14 = emitc.call @XAie_StrmConnCctEnable(%9, %8, %10, %11, %12, %13) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %15 = "emitc.constant"() <{value = 34 : i32}> : () -> i32
    %16 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %17 = emitc.call @XAie_TileLoc(%16, %15) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %18 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %19 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %20 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %21 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %22 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %23 = emitc.call @XAie_StrmConnCctEnable(%18, %17, %19, %20, %21, %22) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %24 = "emitc.constant"() <{value = 34 : i32}> : () -> i32
    %25 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %26 = emitc.call @XAie_TileLoc(%25, %24) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %27 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %28 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %29 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %30 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %31 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %32 = emitc.call @XAie_StrmConnCctEnable(%27, %26, %28, %29, %30, %31) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %33 = "emitc.constant"() <{value = 34 : i32}> : () -> i32
    %34 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %35 = emitc.call @XAie_TileLoc(%34, %33) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %36 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %37 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %38 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %39 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %40 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %41 = emitc.call @XAie_StrmConnCctEnable(%36, %35, %37, %38, %39, %40) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %42 = "emitc.constant"() <{value = 34 : i32}> : () -> i32
    %43 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %44 = emitc.call @XAie_TileLoc(%43, %42) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %45 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %46 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %47 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %48 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %49 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %50 = emitc.call @XAie_StrmConnCctEnable(%45, %44, %46, %47, %48, %49) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %51 = "emitc.constant"() <{value = 34 : i32}> : () -> i32
    %52 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %53 = emitc.call @XAie_TileLoc(%52, %51) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %54 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %55 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %56 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %57 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %58 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %59 = emitc.call @XAie_StrmConnCctEnable(%54, %53, %55, %56, %57, %58) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %60 = "emitc.constant"() <{value = 34 : i32}> : () -> i32
    %61 = "emitc.constant"() <{value = 6 : i32}> : () -> i32
    %62 = emitc.call @XAie_TileLoc(%61, %60) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %63 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %64 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %65 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %66 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %67 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %68 = emitc.call @XAie_StrmConnCctEnable(%63, %62, %64, %65, %66, %67) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %69 = "emitc.constant"() <{value = 34 : i32}> : () -> i32
    %70 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
    %71 = emitc.call @XAie_TileLoc(%70, %69) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %72 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %73 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %74 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %75 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %76 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %77 = emitc.call @XAie_StrmConnCctEnable(%72, %71, %73, %74, %75, %76) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %78 = "emitc.constant"() <{value = 34 : i32}> : () -> i32
    %79 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %80 = emitc.call @XAie_TileLoc(%79, %78) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %81 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %82 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %83 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %84 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %85 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %86 = emitc.call @XAie_StrmConnCctEnable(%81, %80, %82, %83, %84, %85) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %87 = "emitc.constant"() <{value = 34 : i32}> : () -> i32
    %88 = "emitc.constant"() <{value = 9 : i32}> : () -> i32
    %89 = emitc.call @XAie_TileLoc(%88, %87) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %90 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %91 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %92 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %93 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %94 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %95 = emitc.call @XAie_StrmConnCctEnable(%90, %89, %91, %92, %93, %94) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %96 = "emitc.constant"() <{value = 34 : i32}> : () -> i32
    %97 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %98 = emitc.call @XAie_TileLoc(%97, %96) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %99 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %100 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %101 = emitc.call @XAie_EnableShimDmaToAieStrmPort(%99, %98, %100) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
    %102 = "emitc.constant"() <{value = 34 : i32}> : () -> i32
    %103 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %104 = emitc.call @XAie_TileLoc(%103, %102) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %105 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %106 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %107 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %108 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %109 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %110 = emitc.call @XAie_StrmConnCctEnable(%105, %104, %106, %107, %108, %109) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %111 = "emitc.constant"() <{value = 34 : i32}> : () -> i32
    %112 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %113 = emitc.call @XAie_TileLoc(%112, %111) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %114 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %115 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %116 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %117 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %118 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %119 = emitc.call @XAie_StrmConnCctEnable(%114, %113, %115, %116, %117, %118) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %120 = "emitc.constant"() <{value = 34 : i32}> : () -> i32
    %121 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %122 = emitc.call @XAie_TileLoc(%121, %120) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %123 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %124 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %125 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %126 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %127 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %128 = emitc.call @XAie_StrmConnCctEnable(%123, %122, %124, %125, %126, %127) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %129 = "emitc.constant"() <{value = 34 : i32}> : () -> i32
    %130 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %131 = emitc.call @XAie_TileLoc(%130, %129) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %132 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %133 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %134 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %135 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %136 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %137 = emitc.call @XAie_StrmConnCctEnable(%132, %131, %133, %134, %135, %136) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %138 = "emitc.constant"() <{value = 35 : i32}> : () -> i32
    %139 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %140 = emitc.call @XAie_TileLoc(%139, %138) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %141 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %142 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %143 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %144 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %145 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %146 = emitc.call @XAie_StrmConnCctEnable(%141, %140, %142, %143, %144, %145) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %147 = "emitc.constant"() <{value = 35 : i32}> : () -> i32
    %148 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %149 = emitc.call @XAie_TileLoc(%148, %147) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %150 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %151 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %152 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %153 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %154 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %155 = emitc.call @XAie_StrmConnCctEnable(%150, %149, %151, %152, %153, %154) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %156 = "emitc.constant"() <{value = 35 : i32}> : () -> i32
    %157 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %158 = emitc.call @XAie_TileLoc(%157, %156) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %159 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %160 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %161 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %162 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %163 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %164 = emitc.call @XAie_StrmConnCctEnable(%159, %158, %160, %161, %162, %163) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %165 = "emitc.constant"() <{value = 35 : i32}> : () -> i32
    %166 = "emitc.constant"() <{value = 6 : i32}> : () -> i32
    %167 = emitc.call @XAie_TileLoc(%166, %165) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %168 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %169 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %170 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %171 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %172 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %173 = emitc.call @XAie_StrmConnCctEnable(%168, %167, %169, %170, %171, %172) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %174 = "emitc.constant"() <{value = 35 : i32}> : () -> i32
    %175 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
    %176 = emitc.call @XAie_TileLoc(%175, %174) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %177 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %178 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %179 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %180 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %181 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %182 = emitc.call @XAie_StrmConnCctEnable(%177, %176, %178, %179, %180, %181) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %183 = "emitc.constant"() <{value = 35 : i32}> : () -> i32
    %184 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %185 = emitc.call @XAie_TileLoc(%184, %183) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %186 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %187 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %188 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %189 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %190 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %191 = emitc.call @XAie_StrmConnCctEnable(%186, %185, %187, %188, %189, %190) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %192 = "emitc.constant"() <{value = 35 : i32}> : () -> i32
    %193 = "emitc.constant"() <{value = 9 : i32}> : () -> i32
    %194 = emitc.call @XAie_TileLoc(%193, %192) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %195 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %196 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %197 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %198 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %199 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %200 = emitc.call @XAie_StrmConnCctEnable(%195, %194, %196, %197, %198, %199) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %201 = "emitc.constant"() <{value = 34 : i32}> : () -> i32
    %202 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %203 = emitc.call @XAie_TileLoc(%202, %201) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %204 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %205 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %206 = emitc.call @XAie_EnableShimDmaToAieStrmPort(%204, %203, %205) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
    %207 = "emitc.constant"() <{value = 34 : i32}> : () -> i32
    %208 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %209 = emitc.call @XAie_TileLoc(%208, %207) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %210 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %211 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %212 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %213 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %214 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %215 = emitc.call @XAie_StrmConnCctEnable(%210, %209, %211, %212, %213, %214) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %216 = "emitc.constant"() <{value = 34 : i32}> : () -> i32
    %217 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %218 = emitc.call @XAie_TileLoc(%217, %216) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %219 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %220 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %221 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %222 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %223 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %224 = emitc.call @XAie_StrmConnCctEnable(%219, %218, %220, %221, %222, %223) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %225 = "emitc.constant"() <{value = 34 : i32}> : () -> i32
    %226 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %227 = emitc.call @XAie_TileLoc(%226, %225) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %228 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %229 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %230 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %231 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %232 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %233 = emitc.call @XAie_StrmConnCctEnable(%228, %227, %229, %230, %231, %232) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %234 = "emitc.constant"() <{value = 34 : i32}> : () -> i32
    %235 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %236 = emitc.call @XAie_TileLoc(%235, %234) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %237 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %238 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %239 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %240 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %241 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %242 = emitc.call @XAie_StrmConnCctEnable(%237, %236, %238, %239, %240, %241) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %243 = "emitc.constant"() <{value = 33 : i32}> : () -> i32
    %244 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %245 = emitc.call @XAie_TileLoc(%244, %243) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %246 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %247 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %248 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %249 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %250 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %251 = emitc.call @XAie_StrmConnCctEnable(%246, %245, %247, %248, %249, %250) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %252 = "emitc.constant"() <{value = 33 : i32}> : () -> i32
    %253 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %254 = emitc.call @XAie_TileLoc(%253, %252) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %255 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %256 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %257 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %258 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %259 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %260 = emitc.call @XAie_StrmConnCctEnable(%255, %254, %256, %257, %258, %259) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %261 = "emitc.constant"() <{value = 33 : i32}> : () -> i32
    %262 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %263 = emitc.call @XAie_TileLoc(%262, %261) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %264 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %265 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %266 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %267 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %268 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %269 = emitc.call @XAie_StrmConnCctEnable(%264, %263, %265, %266, %267, %268) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %270 = "emitc.constant"() <{value = 33 : i32}> : () -> i32
    %271 = "emitc.constant"() <{value = 6 : i32}> : () -> i32
    %272 = emitc.call @XAie_TileLoc(%271, %270) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %273 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %274 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %275 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %276 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %277 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %278 = emitc.call @XAie_StrmConnCctEnable(%273, %272, %274, %275, %276, %277) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %279 = "emitc.constant"() <{value = 33 : i32}> : () -> i32
    %280 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
    %281 = emitc.call @XAie_TileLoc(%280, %279) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %282 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %283 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %284 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %285 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %286 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %287 = emitc.call @XAie_StrmConnCctEnable(%282, %281, %283, %284, %285, %286) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %288 = "emitc.constant"() <{value = 33 : i32}> : () -> i32
    %289 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %290 = emitc.call @XAie_TileLoc(%289, %288) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %291 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %292 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %293 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %294 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %295 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %296 = emitc.call @XAie_StrmConnCctEnable(%291, %290, %292, %293, %294, %295) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %297 = "emitc.constant"() <{value = 33 : i32}> : () -> i32
    %298 = "emitc.constant"() <{value = 9 : i32}> : () -> i32
    %299 = emitc.call @XAie_TileLoc(%298, %297) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %300 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %301 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %302 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %303 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %304 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %305 = emitc.call @XAie_StrmConnCctEnable(%300, %299, %301, %302, %303, %304) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %306 = "emitc.constant"() <{value = 34 : i32}> : () -> i32
    %307 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %308 = emitc.call @XAie_TileLoc(%307, %306) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %309 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %310 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %311 = emitc.call @XAie_EnableShimDmaToAieStrmPort(%309, %308, %310) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
    %312 = "emitc.constant"() <{value = 34 : i32}> : () -> i32
    %313 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %314 = emitc.call @XAie_TileLoc(%313, %312) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %315 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %316 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %317 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %318 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %319 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %320 = emitc.call @XAie_StrmConnCctEnable(%315, %314, %316, %317, %318, %319) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %321 = "emitc.constant"() <{value = 34 : i32}> : () -> i32
    %322 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %323 = emitc.call @XAie_TileLoc(%322, %321) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %324 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %325 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %326 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %327 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %328 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %329 = emitc.call @XAie_StrmConnCctEnable(%324, %323, %325, %326, %327, %328) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %330 = "emitc.constant"() <{value = 34 : i32}> : () -> i32
    %331 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %332 = emitc.call @XAie_TileLoc(%331, %330) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %333 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %334 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %335 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %336 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %337 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %338 = emitc.call @XAie_StrmConnCctEnable(%333, %332, %334, %335, %336, %337) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %339 = "emitc.constant"() <{value = 34 : i32}> : () -> i32
    %340 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %341 = emitc.call @XAie_TileLoc(%340, %339) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %342 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %343 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %344 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %345 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %346 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %347 = emitc.call @XAie_StrmConnCctEnable(%342, %341, %343, %344, %345, %346) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %348 = "emitc.constant"() <{value = 35 : i32}> : () -> i32
    %349 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %350 = emitc.call @XAie_TileLoc(%349, %348) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %351 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %352 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %353 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %354 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %355 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %356 = emitc.call @XAie_StrmConnCctEnable(%351, %350, %352, %353, %354, %355) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %357 = "emitc.constant"() <{value = 36 : i32}> : () -> i32
    %358 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %359 = emitc.call @XAie_TileLoc(%358, %357) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %360 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %361 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %362 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %363 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %364 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %365 = emitc.call @XAie_StrmConnCctEnable(%360, %359, %361, %362, %363, %364) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %366 = "emitc.constant"() <{value = 36 : i32}> : () -> i32
    %367 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %368 = emitc.call @XAie_TileLoc(%367, %366) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %369 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %370 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %371 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %372 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %373 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %374 = emitc.call @XAie_StrmConnCctEnable(%369, %368, %370, %371, %372, %373) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %375 = "emitc.constant"() <{value = 36 : i32}> : () -> i32
    %376 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %377 = emitc.call @XAie_TileLoc(%376, %375) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %378 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %379 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %380 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %381 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %382 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %383 = emitc.call @XAie_StrmConnCctEnable(%378, %377, %379, %380, %381, %382) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %384 = "emitc.constant"() <{value = 36 : i32}> : () -> i32
    %385 = "emitc.constant"() <{value = 6 : i32}> : () -> i32
    %386 = emitc.call @XAie_TileLoc(%385, %384) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %387 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %388 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %389 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %390 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %391 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %392 = emitc.call @XAie_StrmConnCctEnable(%387, %386, %388, %389, %390, %391) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %393 = "emitc.constant"() <{value = 36 : i32}> : () -> i32
    %394 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
    %395 = emitc.call @XAie_TileLoc(%394, %393) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %396 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %397 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %398 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %399 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %400 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %401 = emitc.call @XAie_StrmConnCctEnable(%396, %395, %397, %398, %399, %400) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %402 = "emitc.constant"() <{value = 36 : i32}> : () -> i32
    %403 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %404 = emitc.call @XAie_TileLoc(%403, %402) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %405 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %406 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %407 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %408 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %409 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %410 = emitc.call @XAie_StrmConnCctEnable(%405, %404, %406, %407, %408, %409) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %411 = "emitc.constant"() <{value = 36 : i32}> : () -> i32
    %412 = "emitc.constant"() <{value = 9 : i32}> : () -> i32
    %413 = emitc.call @XAie_TileLoc(%412, %411) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %414 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %415 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %416 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %417 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %418 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %419 = emitc.call @XAie_StrmConnCctEnable(%414, %413, %415, %416, %417, %418) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %420 = "emitc.constant"() <{value = 31 : i32}> : () -> i32
    %421 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %422 = emitc.call @XAie_TileLoc(%421, %420) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %423 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %424 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %425 = emitc.call @XAie_EnableShimDmaToAieStrmPort(%423, %422, %424) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
    %426 = "emitc.constant"() <{value = 31 : i32}> : () -> i32
    %427 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %428 = emitc.call @XAie_TileLoc(%427, %426) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %429 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %430 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %431 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %432 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %433 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %434 = emitc.call @XAie_StrmConnCctEnable(%429, %428, %430, %431, %432, %433) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %435 = "emitc.constant"() <{value = 31 : i32}> : () -> i32
    %436 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %437 = emitc.call @XAie_TileLoc(%436, %435) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %438 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %439 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %440 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %441 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %442 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %443 = emitc.call @XAie_StrmConnCctEnable(%438, %437, %439, %440, %441, %442) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %444 = "emitc.constant"() <{value = 31 : i32}> : () -> i32
    %445 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %446 = emitc.call @XAie_TileLoc(%445, %444) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %447 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %448 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %449 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %450 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %451 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %452 = emitc.call @XAie_StrmConnCctEnable(%447, %446, %448, %449, %450, %451) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %453 = "emitc.constant"() <{value = 31 : i32}> : () -> i32
    %454 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %455 = emitc.call @XAie_TileLoc(%454, %453) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %456 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %457 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %458 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %459 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %460 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %461 = emitc.call @XAie_StrmConnCctEnable(%456, %455, %457, %458, %459, %460) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %462 = "emitc.constant"() <{value = 31 : i32}> : () -> i32
    %463 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %464 = emitc.call @XAie_TileLoc(%463, %462) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %465 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %466 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %467 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %468 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %469 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %470 = emitc.call @XAie_StrmConnCctEnable(%465, %464, %466, %467, %468, %469) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %471 = "emitc.constant"() <{value = 31 : i32}> : () -> i32
    %472 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %473 = emitc.call @XAie_TileLoc(%472, %471) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %474 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %475 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %476 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %477 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %478 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %479 = emitc.call @XAie_StrmConnCctEnable(%474, %473, %475, %476, %477, %478) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %480 = "emitc.constant"() <{value = 31 : i32}> : () -> i32
    %481 = "emitc.constant"() <{value = 6 : i32}> : () -> i32
    %482 = emitc.call @XAie_TileLoc(%481, %480) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %483 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %484 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %485 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %486 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %487 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %488 = emitc.call @XAie_StrmConnCctEnable(%483, %482, %484, %485, %486, %487) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %489 = "emitc.constant"() <{value = 31 : i32}> : () -> i32
    %490 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
    %491 = emitc.call @XAie_TileLoc(%490, %489) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %492 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %493 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %494 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %495 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %496 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %497 = emitc.call @XAie_StrmConnCctEnable(%492, %491, %493, %494, %495, %496) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %498 = "emitc.constant"() <{value = 31 : i32}> : () -> i32
    %499 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %500 = emitc.call @XAie_TileLoc(%499, %498) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %501 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %502 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %503 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %504 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %505 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %506 = emitc.call @XAie_StrmConnCctEnable(%501, %500, %502, %503, %504, %505) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %507 = "emitc.constant"() <{value = 31 : i32}> : () -> i32
    %508 = "emitc.constant"() <{value = 9 : i32}> : () -> i32
    %509 = emitc.call @XAie_TileLoc(%508, %507) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %510 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %511 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %512 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %513 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %514 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %515 = emitc.call @XAie_StrmConnCctEnable(%510, %509, %511, %512, %513, %514) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %516 = "emitc.constant"() <{value = 31 : i32}> : () -> i32
    %517 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %518 = emitc.call @XAie_TileLoc(%517, %516) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %519 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %520 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %521 = emitc.call @XAie_EnableShimDmaToAieStrmPort(%519, %518, %520) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
    %522 = "emitc.constant"() <{value = 31 : i32}> : () -> i32
    %523 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %524 = emitc.call @XAie_TileLoc(%523, %522) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %525 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %526 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %527 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %528 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %529 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %530 = emitc.call @XAie_StrmConnCctEnable(%525, %524, %526, %527, %528, %529) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %531 = "emitc.constant"() <{value = 31 : i32}> : () -> i32
    %532 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %533 = emitc.call @XAie_TileLoc(%532, %531) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %534 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %535 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %536 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %537 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %538 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %539 = emitc.call @XAie_StrmConnCctEnable(%534, %533, %535, %536, %537, %538) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %540 = "emitc.constant"() <{value = 31 : i32}> : () -> i32
    %541 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %542 = emitc.call @XAie_TileLoc(%541, %540) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %543 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %544 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %545 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %546 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %547 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %548 = emitc.call @XAie_StrmConnCctEnable(%543, %542, %544, %545, %546, %547) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %549 = "emitc.constant"() <{value = 31 : i32}> : () -> i32
    %550 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %551 = emitc.call @XAie_TileLoc(%550, %549) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %552 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %553 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %554 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %555 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %556 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %557 = emitc.call @XAie_StrmConnCctEnable(%552, %551, %553, %554, %555, %556) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %558 = "emitc.constant"() <{value = 32 : i32}> : () -> i32
    %559 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %560 = emitc.call @XAie_TileLoc(%559, %558) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %561 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %562 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %563 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %564 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %565 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %566 = emitc.call @XAie_StrmConnCctEnable(%561, %560, %562, %563, %564, %565) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %567 = "emitc.constant"() <{value = 32 : i32}> : () -> i32
    %568 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %569 = emitc.call @XAie_TileLoc(%568, %567) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %570 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %571 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %572 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %573 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %574 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %575 = emitc.call @XAie_StrmConnCctEnable(%570, %569, %571, %572, %573, %574) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %576 = "emitc.constant"() <{value = 32 : i32}> : () -> i32
    %577 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %578 = emitc.call @XAie_TileLoc(%577, %576) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %579 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %580 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %581 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %582 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %583 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %584 = emitc.call @XAie_StrmConnCctEnable(%579, %578, %580, %581, %582, %583) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %585 = "emitc.constant"() <{value = 32 : i32}> : () -> i32
    %586 = "emitc.constant"() <{value = 6 : i32}> : () -> i32
    %587 = emitc.call @XAie_TileLoc(%586, %585) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %588 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %589 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %590 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %591 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %592 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %593 = emitc.call @XAie_StrmConnCctEnable(%588, %587, %589, %590, %591, %592) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %594 = "emitc.constant"() <{value = 32 : i32}> : () -> i32
    %595 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
    %596 = emitc.call @XAie_TileLoc(%595, %594) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %597 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %598 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %599 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %600 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %601 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %602 = emitc.call @XAie_StrmConnCctEnable(%597, %596, %598, %599, %600, %601) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %603 = "emitc.constant"() <{value = 32 : i32}> : () -> i32
    %604 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %605 = emitc.call @XAie_TileLoc(%604, %603) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %606 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %607 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %608 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %609 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %610 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %611 = emitc.call @XAie_StrmConnCctEnable(%606, %605, %607, %608, %609, %610) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %612 = "emitc.constant"() <{value = 32 : i32}> : () -> i32
    %613 = "emitc.constant"() <{value = 9 : i32}> : () -> i32
    %614 = emitc.call @XAie_TileLoc(%613, %612) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %615 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %616 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %617 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %618 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %619 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %620 = emitc.call @XAie_StrmConnCctEnable(%615, %614, %616, %617, %618, %619) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %621 = "emitc.constant"() <{value = 31 : i32}> : () -> i32
    %622 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %623 = emitc.call @XAie_TileLoc(%622, %621) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %624 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %625 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %626 = emitc.call @XAie_EnableShimDmaToAieStrmPort(%624, %623, %625) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
    %627 = "emitc.constant"() <{value = 31 : i32}> : () -> i32
    %628 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %629 = emitc.call @XAie_TileLoc(%628, %627) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %630 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %631 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %632 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %633 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %634 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %635 = emitc.call @XAie_StrmConnCctEnable(%630, %629, %631, %632, %633, %634) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %636 = "emitc.constant"() <{value = 31 : i32}> : () -> i32
    %637 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %638 = emitc.call @XAie_TileLoc(%637, %636) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %639 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %640 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %641 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %642 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %643 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %644 = emitc.call @XAie_StrmConnCctEnable(%639, %638, %640, %641, %642, %643) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %645 = "emitc.constant"() <{value = 31 : i32}> : () -> i32
    %646 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %647 = emitc.call @XAie_TileLoc(%646, %645) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %648 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %649 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %650 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %651 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %652 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %653 = emitc.call @XAie_StrmConnCctEnable(%648, %647, %649, %650, %651, %652) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %654 = "emitc.constant"() <{value = 31 : i32}> : () -> i32
    %655 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %656 = emitc.call @XAie_TileLoc(%655, %654) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %657 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %658 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %659 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %660 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %661 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %662 = emitc.call @XAie_StrmConnCctEnable(%657, %656, %658, %659, %660, %661) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %663 = "emitc.constant"() <{value = 30 : i32}> : () -> i32
    %664 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %665 = emitc.call @XAie_TileLoc(%664, %663) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %666 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %667 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %668 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %669 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %670 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %671 = emitc.call @XAie_StrmConnCctEnable(%666, %665, %667, %668, %669, %670) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %672 = "emitc.constant"() <{value = 30 : i32}> : () -> i32
    %673 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %674 = emitc.call @XAie_TileLoc(%673, %672) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %675 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %676 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %677 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %678 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %679 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %680 = emitc.call @XAie_StrmConnCctEnable(%675, %674, %676, %677, %678, %679) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %681 = "emitc.constant"() <{value = 30 : i32}> : () -> i32
    %682 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %683 = emitc.call @XAie_TileLoc(%682, %681) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %684 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %685 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %686 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %687 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %688 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %689 = emitc.call @XAie_StrmConnCctEnable(%684, %683, %685, %686, %687, %688) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %690 = "emitc.constant"() <{value = 30 : i32}> : () -> i32
    %691 = "emitc.constant"() <{value = 6 : i32}> : () -> i32
    %692 = emitc.call @XAie_TileLoc(%691, %690) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %693 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %694 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %695 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %696 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %697 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %698 = emitc.call @XAie_StrmConnCctEnable(%693, %692, %694, %695, %696, %697) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %699 = "emitc.constant"() <{value = 30 : i32}> : () -> i32
    %700 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
    %701 = emitc.call @XAie_TileLoc(%700, %699) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %702 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %703 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %704 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %705 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %706 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %707 = emitc.call @XAie_StrmConnCctEnable(%702, %701, %703, %704, %705, %706) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %708 = "emitc.constant"() <{value = 30 : i32}> : () -> i32
    %709 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %710 = emitc.call @XAie_TileLoc(%709, %708) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %711 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %712 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %713 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %714 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %715 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %716 = emitc.call @XAie_StrmConnCctEnable(%711, %710, %712, %713, %714, %715) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %717 = "emitc.constant"() <{value = 30 : i32}> : () -> i32
    %718 = "emitc.constant"() <{value = 9 : i32}> : () -> i32
    %719 = emitc.call @XAie_TileLoc(%718, %717) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %720 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %721 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %722 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %723 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %724 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %725 = emitc.call @XAie_StrmConnCctEnable(%720, %719, %721, %722, %723, %724) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %726 = "emitc.constant"() <{value = 31 : i32}> : () -> i32
    %727 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %728 = emitc.call @XAie_TileLoc(%727, %726) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %729 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %730 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %731 = emitc.call @XAie_EnableShimDmaToAieStrmPort(%729, %728, %730) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
    %732 = "emitc.constant"() <{value = 31 : i32}> : () -> i32
    %733 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %734 = emitc.call @XAie_TileLoc(%733, %732) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %735 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %736 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %737 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %738 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %739 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %740 = emitc.call @XAie_StrmConnCctEnable(%735, %734, %736, %737, %738, %739) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %741 = "emitc.constant"() <{value = 31 : i32}> : () -> i32
    %742 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %743 = emitc.call @XAie_TileLoc(%742, %741) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %744 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %745 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %746 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %747 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %748 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %749 = emitc.call @XAie_StrmConnCctEnable(%744, %743, %745, %746, %747, %748) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %750 = "emitc.constant"() <{value = 31 : i32}> : () -> i32
    %751 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %752 = emitc.call @XAie_TileLoc(%751, %750) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %753 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %754 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %755 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %756 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %757 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %758 = emitc.call @XAie_StrmConnCctEnable(%753, %752, %754, %755, %756, %757) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %759 = "emitc.constant"() <{value = 31 : i32}> : () -> i32
    %760 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %761 = emitc.call @XAie_TileLoc(%760, %759) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %762 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %763 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %764 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %765 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %766 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %767 = emitc.call @XAie_StrmConnCctEnable(%762, %761, %763, %764, %765, %766) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %768 = "emitc.constant"() <{value = 30 : i32}> : () -> i32
    %769 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %770 = emitc.call @XAie_TileLoc(%769, %768) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %771 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %772 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %773 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %774 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %775 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %776 = emitc.call @XAie_StrmConnCctEnable(%771, %770, %772, %773, %774, %775) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %777 = "emitc.constant"() <{value = 29 : i32}> : () -> i32
    %778 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %779 = emitc.call @XAie_TileLoc(%778, %777) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %780 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %781 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %782 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %783 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %784 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %785 = emitc.call @XAie_StrmConnCctEnable(%780, %779, %781, %782, %783, %784) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %786 = "emitc.constant"() <{value = 29 : i32}> : () -> i32
    %787 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %788 = emitc.call @XAie_TileLoc(%787, %786) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %789 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %790 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %791 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %792 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %793 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %794 = emitc.call @XAie_StrmConnCctEnable(%789, %788, %790, %791, %792, %793) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %795 = "emitc.constant"() <{value = 29 : i32}> : () -> i32
    %796 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %797 = emitc.call @XAie_TileLoc(%796, %795) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %798 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %799 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %800 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %801 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %802 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %803 = emitc.call @XAie_StrmConnCctEnable(%798, %797, %799, %800, %801, %802) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %804 = "emitc.constant"() <{value = 29 : i32}> : () -> i32
    %805 = "emitc.constant"() <{value = 6 : i32}> : () -> i32
    %806 = emitc.call @XAie_TileLoc(%805, %804) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %807 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %808 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %809 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %810 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %811 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %812 = emitc.call @XAie_StrmConnCctEnable(%807, %806, %808, %809, %810, %811) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %813 = "emitc.constant"() <{value = 29 : i32}> : () -> i32
    %814 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
    %815 = emitc.call @XAie_TileLoc(%814, %813) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %816 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %817 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %818 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %819 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %820 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %821 = emitc.call @XAie_StrmConnCctEnable(%816, %815, %817, %818, %819, %820) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %822 = "emitc.constant"() <{value = 29 : i32}> : () -> i32
    %823 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %824 = emitc.call @XAie_TileLoc(%823, %822) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %825 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %826 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %827 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %828 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %829 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %830 = emitc.call @XAie_StrmConnCctEnable(%825, %824, %826, %827, %828, %829) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %831 = "emitc.constant"() <{value = 29 : i32}> : () -> i32
    %832 = "emitc.constant"() <{value = 9 : i32}> : () -> i32
    %833 = emitc.call @XAie_TileLoc(%832, %831) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %834 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %835 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %836 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %837 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %838 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %839 = emitc.call @XAie_StrmConnCctEnable(%834, %833, %835, %836, %837, %838) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %840 = "emitc.constant"() <{value = 30 : i32}> : () -> i32
    %841 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %842 = emitc.call @XAie_TileLoc(%841, %840) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %843 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %844 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %845 = emitc.call @XAie_EnableShimDmaToAieStrmPort(%843, %842, %844) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
    %846 = "emitc.constant"() <{value = 30 : i32}> : () -> i32
    %847 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %848 = emitc.call @XAie_TileLoc(%847, %846) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %849 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %850 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %851 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %852 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %853 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %854 = emitc.call @XAie_StrmConnCctEnable(%849, %848, %850, %851, %852, %853) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %855 = "emitc.constant"() <{value = 30 : i32}> : () -> i32
    %856 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %857 = emitc.call @XAie_TileLoc(%856, %855) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %858 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %859 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %860 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %861 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %862 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %863 = emitc.call @XAie_StrmConnCctEnable(%858, %857, %859, %860, %861, %862) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %864 = "emitc.constant"() <{value = 30 : i32}> : () -> i32
    %865 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %866 = emitc.call @XAie_TileLoc(%865, %864) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %867 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %868 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %869 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %870 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %871 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %872 = emitc.call @XAie_StrmConnCctEnable(%867, %866, %868, %869, %870, %871) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %873 = "emitc.constant"() <{value = 30 : i32}> : () -> i32
    %874 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %875 = emitc.call @XAie_TileLoc(%874, %873) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %876 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %877 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %878 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %879 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %880 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %881 = emitc.call @XAie_StrmConnCctEnable(%876, %875, %877, %878, %879, %880) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %882 = "emitc.constant"() <{value = 29 : i32}> : () -> i32
    %883 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %884 = emitc.call @XAie_TileLoc(%883, %882) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %885 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %886 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %887 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %888 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %889 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %890 = emitc.call @XAie_StrmConnCctEnable(%885, %884, %886, %887, %888, %889) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %891 = "emitc.constant"() <{value = 28 : i32}> : () -> i32
    %892 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %893 = emitc.call @XAie_TileLoc(%892, %891) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %894 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %895 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %896 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %897 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %898 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %899 = emitc.call @XAie_StrmConnCctEnable(%894, %893, %895, %896, %897, %898) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %900 = "emitc.constant"() <{value = 28 : i32}> : () -> i32
    %901 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %902 = emitc.call @XAie_TileLoc(%901, %900) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %903 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %904 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %905 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %906 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %907 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %908 = emitc.call @XAie_StrmConnCctEnable(%903, %902, %904, %905, %906, %907) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %909 = "emitc.constant"() <{value = 28 : i32}> : () -> i32
    %910 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %911 = emitc.call @XAie_TileLoc(%910, %909) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %912 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %913 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %914 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %915 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %916 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %917 = emitc.call @XAie_StrmConnCctEnable(%912, %911, %913, %914, %915, %916) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %918 = "emitc.constant"() <{value = 28 : i32}> : () -> i32
    %919 = "emitc.constant"() <{value = 6 : i32}> : () -> i32
    %920 = emitc.call @XAie_TileLoc(%919, %918) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %921 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %922 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %923 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %924 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %925 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %926 = emitc.call @XAie_StrmConnCctEnable(%921, %920, %922, %923, %924, %925) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %927 = "emitc.constant"() <{value = 28 : i32}> : () -> i32
    %928 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
    %929 = emitc.call @XAie_TileLoc(%928, %927) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %930 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %931 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %932 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %933 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %934 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %935 = emitc.call @XAie_StrmConnCctEnable(%930, %929, %931, %932, %933, %934) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %936 = "emitc.constant"() <{value = 28 : i32}> : () -> i32
    %937 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %938 = emitc.call @XAie_TileLoc(%937, %936) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %939 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %940 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %941 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %942 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %943 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %944 = emitc.call @XAie_StrmConnCctEnable(%939, %938, %940, %941, %942, %943) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %945 = "emitc.constant"() <{value = 28 : i32}> : () -> i32
    %946 = "emitc.constant"() <{value = 9 : i32}> : () -> i32
    %947 = emitc.call @XAie_TileLoc(%946, %945) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %948 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %949 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %950 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %951 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %952 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %953 = emitc.call @XAie_StrmConnCctEnable(%948, %947, %949, %950, %951, %952) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %954 = "emitc.constant"() <{value = 30 : i32}> : () -> i32
    %955 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %956 = emitc.call @XAie_TileLoc(%955, %954) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %957 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %958 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %959 = emitc.call @XAie_EnableShimDmaToAieStrmPort(%957, %956, %958) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
    %960 = "emitc.constant"() <{value = 30 : i32}> : () -> i32
    %961 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %962 = emitc.call @XAie_TileLoc(%961, %960) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %963 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %964 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %965 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %966 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %967 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %968 = emitc.call @XAie_StrmConnCctEnable(%963, %962, %964, %965, %966, %967) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %969 = "emitc.constant"() <{value = 30 : i32}> : () -> i32
    %970 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %971 = emitc.call @XAie_TileLoc(%970, %969) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %972 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %973 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %974 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %975 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %976 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %977 = emitc.call @XAie_StrmConnCctEnable(%972, %971, %973, %974, %975, %976) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %978 = "emitc.constant"() <{value = 30 : i32}> : () -> i32
    %979 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %980 = emitc.call @XAie_TileLoc(%979, %978) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %981 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %982 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %983 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %984 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %985 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %986 = emitc.call @XAie_StrmConnCctEnable(%981, %980, %982, %983, %984, %985) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %987 = "emitc.constant"() <{value = 30 : i32}> : () -> i32
    %988 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %989 = emitc.call @XAie_TileLoc(%988, %987) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %990 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %991 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %992 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %993 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %994 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %995 = emitc.call @XAie_StrmConnCctEnable(%990, %989, %991, %992, %993, %994) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %996 = "emitc.constant"() <{value = 29 : i32}> : () -> i32
    %997 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %998 = emitc.call @XAie_TileLoc(%997, %996) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %999 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1000 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1001 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1002 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1003 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1004 = emitc.call @XAie_StrmConnCctEnable(%999, %998, %1000, %1001, %1002, %1003) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1005 = "emitc.constant"() <{value = 28 : i32}> : () -> i32
    %1006 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1007 = emitc.call @XAie_TileLoc(%1006, %1005) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1008 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1009 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1010 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1011 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1012 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1013 = emitc.call @XAie_StrmConnCctEnable(%1008, %1007, %1009, %1010, %1011, %1012) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1014 = "emitc.constant"() <{value = 27 : i32}> : () -> i32
    %1015 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1016 = emitc.call @XAie_TileLoc(%1015, %1014) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1017 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1018 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1019 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1020 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1021 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1022 = emitc.call @XAie_StrmConnCctEnable(%1017, %1016, %1018, %1019, %1020, %1021) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1023 = "emitc.constant"() <{value = 27 : i32}> : () -> i32
    %1024 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1025 = emitc.call @XAie_TileLoc(%1024, %1023) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1026 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1027 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1028 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1029 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1030 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1031 = emitc.call @XAie_StrmConnCctEnable(%1026, %1025, %1027, %1028, %1029, %1030) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1032 = "emitc.constant"() <{value = 27 : i32}> : () -> i32
    %1033 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1034 = emitc.call @XAie_TileLoc(%1033, %1032) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1035 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1036 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1037 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1038 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1039 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1040 = emitc.call @XAie_StrmConnCctEnable(%1035, %1034, %1036, %1037, %1038, %1039) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1041 = "emitc.constant"() <{value = 27 : i32}> : () -> i32
    %1042 = "emitc.constant"() <{value = 6 : i32}> : () -> i32
    %1043 = emitc.call @XAie_TileLoc(%1042, %1041) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1044 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1045 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1046 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1047 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1048 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1049 = emitc.call @XAie_StrmConnCctEnable(%1044, %1043, %1045, %1046, %1047, %1048) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1050 = "emitc.constant"() <{value = 27 : i32}> : () -> i32
    %1051 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
    %1052 = emitc.call @XAie_TileLoc(%1051, %1050) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1053 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1054 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1055 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1056 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1057 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1058 = emitc.call @XAie_StrmConnCctEnable(%1053, %1052, %1054, %1055, %1056, %1057) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1059 = "emitc.constant"() <{value = 27 : i32}> : () -> i32
    %1060 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %1061 = emitc.call @XAie_TileLoc(%1060, %1059) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1062 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1063 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1064 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1065 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1066 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1067 = emitc.call @XAie_StrmConnCctEnable(%1062, %1061, %1063, %1064, %1065, %1066) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1068 = "emitc.constant"() <{value = 27 : i32}> : () -> i32
    %1069 = "emitc.constant"() <{value = 9 : i32}> : () -> i32
    %1070 = emitc.call @XAie_TileLoc(%1069, %1068) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1071 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1072 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1073 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1074 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1075 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1076 = emitc.call @XAie_StrmConnCctEnable(%1071, %1070, %1072, %1073, %1074, %1075) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1077 = "emitc.constant"() <{value = 30 : i32}> : () -> i32
    %1078 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1079 = emitc.call @XAie_TileLoc(%1078, %1077) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1080 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1081 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1082 = emitc.call @XAie_EnableShimDmaToAieStrmPort(%1080, %1079, %1081) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
    %1083 = "emitc.constant"() <{value = 30 : i32}> : () -> i32
    %1084 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1085 = emitc.call @XAie_TileLoc(%1084, %1083) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1086 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1087 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1088 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1089 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1090 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1091 = emitc.call @XAie_StrmConnCctEnable(%1086, %1085, %1087, %1088, %1089, %1090) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1092 = "emitc.constant"() <{value = 30 : i32}> : () -> i32
    %1093 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1094 = emitc.call @XAie_TileLoc(%1093, %1092) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1095 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1096 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1097 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1098 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1099 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1100 = emitc.call @XAie_StrmConnCctEnable(%1095, %1094, %1096, %1097, %1098, %1099) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1101 = "emitc.constant"() <{value = 30 : i32}> : () -> i32
    %1102 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1103 = emitc.call @XAie_TileLoc(%1102, %1101) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1104 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1105 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1106 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1107 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1108 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1109 = emitc.call @XAie_StrmConnCctEnable(%1104, %1103, %1105, %1106, %1107, %1108) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1110 = "emitc.constant"() <{value = 30 : i32}> : () -> i32
    %1111 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1112 = emitc.call @XAie_TileLoc(%1111, %1110) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1113 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1114 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1115 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1116 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1117 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1118 = emitc.call @XAie_StrmConnCctEnable(%1113, %1112, %1114, %1115, %1116, %1117) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1119 = "emitc.constant"() <{value = 29 : i32}> : () -> i32
    %1120 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1121 = emitc.call @XAie_TileLoc(%1120, %1119) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1122 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1123 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1124 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1125 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1126 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1127 = emitc.call @XAie_StrmConnCctEnable(%1122, %1121, %1123, %1124, %1125, %1126) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1128 = "emitc.constant"() <{value = 28 : i32}> : () -> i32
    %1129 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1130 = emitc.call @XAie_TileLoc(%1129, %1128) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1131 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1132 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1133 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1134 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1135 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1136 = emitc.call @XAie_StrmConnCctEnable(%1131, %1130, %1132, %1133, %1134, %1135) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1137 = "emitc.constant"() <{value = 27 : i32}> : () -> i32
    %1138 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1139 = emitc.call @XAie_TileLoc(%1138, %1137) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1140 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1141 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1142 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1143 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1144 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1145 = emitc.call @XAie_StrmConnCctEnable(%1140, %1139, %1141, %1142, %1143, %1144) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1146 = "emitc.constant"() <{value = 26 : i32}> : () -> i32
    %1147 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1148 = emitc.call @XAie_TileLoc(%1147, %1146) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1149 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1150 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1151 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1152 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1153 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1154 = emitc.call @XAie_StrmConnCctEnable(%1149, %1148, %1150, %1151, %1152, %1153) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1155 = "emitc.constant"() <{value = 26 : i32}> : () -> i32
    %1156 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1157 = emitc.call @XAie_TileLoc(%1156, %1155) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1158 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1159 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1160 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1161 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1162 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1163 = emitc.call @XAie_StrmConnCctEnable(%1158, %1157, %1159, %1160, %1161, %1162) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1164 = "emitc.constant"() <{value = 26 : i32}> : () -> i32
    %1165 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1166 = emitc.call @XAie_TileLoc(%1165, %1164) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1167 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1168 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1169 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1170 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1171 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1172 = emitc.call @XAie_StrmConnCctEnable(%1167, %1166, %1168, %1169, %1170, %1171) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1173 = "emitc.constant"() <{value = 26 : i32}> : () -> i32
    %1174 = "emitc.constant"() <{value = 6 : i32}> : () -> i32
    %1175 = emitc.call @XAie_TileLoc(%1174, %1173) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1176 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1177 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1178 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1179 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1180 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1181 = emitc.call @XAie_StrmConnCctEnable(%1176, %1175, %1177, %1178, %1179, %1180) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1182 = "emitc.constant"() <{value = 26 : i32}> : () -> i32
    %1183 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
    %1184 = emitc.call @XAie_TileLoc(%1183, %1182) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1185 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1186 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1187 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1188 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1189 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1190 = emitc.call @XAie_StrmConnCctEnable(%1185, %1184, %1186, %1187, %1188, %1189) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1191 = "emitc.constant"() <{value = 26 : i32}> : () -> i32
    %1192 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %1193 = emitc.call @XAie_TileLoc(%1192, %1191) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1194 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1195 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1196 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1197 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1198 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1199 = emitc.call @XAie_StrmConnCctEnable(%1194, %1193, %1195, %1196, %1197, %1198) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1200 = "emitc.constant"() <{value = 26 : i32}> : () -> i32
    %1201 = "emitc.constant"() <{value = 9 : i32}> : () -> i32
    %1202 = emitc.call @XAie_TileLoc(%1201, %1200) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1203 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1204 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1205 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1206 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1207 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1208 = emitc.call @XAie_StrmConnCctEnable(%1203, %1202, %1204, %1205, %1206, %1207) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1209 = "emitc.constant"() <{value = 30 : i32}> : () -> i32
    %1210 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1211 = emitc.call @XAie_TileLoc(%1210, %1209) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1212 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1213 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1214 = emitc.call @XAie_EnableShimDmaToAieStrmPort(%1212, %1211, %1213) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
    %1215 = "emitc.constant"() <{value = 30 : i32}> : () -> i32
    %1216 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1217 = emitc.call @XAie_TileLoc(%1216, %1215) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1218 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1219 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1220 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1221 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1222 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1223 = emitc.call @XAie_StrmConnCctEnable(%1218, %1217, %1219, %1220, %1221, %1222) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1224 = "emitc.constant"() <{value = 30 : i32}> : () -> i32
    %1225 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1226 = emitc.call @XAie_TileLoc(%1225, %1224) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1227 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1228 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1229 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1230 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1231 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1232 = emitc.call @XAie_StrmConnCctEnable(%1227, %1226, %1228, %1229, %1230, %1231) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1233 = "emitc.constant"() <{value = 30 : i32}> : () -> i32
    %1234 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1235 = emitc.call @XAie_TileLoc(%1234, %1233) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1236 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1237 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1238 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1239 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1240 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1241 = emitc.call @XAie_StrmConnCctEnable(%1236, %1235, %1237, %1238, %1239, %1240) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1242 = "emitc.constant"() <{value = 29 : i32}> : () -> i32
    %1243 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1244 = emitc.call @XAie_TileLoc(%1243, %1242) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1245 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1246 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1247 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1248 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1249 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1250 = emitc.call @XAie_StrmConnCctEnable(%1245, %1244, %1246, %1247, %1248, %1249) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1251 = "emitc.constant"() <{value = 29 : i32}> : () -> i32
    %1252 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1253 = emitc.call @XAie_TileLoc(%1252, %1251) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1254 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1255 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1256 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1257 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1258 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1259 = emitc.call @XAie_StrmConnCctEnable(%1254, %1253, %1255, %1256, %1257, %1258) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1260 = "emitc.constant"() <{value = 28 : i32}> : () -> i32
    %1261 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1262 = emitc.call @XAie_TileLoc(%1261, %1260) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1263 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1264 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1265 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1266 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1267 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1268 = emitc.call @XAie_StrmConnCctEnable(%1263, %1262, %1264, %1265, %1266, %1267) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1269 = "emitc.constant"() <{value = 27 : i32}> : () -> i32
    %1270 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1271 = emitc.call @XAie_TileLoc(%1270, %1269) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1272 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1273 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1274 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1275 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1276 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1277 = emitc.call @XAie_StrmConnCctEnable(%1272, %1271, %1273, %1274, %1275, %1276) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1278 = "emitc.constant"() <{value = 26 : i32}> : () -> i32
    %1279 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1280 = emitc.call @XAie_TileLoc(%1279, %1278) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1281 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1282 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1283 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1284 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1285 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1286 = emitc.call @XAie_StrmConnCctEnable(%1281, %1280, %1282, %1283, %1284, %1285) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1287 = "emitc.constant"() <{value = 25 : i32}> : () -> i32
    %1288 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1289 = emitc.call @XAie_TileLoc(%1288, %1287) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1290 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1291 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1292 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1293 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1294 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1295 = emitc.call @XAie_StrmConnCctEnable(%1290, %1289, %1291, %1292, %1293, %1294) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1296 = "emitc.constant"() <{value = 25 : i32}> : () -> i32
    %1297 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1298 = emitc.call @XAie_TileLoc(%1297, %1296) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1299 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1300 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1301 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1302 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1303 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1304 = emitc.call @XAie_StrmConnCctEnable(%1299, %1298, %1300, %1301, %1302, %1303) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1305 = "emitc.constant"() <{value = 25 : i32}> : () -> i32
    %1306 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1307 = emitc.call @XAie_TileLoc(%1306, %1305) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1308 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1309 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1310 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1311 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1312 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1313 = emitc.call @XAie_StrmConnCctEnable(%1308, %1307, %1309, %1310, %1311, %1312) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1314 = "emitc.constant"() <{value = 25 : i32}> : () -> i32
    %1315 = "emitc.constant"() <{value = 6 : i32}> : () -> i32
    %1316 = emitc.call @XAie_TileLoc(%1315, %1314) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1317 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1318 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1319 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1320 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1321 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1322 = emitc.call @XAie_StrmConnCctEnable(%1317, %1316, %1318, %1319, %1320, %1321) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1323 = "emitc.constant"() <{value = 25 : i32}> : () -> i32
    %1324 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
    %1325 = emitc.call @XAie_TileLoc(%1324, %1323) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1326 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1327 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1328 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1329 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1330 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1331 = emitc.call @XAie_StrmConnCctEnable(%1326, %1325, %1327, %1328, %1329, %1330) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1332 = "emitc.constant"() <{value = 25 : i32}> : () -> i32
    %1333 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %1334 = emitc.call @XAie_TileLoc(%1333, %1332) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1335 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1336 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1337 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1338 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1339 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1340 = emitc.call @XAie_StrmConnCctEnable(%1335, %1334, %1336, %1337, %1338, %1339) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1341 = "emitc.constant"() <{value = 25 : i32}> : () -> i32
    %1342 = "emitc.constant"() <{value = 9 : i32}> : () -> i32
    %1343 = emitc.call @XAie_TileLoc(%1342, %1341) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1344 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1345 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1346 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1347 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1348 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1349 = emitc.call @XAie_StrmConnCctEnable(%1344, %1343, %1345, %1346, %1347, %1348) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1350 = "emitc.constant"() <{value = 23 : i32}> : () -> i32
    %1351 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1352 = emitc.call @XAie_TileLoc(%1351, %1350) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1353 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1354 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1355 = emitc.call @XAie_EnableShimDmaToAieStrmPort(%1353, %1352, %1354) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
    %1356 = "emitc.constant"() <{value = 23 : i32}> : () -> i32
    %1357 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1358 = emitc.call @XAie_TileLoc(%1357, %1356) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1359 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1360 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1361 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1362 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1363 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1364 = emitc.call @XAie_StrmConnCctEnable(%1359, %1358, %1360, %1361, %1362, %1363) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1365 = "emitc.constant"() <{value = 23 : i32}> : () -> i32
    %1366 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1367 = emitc.call @XAie_TileLoc(%1366, %1365) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1368 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1369 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1370 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1371 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1372 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1373 = emitc.call @XAie_StrmConnCctEnable(%1368, %1367, %1369, %1370, %1371, %1372) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1374 = "emitc.constant"() <{value = 23 : i32}> : () -> i32
    %1375 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1376 = emitc.call @XAie_TileLoc(%1375, %1374) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1377 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1378 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1379 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1380 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1381 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1382 = emitc.call @XAie_StrmConnCctEnable(%1377, %1376, %1378, %1379, %1380, %1381) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1383 = "emitc.constant"() <{value = 23 : i32}> : () -> i32
    %1384 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1385 = emitc.call @XAie_TileLoc(%1384, %1383) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1386 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1387 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1388 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1389 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1390 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1391 = emitc.call @XAie_StrmConnCctEnable(%1386, %1385, %1387, %1388, %1389, %1390) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1392 = "emitc.constant"() <{value = 23 : i32}> : () -> i32
    %1393 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1394 = emitc.call @XAie_TileLoc(%1393, %1392) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1395 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1396 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1397 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1398 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1399 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1400 = emitc.call @XAie_StrmConnCctEnable(%1395, %1394, %1396, %1397, %1398, %1399) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1401 = "emitc.constant"() <{value = 23 : i32}> : () -> i32
    %1402 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1403 = emitc.call @XAie_TileLoc(%1402, %1401) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1404 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1405 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1406 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1407 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1408 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1409 = emitc.call @XAie_StrmConnCctEnable(%1404, %1403, %1405, %1406, %1407, %1408) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1410 = "emitc.constant"() <{value = 23 : i32}> : () -> i32
    %1411 = "emitc.constant"() <{value = 6 : i32}> : () -> i32
    %1412 = emitc.call @XAie_TileLoc(%1411, %1410) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1413 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1414 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1415 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1416 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1417 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1418 = emitc.call @XAie_StrmConnCctEnable(%1413, %1412, %1414, %1415, %1416, %1417) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1419 = "emitc.constant"() <{value = 23 : i32}> : () -> i32
    %1420 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
    %1421 = emitc.call @XAie_TileLoc(%1420, %1419) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1422 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1423 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1424 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1425 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1426 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1427 = emitc.call @XAie_StrmConnCctEnable(%1422, %1421, %1423, %1424, %1425, %1426) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1428 = "emitc.constant"() <{value = 23 : i32}> : () -> i32
    %1429 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %1430 = emitc.call @XAie_TileLoc(%1429, %1428) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1431 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1432 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1433 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1434 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1435 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1436 = emitc.call @XAie_StrmConnCctEnable(%1431, %1430, %1432, %1433, %1434, %1435) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1437 = "emitc.constant"() <{value = 23 : i32}> : () -> i32
    %1438 = "emitc.constant"() <{value = 9 : i32}> : () -> i32
    %1439 = emitc.call @XAie_TileLoc(%1438, %1437) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1440 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1441 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1442 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1443 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1444 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1445 = emitc.call @XAie_StrmConnCctEnable(%1440, %1439, %1441, %1442, %1443, %1444) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1446 = "emitc.constant"() <{value = 23 : i32}> : () -> i32
    %1447 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1448 = emitc.call @XAie_TileLoc(%1447, %1446) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1449 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1450 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1451 = emitc.call @XAie_EnableShimDmaToAieStrmPort(%1449, %1448, %1450) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
    %1452 = "emitc.constant"() <{value = 23 : i32}> : () -> i32
    %1453 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1454 = emitc.call @XAie_TileLoc(%1453, %1452) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1455 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1456 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1457 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1458 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1459 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1460 = emitc.call @XAie_StrmConnCctEnable(%1455, %1454, %1456, %1457, %1458, %1459) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1461 = "emitc.constant"() <{value = 23 : i32}> : () -> i32
    %1462 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1463 = emitc.call @XAie_TileLoc(%1462, %1461) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1464 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1465 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1466 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1467 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1468 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1469 = emitc.call @XAie_StrmConnCctEnable(%1464, %1463, %1465, %1466, %1467, %1468) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1470 = "emitc.constant"() <{value = 23 : i32}> : () -> i32
    %1471 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1472 = emitc.call @XAie_TileLoc(%1471, %1470) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1473 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1474 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1475 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1476 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1477 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1478 = emitc.call @XAie_StrmConnCctEnable(%1473, %1472, %1474, %1475, %1476, %1477) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1479 = "emitc.constant"() <{value = 23 : i32}> : () -> i32
    %1480 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1481 = emitc.call @XAie_TileLoc(%1480, %1479) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1482 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1483 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1484 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1485 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1486 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1487 = emitc.call @XAie_StrmConnCctEnable(%1482, %1481, %1483, %1484, %1485, %1486) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1488 = "emitc.constant"() <{value = 24 : i32}> : () -> i32
    %1489 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1490 = emitc.call @XAie_TileLoc(%1489, %1488) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1491 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1492 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1493 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1494 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1495 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1496 = emitc.call @XAie_StrmConnCctEnable(%1491, %1490, %1492, %1493, %1494, %1495) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1497 = "emitc.constant"() <{value = 24 : i32}> : () -> i32
    %1498 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1499 = emitc.call @XAie_TileLoc(%1498, %1497) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1500 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1501 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1502 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1503 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1504 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1505 = emitc.call @XAie_StrmConnCctEnable(%1500, %1499, %1501, %1502, %1503, %1504) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1506 = "emitc.constant"() <{value = 24 : i32}> : () -> i32
    %1507 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1508 = emitc.call @XAie_TileLoc(%1507, %1506) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1509 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1510 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1511 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1512 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1513 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1514 = emitc.call @XAie_StrmConnCctEnable(%1509, %1508, %1510, %1511, %1512, %1513) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1515 = "emitc.constant"() <{value = 24 : i32}> : () -> i32
    %1516 = "emitc.constant"() <{value = 6 : i32}> : () -> i32
    %1517 = emitc.call @XAie_TileLoc(%1516, %1515) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1518 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1519 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1520 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1521 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1522 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1523 = emitc.call @XAie_StrmConnCctEnable(%1518, %1517, %1519, %1520, %1521, %1522) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1524 = "emitc.constant"() <{value = 24 : i32}> : () -> i32
    %1525 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
    %1526 = emitc.call @XAie_TileLoc(%1525, %1524) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1527 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1528 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1529 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1530 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1531 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1532 = emitc.call @XAie_StrmConnCctEnable(%1527, %1526, %1528, %1529, %1530, %1531) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1533 = "emitc.constant"() <{value = 24 : i32}> : () -> i32
    %1534 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %1535 = emitc.call @XAie_TileLoc(%1534, %1533) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1536 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1537 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1538 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1539 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1540 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1541 = emitc.call @XAie_StrmConnCctEnable(%1536, %1535, %1537, %1538, %1539, %1540) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1542 = "emitc.constant"() <{value = 24 : i32}> : () -> i32
    %1543 = "emitc.constant"() <{value = 9 : i32}> : () -> i32
    %1544 = emitc.call @XAie_TileLoc(%1543, %1542) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1545 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1546 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1547 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1548 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1549 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1550 = emitc.call @XAie_StrmConnCctEnable(%1545, %1544, %1546, %1547, %1548, %1549) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1551 = "emitc.constant"() <{value = 23 : i32}> : () -> i32
    %1552 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1553 = emitc.call @XAie_TileLoc(%1552, %1551) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1554 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1555 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1556 = emitc.call @XAie_EnableShimDmaToAieStrmPort(%1554, %1553, %1555) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
    %1557 = "emitc.constant"() <{value = 23 : i32}> : () -> i32
    %1558 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1559 = emitc.call @XAie_TileLoc(%1558, %1557) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1560 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1561 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1562 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1563 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1564 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1565 = emitc.call @XAie_StrmConnCctEnable(%1560, %1559, %1561, %1562, %1563, %1564) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1566 = "emitc.constant"() <{value = 23 : i32}> : () -> i32
    %1567 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1568 = emitc.call @XAie_TileLoc(%1567, %1566) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1569 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1570 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1571 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1572 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1573 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1574 = emitc.call @XAie_StrmConnCctEnable(%1569, %1568, %1570, %1571, %1572, %1573) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1575 = "emitc.constant"() <{value = 23 : i32}> : () -> i32
    %1576 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1577 = emitc.call @XAie_TileLoc(%1576, %1575) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1578 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1579 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1580 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1581 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1582 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1583 = emitc.call @XAie_StrmConnCctEnable(%1578, %1577, %1579, %1580, %1581, %1582) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1584 = "emitc.constant"() <{value = 23 : i32}> : () -> i32
    %1585 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1586 = emitc.call @XAie_TileLoc(%1585, %1584) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1587 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1588 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1589 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1590 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1591 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1592 = emitc.call @XAie_StrmConnCctEnable(%1587, %1586, %1588, %1589, %1590, %1591) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1593 = "emitc.constant"() <{value = 22 : i32}> : () -> i32
    %1594 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1595 = emitc.call @XAie_TileLoc(%1594, %1593) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1596 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1597 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1598 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1599 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1600 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1601 = emitc.call @XAie_StrmConnCctEnable(%1596, %1595, %1597, %1598, %1599, %1600) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1602 = "emitc.constant"() <{value = 22 : i32}> : () -> i32
    %1603 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1604 = emitc.call @XAie_TileLoc(%1603, %1602) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1605 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1606 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1607 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1608 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1609 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1610 = emitc.call @XAie_StrmConnCctEnable(%1605, %1604, %1606, %1607, %1608, %1609) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1611 = "emitc.constant"() <{value = 22 : i32}> : () -> i32
    %1612 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1613 = emitc.call @XAie_TileLoc(%1612, %1611) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1614 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1615 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1616 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1617 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1618 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1619 = emitc.call @XAie_StrmConnCctEnable(%1614, %1613, %1615, %1616, %1617, %1618) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1620 = "emitc.constant"() <{value = 22 : i32}> : () -> i32
    %1621 = "emitc.constant"() <{value = 6 : i32}> : () -> i32
    %1622 = emitc.call @XAie_TileLoc(%1621, %1620) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1623 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1624 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1625 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1626 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1627 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1628 = emitc.call @XAie_StrmConnCctEnable(%1623, %1622, %1624, %1625, %1626, %1627) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1629 = "emitc.constant"() <{value = 22 : i32}> : () -> i32
    %1630 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
    %1631 = emitc.call @XAie_TileLoc(%1630, %1629) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1632 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1633 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1634 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1635 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1636 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1637 = emitc.call @XAie_StrmConnCctEnable(%1632, %1631, %1633, %1634, %1635, %1636) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1638 = "emitc.constant"() <{value = 22 : i32}> : () -> i32
    %1639 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %1640 = emitc.call @XAie_TileLoc(%1639, %1638) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1641 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1642 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1643 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1644 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1645 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1646 = emitc.call @XAie_StrmConnCctEnable(%1641, %1640, %1642, %1643, %1644, %1645) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1647 = "emitc.constant"() <{value = 22 : i32}> : () -> i32
    %1648 = "emitc.constant"() <{value = 9 : i32}> : () -> i32
    %1649 = emitc.call @XAie_TileLoc(%1648, %1647) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1650 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1651 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1652 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1653 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1654 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1655 = emitc.call @XAie_StrmConnCctEnable(%1650, %1649, %1651, %1652, %1653, %1654) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1656 = "emitc.constant"() <{value = 23 : i32}> : () -> i32
    %1657 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1658 = emitc.call @XAie_TileLoc(%1657, %1656) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1659 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1660 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1661 = emitc.call @XAie_EnableShimDmaToAieStrmPort(%1659, %1658, %1660) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32) -> i32
    %1662 = "emitc.constant"() <{value = 23 : i32}> : () -> i32
    %1663 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1664 = emitc.call @XAie_TileLoc(%1663, %1662) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1665 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1666 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1667 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1668 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1669 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1670 = emitc.call @XAie_StrmConnCctEnable(%1665, %1664, %1666, %1667, %1668, %1669) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1671 = "emitc.constant"() <{value = 23 : i32}> : () -> i32
    %1672 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1673 = emitc.call @XAie_TileLoc(%1672, %1671) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1674 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1675 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1676 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1677 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1678 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1679 = emitc.call @XAie_StrmConnCctEnable(%1674, %1673, %1675, %1676, %1677, %1678) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1680 = "emitc.constant"() <{value = 23 : i32}> : () -> i32
    %1681 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1682 = emitc.call @XAie_TileLoc(%1681, %1680) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1683 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1684 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1685 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1686 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1687 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1688 = emitc.call @XAie_StrmConnCctEnable(%1683, %1682, %1684, %1685, %1686, %1687) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1689 = "emitc.constant"() <{value = 23 : i32}> : () -> i32
    %1690 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1691 = emitc.call @XAie_TileLoc(%1690, %1689) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1692 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1693 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1694 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1695 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1696 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1697 = emitc.call @XAie_StrmConnCctEnable(%1692, %1691, %1693, %1694, %1695, %1696) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1698 = "emitc.constant"() <{value = 22 : i32}> : () -> i32
    %1699 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1700 = emitc.call @XAie_TileLoc(%1699, %1698) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1701 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1702 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1703 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1704 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1705 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1706 = emitc.call @XAie_StrmConnCctEnable(%1701, %1700, %1702, %1703, %1704, %1705) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1707 = "emitc.constant"() <{value = 21 : i32}> : () -> i32
    %1708 = "emitc.constant"() <{value = 3 : i32}> : () -> i32
    %1709 = emitc.call @XAie_TileLoc(%1708, %1707) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1710 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1711 = "emitc.constant"() <{value = 2 : i32}> : () -> i32
    %1712 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1713 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1714 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1715 = emitc.call @XAie_StrmConnCctEnable(%1710, %1709, %1711, %1712, %1713, %1714) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1716 = "emitc.constant"() <{value = 21 : i32}> : () -> i32
    %1717 = "emitc.constant"() <{value = 4 : i32}> : () -> i32
    %1718 = emitc.call @XAie_TileLoc(%1717, %1716) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1719 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1720 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1721 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1722 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1723 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1724 = emitc.call @XAie_StrmConnCctEnable(%1719, %1718, %1720, %1721, %1722, %1723) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1725 = "emitc.constant"() <{value = 21 : i32}> : () -> i32
    %1726 = "emitc.constant"() <{value = 5 : i32}> : () -> i32
    %1727 = emitc.call @XAie_TileLoc(%1726, %1725) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1728 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1729 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1730 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1731 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1732 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1733 = emitc.call @XAie_StrmConnCctEnable(%1728, %1727, %1729, %1730, %1731, %1732) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1734 = "emitc.constant"() <{value = 21 : i32}> : () -> i32
    %1735 = "emitc.constant"() <{value = 6 : i32}> : () -> i32
    %1736 = emitc.call @XAie_TileLoc(%1735, %1734) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1737 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1738 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1739 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1740 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1741 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1742 = emitc.call @XAie_StrmConnCctEnable(%1737, %1736, %1738, %1739, %1740, %1741) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1743 = "emitc.constant"() <{value = 21 : i32}> : () -> i32
    %1744 = "emitc.constant"() <{value = 7 : i32}> : () -> i32
    %1745 = emitc.call @XAie_TileLoc(%1744, %1743) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1746 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1747 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1748 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1749 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1750 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1751 = emitc.call @XAie_StrmConnCctEnable(%1746, %1745, %1747, %1748, %1749, %1750) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1752 = "emitc.constant"() <{value = 21 : i32}> : () -> i32
    %1753 = "emitc.constant"() <{value = 8 : i32}> : () -> i32
    %1754 = emitc.call @XAie_TileLoc(%1753, %1752) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1755 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1756 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1757 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1758 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1759 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1760 = emitc.call @XAie_StrmConnCctEnable(%1755, %1754, %1756, %1757, %1758, %1759) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    %1761 = "emitc.constant"() <{value = 21 : i32}> : () -> i32
    %1762 = "emitc.constant"() <{value = 9 : i32}> : () -> i32
    %1763 = emitc.call @XAie_TileLoc(%1762, %1761) : (i32, i32) -> !emitc.opaque<"XAie_LocType">
    %1764 = emitc.call @getOrCreateDeviceInstance() : () -> !emitc.ptr<!emitc.opaque<"XAie_DevInst">>
    %1765 = "emitc.constant"() <{value = 1 : i32}> : () -> i32
    %1766 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1767 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1768 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
    %1769 = emitc.call @XAie_StrmConnCctEnable(%1764, %1763, %1765, %1766, %1767, %1768) : (!emitc.ptr<!emitc.opaque<"XAie_DevInst">>, !emitc.opaque<"XAie_LocType">, i32, i32, i32, i32) -> i32
    return
  }
  }