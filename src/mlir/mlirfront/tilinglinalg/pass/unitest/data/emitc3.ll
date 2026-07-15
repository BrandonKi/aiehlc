/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/
%11 = "emitc.constant"() <{value = 0 : i32}> : () -> i32
%12 = "emitc.constant"() <{value = 0 : i64}> : () -> i64
emitc.call_opaque "f"() {args = [#emitc.opaque<"1">,#emitc.opaque<"1">]} : () -> ()
%80 = emitc.call_opaque "XAie_StrmConnCctEnable"(%11) {args =[ 1 : i32, 0 : i32, 0 : i32, 0 : i32]} : (i32) -> i64
 %0 = emitc.call_opaque "mhlo::add"(%80, %12) {args = [9 : i32,  1:index,0:index]} : (i64, i64) -> i64