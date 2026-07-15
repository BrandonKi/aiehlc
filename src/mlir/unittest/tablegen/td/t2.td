/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

include "testattr.td"
def Test_createtilearrayOp : testop<"testcreatetile"> {
  let summary = "Elementwise add with optional scale";
  let description = [{
    **MyAddOp** 计算 `result = (lhs + rhs) * scale`  
    - `lhs` / `rhs` 为 `tensor<f32>`  
    - `scale` 为 `f32` 属性，默认值 1.0  
    - `mode` 为枚举属性 `RoundingMode`
  }];

  //---------------------------------------------------------------------//
  // 4.1 接口、trait
  //---------------------------------------------------------------------//
  let arguments = (ins
    TensorOf<[F32]>:$lhs,
    TensorOf<[F32]>:$rhs,
    F32Attr:$scale
  );
  let results = (outs
    TensorOf<[F32]>:$result
  );
}
