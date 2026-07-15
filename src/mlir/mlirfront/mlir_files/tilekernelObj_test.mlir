// ******************************************************************************
// * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
// * SPDX-License-Identifier: Apache-2.0
// ******************************************************************************

module {
  %0 = Aie.constant_integer32_op 3 : i32
  %1 = Aie.constant_integer32_op 4 : i32
  %2 = Aie.create_tilekernelobject_op %0, %1, "matrix_mult" : !Aie.TileKernelObj<3, 4, "matrix_mult">
}


