// ******************************************************************************
// * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
// * SPDX-License-Identifier: Apache-2.0
// ******************************************************************************

module {
  %0 = Aie.create_window 1, 0, 1, 1, 0, 1, "create_window_operation" : !Aie.Window<1, 0, 1, 1, 0, 1, "mywindow">
  %1 = Aie.create_window 1, 0, 1, 1, 0, 1, "create_window_operation" : !Aie.Window<1, 0, 1, 1, 0, 1, "mywindow">
  %2 = Aie.create_kernelobject_op 1, 1(%0, %1) : (!Aie.Window<1, 0, 1, 1, 0, 1, "mywindow">, !Aie.Window<1, 0, 1, 1, 0, 1, "mywindow">) -> !Aie.KernelObj<1, 1, <1, 0, 1, 1, 0, 1, "mywindow">, <1, 0, 1, 1, 0, 1, "mywindow">>
}


