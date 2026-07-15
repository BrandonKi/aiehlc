// ******************************************************************************
// * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
// * SPDX-License-Identifier: Apache-2.0
// ******************************************************************************

// This is Manually generated IR File, to represent certain logic.
// This part should be emitted by Frontend AST of the Language (TODO: do it via Clang AST)
module {
  %0 = Aie.create_window 1, 0, 1, 1, 0, 1, "create_window_operation" : !Aie.Window<1, 0, 1, 1, 0, 1, "mywindow">
  %1 = Aie.create_window 0, 0, 1, 1, 0, 1, "create_window_operation" : !Aie.Window<0, 0, 1, 1, 0, 1, "mywindow">
  %2 = Aie.create_window 0, 0, 1, 1, 0, 1, "create_window_operation" : !Aie.Window<0, 0, 1, 1, 0, 1, "mywindow">
  %3 = Aie.create_kernelobject_op 2, 1(%0, %1, %2) : (!Aie.Window<1, 0, 1, 1, 0, 1, "mywindow">, !Aie.Window<0, 0, 1, 1, 0, 1, "mywindow">, !Aie.Window<0, 0, 1, 1, 0, 1, "mywindow">) -> !Aie.KernelObj<2,1,<1, 0, 1, 1, 0, 1, "mywindow">,<0, 0, 1, 1, 0, 1, "mywindow">, <0, 0, 1, 1, 0, 1, "mywindow">>
  Aie.load_kernel_op "aa", 37, 21(%3) : (!Aie.KernelObj<2,1,<1, 0, 1, 1, 0, 1, "mywindow">,<0, 0, 1, 1, 0, 1, "mywindow">, <0, 0, 1, 1, 0, 1, "mywindow">>) -> ()
}