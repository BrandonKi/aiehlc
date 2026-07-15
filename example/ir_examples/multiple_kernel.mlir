// ******************************************************************************
// * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
// * SPDX-License-Identifier: Apache-2.0
// ******************************************************************************

module {
  %0 = Aie.create_kernelfuncname_op{"k1"} () : !Aie.KernelFuncName<"k1">
  %1 = Aie.create_kernelfilename_op{"./k1.c"} () : !Aie.KernelFileName<"./k1.c">
  %2 = Aie.create_window 0, 8192, 4096, 12288, 0, 1, "input" : !Aie.Window<0, 8192, 4096, 12288, 0, 1, "input">
  %3 = Aie.create_window 2052755344, 8192, 24576, 32768, 2, 3, "output" : !Aie.Window<2052755344, 8192, 24576, 32768, 2, 3, "output">
  %4 = Aie.create_kernelobject_op{0, 1}(%0, %1, %2, %3) : (!Aie.KernelFuncName<"k1">, !Aie.KernelFileName<"./k1.c">, !Aie.Window<0, 8192, 4096, 12288, 0, 1, "input">, !Aie.Window<2052755344, 8192, 24576, 32768, 2, 3, "output">) -> !Aie.KernelObj<0, 1, <"k1">, <"./k1.c">, <0, 8192, 4096, 12288, 0, 1, "input">, <2052755344, 8192, 24576, 32768, 2, 3, "output">>
  
  %5 = Aie.create_kernelfuncname_op{"k2"} () : !Aie.KernelFuncName<"k2">
  %6 = Aie.create_kernelfilename_op{"./k2.c"} () : !Aie.KernelFileName<"./k2.c">
  %7 = Aie.create_window 0, 8192, 4096, 12288, 0, 1, "input" : !Aie.Window<0, 8192, 4096, 12288, 0, 1, "input">
  %8 = Aie.create_window 2052755344, 8192, 24576, 32768, 2, 3, "output" : !Aie.Window<2052755344, 8192, 24576, 32768, 2, 3, "output">
  %9 = Aie.create_kernelobject_op{0, 1}(%5, %6, %7, %8) : (!Aie.KernelFuncName<"k2">, !Aie.KernelFileName<"./k2.c">, !Aie.Window<0, 8192, 4096, 12288, 0, 1, "input">, !Aie.Window<2052755344, 8192, 24576, 32768, 2, 3, "output">) -> !Aie.KernelObj<0, 1, <"k2">, <"./k2.c">, <0, 8192, 4096, 12288, 0, 1, "input">, <2052755344, 8192, 24576, 32768, 2, 3, "output">>
}
