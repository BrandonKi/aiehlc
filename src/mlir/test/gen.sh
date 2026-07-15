###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################

/mlir/llvm-project/build/bin/mlir-tblgen -gen-dialect-defs ./mdialect.td -I /mlir/llvm-project/mlir/include/ > ./aiedialect.cc.inc
/mlir/llvm-project/build/bin/mlir-tblgen -gen-dialect-decls ./mdialect.td -I /mlir/llvm-project/mlir/include/ > ./aiedialect.h.inc
/mlir/llvm-project/build/bin/mlir-tblgen -gen-op-defs ./mdialect.td -I /mlir/llvm-project/mlir/include/ > ./aieop.cc.inc
/mlir/llvm-project/build/bin/mlir-tblgen -gen-op-decls ./mdialect.td -I /mlir/llvm-project/mlir/include/ > ./aieop.h.inc

