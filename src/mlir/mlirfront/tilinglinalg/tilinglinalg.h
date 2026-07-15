/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef __TILING_LINALG_CLASS__
#define __TILING_LINALG_CLASS__
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/IR/Function.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
// #include "mlir/Dialect/Transform/Transforms/TransformInterpreterPass.h"
#include "mlir/Pass/PassManager.h" 
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h" 
#include "tilingpass.h"
using namespace mlir;
class tilinglinalg {
public:
    tilinglinalg();
    void creatematmul(int m, int k, int n);
private:
    mlir::MLIRContext context;
    mlir::ModuleOp module;
};
#endif