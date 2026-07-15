/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef __TILING_PASS_CLASS__
#define __TILING_PASS_CLASS__
#include "../../common.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include <sstream>
#include "mlir/IR/Matchers.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include <sstream>
#include "routingmanager.h"
using namespace mlir;
class TilingCodePass : public PassWrapper<TilingCodePass, OperationPass<>> {
    void runOnOperation() override;
};
#endif