/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef __DMAPHOP_TODFSCHEBLUEPRINT_PASS_H__
#define __DMAPHOP_TODFSCHEBLUEPRINT_PASS_H__

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "dmaphopmanager.h"
#include "dfscheblueprintmanager.h"
#include "routingmanager.h"

namespace mlir {

class DmaphopTodfscheblueprintPass : public PassWrapper<DmaphopTodfscheblueprintPass, OperationPass<>> {
public:
    DmaphopTodfscheblueprintPass() = default;
    
    void runOnOperation() override;
    
    StringRef getArgument() const override { return "dmaphop-to-dfscheblueprint"; }
    StringRef getDescription() const override { return "Convert dmaphop to dfscheblueprint"; }
    
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<dmaphop::dmaphopdialect>();
        registry.insert<dfscheblueprint::dfscheblueprintdialect>();
        registry.insert<routing::routingdialect>();
        registry.insert<func::FuncDialect>();
        registry.insert<memref::MemRefDialect>();
        registry.insert<arith::ArithDialect>();
        registry.insert<tensor::TensorDialect>();
    }
};

std::unique_ptr<Pass> createDmaphopTodfscheblueprintPass();

} // namespace mlir

#endif // __DMAPHOP_TODFSCHEBLUEPRINT_PASS_H__
