/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef __DMAPHOP_TOROUTINGHW_PASS_H__
#define __DMAPHOP_TOROUTINGHW_PASS_H__

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "dmapmanager.h"
#include "dmaphopmanager.h"
#include "routingmanager.h"
#include "routing/routingpath.h"
#include "routing/routingtopology.h"

namespace mlir {

class DmaphopToRoutinghwPass : public PassWrapper<DmaphopToRoutinghwPass, OperationPass<>> {
public:
    DmaphopToRoutinghwPass(RoutingTopology& rtopology);
    //MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DmaphopToRoutinghwPass)

    StringRef getArgument() const final { return "lower-dmaphop-to-routinghw"; }
    StringRef getDescription() const final { return "Lower dmaphop dialect to routinghw dialect"; }

    void runOnOperation() override;

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<dmaphop::dmaphopdialect, func::FuncDialect, memref::MemRefDialect>();
    }
    RoutingTopology& rtopology_;
};

} // namespace mlir

#endif // __DMAPHOP_TOROUTINGHW_PASS_H__

