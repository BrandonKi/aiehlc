/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef __DMAP_TODMAPHOP_PASS_H__
#define __DMAP_TODMAPHOP_PASS_H__

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "dmapmanager.h"
#include "dmaphopmanager.h"
#include "routingmanager.h"
#include "routing/routingpath.h"
#include "routing/routingtopology.h"

namespace mlir {

class DmapToDmaphopPass : public PassWrapper<DmapToDmaphopPass, OperationPass<>> {
public:
    DmapToDmaphopPass(RoutingTopology& rtopology);
    //MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DmapToDmaphopPass)

    StringRef getArgument() const final { return "lower-dmap-to-dmaphop"; }
    StringRef getDescription() const final { return "Lower dmap dialect to dmaphop dialect"; }

    void runOnOperation() override;

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<dmap::dmapdialect, dmaphop::dmaphopdialect, func::FuncDialect, memref::MemRefDialect>();
    }
    RoutingTopology& rtopology_;
};

} // namespace mlir

#endif // __DMAP_TODMAPHOP_PASS_H__