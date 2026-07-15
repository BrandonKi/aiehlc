/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef __WAIT_MERGE_PASS_H__
#define __WAIT_MERGE_PASS_H__

#include "dfschedulemanager.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

/// Pass to merge consecutive scf.for loops that contain dfschedule.schedule.wait
/// operations. When multiple adjacent scf.for loops share the same loop bounds
/// (lower bound, upper bound, step) and each ends with a wait, they are merged
/// into a single scf.for loop. All matching consecutive for-loops are merged
/// into one loop regardless of count.
class WaitMergePass : public PassWrapper<WaitMergePass, OperationPass<ModuleOp>> {
  public:
    WaitMergePass() = default;

    StringRef getArgument() const final { return "wait-merge"; }
    StringRef getDescription() const final {
        return "Merge consecutive scf.for loops with matching bounds that contain "
               "dfschedule.schedule.wait into a single loop";
    }

    void runOnOperation() override;

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<dfschedule::dfscheduledialect, func::FuncDialect, memref::MemRefDialect, arith::ArithDialect,
                        scf::SCFDialect>();
    }
};

} // namespace mlir

#endif // __WAIT_MERGE_PASS_H__
