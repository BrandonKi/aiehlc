/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef __BLUEPRINT_TOSCHEDULE_KERNEL_PASS_H__
#define __BLUEPRINT_TOSCHEDULE_KERNEL_PASS_H__

#include "dfscheblueprintmanager.h"
#include "dfschedulemanager.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

class BlueprintToScheduleKernelPass : public PassWrapper<BlueprintToScheduleKernelPass, OperationPass<>> {
  public:
    BlueprintToScheduleKernelPass() = default;
    BlueprintToScheduleKernelPass(double ratio) : bufferRatio_(ratio) {}
    BlueprintToScheduleKernelPass(double ratio, int64_t maxBytes) : bufferRatio_(ratio), maxPingPongBytes_(maxBytes) {}
    // MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BlueprintToScheduleKernelPass)

    StringRef getArgument() const final { return "lower-blueprint-to-schedule"; }
    StringRef getDescription() const final { return "Lower dfscheblueprint dialect to dfschedule dialect"; }

    double getBufferRatio() const { return bufferRatio_; }
    int64_t getMaxPingPongBytes() const { return maxPingPongBytes_; }

    void runOnOperation() override;

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<dfscheblueprint::dfscheblueprintdialect, dfschedule::dfscheduledialect, func::FuncDialect,
                        memref::MemRefDialect, arith::ArithDialect, scf::SCFDialect, tensor::TensorDialect>();
    }

  private:
    double bufferRatio_ = 0.5;
    int64_t maxPingPongBytes_ = 4096;
};

} // namespace mlir

#endif // __BLUEPRINT_TOSCHEDULE_KERNEL_PASS_H__
