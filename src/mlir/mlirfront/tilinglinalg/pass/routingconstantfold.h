/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef __ROUTING_CONSTANTFOLD_CLEAN__
#define __ROUTING_CONSTANTFOLD_CLEAN__
#include "common.h"
#include "routing/routingtopology.h"
using namespace mlir;
class RoutingConstantFoldPass : public PassWrapper<RoutingConstantFoldPass, OperationPass<ModuleOp>> {
public:
  RoutingConstantFoldPass(std::string gen = "Gen2") : aieGen_(std::move(gen)) {};

private:
  std::string aieGen_;
  void runOnOperation() override;
  mlir::StringRef getArgument() const final { return "emitc-constant-fold"; }
  mlir::StringRef getDescription() const final { return "Folds emitc.constant operands into emitc.call operations."; }
};
#endif //