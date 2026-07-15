/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef __DFSCHEDULE_PROVENANCEMAP_PASS_H__
#define __DFSCHEDULE_PROVENANCEMAP_PASS_H__

#include "dfschedulemanager.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <string>

namespace mlir {

class DfscheduleProvenanceMapPass : public PassWrapper<DfscheduleProvenanceMapPass, OperationPass<>> {
  public:
    DfscheduleProvenanceMapPass() = default;
    explicit DfscheduleProvenanceMapPass(const std::string &outputDir) : outputDir(outputDir) {}
    DfscheduleProvenanceMapPass(const std::string &outputDir, int startCol)
        : outputDir(outputDir), startCol(startCol) {}
    DfscheduleProvenanceMapPass(const std::string &outputDir, int startCol, const std::string &aieGen)
        : outputDir(outputDir), startCol(startCol), aieGen(aieGen) {}

    void runOnOperation() override;

    StringRef getArgument() const override { return "dfschedule-provenance-map"; }
    StringRef getDescription() const override { return "Generate low-level provenance map JSON from dfschedule IR"; }

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<dfschedule::dfscheduledialect>();
        registry.insert<func::FuncDialect>();
        registry.insert<memref::MemRefDialect>();
        registry.insert<arith::ArithDialect>();
        registry.insert<scf::SCFDialect>();
    }

  private:
    std::string outputDir;
    int startCol = -1;
    std::string aieGen;
};

} // namespace mlir

#endif // __DFSCHEDULE_PROVENANCEMAP_PASS_H__
