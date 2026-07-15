/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include <map>
#include <string>
#include <vector>

/// Lightweight kernel IR op -- mirrors Python KernelOp dict.
struct KernelOp {
    std::string op;                            // "get_coreid", "acquire_input", etc.
    std::map<std::string, int64_t> intArgs;    // "window_idx", "trip_count", "m", "n", "k"
    std::map<std::string, std::string> strArgs; // "var"
};

/// Builds MLIR EmitC ops from a list of KernelOps, then translateToCpp -> C string.
///
/// Mirrors the VerbatimOp pattern used by DfscheduleToKernelApiPass
/// (passdfscheduletokernelapi.cpp): each kernel construct becomes an
/// emitc::VerbatimOp wrapping a C code string, then translateToCpp
/// emits them verbatim into the output file.
class KernelBodyEmitter {
public:
    KernelBodyEmitter(mlir::MLIRContext &ctx,
                      const std::string &kernelName,
                      const std::string &elementType,
                      int numInputWindows, int numOutputWindows);

    /// Build MLIR EmitC module from KernelOps, translateToCpp -> C string.
    std::string emit(const std::vector<KernelOp> &ops);

private:
    mlir::MLIRContext &ctx_;
    std::string kernelName_;
    std::string elementType_;  // "int8", "int16", "int32", "float"
    std::string cType_;        // "int8_t", "int16_t", etc.
    std::string winInType_;    // "input_window_int8", etc.
    std::string winOutType_;   // "output_window_int8", etc.
    int numInputWindows_;
    int numOutputWindows_;

    void emitFunctionHeader(mlir::OpBuilder &b, mlir::Location loc);
    void emitGetCoreId(mlir::OpBuilder &b, mlir::Location loc);
    void emitForRange(mlir::OpBuilder &b, mlir::Location loc, int tripCount);
    void emitAcquireInput(mlir::OpBuilder &b, mlir::Location loc, int windowIdx);
    void emitAcquireOutput(mlir::OpBuilder &b, mlir::Location loc, int windowIdx);
    void emitGemmBody(mlir::OpBuilder &b, mlir::Location loc, int m, int n, int k);
    void emitReleaseInput(mlir::OpBuilder &b, mlir::Location loc, int windowIdx);
    void emitReleaseOutput(mlir::OpBuilder &b, mlir::Location loc, int windowIdx);
    void emitEndFor(mlir::OpBuilder &b, mlir::Location loc);
    void emitFunctionFooter(mlir::OpBuilder &b, mlir::Location loc);
};
