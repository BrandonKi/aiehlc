/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "kernel_body_emitter.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

KernelBodyEmitter::KernelBodyEmitter(MLIRContext &ctx,
                                     const std::string &kernelName,
                                     const std::string &elementType,
                                     int numInputWindows,
                                     int numOutputWindows)
    : ctx_(ctx), kernelName_(kernelName), elementType_(elementType),
      numInputWindows_(numInputWindows), numOutputWindows_(numOutputWindows) {
    // Map element type to C type and window type names
    if (elementType == "int8") {
        cType_ = "int8_t";
    } else if (elementType == "int16") {
        cType_ = "int16_t";
    } else if (elementType == "int32") {
        cType_ = "int32_t";
    } else if (elementType == "float") {
        cType_ = "float";
    } else {
        cType_ = "int8_t"; // default
    }
    winInType_ = "input_window_" + elementType;
    winOutType_ = "output_window_" + elementType;
}

std::string KernelBodyEmitter::emit(const std::vector<KernelOp> &ops) {
    // Create a temporary module to hold EmitC ops
    auto module = ModuleOp::create(UnknownLoc::get(&ctx_));
    OpBuilder builder(&ctx_);
    builder.setInsertionPointToEnd(module.getBody());
    Location loc = UnknownLoc::get(&ctx_);

    // Emit function header
    emitFunctionHeader(builder, loc);

    // Walk KernelOps and dispatch
    for (const auto &op : ops) {
        if (op.op == "get_coreid") {
            emitGetCoreId(builder, loc);
        } else if (op.op == "for_range") {
            int tripCount = 2;
            auto it = op.intArgs.find("trip_count");
            if (it != op.intArgs.end())
                tripCount = static_cast<int>(it->second);
            emitForRange(builder, loc, tripCount);
        } else if (op.op == "acquire_input") {
            int windowIdx = 0;
            auto it = op.intArgs.find("window_idx");
            if (it != op.intArgs.end())
                windowIdx = static_cast<int>(it->second);
            emitAcquireInput(builder, loc, windowIdx);
        } else if (op.op == "acquire_output") {
            int windowIdx = 0;
            auto it = op.intArgs.find("window_idx");
            if (it != op.intArgs.end())
                windowIdx = static_cast<int>(it->second);
            emitAcquireOutput(builder, loc, windowIdx);
        } else if (op.op == "gemm_body") {
            int m = 8, n = 8, k = 8;
            auto im = op.intArgs.find("m");
            auto in = op.intArgs.find("n");
            auto ik = op.intArgs.find("k");
            if (im != op.intArgs.end()) m = static_cast<int>(im->second);
            if (in != op.intArgs.end()) n = static_cast<int>(in->second);
            if (ik != op.intArgs.end()) k = static_cast<int>(ik->second);
            emitGemmBody(builder, loc, m, n, k);
        } else if (op.op == "release_input") {
            int windowIdx = 0;
            auto it = op.intArgs.find("window_idx");
            if (it != op.intArgs.end())
                windowIdx = static_cast<int>(it->second);
            emitReleaseInput(builder, loc, windowIdx);
        } else if (op.op == "release_output") {
            int windowIdx = 0;
            auto it = op.intArgs.find("window_idx");
            if (it != op.intArgs.end())
                windowIdx = static_cast<int>(it->second);
            emitReleaseOutput(builder, loc, windowIdx);
        } else if (op.op == "end_for") {
            emitEndFor(builder, loc);
        }
    }

    // Emit function footer
    emitFunctionFooter(builder, loc);

    // Translate to C++
    std::string result;
    llvm::raw_string_ostream stream(result);
    if (failed(emitc::translateToCpp(module, stream)))
        return "";

    module->erase();
    return result;
}

void KernelBodyEmitter::emitFunctionHeader(OpBuilder &b, Location loc) {
    // Build function signature:
    //   void matmul(input_window_int8 *window_in_0,
    //               input_window_int8 *window_in_1,
    //               output_window_int8 *window_out_0) {
    std::string sig = "void " + kernelName_ + "(";
    int paramIdx = 0;
    for (int i = 0; i < numInputWindows_; ++i) {
        if (paramIdx > 0) sig += ",\n" + std::string(5 + kernelName_.size(), ' ');
        sig += winInType_ + " *window_in_" + std::to_string(i);
        paramIdx++;
    }
    for (int i = 0; i < numOutputWindows_; ++i) {
        if (paramIdx > 0) sig += ",\n" + std::string(5 + kernelName_.size(), ' ');
        sig += winOutType_ + " *window_out_" + std::to_string(i);
        paramIdx++;
    }
    sig += ") {";
    b.create<emitc::VerbatimOp>(loc, sig);
}

void KernelBodyEmitter::emitGetCoreId(OpBuilder &b, Location loc) {
    b.create<emitc::VerbatimOp>(loc,
        "    unsigned coreid = get_coreid();\n"
        "    int col = coreid >> 16;\n"
        "    int row = coreid & 0x1F;");
}

void KernelBodyEmitter::emitForRange(OpBuilder &b, Location loc, int tripCount) {
    b.create<emitc::VerbatimOp>(loc,
        "    for (int iter = 0; iter < " + std::to_string(tripCount) + "; iter++) {");
    b.create<emitc::VerbatimOp>(loc,
        "        klog(\"CENk\", iter);");
}

void KernelBodyEmitter::emitAcquireInput(OpBuilder &b, Location loc, int windowIdx) {
    std::string varName = "in" + std::to_string(windowIdx);
    b.create<emitc::VerbatimOp>(loc,
        "        " + cType_ + " *" + varName + " = (" + cType_ +
        " *)acquire_input_window(window_in_" + std::to_string(windowIdx) + ");");
}

void KernelBodyEmitter::emitAcquireOutput(OpBuilder &b, Location loc, int windowIdx) {
    std::string varName = "out" + std::to_string(windowIdx);
    b.create<emitc::VerbatimOp>(loc,
        "        " + cType_ + " *" + varName +
        " = acquire_output_window(window_out_" + std::to_string(windowIdx) + ");");
}

void KernelBodyEmitter::emitGemmBody(OpBuilder &b, Location loc, int m, int n, int k) {
    std::string body =
        "        for (int i = 0; i < " + std::to_string(m) + "; i++) {\n"
        "            for (int j = 0; j < " + std::to_string(n) + "; j++) {\n"
        "                int16_t sum = 0;\n"
        "                for (int kk = 0; kk < " + std::to_string(k) + "; kk++)\n"
        "                    sum += (int16_t)in0[i * " + std::to_string(k) + " + kk] * "
        "(int16_t)in1[kk * " + std::to_string(n) + " + j];\n"
        "                if (sum > 127) sum = 127;\n"
        "                else if (sum < -128) sum = -128;\n"
        "                out0[i * " + std::to_string(n) + " + j] = (int8_t)sum;\n"
        "            }\n"
        "        }";
    b.create<emitc::VerbatimOp>(loc, body);
    b.create<emitc::VerbatimOp>(loc,
        "        klog(\"CLOP\", " + std::to_string(m * n) + ");");
}

void KernelBodyEmitter::emitReleaseInput(OpBuilder &b, Location loc, int windowIdx) {
    b.create<emitc::VerbatimOp>(loc,
        "        release_input_window(window_in_" + std::to_string(windowIdx) + ");");
}

void KernelBodyEmitter::emitReleaseOutput(OpBuilder &b, Location loc, int windowIdx) {
    b.create<emitc::VerbatimOp>(loc,
        "        release_output_window(window_out_" + std::to_string(windowIdx) + ");");
}

void KernelBodyEmitter::emitEndFor(OpBuilder &b, Location loc) {
    b.create<emitc::VerbatimOp>(loc,
        "        klog(\"CEXT\", 1);");
    b.create<emitc::VerbatimOp>(loc,
        "    }");
}

void KernelBodyEmitter::emitFunctionFooter(OpBuilder &b, Location loc) {
    b.create<emitc::VerbatimOp>(loc, "}");
}
