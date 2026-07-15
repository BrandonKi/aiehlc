/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tilinglinalg_pipeline.h"
#include "aie_pass/kernel_body_emitter.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace py = pybind11;

static bool run_aie_pipeline(int meshRows, int meshCols,
                             const std::vector<std::tuple<std::vector<int64_t>, int, bool>> &tensorSpecs,
                             const std::string &outputDir, const std::string &userKernelBody,
                             const std::string &userKernelFuncName,
                             const std::vector<std::tuple<int, std::string, std::string>> &splitSpecs = {}) {
    mlir::MLIRContext ctx;
    TilingLinalgPipeline::registerDialects(ctx);

    std::vector<TensorParam> tensors;
    for (auto &[shape, bw, isIn] : tensorSpecs)
        tensors.push_back({shape, bw, isIn});

    // Build SplitModel from optional split specs
    SplitModel splitModel;
    if (!splitSpecs.empty()) {
        for (auto &[dim, axis, replicate] : splitSpecs)
            splitModel.tensorSplits.push_back({dim, axis, replicate});
    } else {
        splitModel = SplitModel::gemm();
    }

    auto module = TilingLinalgPipeline::buildRoutingIR(ctx, meshRows, meshCols, tensors, splitModel);
    return TilingLinalgPipeline::runPipeline(ctx, module, outputDir, userKernelBody, userKernelFuncName);
}

/// Build a C kernel body string from a list of KernelOp dicts via MLIR EmitC.
///
/// Python passes a list[dict] where each dict has an "op" key plus op-specific
/// int/str fields. This function converts them to C++ KernelOp structs, feeds
/// them to KernelBodyEmitter, which builds EmitC VerbatimOps and calls
/// translateToCpp to produce the final C string.
static std::string build_kernel_body(
    const std::string &kernelName,
    const std::string &elementType,
    int numInputWindows,
    int numOutputWindows,
    const py::list &kernelOpsList)
{
    // Convert Python list[dict] -> vector<KernelOp>
    std::vector<KernelOp> ops;
    for (auto item : kernelOpsList) {
        auto d = item.cast<py::dict>();
        KernelOp op;
        op.op = d["op"].cast<std::string>();
        for (auto &kv : d) {
            std::string key = kv.first.cast<std::string>();
            if (key == "op") continue;
            if (py::isinstance<py::int_>(kv.second))
                op.intArgs[key] = kv.second.cast<int64_t>();
            else if (py::isinstance<py::str>(kv.second))
                op.strArgs[key] = kv.second.cast<std::string>();
        }
        ops.push_back(std::move(op));
    }

    mlir::MLIRContext ctx;
    ctx.getOrLoadDialect<mlir::emitc::EmitCDialect>();
    ctx.getOrLoadDialect<mlir::func::FuncDialect>();

    KernelBodyEmitter emitter(ctx, kernelName, elementType,
                              numInputWindows, numOutputWindows);
    return emitter.emit(ops);
}

PYBIND11_MODULE(_aietriton_core, m) {
    m.doc() = "AIE Triton pybind11 bindings to TilingLinalgPipeline";
    m.def("run_aie_pipeline", &run_aie_pipeline, py::arg("mesh_rows"), py::arg("mesh_cols"), py::arg("tensor_specs"),
          py::arg("output_dir"), py::arg("user_kernel_body") = "", py::arg("user_kernel_func_name") = "",
          py::arg("split_specs") = std::vector<std::tuple<int, std::string, std::string>>{},
          "Run TilingLinalgPipeline: buildRoutingIR + runPipeline.\n"
          "split_specs: optional list of (splitDim, hwAxisOwner, replicateOn) tuples, one per tensor.");
    m.def("build_kernel_body", &build_kernel_body,
          py::arg("kernel_name"),
          py::arg("element_type"),
          py::arg("num_input_windows"),
          py::arg("num_output_windows"),
          py::arg("kernel_ops"),
          "Build C kernel body from KernelOp list via MLIR EmitC");
}
