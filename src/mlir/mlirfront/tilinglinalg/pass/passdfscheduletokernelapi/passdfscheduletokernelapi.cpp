/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "passdfscheduletokernelapi.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace dfschedule;

namespace {

// ---------------------------------------------------------------------------
// Single pattern for dfschedule.module: convert body line-by-line (one emitc op
// per dfschedule op) then erase module. Always matches so conversion does not
// depend on nested ops being converted first (driver may try parent first).
// ---------------------------------------------------------------------------

/// Window info from WindowDefOp for emitting correct buffer/lock names.
struct WindowInfo {
    std::string pingBuffer;
    std::string pongBuffer;
    std::string acquireLock;
    std::string releaseLock;
    int32_t bufferSize = 0; // Per-window buffer size from window_def attribute
    int32_t numRounds = 0;  // Number of ping-pong rounds (0 = use bufferSize as fallback)
    std::string direction;  // "in" or "out"
};

/// dfschedule.module -> convert entire body line-by-line then erase module.
/// Always matches and does full conversion so we do not depend on nested patterns
/// firing first (conversion may try the parent before descending into regions).
struct KernelModuleToEmitCPattern : public OpConversionPattern<KernelModuleOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(KernelModuleOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        Location loc = op.getLoc();
        Block &body = op.getBody().front();

        // Build map: window symbol name -> (ping_buffer, pong_buffer, acquire_lock, release_lock)
        llvm::StringMap<WindowInfo> windowInfoMap;
        for (Operation &inner : body) {
            if (auto windowDefOp = dyn_cast<WindowDefOp>(&inner)) {
                DictionaryAttr winAttrs = windowDefOp.getWindowAttrs();
                WindowInfo info;
                if (auto a = winAttrs.getAs<SymbolRefAttr>("ping_buffer"))
                    info.pingBuffer = a.getRootReference().getValue().str();
                if (auto a = winAttrs.getAs<SymbolRefAttr>("pong_buffer"))
                    info.pongBuffer = a.getRootReference().getValue().str();
                if (auto a = winAttrs.getAs<SymbolRefAttr>("acquire_lock"))
                    info.acquireLock = a.getRootReference().getValue().str();
                if (auto a = winAttrs.getAs<SymbolRefAttr>("release_lock"))
                    info.releaseLock = a.getRootReference().getValue().str();
                if (auto a = winAttrs.getAs<IntegerAttr>("buffer_size"))
                    info.bufferSize = a.getInt();
                if (auto a = winAttrs.getAs<IntegerAttr>("num_rounds"))
                    info.numRounds = a.getInt();
                if (auto a = winAttrs.getAs<StringAttr>("direction"))
                    info.direction = a.getValue().str();
                windowInfoMap[windowDefOp.getSymName().str()] = info;
            }
        }

        // Config state (from kernel_config_def) for buffer_def emission
        int32_t bufferSize = 256;
        int32_t vectorWidth = 4;
        std::string elementType = "int32";
        std::string kernelFileName = "compute_kernel.cc";

        rewriter.setInsertionPoint(op);

        for (Operation &inner : body) {
            if (auto configOp = dyn_cast<KernelConfigDefOp>(&inner)) {
                DictionaryAttr attrs = configOp.getConfigAttrs();
                if (auto a = attrs.getAs<IntegerAttr>("buffer_size"))
                    bufferSize = a.getInt();
                if (auto a = attrs.getAs<IntegerAttr>("vector_width"))
                    vectorWidth = a.getInt();
                if (auto a = attrs.getAs<TypeAttr>("element_type")) {
                    Type t = a.getValue();
                    if (t.isInteger(32))
                        elementType = "int32";
                    else if (t.isInteger(16))
                        elementType = "int16";
                    else if (t.isInteger(8))
                        elementType = "int8";
                    else if (t.isF32())
                        elementType = "float";
                }
                if (auto a = attrs.getAs<StringAttr>("kernel_file"))
                    kernelFileName = a.getValue().str();
                rewriter.create<emitc::VerbatimOp>(loc, "#include <stdint.h>");
                rewriter.create<emitc::VerbatimOp>(loc, "#include <adf.h>");
                rewriter.create<emitc::VerbatimOp>(loc, "#include <aie_api/aie.hpp>");
                rewriter.create<emitc::VerbatimOp>(loc, "#include <aie_api/aie_adf.hpp>");
                rewriter.create<emitc::VerbatimOp>(loc, "#define FOR_READ  1");
                rewriter.create<emitc::VerbatimOp>(loc, "#define FOR_WRITE 0");
                // Emit per-window BUF_SZ defines (e.g. BUF_SZ_IN_0, BUF_SZ_OUT_0)
                // Collect window_def buffer_size attributes first
                {
                    int inIdx = 0, outIdx = 0;
                    int32_t maxBufSz = bufferSize;
                    for (Operation &winInner : body) {
                        if (auto wOp = dyn_cast<WindowDefOp>(&winInner)) {
                            DictionaryAttr wAttrs = wOp.getWindowAttrs();
                            int32_t wBufSz = bufferSize;
                            std::string wDir = "in";
                            if (auto a = wAttrs.getAs<IntegerAttr>("buffer_size"))
                                wBufSz = a.getInt();
                            if (auto a = wAttrs.getAs<StringAttr>("direction"))
                                wDir = a.getValue().str();
                            std::string macroName;
                            if (wDir == "out") {
                                macroName = "BUF_SZ_OUT_" + std::to_string(outIdx++);
                            } else {
                                macroName = "BUF_SZ_IN_" + std::to_string(inIdx++);
                            }
                            rewriter.create<emitc::VerbatimOp>(loc,
                                                               "#define " + macroName + " " + std::to_string(wBufSz));
                            if (wBufSz > maxBufSz)
                                maxBufSz = wBufSz;
                        }
                    }
                    // Backward-compat: BUF_SZ = max of all per-window sizes
                    rewriter.create<emitc::VerbatimOp>(loc, "#define BUF_SZ " + std::to_string(maxBufSz));
                }
                // ADF kernel header: window types and helpers (required for xchesscc)
                /*
                rewriter.create<emitc::VerbatimOp>(loc,
                    "typedef struct { void* ptr; int ping_acq; int ping_rel; int pong_acq; int pong_rel; int size; }
                window_internal;"); rewriter.create<emitc::VerbatimOp>(loc, "typedef void* output_window_int8;");
                rewriter.create<emitc::VerbatimOp>(loc,
                    "inline void window_init(window_internal* win, int count, void* ping, int ping_acq_lock, void* pong,
                int ping_rel_lock, int ping_size, int pong_size) { win->ptr = ping; win->ping_acq = ping_acq_lock;
                win->ping_rel = ping_rel_lock; win->size = ping_size; }"); rewriter.create<emitc::VerbatimOp>(loc,
                    "inline output_window_int8 get_output_async_window_int8(window_internal* win) { return win->ptr;
                }");
                */
                // AIEML Gen-2 lock protocol (acquire_greater_equal / release with value).
                // Reference: cardano/src/windowfunctions/window.h lines 1232-1286.
                // Input window: DMA S2MM writes data then releases lockids[1].
                //   Kernel acquire: acquire_greater_equal(lockids[1], 1)  -- wait for "data ready"
                //   Kernel release: release(lockids[0], 1)               -- signal "buffer free"
                // Output window: Kernel writes data then releases lockids[1].
                //   Kernel acquire: acquire_greater_equal(lockids[0], 1)  -- wait for "buffer free"
                //   Kernel release: release(lockids[1], 1)               -- signal "data ready"
                std::string cType = elementType + "_t"; // e.g. "int32_t", "int8_t"
                std::string winInType = "input_window_" + elementType;   // e.g. "input_window_int32"
                std::string winOutType = "output_window_" + elementType; // e.g. "output_window_int32"
                rewriter.create<emitc::VerbatimOp>(
                    loc, "inline " + cType + "* acquire_output_window(" + winOutType + "* win) {\n"
                         "  window_internal* w = (window_internal*)win;\n"
                         "  w->buffer = (window_datatype*)select(w->current_bufid, w->buffers[1], w->buffers[0]);\n"
                         "  w->head = w->ptr = (window_datatype*)select(w->current_bufid, w->heads[1], w->heads[0]);\n"
                         "  acquire_greater_equal(w->lockids[0], 1);\n"
                         "  return (" + cType + "*)w->ptr;\n"
                         "}");
                rewriter.create<emitc::VerbatimOp>(
                    loc, "inline void release_output_window(" + winOutType + "* win) {\n"
                         "  chess_memory_fence();\n"
                         "  window_internal* w = (window_internal*)win;\n"
                         "  release(w->lockids[1], 1);\n"
                         "  w->heads[w->current_bufid] = w->head;\n"
                         "  w->current_bufid = select((w->heads[1] == 0), w->current_bufid, 1 - w->current_bufid);\n"
                         "}");
                rewriter.create<emitc::VerbatimOp>(
                    loc, "inline " + cType + "* acquire_input_window(" + winInType + "* win) {\n"
                         "  window_internal* w = (window_internal*)win;\n"
                         "  w->buffer = (window_datatype*)select(w->current_bufid, w->buffers[1], w->buffers[0]);\n"
                         "  w->head = w->ptr = (window_datatype*)select(w->current_bufid, w->heads[1], w->heads[0]);\n"
                         "  acquire_greater_equal(w->lockids[1], 1);\n"
                         "  return (" + cType + "*)w->ptr;\n"
                         "}");
                rewriter.create<emitc::VerbatimOp>(
                    loc, "inline void release_input_window(" + winInType + "* win) {\n"
                         "  chess_memory_fence();\n"
                         "  window_internal* w = (window_internal*)win;\n"
                         "  release(w->lockids[0], 1);\n"
                         "  w->heads[w->current_bufid] = w->head;\n"
                         "  w->current_bufid = select((w->heads[1] == 0), w->current_bufid, 1 - w->current_bufid);\n"
                         "}");

                continue;
            }
            if (auto lockOp = dyn_cast<LockDefOp>(&inner)) {
                std::string line = "#define " + lockOp.getSymName().str() + " " + std::to_string(lockOp.getId());
                rewriter.create<emitc::VerbatimOp>(loc, line);
                continue;
            }
            if (auto bufferOp = dyn_cast<BufferDefOp>(&inner)) {
                Type t = bufferOp.getBufferTypeAttr().getValue();
                std::string vecType = "v4int32";
                int64_t bufSize = bufferSize; // fallback to global
                if (auto memref = dyn_cast<MemRefType>(t)) {
                    // Use the memref dimension as the per-buffer size
                    if (!memref.getShape().empty())
                        bufSize = memref.getShape()[0];
                    Type elem = memref.getElementType();
                    if (auto vec = dyn_cast<VectorType>(elem)) {
                        int w = vec.getShape().empty() ? 4 : vec.getShape()[0];
                        if (vec.getElementType().isInteger(32))
                            vecType = "v" + std::to_string(w) + "int32";
                        else if (vec.getElementType().isInteger(16))
                            vecType = "v" + std::to_string(w) + "int16";
                        else if (vec.getElementType().isInteger(8))
                            vecType = "v" + std::to_string(w) + "int8";
                        else if (vec.getElementType().isF32())
                            vecType = "v" + std::to_string(w) + "float";
                    }
                }
                std::string line = vecType + " " + bufferOp.getSymName().str() + "[" + std::to_string(bufSize) + "];";
                rewriter.create<emitc::VerbatimOp>(loc, line);
                continue;
            }
            if (auto windowOp = dyn_cast<WindowDefOp>(&inner)) {
                rewriter.create<emitc::VerbatimOp>(loc, "// window_def " + windowOp.getSymName().str());
                continue;
            }
            if (auto declOp = dyn_cast<KernelDeclOp>(&inner)) {
                // Emit kernel_log.h include for klog() debug logging.
                // klog writes (tag, value) pairs to fixed DM address 0x7F800.
                // Host reads via __Runtime_read_kernel_log() at DM offset 0xF800.
                rewriter.create<emitc::VerbatimOp>(loc, "#include \"kernel_log.h\"");
                rewriter.create<emitc::VerbatimOp>(loc, "#include \"" + kernelFileName + "\"");
                rewriter.create<emitc::VerbatimOp>(loc, "// kernel_decl " + declOp.getSymName().str());
                continue;
            }
            if (auto mainOp = dyn_cast<KernelMainOp>(&inner)) {
                convertMainToEmitC(rewriter, mainOp, op, windowInfoMap, elementType);
                continue;
            }
        }

        rewriter.eraseOp(op);
        return success();
    }

    void convertMainToEmitC(ConversionPatternRewriter &rewriter, KernelMainOp mainOp, KernelModuleOp kernelModuleOp,
                            const llvm::StringMap<WindowInfo> &windowInfoMap,
                            const std::string &elementType) const {
        Location loc = mainOp.getLoc();
        Block &mainBody = mainOp.getBody().front();

        // Map SSA values to C expression names (e.g. window_init result -> "window_win")
        llvm::DenseMap<Value, std::string> valueToCName;

        // Insert main func in parent block (before kernelModuleOp), not inside module body
        rewriter.setInsertionPoint(kernelModuleOp);
        auto funcType = rewriter.getFunctionType({}, rewriter.getI32Type());
        auto emitcMain = rewriter.create<emitc::FuncOp>(loc, "main", funcType);
        Block *entry = emitcMain.addEntryBlock();
        rewriter.setInsertionPointToStart(entry);

        // avoids sim crash
        rewriter.create<emitc::VerbatimOp>(loc, "#ifdef AIEHLC_KERNEL_SIM");
        rewriter.create<emitc::VerbatimOp>(loc, "volatile int sync_buffer[8];");
        rewriter.create<emitc::VerbatimOp>(loc, "#else");
        rewriter.create<emitc::VerbatimOp>(loc, "volatile static int sync_buffer[8] = {0, -1};");
        rewriter.create<emitc::VerbatimOp>(loc, "#endif");
        rewriter.create<emitc::VerbatimOp>(loc, "sync_buffer[0] = 0;");
        rewriter.create<emitc::VerbatimOp>(loc, "sync_buffer[1] = -1;");
        rewriter.create<emitc::VerbatimOp>(loc, "klog_init();");

        for (Operation &inner : mainBody) {
            if (isa<AllocSyncBufferOp>(&inner)) {
                rewriter.create<emitc::VerbatimOp>(loc, "// alloc_sync_buffer");
                continue;
            }
            if (isa<SyncBufferWriteOp>(&inner)) {
                rewriter.create<emitc::VerbatimOp>(loc, "// sync_buffer_write");
                continue;
            }
            if (isa<LogOp>(&inner)) {
                rewriter.create<emitc::VerbatimOp>(loc, "// log(...)");
                continue;
            }
            if (auto winInit = dyn_cast<WindowInitOp>(&inner)) {
                std::string winSym = winInit.getWindowRefAttr().getRootReference().getValue().str();
                valueToCName[winInit.getResult()] = "window_" + winSym;
                rewriter.create<emitc::VerbatimOp>(loc, "window_internal window_" + winSym + "[1];");
                std::string pingBuf, pongBuf, acqLock, relLock;
                auto it = windowInfoMap.find(winSym);
                if (it != windowInfoMap.end()) {
                    pingBuf = it->second.pingBuffer;
                    pongBuf = it->second.pongBuffer;
                    acqLock = it->second.acquireLock;
                    relLock = it->second.releaseLock;
                } else {
                    pingBuf = winSym + "_ping";
                    pongBuf = winSym + "_pong";
                    acqLock = "LOCK_" + winSym + "_ACQ";
                    relLock = "LOCK_" + winSym + "_REL";
                }
                // Use per-window buffer size and num_rounds from window_def attribute
                std::string winBufSzStr = "BUF_SZ"; // fallback
                std::string winNumRoundsStr = winBufSzStr; // fallback: same as bufferSize
                if (it != windowInfoMap.end() && it->second.bufferSize > 0)
                    winBufSzStr = std::to_string(it->second.bufferSize);
                if (it != windowInfoMap.end() && it->second.numRounds > 0)
                    winNumRoundsStr = std::to_string(it->second.numRounds);
                else
                    winNumRoundsStr = winBufSzStr; // backward compat: numRounds = bufferSize
                rewriter.create<emitc::VerbatimOp>(loc, "window_init(window_" + winSym + ", 1, " + pingBuf + ", " +
                                                            acqLock + ", " + pongBuf + ", " + relLock + ", " +
                                                            winBufSzStr + ", " + winNumRoundsStr + ");");
                continue;
            }
            if (auto invokeOp = dyn_cast<KernelInvokeOp>(&inner)) {
                std::string callee = invokeOp.getKernelRefAttr().getRootReference().getValue().str();
                std::string argList;
                for (Value arg : invokeOp.getArgs()) {
                    auto it = valueToCName.find(arg);
                    std::string cArg = (it != valueToCName.end()) ? it->second : "/*unknown*/";
                    if (!argList.empty())
                        argList += ", ";
                    if (isa<InputWindowType>(arg.getType())) {
                        argList += "get_input_async_window_" + elementType + "(" + cArg + ")";
                    } else {
                        argList += "get_output_async_window_" + elementType + "(" + cArg + ")";
                    }
                }
                rewriter.create<emitc::VerbatimOp>(loc, "// kernel_invoke " + callee);
                rewriter.create<emitc::VerbatimOp>(loc, callee + "(" + argList + ");");
                continue;
            }
            if (isa<DoneOp>(&inner)) {
                rewriter.create<emitc::VerbatimOp>(loc, "done();");
                continue;
            }
            if (isa<KernelReturnOp>(&inner)) {
                auto c0 = rewriter.create<emitc::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
                rewriter.create<emitc::ReturnOp>(loc, c0.getResult());
                break;
            }
        }

        if (!entry->back().hasTrait<OpTrait::IsTerminator>()) {
            auto c0 = rewriter.create<emitc::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
            rewriter.create<emitc::ReturnOp>(loc, c0.getResult());
        }
    }
};

} // namespace

namespace mlir {

void DfscheduleToKernelApiPass::runOnOperation() {
    ModuleOp moduleOp = getOperation();
    MLIRContext *ctx = &getContext();

    ConversionTarget target(*ctx);
    target.addLegalDialect<emitc::EmitCDialect, func::FuncDialect, arith::ArithDialect, memref::MemRefDialect>();
    target.addIllegalOp<KernelModuleOp>();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    RewritePatternSet patterns(ctx);
    patterns.add<KernelModuleToEmitCPattern>(typeConverter, ctx);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
        signalPassFailure();
}

} // namespace mlir
