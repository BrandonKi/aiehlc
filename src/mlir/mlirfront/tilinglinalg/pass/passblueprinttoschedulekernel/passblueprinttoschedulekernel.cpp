/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "passblueprinttoschedulekernel.h"
#include "dfscheblueprintmanager.h"
#include "dfschedulemanager.h"
#include "hw/ResourceManager.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

using namespace mlir;
using namespace dfscheblueprint;
using namespace dfschedule;

namespace {

// Trace a Value back through the SSA chain to find the originating function
// argument index.  This walks through bufferization.to_tensor,
// routing.routingcreatescheduletensor, routing.partitiontensor,
// routing.routingextract_data, dfscheblueprint.declare_data, and
// scf.execute_region captures until it reaches a BlockArgument of a func::FuncOp.
// Returns the argument index (0-based), or -1 if the chain cannot be resolved.
static int traceToFuncArgIndex(Value v) {
    // Walk up the def chain, max 20 hops to avoid infinite loops
    for (int depth = 0; depth < 20; ++depth) {
        // If v is a block argument of a func op, we found it
        if (auto blockArg = dyn_cast<BlockArgument>(v)) {
            if (auto funcOp = dyn_cast<func::FuncOp>(blockArg.getOwner()->getParentOp()))
                return static_cast<int>(blockArg.getArgNumber());
            break; // can't follow further through a block argument
        }
        Operation *defOp = v.getDefiningOp();
        if (!defOp)
            break;

        // routing ops that forward data through operand 0
        if (defOp->getName().getStringRef() == "routing.routingextract_data" ||
            defOp->getName().getStringRef() == "routing.routingcreatescheduletensor" ||
            defOp->getName().getStringRef() == "routing.partitiontensor") {
            v = defOp->getOperand(0);
            continue;
        }
        // bufferization.to_tensor %memref -> follow %memref (operand 0)
        if (defOp->getName().getStringRef() == "bufferization.to_tensor") {
            v = defOp->getOperand(0);
            continue;
        }
        // dfscheblueprint.declare_data -> follow init_tensor (operand 0)
        if (defOp->getName().getStringRef() == "dfscheblueprint.declare_data") {
            v = defOp->getOperand(0);
            continue;
        }
        // tensor.extract_slice -> follow source tensor (operand 0)
        if (defOp->getName().getStringRef() == "tensor.extract_slice") {
            v = defOp->getOperand(0);
            continue;
        }
        // Generic single-result ops that just forward operand 0
        if (defOp->getNumOperands() > 0) {
            v = defOp->getOperand(0);
            continue;
        }
        break;
    }
    return -1; // unable to resolve
}

// ============================================================================
// Fake Resource Manager for Lock and BD ID allocation
// ============================================================================
class KernelResourceManager {
  public:
    // Lock base offset for kernel-side lock intrinsics.
    // On AIE2/AIE2PS, core tile local locks start at hardware ID 48.
    // The host side uses XAie_LockSetValue with lock IDs 0,1,2,... which the
    // driver maps to the tile's memory module locks.  From the kernel's
    // perspective (acquire_greater_equal / release intrinsics), these same
    // locks are accessed at ID 48+N where N is the host-side lock ID.
    static constexpr int32_t LOCK_BASE = 48;

    KernelResourceManager() : nextBdId(0), nextLockId(0), nextLockOffset(0) {}

    // Allocate next BD ID (0, 1 for ping-pong)
    int32_t allocateBdId() { return nextBdId++; }

    // Allocate next Lock ID
    int64_t allocateLockId() { return nextLockId++; }

    // Allocate a lock pair for a window (input or output).
    // Returns sequential lock IDs: lock 0, lock 1, lock 2, ...
    // acquireLockId gets the first lock, releaseLockId gets the second.
    void allocateLockPair(int32_t &acquireLockId, int32_t &releaseLockId) {
        acquireLockId = LOCK_BASE + nextLockOffset++;
        releaseLockId = LOCK_BASE + nextLockOffset++;
    }

    // Convenience wrappers for input/output (both use same sequential allocation)
    int32_t allocateInputAcquireLock() { return LOCK_BASE + nextLockOffset++; }
    int32_t allocateInputReleaseLock() { return LOCK_BASE + nextLockOffset++; }
    int32_t allocateOutputAcquireLock() { return LOCK_BASE + nextLockOffset++; }
    int32_t allocateOutputReleaseLock() { return LOCK_BASE + nextLockOffset++; }

    // Reset for new kernel
    void reset() {
        nextBdId = 0;
        nextLockId = 0;
        nextLockOffset = 0;
    }

  private:
    int32_t nextBdId;
    int64_t nextLockId;
    int32_t nextLockOffset; // Sequential offset from LOCK_BASE (48)
};

// Generic template function to look up any operation by symbol reference
// This function searches for an operation of type OpTy with a matching symbol name
// The search starts in the same block as rootOp and then searches parent regions
template <typename OpTy> static OpTy lookupSymbolOp(Operation *rootOp, SymbolRefAttr target) {
    StringRef targetName = target.getRootReference().getValue();

    // First, search in the same block as the rootOp
    Block *parentBlock = rootOp->getBlock();
    if (parentBlock) {
        for (Operation &op : *parentBlock) {
            if (auto targetOp = dyn_cast<OpTy>(&op)) {
                if (targetOp.getSymName() == targetName) {
                    return targetOp;
                }
            }
        }
    }

    // If not found, try searching in parent regions (for nested structures)
    Operation *parentOp = rootOp->getParentOp();
    while (parentOp) {
        for (Region &region : parentOp->getRegions()) {
            for (Block &block : region) {
                for (Operation &op : block) {
                    if (auto targetOp = dyn_cast<OpTy>(&op)) {
                        if (targetOp.getSymName() == targetName) {
                            return targetOp;
                        }
                    }
                }
            }
        }
        parentOp = parentOp->getParentOp();
    }

    return nullptr;
}

// Helper function to look up TileGroupOp by symbol reference (wrapper for backward compatibility)
static dfscheblueprint::TileGroupOp lookupTileGroup(Operation *rootOp, SymbolRefAttr target) {
    return lookupSymbolOp<dfscheblueprint::TileGroupOp>(rootOp, target);
}

// Unified template pattern to erase dfscheblueprint operations
// FlowConfigOp is just erased since FlowTransferConversion reads its attributes
// and generates all the DMA BD configuration logic
template <typename OpTy> struct EraseOpPattern : public OpConversionPattern<OpTy> {
    using OpConversionPattern<OpTy>::OpConversionPattern;

    LogicalResult matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        rewriter.eraseOp(op);
        return success();
    }
};

// Helper function to check if dskernel_receiver already exists in the module
static bool hasDSKernelReceiver(Operation *rootOp, StringRef kernelName) {
    // Find the module-level operation
    Operation *moduleOp = rootOp;
    while (moduleOp->getParentOp()) {
        moduleOp = moduleOp->getParentOp();
    }

    // Search for existing dskernel_receiver with the given name
    for (Region &region : moduleOp->getRegions()) {
        for (Block &block : region) {
            for (Operation &op : block) {
                if (auto receiver = dyn_cast<dfschedule::DSKernelReceiverOp>(&op)) {
                    if (receiver.getSymName() == kernelName) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

// Helper function to check if kernel module already exists in the module
static bool hasKernelModule(Operation *rootOp, StringRef moduleName) {
    // Find the module-level operation
    Operation *moduleOp = rootOp;
    while (moduleOp->getParentOp()) {
        moduleOp = moduleOp->getParentOp();
    }

    // Search for existing kernel module with the given name
    for (Region &region : moduleOp->getRegions()) {
        for (Block &block : region) {
            for (Operation &op : block) {
                if (auto kernelModule = dyn_cast<dfschedule::KernelModuleOp>(&op)) {
                    if (kernelModule.getSymName() == moduleName) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

// Helper function to get the module-level insertion point
static Operation *getModuleOp(Operation *rootOp) {
    Operation *moduleOp = rootOp;
    while (moduleOp->getParentOp()) {
        moduleOp = moduleOp->getParentOp();
    }
    return moduleOp;
}

// =============================================================================
// Kernel Module IR Generation
// =============================================================================
// Generates a general-purpose kernel module IR that can be lowered by different
// passes to produce either:
//   - LegacyKernelPass  -> adfkernellegacy.cc style (loop inside kernel)
//   - AdfKernelPass     -> adfkernel.cc style (loop in wrapper)
//
// The generated IR uses abstract operations:
//   - dfschedule.module           : top-level kernel module
//   - dfschedule.kernel_config    : kernel metadata
//   - dfschedule.lock_def         : named lock definitions
//   - dfschedule.buffer           : named buffer declarations
//   - dfschedule.window           : abstract window (ping/pong + locks)
//   - dfschedule.kernel_decl      : kernel signature with iteration_style
//   - dfschedule.main             : entry point with kernel_invoke
// =============================================================================

// Structure to hold individual kernel parameter info (input/output window)
struct KernelParamInfo {
    std::string windowName;     // e.g., "window_in_0", "window_out_0"
    std::string bufferPingName; // e.g., "buf_in_ping_0"
    std::string bufferPongName; // e.g., "buf_in_pong_0"
    bool isInput;               // true = input parameter, false = output parameter
    int32_t acquireLockId;      // Lock ID for acquire operation
    int32_t releaseLockId;      // Lock ID for release operation
    Type elementType;           // Element type of the buffer
    int64_t bufferSize;         // Size of the buffer
    int32_t vectorWidth;        // Vector width (e.g., 4 for v4int32)
    int32_t numRounds = 0;      // Number of ping-pong rounds (ppDepth)
    uint32_t pingAddress = 0;   // BCF symbol address for ping buffer (from CoreMemAllocator)
    uint32_t pongAddress = 0;   // BCF symbol address for pong buffer (from CoreMemAllocator)
    int funcArgIndex = -1;      // Function argument index (0=A, 1=B, 2=C)
    bool singleBuffer = false;  // Single-buffer (no pong) mode, e.g. spatial-halo IFM slab
                                // that is received once per invocation (numRounds==1).
};

// Structure to hold kernel generation parameters
struct KernelGenParams {
    StringRef kernelName;        // Wrapper function name, e.g., "dskernel_receiver"
    StringRef computeKernelName; // Actual compute kernel name, e.g., "perf"
    StringRef kernelFile;        // e.g., "perf.cc"
    int64_t bufferSize;          // e.g., 256
    Type elementType;            // e.g., i32
    int32_t vectorWidth;         // e.g., 4
    StringRef iterationStyle;    // "internal" or "external"

    // Lock IDs (base values, will be offset for ping/pong) - used when params is empty
    int32_t inputAcquireLockId;  // e.g., 48
    int32_t inputReleaseLockId;  // e.g., 49
    int32_t outputAcquireLockId; // e.g., 51
    int32_t outputReleaseLockId; // e.g., 50

    // Dynamic parameter list - when non-empty, overrides the hardcoded lock IDs
    SmallVector<KernelParamInfo> kernelParams;
};

// Generate dfschedule.module with kernel_config, locks, buffers, windows, kernel_decl, and main
// This is the general-purpose kernel module that can be lowered by different passes
static void generateKernelModule(ConversionPatternRewriter &rewriter, Location loc, Operation *insertBeforeOp,
                                 const KernelGenParams &params, RankedTensorType tensorType) {

    // Create module name from kernel name
    std::string moduleName = "kernel_driver_" + params.kernelName.str();

    // Check if kernel module already exists - skip if duplicate
    if (hasKernelModule(insertBeforeOp, moduleName)) {
        return;
    }

    // Save current insertion point
    OpBuilder::InsertionGuard guard(rewriter);

    // Find module and insert at module level
    Operation *rootModuleOp = getModuleOp(insertBeforeOp);
    Block &moduleBlock = rootModuleOp->getRegions().front().front();

    // Check if block has a terminator, insert before it; otherwise insert at end
    if (!moduleBlock.empty() && moduleBlock.back().hasTrait<OpTrait::IsTerminator>()) {
        rewriter.setInsertionPoint(&moduleBlock.back());
    } else {
        rewriter.setInsertionPointToEnd(&moduleBlock);
    }

    // Create the dfschedule.module operation
    auto kernelModuleOp = rewriter.create<dfschedule::KernelModuleOp>(loc, rewriter.getStringAttr(moduleName));

    // Create the body block for the module
    Block *body = &kernelModuleOp.getBody().emplaceBlock();
    rewriter.setInsertionPointToStart(body);

    // =========================================================================
    // 1. Kernel Config (metadata)
    // =========================================================================
    NamedAttrList configAttrs;
    configAttrs.append("kernel_name", rewriter.getStringAttr(params.computeKernelName));
    configAttrs.append("kernel_file", rewriter.getStringAttr(params.kernelFile));
    configAttrs.append("buffer_size", rewriter.getI32IntegerAttr(params.bufferSize));
    configAttrs.append("element_type", TypeAttr::get(params.elementType));
    configAttrs.append("vector_width", rewriter.getI32IntegerAttr(params.vectorWidth));

    rewriter.create<dfschedule::KernelConfigDefOp>(loc, rewriter.getStringAttr("config"),
                                                   rewriter.getDictionaryAttr(configAttrs));

    // =========================================================================
    // 2-5. Generate Locks, Buffers, Windows, and Kernel Declaration
    // =========================================================================
    // Use dynamic parameters if provided, otherwise fall back to hardcoded values

    SmallVector<Attribute> inputWindowRefs;
    SmallVector<Attribute> outputWindowRefs;
    SmallVector<std::string> allWindowNames; // Track window names for main block

    if (!params.kernelParams.empty()) {
        // === DYNAMIC PARAMETER MODE ===
        // Generate locks, buffers, and windows for each kernel parameter

        for (const auto &paramInfo : params.kernelParams) {
            // Lock definitions for this parameter
            std::string acqLockName = "LOCK_" + paramInfo.windowName + "_ACQ";
            std::string relLockName = "LOCK_" + paramInfo.windowName + "_REL";

            auto acqLockOp = rewriter.create<dfschedule::LockDefOp>(
                loc, rewriter.getStringAttr(acqLockName), rewriter.getI32IntegerAttr(paramInfo.acquireLockId));
            acqLockOp->setAttr("init_value", rewriter.getI32IntegerAttr(paramInfo.isInput ? 2 : 0));

            rewriter.create<dfschedule::LockDefOp>(loc, rewriter.getStringAttr(relLockName),
                                                   rewriter.getI32IntegerAttr(paramInfo.releaseLockId));

            // Buffer definitions (ping/pong pair)
            Type elemType = paramInfo.elementType ? paramInfo.elementType : params.elementType;
            auto vectorType = VectorType::get({paramInfo.vectorWidth}, elemType);
            auto localMemRefType =
                MemRefType::get({paramInfo.bufferSize}, vectorType, AffineMap(), rewriter.getStringAttr("LOCAL"));

            auto pingBufDef = rewriter.create<dfschedule::BufferDefOp>(
                loc, rewriter.getStringAttr(paramInfo.bufferPingName), TypeAttr::get(localMemRefType));
            // Annotate buffer addresses from CoreMemAllocator (for BCF generation)
            if (paramInfo.pingAddress != 0) {
                pingBufDef->setAttr("address", rewriter.getI64IntegerAttr(paramInfo.pingAddress));
            }
            // Single-buffer ports (bufferPongName == bufferPingName) declare only
            // one global array; the window's pong slot reuses the ping buffer.
            if (!paramInfo.singleBuffer) {
                auto pongBufDef = rewriter.create<dfschedule::BufferDefOp>(
                    loc, rewriter.getStringAttr(paramInfo.bufferPongName), TypeAttr::get(localMemRefType));
                if (paramInfo.pongAddress != 0) {
                    pongBufDef->setAttr("address", rewriter.getI64IntegerAttr(paramInfo.pongAddress));
                }
            }

            // Window definition
            NamedAttrList winAttrs;
            winAttrs.append("direction", rewriter.getStringAttr(paramInfo.isInput ? "in" : "out"));
            winAttrs.append("ping_buffer", SymbolRefAttr::get(rewriter.getContext(), paramInfo.bufferPingName));
            winAttrs.append("pong_buffer", SymbolRefAttr::get(rewriter.getContext(), paramInfo.bufferPongName));
            winAttrs.append("acquire_lock", SymbolRefAttr::get(rewriter.getContext(), acqLockName));
            winAttrs.append("release_lock", SymbolRefAttr::get(rewriter.getContext(), relLockName));
            winAttrs.append("buffer_size", rewriter.getI32IntegerAttr(paramInfo.bufferSize));
            if (paramInfo.numRounds > 0)
                winAttrs.append("num_rounds", rewriter.getI32IntegerAttr(paramInfo.numRounds));
            winAttrs.append("async", rewriter.getBoolAttr(true));

            rewriter.create<dfschedule::WindowDefOp>(loc, rewriter.getStringAttr(paramInfo.windowName),
                                                     rewriter.getDictionaryAttr(winAttrs));

            // Track for kernel declaration
            if (paramInfo.isInput) {
                inputWindowRefs.push_back(SymbolRefAttr::get(rewriter.getContext(), paramInfo.windowName));
            } else {
                outputWindowRefs.push_back(SymbolRefAttr::get(rewriter.getContext(), paramInfo.windowName));
            }
            allWindowNames.push_back(paramInfo.windowName);
        }
    } else {
        // === HARDCODED MODE (backward compatibility) ===
        // Generate fixed input/output locks, buffers, and windows

        // Lock definitions (acquire locks get init_value=2 for ping-pong)
        auto winPingAcqLock = rewriter.create<dfschedule::LockDefOp>(
            loc, rewriter.getStringAttr("LOCK_win_ping_ACQ"), rewriter.getI32IntegerAttr(params.inputAcquireLockId));
        winPingAcqLock->setAttr("init_value", rewriter.getI32IntegerAttr(2));

        rewriter.create<dfschedule::LockDefOp>(loc, rewriter.getStringAttr("LOCK_win_pong_REL"),
                                               rewriter.getI32IntegerAttr(params.inputReleaseLockId));

        auto outPingAcqLock = rewriter.create<dfschedule::LockDefOp>(
            loc, rewriter.getStringAttr("LOCK_out_ping_ACQ"), rewriter.getI32IntegerAttr(params.outputAcquireLockId));
        outPingAcqLock->setAttr("init_value", rewriter.getI32IntegerAttr(0));

        rewriter.create<dfschedule::LockDefOp>(loc, rewriter.getStringAttr("LOCK_out_pong_REL"),
                                               rewriter.getI32IntegerAttr(params.outputReleaseLockId));

        // Buffer definitions
        auto vectorType = VectorType::get({params.vectorWidth}, params.elementType);
        auto localMemRefType =
            MemRefType::get({params.bufferSize}, vectorType, AffineMap(), rewriter.getStringAttr("LOCAL"));

        rewriter.create<dfschedule::BufferDefOp>(loc, rewriter.getStringAttr("win_ping"),
                                                 TypeAttr::get(localMemRefType));
        rewriter.create<dfschedule::BufferDefOp>(loc, rewriter.getStringAttr("win_pong"),
                                                 TypeAttr::get(localMemRefType));
        rewriter.create<dfschedule::BufferDefOp>(loc, rewriter.getStringAttr("out_ping"),
                                                 TypeAttr::get(localMemRefType));
        rewriter.create<dfschedule::BufferDefOp>(loc, rewriter.getStringAttr("out_pong"),
                                                 TypeAttr::get(localMemRefType));

        // Window definitions
        NamedAttrList winInAttrs;
        winInAttrs.append("direction", rewriter.getStringAttr("in"));
        winInAttrs.append("ping_buffer", SymbolRefAttr::get(rewriter.getContext(), "win_ping"));
        winInAttrs.append("pong_buffer", SymbolRefAttr::get(rewriter.getContext(), "win_pong"));
        winInAttrs.append("acquire_lock", SymbolRefAttr::get(rewriter.getContext(), "LOCK_win_ping_ACQ"));
        winInAttrs.append("release_lock", SymbolRefAttr::get(rewriter.getContext(), "LOCK_win_pong_REL"));
        winInAttrs.append("buffer_size", rewriter.getI32IntegerAttr(params.bufferSize));
        winInAttrs.append("async", rewriter.getBoolAttr(true));
        rewriter.create<dfschedule::WindowDefOp>(loc, rewriter.getStringAttr("window_in"),
                                                 rewriter.getDictionaryAttr(winInAttrs));

        NamedAttrList winOutAttrs;
        winOutAttrs.append("direction", rewriter.getStringAttr("out"));
        winOutAttrs.append("ping_buffer", SymbolRefAttr::get(rewriter.getContext(), "out_ping"));
        winOutAttrs.append("pong_buffer", SymbolRefAttr::get(rewriter.getContext(), "out_pong"));
        winOutAttrs.append("acquire_lock", SymbolRefAttr::get(rewriter.getContext(), "LOCK_out_ping_ACQ"));
        winOutAttrs.append("release_lock", SymbolRefAttr::get(rewriter.getContext(), "LOCK_out_pong_REL"));
        winOutAttrs.append("buffer_size", rewriter.getI32IntegerAttr(params.bufferSize));
        winOutAttrs.append("async", rewriter.getBoolAttr(true));
        rewriter.create<dfschedule::WindowDefOp>(loc, rewriter.getStringAttr("window_out"),
                                                 rewriter.getDictionaryAttr(winOutAttrs));

        inputWindowRefs.push_back(SymbolRefAttr::get(rewriter.getContext(), "window_in"));
        outputWindowRefs.push_back(SymbolRefAttr::get(rewriter.getContext(), "window_out"));
        allWindowNames.push_back("window_in");
        allWindowNames.push_back("window_out");
    }

    // =========================================================================
    // 5. Kernel Declaration
    // =========================================================================
    NamedAttrList kernelDeclAttrs;
    kernelDeclAttrs.append("inputs", rewriter.getArrayAttr(inputWindowRefs));
    kernelDeclAttrs.append("outputs", rewriter.getArrayAttr(outputWindowRefs));
    kernelDeclAttrs.append("iteration_style", rewriter.getStringAttr(params.iterationStyle));

    rewriter.create<dfschedule::KernelDeclOp>(loc, rewriter.getStringAttr(params.computeKernelName),
                                              rewriter.getDictionaryAttr(kernelDeclAttrs));

    // =========================================================================
    // 6. Main Entry Point
    // =========================================================================
    auto mainOp = rewriter.create<dfschedule::KernelMainOp>(loc, rewriter.getStringAttr("main"));

    // Create the body block for main
    Block *mainBody = &mainOp.getBody().emplaceBlock();
    rewriter.setInsertionPointToStart(mainBody);

    // --- Sync buffer ---
    auto syncBufferType = dfschedule::SyncBufferType::get(rewriter.getContext());
    auto syncBufferOp =
        rewriter.create<dfschedule::AllocSyncBufferOp>(loc, syncBufferType, rewriter.getI32IntegerAttr(8));

    // Reset end signal: sync_buffer[0] = 0
    auto c0_i32 = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
    rewriter.create<dfschedule::SyncBufferWriteOp>(loc, syncBufferOp.getResult(), c0_i32.getResult(),
                                                   rewriter.getI32IntegerAttr(0));

    // --- Debug logging ---
    auto c1_i32 = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
    rewriter.create<dfschedule::LogOp>(loc, c1_i32.getResult());

    // --- Window initialization and kernel args ---
    SmallVector<Value> kernelArgs;

    if (!params.kernelParams.empty()) {
        // Dynamic: Initialize each window based on kernelParams
        for (const auto &paramInfo : params.kernelParams) {
            Type elemType = paramInfo.elementType ? paramInfo.elementType : params.elementType;
            if (paramInfo.isInput) {
                auto inputWindowType = dfschedule::InputWindowType::get(rewriter.getContext(), elemType);
                auto winPtrOp = rewriter.create<dfschedule::WindowInitOp>(
                    loc, inputWindowType, SymbolRefAttr::get(rewriter.getContext(), paramInfo.windowName));
                kernelArgs.push_back(winPtrOp.getResult());
            } else {
                auto outputWindowType = dfschedule::OutputWindowType::get(rewriter.getContext(), elemType);
                auto outPtrOp = rewriter.create<dfschedule::WindowInitOp>(
                    loc, outputWindowType, SymbolRefAttr::get(rewriter.getContext(), paramInfo.windowName));
                kernelArgs.push_back(outPtrOp.getResult());
            }
        }
    } else {
        // Hardcoded: Initialize fixed input/output windows
        auto inputWindowType = dfschedule::InputWindowType::get(rewriter.getContext(), params.elementType);
        auto winPtrOp = rewriter.create<dfschedule::WindowInitOp>(
            loc, inputWindowType, SymbolRefAttr::get(rewriter.getContext(), "window_in"));

        auto outputWindowType = dfschedule::OutputWindowType::get(rewriter.getContext(), params.elementType);
        auto outPtrOp = rewriter.create<dfschedule::WindowInitOp>(
            loc, outputWindowType, SymbolRefAttr::get(rewriter.getContext(), "window_out"));

        kernelArgs.push_back(winPtrOp.getResult());
        kernelArgs.push_back(outPtrOp.getResult());
    }

    // --- Kernel invocation ---
    rewriter.create<dfschedule::KernelInvokeOp>(
        loc, SymbolRefAttr::get(rewriter.getContext(), params.computeKernelName), kernelArgs);

    // --- Signal completion ---
    rewriter.create<dfschedule::DoneOp>(loc);

    // --- Return ---
    rewriter.create<dfschedule::KernelReturnOp>(loc);
}

// Forward declarations for functions used in generateDSKernelReceiver
static dfscheblueprint::FlowConfigOp lookupFlowConfig(Operation *rootOp, SymbolRefAttr target);
static SmallVector<KernelParamInfo> analyzeKernelParams(Operation *rootOp, KernelResourceManager &resourceMgr,
                                                        Type defaultElementType, int64_t defaultBufferSize,
                                                        int32_t defaultVectorWidth, double bufferRatio,
                                                        int64_t maxPingPongBytes);

// Generate dfschedule.dskernel_receiver function (legacy style)
// This is kept for backward compatibility and will call generateKernelModule internally
// Parameters:
//   - kernelName: symbol name for the kernel (same as load_kernel_group callee)
//   - tensorType: the tensor type for data
//   - bufferLen: buffer length for DMA BD
//   - basePacketId: base packet ID from FlowTransferOp
//   - coreChannel: DMA channel from core FlowConfig
//   - flowIndex: unique index for this flow, used to access per-flow DMA configs
static void generateDSKernelReceiver(ConversionPatternRewriter &rewriter, Location loc, Operation *insertBeforeOp,
                                     StringRef kernelName, RankedTensorType tensorType, int64_t bufferLen,
                                     uint32_t basePacketId, int64_t coreChannel, uint32_t flowIndex,
                                     KernelResourceManager &resourceMgr, double bufferRatio, int64_t maxPingPongBytes) {

    // Build kernel generation parameters
    KernelGenParams params;
    params.kernelName = kernelName;              // Wrapper function name (e.g., "dskernel_receiver")
    params.computeKernelName = "computekernel";  // Actual compute kernel name
    params.kernelFile = "computekernel.cc";      // Kernel source file
    params.bufferSize = 256;                     // Will be overridden by analyzeKernelParams
    params.elementType = rewriter.getI32Type();
    params.vectorWidth = 4;
    params.iterationStyle = "internal"; // Legacy style: loop inside kernel

    // Lock IDs (fallback - used when kernelParams is empty)
    // Sequential allocation from lock base 48: input pair (48,49), output pair (50,51)
    params.inputAcquireLockId = 48;  // LOCK_win_ping_ACQ  (lock 0)
    params.inputReleaseLockId = 49;  // LOCK_win_pong_REL  (lock 1)
    params.outputAcquireLockId = 50; // LOCK_out_ping_ACQ  (lock 2)
    params.outputReleaseLockId = 51; // LOCK_out_pong_REL  (lock 3)

    // Dynamically analyze flow_transfer operations to determine kernel parameters
    // Walk from the module root to collect all shim<->core data flows
    Operation *rootOp = getModuleOp(insertBeforeOp);
    params.kernelParams = analyzeKernelParams(rootOp, resourceMgr, params.elementType, params.bufferSize,
                                              params.vectorWidth, bufferRatio, maxPingPongBytes);

    // Update params.bufferSize to the max of all per-window sizes
    // (used for kernel_config_def's backward-compat BUF_SZ global define)
    if (!params.kernelParams.empty()) {
        int64_t maxBufSize = 0;
        for (auto &kp : params.kernelParams)
            maxBufSize = std::max(maxBufSize, kp.bufferSize);
        params.bufferSize = maxBufSize;
        // Update element type from the actual tensor (e.g., i8 instead of default i32)
        if (params.kernelParams[0].elementType)
            params.elementType = params.kernelParams[0].elementType;
    }

    // Generate the kernel module IR
    generateKernelModule(rewriter, loc, insertBeforeOp, params, tensorType);
}

static dfscheblueprint::DataSliceOp lookupDataSlice(Operation *rootOp, SymbolRefAttr target) {
    return lookupSymbolOp<dfscheblueprint::DataSliceOp>(rootOp, target);
}

// Helper function to look up FlowConfigOp by symbol reference (wrapper for backward compatibility)
static dfscheblueprint::FlowConfigOp lookupFlowConfig(Operation *rootOp, SymbolRefAttr target) {
    return lookupSymbolOp<dfscheblueprint::FlowConfigOp>(rootOp, target);
}

// Helper function to trace a value through the IR to find FlowConfigOp that uses it
// This follows the value chain from declare_data result to FlowConfig.view
static dfscheblueprint::FlowConfigOp traceToFlowConfig(Value dataValue, Operation *rootOp) {
    // Track visited values to avoid infinite loops
    SmallPtrSet<Value, 16> visited;
    SmallVector<Value, 16> worklist;
    worklist.push_back(dataValue);

    while (!worklist.empty()) {
        Value currentValue = worklist.pop_back_val();
        if (visited.contains(currentValue))
            continue;
        visited.insert(currentValue);

        // Check all users of this value
        for (Operation *user : currentValue.getUsers()) {
            // Check if user is a FlowConfigOp and this value is its view operand
            if (auto flowConfig = dyn_cast<dfscheblueprint::FlowConfigOp>(user)) {
                if (flowConfig.getView() == currentValue) {
                    return flowConfig;
                }
            }

            // For other ops, add their results to the worklist to continue tracing
            for (Value result : user->getResults()) {
                if (!visited.contains(result)) {
                    worklist.push_back(result);
                }
            }
        }
    }

    return nullptr; // No FlowConfigOp found using this data
}

// Helper function to find FlowTransferOp that references a given FlowConfigOp
// Searches for flow_transfer ops where 'from' or 'to' matches the FlowConfig's symbol
static dfscheblueprint::FlowTransferOp
findFlowTransferFor(dfscheblueprint::FlowConfigOp flowConfig, Operation *rootOp,
                    bool &isFromConfig) { // Output: true if flowConfig is the 'from', false if 'to'

    // Get the symbol name of the FlowConfigOp
    StringRef configSymbol = flowConfig.getSymName();
    if (configSymbol.empty())
        return nullptr;

    dfscheblueprint::FlowTransferOp result = nullptr;
    isFromConfig = false;

    // Walk all flow_transfer operations to find one that references this FlowConfig
    rootOp->walk([&](dfscheblueprint::FlowTransferOp transferOp) {
        if (result)
            return; // Already found

        // Check if 'from' matches
        SymbolRefAttr fromRef = transferOp.getFrom();
        if (fromRef && fromRef.getRootReference().getValue() == configSymbol) {
            result = transferOp;
            isFromConfig = true;
            return;
        }

        // Check if 'to' matches
        SymbolRefAttr toRef = transferOp.getTo();
        if (toRef && toRef.getRootReference().getValue() == configSymbol) {
            result = transferOp;
            isFromConfig = false;
            return;
        }
    });

    return result;
}

// Analyze all declare_data operations to determine kernel parameters dynamically
// Returns a list of KernelParamInfo based on actual data flows:
// - For each declare_data, trace value chain to find FlowConfigOp
// - Find flow_transfer referencing that FlowConfig
// - shim -> core: Input parameter
// - core -> shim: Output parameter
// - No flow_transfer or core -> core: Skip (not a kernel parameter)
static SmallVector<KernelParamInfo> analyzeKernelParams(Operation *rootOp, KernelResourceManager &resourceMgr,
                                                        Type defaultElementType, int64_t defaultBufferSize,
                                                        int32_t defaultVectorWidth, double bufferRatio,
                                                        int64_t maxPingPongBytes) {

    SmallVector<KernelParamInfo> params;

    // Track which declare_data results we've already created a window for.
    // With row partitioning, the same declare_data feeds multiple flow_transfers
    // (one per row partition) — we keep only the first.
    llvm::DenseSet<Value> processedDeclareData;

    // Walk all declare_data operations in the module
    rootOp->walk([&](dfscheblueprint::DeclareDataOp declareDataOp) {
        // Get the result value of declare_data
        Value dataValue = declareDataOp.getResult();

        // Trace the value chain to find FlowConfigOp that uses this data
        auto flowConfig = traceToFlowConfig(dataValue, rootOp);
        if (!flowConfig) {
            return; // No FlowConfig found - not a kernel parameter
        }

        // Find flow_transfer that references this FlowConfig
        bool isFromConfig = false;
        auto flowTransfer = findFlowTransferFor(flowConfig, rootOp, isFromConfig);
        if (!flowTransfer) {
            return; // No flow_transfer found - not a kernel parameter
        }

        // Get the 'from' and 'to' FlowConfigOps from the flow_transfer
        SymbolRefAttr fromRef = flowTransfer.getFrom();
        auto fromFlowConfig = lookupFlowConfig(flowTransfer.getOperation(), fromRef);
        if (!fromFlowConfig) {
            return;
        }

        SymbolRefAttr toRef = flowTransfer.getTo();
        auto toFlowConfig = lookupFlowConfig(flowTransfer.getOperation(), toRef);
        if (!toFlowConfig) {
            return;
        }

        // Determine direction based on shim/core types
        auto fromType = fromFlowConfig.getType();
        auto toType = toFlowConfig.getType();

        bool isInput = false;
        bool isValidParam = false;

        if (fromType && *fromType == "shim" && toType && *toType == "core") {
            // shim -> core: Input parameter (MM2S - data flows into kernel)
            isInput = true;
            isValidParam = true;
        } else if (fromType && *fromType == "core" && toType && *toType == "shim") {
            // core -> shim: Output parameter (S2MM - data flows out of kernel)
            isInput = false;
            isValidParam = true;
        }
        // Note: core -> core is skipped (inter-tile transfer)

        if (isValidParam) {
            // Deduplicate row partitions: within one declare_data's routing
            // region there may be multiple flow_transfers (one per row partition).
            // We only need one window per declare_data result.
            if (processedDeclareData.contains(dataValue)) {
                return; // Already created a window for this declare_data
            }
            processedDeclareData.insert(dataValue);

            KernelParamInfo paramInfo;
            paramInfo.isInput = isInput;

            // Trace from declare_data's init_tensor operand back to the
            // function argument to determine canonical ordering.
            paramInfo.funcArgIndex = traceToFuncArgIndex(declareDataOp.getOperand());

            // Temporary placeholders - will be reassigned after sorting
            paramInfo.windowName = "";
            paramInfo.bufferPingName = "";
            paramInfo.bufferPongName = "";
            paramInfo.acquireLockId = 0;
            paramInfo.releaseLockId = 0;

            // Get element type and partition size from the core FlowConfig's view
            // (not the full tensor from declare_data, which is the root tensor)
            auto coreFlowConfig = isInput ? toFlowConfig : fromFlowConfig;
            Value viewValue = coreFlowConfig.getView();
            Type viewType = viewValue ? viewValue.getType() : Type();
            if (auto tensorType = dyn_cast_or_null<RankedTensorType>(viewType)) {
                paramInfo.elementType = tensorType.getElementType();
                // Calculate partition size from the flow's view tensor shape
                int64_t partitionSize = 1;
                for (int64_t dim : tensorType.getShape()) {
                    partitionSize *= dim;
                }
                // Get number of core tiles from the core TileGroup
                int64_t numCoreTiles = 1;
                if (auto coreTG = lookupTileGroup(coreFlowConfig.getOperation(), coreFlowConfig.getTarget())) {
                    numCoreTiles = coreTG.getTiles().size();
                    if (numCoreTiles <= 0)
                        numCoreTiles = 1;
                }
                // Per-core data depends on transfer type:
                // many_to_one (gather/output): each core produces partitionSize / numCoreTiles.
                // one_to_many (broadcast/input): each core receives the full partition.
                StringRef transferType = flowTransfer.getType();
                int64_t perCoreSize = partitionSize;
                if (transferType == "many_to_one")
                    perCoreSize = partitionSize / numCoreTiles;

                // K-round adjustment: for input flows, use per-k-round data
                // size so pingPongBufSize matches the kernel's buf_sz (which
                // is computed from effectiveK, not fullK).
                int64_t perCoreSizeForBuf = perCoreSize;

                // M/N-round adjustment for output: when tile_m < tileRows
                // or tile_n < tileCols, the per-round output buffer is one
                // sub-tile (tileM * tileN_sub), not the full tile output.
                // Divide by mRounds * nRounds so pingPongBufSize matches
                // one kernel output window.
                if (!paramInfo.isInput) {
                    auto moduleOp2 = declareDataOp->getParentOfType<ModuleOp>();
                    if (moduleOp2) {
                        auto tileMAttr = moduleOp2->getAttrOfType<IntegerAttr>("routing.tile_m");
                        auto tileRowsAttr = moduleOp2->getAttrOfType<IntegerAttr>("routing.tile_rows");
                        auto tileNAttr = moduleOp2->getAttrOfType<IntegerAttr>("routing.tile_n");
                        auto tileColsAttr = moduleOp2->getAttrOfType<IntegerAttr>("routing.tile_cols");
                        int64_t tileM = tileMAttr ? tileMAttr.getInt() : 0;
                        int64_t tileRows = tileRowsAttr ? tileRowsAttr.getInt() : 0;
                        int64_t tileN = tileNAttr ? tileNAttr.getInt() : 0;
                        int64_t tileCols = tileColsAttr ? tileColsAttr.getInt() : 0;
                        int64_t mRounds = (tileM > 0 && tileM < tileRows) ? (tileRows / tileM) : 1;
                        int64_t nRounds = (tileN > 0 && tileN < tileCols) ? (tileCols / tileN) : 1;
                        int64_t outDivisor = mRounds * nRounds;
                        // Spatial-halo conv: tile_m/tile_rows/tile_n attrs are DROPPED under
                        // fullconnect_auto=0, collapsing the divisor to 1 and leaving the
                        // output window at the full per-core partition (clamped to 4096).
                        // The conv2d_spatial kernel still writes one [oh_per_row*ow_t, tile_n]
                        // slab per on-core round, so honor the authoritative per-slab count
                        // in "routing.spatial_out_rounds" (= spatialMRounds*spatialNRounds).
                        // Kept in lockstep with the host flowtransfer_kernel.cpp output path.
                        if (auto outRoundsAttr = moduleOp2->getAttrOfType<IntegerAttr>("routing.spatial_out_rounds")) {
                            int64_t outRounds = outRoundsAttr.getInt();
                            if (outRounds > outDivisor)
                                outDivisor = outRounds;
                        }
                        if (outDivisor > 1) {
                            perCoreSizeForBuf = perCoreSize / outDivisor;
                        }
                    }
                }

                if (paramInfo.isInput) {
                    auto moduleOp = declareDataOp->getParentOfType<ModuleOp>();
                    if (moduleOp) {
                        if (auto kRoundsAttr = moduleOp->getAttrOfType<IntegerAttr>("routing.k_rounds")) {
                            int64_t kRounds = kRoundsAttr.getInt();
                            if (kRounds > 1) {
                                perCoreSizeForBuf = perCoreSize / kRounds;
                                // When tile_m < tileRows, each k-round only needs
                                // tile_m rows (not partRows). Divide by mRounds
                                // so pingPongBufSize = tile_m * effectiveK.
                                auto tileMAttr = moduleOp->getAttrOfType<IntegerAttr>("routing.tile_m");
                                auto tileRowsAttr = moduleOp->getAttrOfType<IntegerAttr>("routing.tile_rows");
                                int64_t tileM = tileMAttr ? tileMAttr.getInt() : 0;
                                int64_t tileRows = tileRowsAttr ? tileRowsAttr.getInt() : 0;
                                if (tileM > 0 && tileM < tileRows) {
                                    int64_t mRounds = tileRows / tileM;
                                    perCoreSizeForBuf = perCoreSizeForBuf / mRounds;
                                }
                            }
                        }
                    }
                }

                // pp_depth controls physical ping-pong buffer count (for DMA/compute
                // overlap), NOT data splitting.  Buffer size = full per-k-round data,
                // clamped only by maxPingPongBytes when the data exceeds tile memory.
                int64_t pingPongBufSize = perCoreSizeForBuf;
                if (pingPongBufSize <= 0)
                    pingPongBufSize = 1;

                // Spatial-halo conv IFM override: the kernel performs on-chip im2col
                // and needs the WHOLE contiguous halo slab (halo_slice * raw_wc) in a
                // single window buffer. The GEMM flow-view partition size above does
                // not reflect this slab, so the window allocation must use the
                // authoritative slab size carried via "routing.spatial_halo_buf_size".
                // Identify the spatial-halo IFM port by the per-tensor "tensor_N.halo"
                // attribute (N = funcArgIndex). This override bypasses the maxPingPong
                // clamp because the slab must stay contiguous for windowing.
                bool isSpatialHaloPort = false;
                int64_t l2RoundsKernel = 0; // >1 ⇒ two-level (nested) halo IFM port
                if (paramInfo.isInput && paramInfo.funcArgIndex >= 0) {
                    auto moduleOpHalo = declareDataOp->getParentOfType<ModuleOp>();
                    if (moduleOpHalo) {
                        std::string haloAttrName = "tensor_" + std::to_string(paramInfo.funcArgIndex) + ".halo";
                        auto haloDict = moduleOpHalo->getAttrOfType<DictionaryAttr>(haloAttrName);
                        auto haloBufAttr = moduleOpHalo->getAttrOfType<IntegerAttr>("routing.spatial_halo_buf_size");
                        if (haloDict && haloBufAttr && haloBufAttr.getInt() > 0) {
                            pingPongBufSize = haloBufAttr.getInt();
                            isSpatialHaloPort = true;
                            // Two-level (nested) halo: when l2_rounds>1 the kernel
                            // receives the slab in l2_rounds shifted windows of
                            // l2_slice*raw_wc each (instead of one full halo_slice slab).
                            // Note: routing.spatial_halo_buf_size already carries the
                            // per-round (l2_slice*raw_wc) size because aiehlc.cc sizes
                            // the A-port bufferSize from l2_slice when L2 is active; we
                            // only need the round multiplier here.
                            if (auto a = haloDict.getAs<IntegerAttr>("l2_rounds"))
                                l2RoundsKernel = a.getInt();
                        }
                    }
                }

                // K-contraction split halo IFM (tensor_N.halo k_rounds>1): unlike the
                // single-slab spatial-halo path above, this config has NO
                // routing.spatial_halo_buf_size. The kernel receives the IFM as
                // l2_rounds*k_rounds windows, each a [l2_slice rows x k_slice width-elems]
                // band. Per-window buffer = l2_slice*k_slice elements; numRounds =
                // l2_rounds*k_rounds. pp_depth stays >=2 (do NOT set singleBuffer) so the
                // window double-buffers across rounds — kept in lockstep with host B2/B3.
                bool isKSplitHaloPort = false;
                int64_t kSplitNumRounds = 0;
                if (!isSpatialHaloPort && paramInfo.isInput && paramInfo.funcArgIndex >= 0) {
                    auto moduleOpK = declareDataOp->getParentOfType<ModuleOp>();
                    if (moduleOpK) {
                        std::string haloAttrName = "tensor_" + std::to_string(paramInfo.funcArgIndex) + ".halo";
                        if (auto haloDict = moduleOpK->getAttrOfType<DictionaryAttr>(haloAttrName)) {
                            auto l2SliceA = haloDict.getAs<IntegerAttr>("l2_slice");
                            auto kSliceA = haloDict.getAs<IntegerAttr>("k_slice");
                            auto l2RoundsA = haloDict.getAs<IntegerAttr>("l2_rounds");
                            auto kRoundsA = haloDict.getAs<IntegerAttr>("k_rounds");
                            if (kRoundsA && kRoundsA.getInt() > 1 && kSliceA && kSliceA.getInt() > 0 && l2SliceA &&
                                l2SliceA.getInt() > 0) {
                                pingPongBufSize = l2SliceA.getInt() * kSliceA.getInt(); // 19*244 = 4636 elements
                                int64_t l2r = l2RoundsA ? l2RoundsA.getInt() : 1;
                                kSplitNumRounds = (l2r > 0 ? l2r : 1) * kRoundsA.getInt(); // 4*4 = 16
                                isKSplitHaloPort = true;
                                llvm::errs()
                                    << "[BlueprintToScheduleKernel] K-split halo IFM: bufSize=" << pingPongBufSize
                                    << " (l2_slice*k_slice) numRounds=" << kSplitNumRounds << " (l2_rounds*k_rounds)\n";
                            }
                        }
                    }
                }

                // Clamp to maxPingPongBytes to prevent exceeding core tile memory.
                // Skip the clamp for the spatial-halo IFM slab and the K-split per-round
                // slab (both must match the host per-round len exactly).
                if (!isSpatialHaloPort && !isKSplitHaloPort && maxPingPongBytes > 0) {
                    int64_t elemBytes =
                        paramInfo.elementType.isIntOrFloat() ? paramInfo.elementType.getIntOrFloatBitWidth() / 8 : 1;
                    if (elemBytes > 0) {
                        int64_t maxElements = maxPingPongBytes / elemBytes;
                        if (maxElements > 0 && pingPongBufSize > maxElements)
                            pingPongBufSize = maxElements;
                    }
                }
                // bufferSize is in units of vectors (BUF_SZ = elements / vectorWidth)
                paramInfo.bufferSize = pingPongBufSize / defaultVectorWidth;
                if (paramInfo.bufferSize <= 0)
                    paramInfo.bufferSize = 1;
                // numRounds = rounds per k-round = perCoreSizeForBuf / pingPongBufSize
                // The kRounds multiplier below scales to total across all k-rounds.
                paramInfo.numRounds =
                    (pingPongBufSize > 0) ? static_cast<int32_t>(perCoreSizeForBuf / pingPongBufSize) : 1;

                // Two-level (nested) halo IFM: the per-round window is l2_slice*raw_wc
                // (already reflected in pingPongBufSize via spatial_halo_buf_size) and
                // the kernel acquires/releases this window l2_rounds times — once per
                // on-core temporal round. Set numRounds = l2_rounds directly so the
                // window_init covers every L2 acquire/release cycle. (The GEMM
                // flow-view ratio above does not capture the L2 round count.)
                if (l2RoundsKernel > 1) {
                    llvm::errs() << "[BlueprintToScheduleKernel] L2 halo IFM: numRounds " << paramInfo.numRounds
                                 << " -> l2_rounds " << l2RoundsKernel << "\n";
                    paramInfo.numRounds = static_cast<int32_t>(l2RoundsKernel);
                }

                // K-contraction split halo IFM: the kernel acquires/releases the per-round
                // window l2_rounds*k_rounds times (= 16). Set numRounds directly so
                // window_init arms every acquire/release cycle — symmetric with the host
                // core num_iterations (B3). Must run after the GEMM-ratio numRounds compute
                // and BEFORE the routing.k_rounds multiplier below (absent here anyway).
                if (isKSplitHaloPort && kSplitNumRounds > 1) {
                    llvm::errs() << "[BlueprintToScheduleKernel] K-split halo IFM: numRounds " << paramInfo.numRounds
                                 << " -> l2_rounds*k_rounds " << kSplitNumRounds << "\n";
                    paramInfo.numRounds = static_cast<int32_t>(kSplitNumRounds);
                }

                // M-round multiplication for output: when tile_m < tileRows,
                // the kernel outputs mRounds sub-tiles per GEMM invocation.
                // numRounds must cover all m-round iterations.
                if (!paramInfo.isInput) {
                    auto moduleOp3 = declareDataOp->getParentOfType<ModuleOp>();
                    if (moduleOp3) {
                        auto tileMAttr = moduleOp3->getAttrOfType<IntegerAttr>("routing.tile_m");
                        auto tileRowsAttr = moduleOp3->getAttrOfType<IntegerAttr>("routing.tile_rows");
                        int64_t tileM = tileMAttr ? tileMAttr.getInt() : 0;
                        int64_t tileRows = tileRowsAttr ? tileRowsAttr.getInt() : 0;
                        int64_t mRounds = (tileM > 0 && tileM < tileRows) ? (tileRows / tileM) : 1;
                        // Spatial-halo conv (fullconnect_auto=0): tile_m/tile_rows dropped
                        // so mRounds collapses to 1; use the authoritative per-slab round
                        // count so the output window arms once per kernel slab (16), matching
                        // the per-slab buffer size divided above. In lockstep with the host.
                        if (auto outRoundsAttr = moduleOp3->getAttrOfType<IntegerAttr>("routing.spatial_out_rounds")) {
                            int64_t outRounds = outRoundsAttr.getInt();
                            if (outRounds > mRounds)
                                mRounds = outRounds;
                        }
                        if (mRounds > 1) {
                            llvm::errs() << "[BlueprintToScheduleKernel] M-round: output numRounds "
                                         << paramInfo.numRounds << " * mRounds " << mRounds << " = "
                                         << paramInfo.numRounds * mRounds << "\n";
                            paramInfo.numRounds *= static_cast<int32_t>(mRounds);
                        }
                    }
                }

                // K-round multiplication: when effectiveK < K, the kernel runs
                // kRounds iterations. window_init numRounds must cover the total
                // acquire/release cycles across all k-rounds.
                if (paramInfo.isInput) {
                    auto moduleOp = declareDataOp->getParentOfType<ModuleOp>();
                    if (moduleOp) {
                        if (auto kRoundsAttr = moduleOp->getAttrOfType<IntegerAttr>("routing.k_rounds")) {
                            int64_t kRounds = kRoundsAttr.getInt();
                            if (kRounds > 1) {
                                llvm::errs()
                                    << "[BlueprintToScheduleKernel] K-round: input numRounds " << paramInfo.numRounds
                                    << " * kRounds " << kRounds << " = " << paramInfo.numRounds * kRounds << "\n";
                                paramInfo.numRounds *= static_cast<int32_t>(kRounds);
                            }
                        }
                    }
                }

                // M-round multiplication for input: when tile_m < tileRows,
                // the kernel acquires A data mRounds times per k-round.
                // window_init numRounds must cover the total across all (mr, kr).
                if (paramInfo.isInput) {
                    auto moduleOp4 = declareDataOp->getParentOfType<ModuleOp>();
                    if (moduleOp4) {
                        auto tileMAttr = moduleOp4->getAttrOfType<IntegerAttr>("routing.tile_m");
                        auto tileRowsAttr = moduleOp4->getAttrOfType<IntegerAttr>("routing.tile_rows");
                        int64_t tileM = tileMAttr ? tileMAttr.getInt() : 0;
                        int64_t tileRows = tileRowsAttr ? tileRowsAttr.getInt() : 0;
                        if (tileM > 0 && tileM < tileRows) {
                            int64_t mRounds = tileRows / tileM;
                            llvm::errs() << "[BlueprintToScheduleKernel] M-round: input numRounds "
                                         << paramInfo.numRounds << " * mRounds " << mRounds << " = "
                                         << paramInfo.numRounds * mRounds << "\n";
                            paramInfo.numRounds *= static_cast<int32_t>(mRounds);
                        }
                    }
                }
            } else {
                // Fallback: try declare_data operand
                Value srcValue = declareDataOp.getOperand();
                if (srcValue) {
                    if (auto fallbackType = dyn_cast<RankedTensorType>(srcValue.getType())) {
                        paramInfo.elementType = fallbackType.getElementType();
                    } else {
                        paramInfo.elementType = defaultElementType;
                    }
                } else {
                    paramInfo.elementType = defaultElementType;
                }
                paramInfo.bufferSize = static_cast<int64_t>(defaultBufferSize * bufferRatio);
            }

            paramInfo.vectorWidth = defaultVectorWidth;

            params.push_back(paramInfo);
        }
    });

    // =========================================================================
    // Phase 2: Sort by funcArgIndex and assign names, locks, and addresses
    // =========================================================================
    // Separate inputs and outputs, sort each by funcArgIndex, then assign
    // sequential names (window_in_0, window_in_1, ...) and lock IDs.
    SmallVector<KernelParamInfo *> inputParams, outputParams;
    for (auto &p : params) {
        if (p.isInput)
            inputParams.push_back(&p);
        else
            outputParams.push_back(&p);
    }

    // Sort by funcArgIndex (stable sort preserves encounter order for ties)
    llvm::sort(inputParams,
               [](const KernelParamInfo *a, const KernelParamInfo *b) { return a->funcArgIndex < b->funcArgIndex; });
    llvm::sort(outputParams,
               [](const KernelParamInfo *a, const KernelParamInfo *b) { return a->funcArgIndex < b->funcArgIndex; });

    // Reset resource manager to allocate locks in sorted order
    resourceMgr.reset();

    int sortedInputCount = 0;
    for (auto *p : inputParams) {
        p->windowName = "window_in_" + std::to_string(sortedInputCount);
        p->bufferPingName = "buf_in_ping_" + std::to_string(sortedInputCount);
        // Single-buffer ports reuse the ping buffer for the pong slot so no
        // second global array is declared (and no floating BCF symbol).
        p->bufferPongName = p->singleBuffer ? p->bufferPingName : "buf_in_pong_" + std::to_string(sortedInputCount);
        p->acquireLockId = resourceMgr.allocateInputAcquireLock();
        p->releaseLockId = resourceMgr.allocateInputReleaseLock();
        sortedInputCount++;
    }
    int sortedOutputCount = 0;
    for (auto *p : outputParams) {
        p->windowName = "window_out_" + std::to_string(sortedOutputCount);
        p->bufferPingName = "buf_out_ping_" + std::to_string(sortedOutputCount);
        p->bufferPongName = "buf_out_pong_" + std::to_string(sortedOutputCount);
        p->acquireLockId = resourceMgr.allocateOutputAcquireLock();
        p->releaseLockId = resourceMgr.allocateOutputReleaseLock();
        sortedOutputCount++;
    }

    // Read buffer addresses from CoreMemAllocator (allocated by host pass)
    for (auto &p : params) {
        try {
            auto &allocator = ResourceMgr::instance()->coreMemAllocator();
            p.pingAddress = allocator.getAddress(p.bufferPingName);
            p.pongAddress = allocator.getAddress(p.bufferPongName);
        } catch (...) {
            // ResourceMgr singleton not initialized; addresses remain 0
        }
    }

    // Debug: print parameter mapping chain
    llvm::errs() << "[KernelParamMapping] Parameter order mapping:\n";
    for (const auto &p : params) {
        llvm::errs() << "  funcArg=" << p.funcArgIndex << " -> " << p.windowName << " -> " << p.bufferPingName << "/"
                     << p.bufferPongName << " -> addr=0x" << llvm::utohexstr(p.pingAddress) << "/0x"
                     << llvm::utohexstr(p.pongAddress) << " -> lock=" << p.acquireLockId << "/" << p.releaseLockId
                     << "\n";
    }

    return params;
}

// Pattern to convert dfscheblueprint::FlowTransferOp to dfschedule operations.
// Kernel path: generates dfschedule.module (kernel driver) via generateDSKernelReceiver,
// plus core tile DMA IO configuration (ConfigDmaBd, ConfigCreateIo, GetBdId, StartIo).
struct FlowTransferConversion : public OpConversionPattern<dfscheblueprint::FlowTransferOp> {
    double bufferRatio;
    int64_t maxPingPongBytes;

    FlowTransferConversion(MLIRContext *ctx, double ratio, int64_t maxPPBytes)
        : OpConversionPattern<dfscheblueprint::FlowTransferOp>(ctx), bufferRatio(ratio), maxPingPongBytes(maxPPBytes) {}

    mutable KernelResourceManager resourceMgr;

    LogicalResult matchAndRewrite(dfscheblueprint::FlowTransferOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        auto loc = op.getLoc();

        SymbolRefAttr fromRef = op.getFrom();
        auto fromFlowConfig = lookupFlowConfig(op.getOperation(), fromRef);
        if (!fromFlowConfig) {
            rewriter.eraseOp(op);
            return success();
        }

        SymbolRefAttr toRef = op.getTo();
        auto toFlowConfig = lookupFlowConfig(op.getOperation(), toRef);
        if (!toFlowConfig) {
            rewriter.eraseOp(op);
            return success();
        }

        dfscheblueprint::FlowConfigOp shimFlowConfig = nullptr;
        dfscheblueprint::FlowConfigOp coreFlowConfig = nullptr;

        auto fromType = fromFlowConfig.getType();
        auto toType = toFlowConfig.getType();

        if (fromType && *fromType == "shim") {
            shimFlowConfig = fromFlowConfig;
            coreFlowConfig = toFlowConfig;
        } else if (toType && *toType == "shim") {
            shimFlowConfig = toFlowConfig;
            coreFlowConfig = fromFlowConfig;
        } else {
            rewriter.eraseOp(op);
            return success();
        }

        auto shimTileGroup = lookupTileGroup(shimFlowConfig.getOperation(), shimFlowConfig.getTarget());
        auto coreTileGroup = lookupTileGroup(coreFlowConfig.getOperation(), coreFlowConfig.getTarget());
        if (!shimTileGroup || !coreTileGroup) {
            rewriter.eraseOp(op);
            return success();
        }

        uint32_t basePacketId = op.getBasePacketId();
        uint32_t flowIndex = op.getFlowIndex();

        Value viewValue = shimFlowConfig.getView();
        Type viewType = viewValue.getType();

        // Compute bufferLen and kernelTensorType from view (no host IR)
        int64_t bufferLen = 1;
        RankedTensorType kernelTensorType;
        if (auto tensorType = dyn_cast<RankedTensorType>(viewType)) {
            kernelTensorType = tensorType;
            for (int64_t dim : tensorType.getShape()) {
                bufferLen *= dim;
            }
        } else if (auto mrType = dyn_cast<MemRefType>(viewType)) {
            kernelTensorType = RankedTensorType::get(mrType.getShape(), mrType.getElementType());
            for (int64_t dim : mrType.getShape()) {
                bufferLen *= dim;
            }
        } else {
            rewriter.eraseOp(op);
            return success();
        }

        auto coreDmaAttr = coreFlowConfig.getDma();
        auto coreDmaChannels = coreDmaAttr.getChannels();
        int64_t coreChannel = coreDmaChannels.empty() ? 0 : coreDmaChannels[0];

        StringRef kernelName = "dskernel_receiver";
        if (!hasDSKernelReceiver(op.getOperation(), kernelName)) {
            generateDSKernelReceiver(rewriter, loc, op.getOperation(), kernelName, kernelTensorType, bufferLen,
                                     basePacketId, coreChannel, flowIndex, resourceMgr, bufferRatio, maxPingPongBytes);
        }

        // --- Core tile DMA IO configuration (create_io + start_io) ---
        {
            ArrayAttr coreTilesAttr = coreTileGroup.getTiles();
            auto coreDmaDir = coreDmaAttr.getDirection();
            StringRef coreDmaDirection = (coreDmaDir == dfscheblueprint::bp_direction::MM2S) ? "MM2S" : "S2MM";
            StringRef coreIoOperation = (coreDmaDir == dfscheblueprint::bp_direction::MM2S) ? "SEND" : "RECV";

            int64_t numCoreTiles = coreTilesAttr.size();
            int64_t perTileLen = (numCoreTiles > 0) ? bufferLen / numCoreTiles : bufferLen;
            Type elemType = kernelTensorType.getElementType();

            int tileIdx = 0;
            for (auto tileAttr : coreTilesAttr) {
                auto tileArray = dyn_cast<ArrayAttr>(tileAttr);
                if (!tileArray || tileArray.size() < 2)
                    continue;

                int64_t col = cast<IntegerAttr>(tileArray[0]).getInt();
                int64_t row = cast<IntegerAttr>(tileArray[1]).getInt();

                auto coreTileOp = rewriter.create<dfschedule::DeclareTileOp>(
                    loc, dfschedule::TileType::get(rewriter.getContext()), rewriter.getI32IntegerAttr(col),
                    rewriter.getI32IntegerAttr(row));

                MemRefType coreBufType = MemRefType::get({perTileLen}, elemType);
                Value coreBuf = rewriter.create<memref::AllocOp>(loc, coreBufType);

                auto bdIdConst = rewriter.create<arith::ConstantOp>(
                    loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(resourceMgr.allocateBdId()));

                auto kernelOffsetConst =
                    rewriter.create<arith::ConstantOp>(loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
                auto coreBdOp = rewriter.create<dfschedule::ConfigDmaBdOp>(
                    loc, dfschedule::BdHandleType::get(rewriter.getContext()), coreBuf, coreTileOp.getTile(), bdIdConst,
                    kernelOffsetConst.getResult(), rewriter.getI32IntegerAttr(perTileLen), rewriter.getBoolAttr(true),
                    rewriter.getI32IntegerAttr(basePacketId + tileIdx), rewriter.getI32IntegerAttr(4294967295),
                    rewriter.getI32IntegerAttr(0), rewriter.getI32IntegerAttr(0), rewriter.getI32IntegerAttr(0),
                    rewriter.getI32IntegerAttr(0), rewriter.getI32IntegerAttr(-1), Value(),
                    rewriter.getI32IntegerAttr(-1), // out_of_order_bd_id
                    /*dim_strides=*/nullptr, /*dim_wraps=*/nullptr,
                    rewriter.getI32IntegerAttr(0),  // iter_step_size (no iteration)
                    rewriter.getI32IntegerAttr(0)); // iter_wrap (no iteration)

                auto createIoOp = rewriter.create<dfschedule::ConfigCreateIoOp>(
                    loc, dfschedule::IoHandleType::get(rewriter.getContext()), coreBdOp.getBdHandle(),
                    coreTileOp.getTile(), rewriter.getI32IntegerAttr(coreChannel),
                    rewriter.getStringAttr(coreDmaDirection), rewriter.getStringAttr(coreIoOperation),
                    rewriter.getBoolAttr(false)); // enable_out_of_order

                auto getBdIdOp =
                    rewriter.create<dfschedule::GetBdIdOp>(loc, rewriter.getI32Type(), coreTileOp.getTile());

                // Core MM2S (output) repeat: when tile_m < tileRows, the kernel
                // outputs mRounds sub-tiles. The DMA must repeat accordingly.
                int32_t coreRepeatCount = 1;
                if (coreDmaDir == dfscheblueprint::bp_direction::MM2S) {
                    auto moduleOp = op->getParentOfType<ModuleOp>();
                    if (moduleOp) {
                        auto tileMAttr = moduleOp->getAttrOfType<IntegerAttr>("routing.tile_m");
                        auto tileRowsAttr = moduleOp->getAttrOfType<IntegerAttr>("routing.tile_rows");
                        int64_t tileM = tileMAttr ? tileMAttr.getInt() : 0;
                        int64_t tileRows = tileRowsAttr ? tileRowsAttr.getInt() : 0;
                        if (tileM > 0 && tileM < tileRows) {
                            coreRepeatCount = static_cast<int32_t>(tileRows / tileM);
                        }
                    }
                }

                rewriter.create<dfschedule::StartIoOp>(loc, dfschedule::EventType::get(rewriter.getContext()),
                                                       createIoOp.getIoHandle(), getBdIdOp.getBdId(),
                                                       rewriter.getI32IntegerAttr(flowIndex),
                                                       rewriter.getI32IntegerAttr(coreRepeatCount));

                tileIdx++;
            }
        }

        rewriter.eraseOp(op);
        return success();
    }
};

// Special pattern for DataSliceOp - replaces with input tensor instead of erasing
struct DataSliceOpConversion : public OpConversionPattern<dfscheblueprint::DataSliceOp> {
    using OpConversionPattern<dfscheblueprint::DataSliceOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(dfscheblueprint::DataSliceOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {
        // DataSliceOp is used for symbol references, replace with the input tensor
        rewriter.replaceOp(op, adaptor.getTensorSlice());
        return success();
    }
};

} // namespace

namespace mlir {

void BlueprintToScheduleKernelPass::runOnOperation() {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);

    // Mark target dialects as legal
    target
        .addLegalDialect<dfschedule::dfscheduledialect, func::FuncDialect, memref::MemRefDialect, arith::ArithDialect,
                         scf::SCFDialect, tensor::TensorDialect, bufferization::BufferizationDialect, BuiltinDialect>();

    // Mark all dfscheblueprint operations as illegal to trigger conversion/erasure
    target.addIllegalOp<dfscheblueprint::FlowConfigOp>();
    target.addIllegalOp<dfscheblueprint::TileGroupOp>();
    // target.addIllegalOp<dfscheblueprint::DeclareDataOp>();
    // target.addIllegalOp<dfscheblueprint::DataSliceOp>();
    target.addIllegalOp<dfscheblueprint::FlowTransferOp>();
    // target.addIllegalOp<dfscheblueprint::TransferManifestOp>();

    // Type converter
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) { return type; });

    // Convert tensor types to memref types where needed
    typeConverter.addConversion([](RankedTensorType tensorType) -> Type {
        return MemRefType::get(tensorType.getShape(), tensorType.getElementType());
    });

    RewritePatternSet patterns(context);
    // FlowTransferConversion converts flow_transfer to dfschedule operations
    // It reads from FlowConfigOps to get DMA configuration
    patterns.add<FlowTransferConversion>(context, bufferRatio_, maxPingPongBytes_);
    // DataSliceOp replaces with input tensor
    patterns.add<DataSliceOpConversion>(context);
    // Use unified erase pattern for ops that just need to be removed
    // FlowConfigOp is erased since FlowTransferConversion reads its attributes directly
    patterns.add<EraseOpPattern<dfscheblueprint::FlowConfigOp>>(context);
    patterns.add<EraseOpPattern<dfscheblueprint::TileGroupOp>>(context);
    // patterns.add<EraseOpPattern<dfscheblueprint::DeclareDataOp>>(context);
    // patterns.add<EraseOpPattern<dfscheblueprint::TransferManifestOp>>(context);

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
        return;
    }

    // Kernel-only: remove all top-level ops that are not dfschedule kernel logic.
    // Keep DSKernelReceiverOp, KernelModuleOp, and DMA IO config ops
    // (DeclareTile, ConfigDmaBd, ConfigCreateIo, GetBdId, StartIo + their operand producers).
    Operation *root = getOperation();
    while (root->getParentOp())
        root = root->getParentOp();
    Block &body = root->getRegion(0).front();
    SmallVector<Operation *> toErase;
    for (Operation &op : body)
        if (!isa<dfschedule::DSKernelReceiverOp>(&op) && !isa<dfschedule::KernelModuleOp>(&op) &&
            !isa<dfschedule::DeclareTileOp>(&op) && !isa<dfschedule::ConfigDmaBdOp>(&op) &&
            !isa<dfschedule::ConfigCreateIoOp>(&op) && !isa<dfschedule::GetBdIdOp>(&op) &&
            !isa<dfschedule::StartIoOp>(&op) && !isa<arith::ConstantOp>(&op) && !isa<memref::AllocOp>(&op))
            toErase.push_back(&op);
    for (auto it = toErase.rbegin(); it != toErase.rend(); ++it)
        (*it)->erase();
}

} // namespace mlir
