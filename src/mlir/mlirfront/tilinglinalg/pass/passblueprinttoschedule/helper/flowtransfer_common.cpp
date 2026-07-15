/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

// Shared utilities for the split FlowTransferConversion implementation.
//
// These were formerly file-local `static` helpers at the top of
// passblueprinttoschedule.cpp. They are promoted to external linkage (inside
// namespace blueprint_sched to avoid ODR clashes with the identically-named
// statics in sibling passes) so the host/kernel helper TUs can share them.
//
// Bodies are moved verbatim: only the `static` keyword is removed and the
// namespace wrapper added. No behavioral change.

#include "flowtransfer_internal.h"
#include "routingmanager.h"

namespace blueprint_sched {

using namespace mlir;
using namespace dfscheblueprint;
using namespace dfschedule;

// Trace a Value back through the SSA chain to find the originating function
// argument index.  This walks through bufferization.to_tensor,
// routing.routingcreatescheduletensor, routing.partitiontensor,
// routing.routingextract_data, dfscheblueprint.declare_data, and
// scf.execute_region captures until it reaches a BlockArgument of a func::FuncOp.
// Returns the argument index (0-based), or -1 if the chain cannot be resolved.
int traceToFuncArgIndex(Value v) {
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

// Trace from a FlowConfigOp's view value back through the SSA chain to find
// the originating function argument index.
int traceFlowConfigToFuncArgIndex(dfscheblueprint::FlowConfigOp flowConfig) {
    Value viewValue = flowConfig.getView();
    if (!viewValue)
        return -1;
    return traceToFuncArgIndex(viewValue);
}

// Helper function to look up TileGroupOp by symbol reference (wrapper for backward compatibility)
dfscheblueprint::TileGroupOp lookupTileGroup(Operation *rootOp, SymbolRefAttr target) {
    return lookupSymbolOp<dfscheblueprint::TileGroupOp>(rootOp, target);
}

// Helper function to check if dskernel_receiver already exists in the module
bool hasDSKernelReceiver(Operation *rootOp, StringRef kernelName) {
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

// Helper function to get the module-level insertion point
Operation *getModuleOp(Operation *rootOp) {
    Operation *moduleOp = rootOp;
    while (moduleOp->getParentOp()) {
        moduleOp = moduleOp->getParentOp();
    }
    return moduleOp;
}

// Generate a kernel symbol declaration only (dfschedule.dskernel_receiver with empty body).
// Details (kernel module, buffers, locks, etc.) are filled in by a separate pass.
// Parameters:
//   - kernelName: symbol name for the kernel (same as load_kernel_group callee)
//   - insertBeforeOp: used to find module and insertion point
void generateDSKernelReceiver(ConversionPatternRewriter &rewriter, Location loc, Operation *insertBeforeOp,
                              StringRef kernelName, RankedTensorType tensorType, int64_t bufferLen,
                              uint32_t basePacketId, int64_t coreChannel, uint32_t flowIndex) {
    (void)tensorType;
    (void)bufferLen;
    (void)basePacketId;
    (void)coreChannel;
    (void)flowIndex;

    Operation *rootModuleOp = getModuleOp(insertBeforeOp);
    Block &moduleBlock = rootModuleOp->getRegions().front().front();
    if (!moduleBlock.empty() && moduleBlock.back().hasTrait<OpTrait::IsTerminator>())
        rewriter.setInsertionPoint(&moduleBlock.back());
    else
        rewriter.setInsertionPointToEnd(&moduleBlock);

    auto receiverOp = rewriter.create<dfschedule::DSKernelReceiverOp>(loc, kernelName);
    receiverOp.getBody().emplaceBlock();
}

// Helper function to look up FlowConfigOp by symbol reference (wrapper for backward compatibility)
dfscheblueprint::FlowConfigOp lookupFlowConfig(Operation *rootOp, SymbolRefAttr target) {
    return lookupSymbolOp<dfscheblueprint::FlowConfigOp>(rootOp, target);
}

dfscheblueprint::DataSliceOp lookupDataSlice(Operation *rootOp, SymbolRefAttr target) {
    return lookupSymbolOp<dfscheblueprint::DataSliceOp>(rootOp, target);
}

SmallVector<OpFoldResult> toOpFoldResult(ArrayRef<int64_t> values, OpBuilder &b) {
    SmallVector<OpFoldResult> result;
    for (int64_t v : values)
        result.push_back(b.getI64IntegerAttr(v));
    return result;
}

// Pre-processing: lower arith.constant dense tensors to memrefs via
// bufferization.to_memref, keeping everything inside @main with no module-level
// memref.global.  Processes ALL DeclareDataOps so each data tensor (input A,
// input B, output C, etc.) gets its own backing memref.
LogicalResult preprocessConstantToMemref(Operation *topLevel, std::shared_ptr<BlueprintPassState> state) {
    func::FuncOp mainFunc = nullptr;
    topLevel->walk([&](func::FuncOp f) {
        if (f.getName() == "main" || f.getName() == "host_canonicalized")
            mainFunc = f;
    });
    if (!mainFunc)
        return success();

    // Collect ALL DeclareDataOps
    SmallVector<dfscheblueprint::DeclareDataOp> declareDataOps;
    mainFunc.walk([&](dfscheblueprint::DeclareDataOp op) { declareDataOps.push_back(op); });
    if (declareDataOps.empty())
        return success();

    // Track which constants have already been lowered to avoid duplicates
    // (multiple DeclareDataOps may reference the same constant)
    llvm::DenseSet<Operation *> processedConstants;
    SmallVector<Value> allocatedMemrefs;

    for (auto declareDataOp : declareDataOps) {
        Value initTensor = declareDataOp.getInitTensor();

        // Case 1: arith.constant (original path — hardcoded data)
        if (auto constantOp = initTensor.getDefiningOp<arith::ConstantOp>()) {
            // Skip if this constant was already lowered
            if (processedConstants.count(constantOp.getOperation()))
                continue;
            processedConstants.insert(constantOp.getOperation());

            auto tensorType = dyn_cast<RankedTensorType>(constantOp.getType());
            if (!tensorType)
                continue;

            // Use the first constant's shape/type for the legacy rootShape/elementType fields
            if (!state->elementType) {
                state->rootShape.assign(tensorType.getShape().begin(), tensorType.getShape().end());
                state->elementType = tensorType.getElementType();
            }

            MemRefType memrefType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());

            // Lower the dense constant: get a read-only memref view via bufferization.to_memref,
            // then alloc a writable buffer and copy into it.
            OpBuilder builder(constantOp.getOperation());
            builder.setInsertionPointAfter(constantOp.getOperation());
            auto loc = constantOp.getLoc();
            auto toMemref = builder.create<bufferization::ToMemrefOp>(loc, memrefType, constantOp.getResult());
            auto allocOp = builder.create<memref::AllocOp>(loc, memrefType);
            builder.create<memref::CopyOp>(loc, toMemref.getResult(), allocOp.getResult());

            // Register this constant's memref in the map
            state->constantToMemref[constantOp.getResult()] = allocOp.getResult();
            allocatedMemrefs.push_back(allocOp.getResult());

            // Keep legacy rootMemref/rootMemrefType pointing to the first allocation
            if (!state->rootMemref) {
                state->rootMemref = allocOp.getResult();
                state->rootMemrefType = memrefType;
            }
        }
        // Case 2: bufferization.to_tensor of func arg (external DDR pointer)
        // The user's pointer IS the DDR buffer — no alloc, no copy needed.
        else if (auto toTensorOp = initTensor.getDefiningOp<bufferization::ToTensorOp>()) {
            Value srcMemref = toTensorOp.getMemref();

            auto tensorType = dyn_cast<RankedTensorType>(toTensorOp.getType());
            if (!tensorType)
                continue;

            // Skip if already processed
            if (processedConstants.count(toTensorOp.getOperation()))
                continue;
            processedConstants.insert(toTensorOp.getOperation());

            if (!state->elementType) {
                state->rootShape.assign(tensorType.getShape().begin(), tensorType.getShape().end());
                state->elementType = tensorType.getElementType();
            }

            MemRefType memrefType = cast<MemRefType>(srcMemref.getType());

            // Use the func arg memref DIRECTLY — no alloc, no copy needed.
            state->constantToMemref[initTensor] = srcMemref;

            if (!state->rootMemref) {
                state->rootMemref = srcMemref;
                state->rootMemrefType = memrefType;
            }
        }
    }

    // Insert memref.dealloc for ALL allocated memrefs before the return op.
    if (!allocatedMemrefs.empty()) {
        Block &mainBlock = mainFunc.getBody().front();
        for (auto &op : mainBlock) {
            if (isa<func::ReturnOp>(op)) {
                OpBuilder builder(&op);
                for (auto memref : allocatedMemrefs) {
                    builder.create<memref::DeallocOp>(op.getLoc(), memref);
                }
                break;
            }
        }
    }

    return success();
}

// Trace a view value back through tensor.extract_slice → routing.partitiontensor
// → dfscheblueprint.declare_data → arith.constant to find the originating constant,
// then look up its memref in the constantToMemref map.
Value resolveMemrefForView(Value viewValue, const BlueprintPassState &state) {
    // Walk backwards through extract_slice chain
    Value current = viewValue;
    while (auto extractSlice = current.getDefiningOp<tensor::ExtractSliceOp>()) {
        current = extractSlice.getSource();
    }
    // Now current should be the partition tensor result.
    // Walk through routing.partitiontensor to find its source tensor.
    if (auto partitionOp = current.getDefiningOp<routing::partitiontensor>()) {
        current = partitionOp.getTensor();
    }
    // Walk through dfscheblueprint.declare_data to find the init_tensor.
    if (auto declareOp = current.getDefiningOp<dfscheblueprint::DeclareDataOp>()) {
        current = declareOp.getInitTensor();
    }
    // Now current should be the arith.constant result or bufferization.to_tensor
    // result — look it up in the map.
    auto it = state.constantToMemref.find(current);
    if (it != state.constantToMemref.end())
        return it->second;
    return Value();
}

// Reads routing.fullconnect_auto (default true = M×N cartesian repeat / M-N round
// auto-generation enabled). false = A/B streamed once, no M/N round split; conv2d
// (halo) output tiling is halo-driven, never derived from tile_m/tile_rows.
bool isFullConnectAuto(ModuleOp moduleOp) {
    if (!moduleOp)
        return true;
    if (auto a = moduleOp->getAttrOfType<IntegerAttr>("routing.fullconnect_auto"))
        return a.getInt() != 0;
    return true;
}

TilingClassification classifyTiling(ModuleOp moduleOp) {
    TilingClassification result;
    result.mMode = TilingMode::Match;
    result.mRounds = 1;
    result.nMode = TilingMode::Match;
    result.nRounds = 1;

    if (!moduleOp)
        return result;

    // fca=0: no M/N round auto-generation. Force Match/1/1 even if tile_m/tile_rows
    // are present (result is already initialized to Match, mRounds=1, nRounds=1).
    if (!isFullConnectAuto(moduleOp))
        return result;

    // M-dimension classification
    auto tileMAttr = moduleOp->getAttrOfType<IntegerAttr>("routing.tile_m");
    auto tileRowsAttr = moduleOp->getAttrOfType<IntegerAttr>("routing.tile_rows");

    int64_t m = tileMAttr ? tileMAttr.getInt() : 0;
    int64_t rows = tileRowsAttr ? tileRowsAttr.getInt() : 0;

    if (m == 0 || m == rows) {
        result.mMode = TilingMode::Match;
        result.mRounds = 1;
    } else if (m < rows && rows % m == 0) {
        result.mMode = TilingMode::Multiple;
        result.mRounds = rows / m;
    } else {
        result.mMode = TilingMode::Invalid;
        result.mRounds = 0;
    }

    // N-dimension classification
    auto tileNAttr = moduleOp->getAttrOfType<IntegerAttr>("routing.tile_n");
    auto tileColsAttr = moduleOp->getAttrOfType<IntegerAttr>("routing.tile_cols");

    int64_t n = tileNAttr ? tileNAttr.getInt() : 0;
    int64_t cols = tileColsAttr ? tileColsAttr.getInt() : 0;

    if (n == 0 || n == cols) {
        result.nMode = TilingMode::Match;
        result.nRounds = 1;
    } else if (n < cols && cols % n == 0) {
        result.nMode = TilingMode::Multiple;
        result.nRounds = cols / n;
    } else {
        result.nMode = TilingMode::Invalid;
        result.nRounds = 0;
    }

    return result;
}

// Returns true if policy is "n_outer_m_inner" (N is outer loop, M is inner/iter)
bool isNOuterPolicy(ModuleOp moduleOp) {
    if (!moduleOp)
        return false;
    auto attr = moduleOp->getAttrOfType<StringAttr>("routing.iter_policy");
    if (attr && attr.getValue() == "n_outer_m_inner")
        return true;
    return false; // default: m_outer_n_inner
}

// Build the output tile descriptor from the true output geometry. All conv-vs-gemm
// geometry knowledge lives here. The output memref is modeled as [M, N] (row-major),
// with N = output row width (= OC for conv) and M split into mRounds x tileM. The
// per-tile column split (numCoreTiles) handles the N dimension via per-tile offsets.
OutputTileDescriptor buildOutputTileDescriptor(const BlueprintPassState &passState, MemRefType memrefType,
                                               int64_t numCoreTiles, ModuleOp moduleOp, int64_t ooElementSizeBytes) {
    OutputTileDescriptor desc;

    int64_t tileM = passState.tileM;
    int64_t tileRows = passState.tileRows;
    int64_t mRounds = tileRows / tileM;

    unsigned bitWidth = memrefType.getElementTypeBitWidth();
    int64_t elemsPerWord = 32 / bitWidth;
    constexpr int64_t wordBytes = 4;
    int64_t outW = memrefType.getDimSize(1);  // full output row width (elements)
    int64_t outW_w = outW / elemsPerWord;     // ... in 32-bit words
    int64_t tileN_full = outW / numCoreTiles; // per-tile column width (elements)

    // Per-tile N sub-tiling (nRounds): when tile_n < tile_cols, the per-tile width is
    // further split into nRounds iterations of tileN_sub columns.
    int64_t nRounds = 1;
    int64_t tileN_sub = tileN_full;
    if (passState.tileN > 0 && passState.tileN < passState.tileCols) {
        nRounds = passState.tileCols / passState.tileN;
        tileN_sub = passState.tileN;
    }
    int64_t tileN_sub_w = tileN_sub / elemsPerWord;

    int64_t elemBytes = bitWidth / 8;
    if (elemBytes == 0)
        elemBytes = 1;

    // One d0xd1(xd2) block per BD activation (single OOO packet).
    desc.bdLenBytes = tileM * tileN_sub * ooElementSizeBytes;

    // Policy-aware iteration assignment (shared by conv + gemm). The "outer" loop is
    // the scf.for round dimension; the "inner" dimension is folded into the BD's
    // iter_step/iter_wrap.
    bool nOuterPolicy = moduleOp ? isNOuterPolicy(moduleOp) : false;
    int64_t outerRounds;
    int64_t outerStrideBytes;
    if (nOuterPolicy) {
        // n_outer_m_inner: scf.for over nRounds, iter over mRounds
        desc.iterStep = static_cast<int32_t>(tileM * outW_w * wordBytes);
        desc.iterWrap = static_cast<int32_t>(mRounds);
        outerRounds = nRounds;
        outerStrideBytes = tileN_sub_w * wordBytes;
    } else {
        // m_outer_n_inner (default): scf.for over mRounds, iter over nRounds
        desc.iterStep = static_cast<int32_t>(tileN_sub_w * wordBytes);
        desc.iterWrap = static_cast<int32_t>(nRounds);
        outerRounds = mRounds;
        outerStrideBytes = tileM * outW_w * wordBytes;
    }
    desc.totalRounds = outerRounds;

    // Detect conv 2D width-split halo: any tensor_N.halo with w_rounds>1 (gated on
    // routing.spatial_halo_buf_size>0) means the per-round output is a scattered
    // [oh_per_row, OW_T] sub-block of the [OH, OW, OC] image.
    int64_t haloOwT = 0, haloWRounds = 0;
    if (moduleOp) {
        int64_t convBufSize = 0;
        if (auto b = moduleOp->getAttrOfType<IntegerAttr>("routing.spatial_halo_buf_size"))
            convBufSize = b.getInt();
        if (convBufSize > 0) {
            for (const NamedAttribute &na : moduleOp->getAttrs()) {
                if (!na.getName().getValue().ends_with(".halo"))
                    continue;
                auto hd = dyn_cast<DictionaryAttr>(na.getValue());
                if (!hd)
                    continue;
                auto wr = hd.getAs<IntegerAttr>("w_rounds");
                auto ot = hd.getAs<IntegerAttr>("ow_t");
                if (wr && wr.getInt() > 1 && ot) {
                    haloWRounds = wr.getInt();
                    haloOwT = ot.getInt();
                    break;
                }
            }
        }
    }

    if (haloWRounds > 1 && haloOwT > 0 && (tileM % haloOwT) == 0) {
        // Conv width-split scatter. The kernel streams
        // local_out[oh_per_row][OW_T][OC_PER_G] (channel innermost):
        int64_t ohPerRow = tileM / haloOwT;     // e.g. 196/28 = 7
        int64_t fullOW = haloOwT * haloWRounds; // e.g. 28*4 = 112
        // BD scatter dims, innermost-first.
        desc.bdDims.push_back({1 * wordBytes, tileN_sub_w});          // D0: channel-group words
        desc.bdDims.push_back({outW * elemBytes, haloOwT});           // D1: OW_T columns, one pixel (OC) apart
        desc.bdDims.push_back({fullOW * outW * elemBytes, ohPerRow}); // D2: oh_per_row rows, one image row apart
        // Per-round base-offset dims, outermost-first.
        int64_t hChunks = outerRounds / haloWRounds;
        desc.roundDims.push_back({hChunks, ohPerRow * fullOW * outW * elemBytes}); // hc stride
        desc.roundDims.push_back({haloWRounds, haloOwT * outW * elemBytes});       // wc stride
    } else {
        // GEMM contiguous-M: each round writes tileM contiguous M-rows.
        desc.bdDims.push_back({1 * wordBytes, tileN_sub_w}); // D0: contiguous word step
        desc.bdDims.push_back({outW_w * wordBytes, tileM});  // D1: DDR row stride, tileM rows
        desc.roundDims.push_back({outerRounds, outerStrideBytes});
    }

    return desc;
}

} // namespace blueprint_sched
