/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

// Shared internal header for the split FlowTransferConversion implementation.
//
// The single ~2200-line FlowTransferConversion::matchAndRewrite has been broken
// into a thin dispatcher (passblueprinttoschedule.cpp) plus host / kernel helper
// member methods defined across:
//   helper/flowtransfer_common.cpp  — shared utilities (traceToFuncArgIndex, ...)
//   helper/flowtransfer_host.cpp    — shim DMA + schedule orchestration
//   helper/flowtransfer_kernel.cpp  — core-tile config (ping-pong / single-buffer)
//
// All extracted helpers are member methods of FlowTransferConversion so they can
// naturally access the pattern's mutable buffer-index bookkeeping members. Locals
// that cross section boundaries are carried in FlowLoweringCtx; per-core-tile
// locals are carried in CoreTileCtx.
//
// Everything shared lives in namespace blueprint_sched to avoid ODR clashes with
// the identically-named static helpers in sibling passes.

#ifndef __FLOWTRANSFER_INTERNAL_H__
#define __FLOWTRANSFER_INTERNAL_H__

#include "dfscheblueprintmanager.h"
#include "dfschedulemanager.h"
#include "hw/ResourceManager.h"
#include "hw/hwresource.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "routingmanager.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <memory>
#include <string>
#include <unordered_map>

namespace blueprint_sched {

using namespace mlir;

// ---------------------------------------------------------------------------
// Shared state between pre-processing, FlowTransferConversion, and post-processing.
// ---------------------------------------------------------------------------
struct BlueprintPassState {
    Value rootMemref;
    MemRefType rootMemrefType;
    SmallVector<int64_t> rootShape;
    Type elementType;
    // Map from each arith.constant result to its allocated memref.
    // When multiple data tensors exist (e.g. input A, input B, output C),
    // each has its own backing memref.
    llvm::DenseMap<Value, Value> constantToMemref;
    // Routing module attributes, cached before conversion (module attrs may be
    // stripped during applyPartialConversion).
    int64_t tileM = 0;
    int64_t tileRows = 0;
    int64_t tileN = 0;
    int64_t tileCols = 0;
    int64_t effectiveK = 0;
    int64_t fullK = 0;
    int64_t kRounds = 0;
};

// === Tiling Classification ===
// Determines whether the M/N-dimension tiling requires host-side SCF loops.
enum class TilingMode { Match, Multiple, Invalid };

struct TilingClassification {
    TilingMode mMode;
    int64_t mRounds; // 1 for Match, tileRows/tileM for Multiple
    TilingMode nMode;
    int64_t nRounds; // 1 for Match, tileCols/tileN for Multiple
};

// === Generic N-D output access-pattern descriptor ===
struct OutputTileDescriptor {
    // BD scatter dims, innermost-first: {strideBytes, wrap}.
    SmallVector<std::pair<int64_t, int64_t>> bdDims;
    // Per-round base-offset dims, OUTERMOST-first: {wrap(extent), strideBytes}.
    SmallVector<std::pair<int64_t, int64_t>> roundDims;
    int64_t perTileStrideBytes = 0; // per-tile channel-column step (offset += t*this)
    int64_t bdLenBytes = 0;         // -> perRoundBytes
    int64_t totalRounds = 1;        // -> scf.for ub (= product of roundDims wraps)
    int32_t iterStep = 0;           // -> oooIterStepSize
    int32_t iterWrap = 1;           // -> oooIterWrap
};

// Deferred core StartIoOp data: collected inside the per-tile loop and emitted
// AFTER LoadKernelGroup/LaunchKernelGroup so ELF BSS init does not overwrite DMA.
struct DeferredCoreStartIo {
    Value ioHandle;
    Value bdId;
    uint32_t flowIdx;
    int32_t repeatCount;
};

// Offset / repeat parameters computed by computeMultipleInputOffsetParams and
// consumed by the op-emission tail of emitScheduleMultipleInput.
struct MultipleInputOffsetParams {
    int64_t outerRounds = 1;
    int32_t perIterRepeat = 1;
    int64_t subTileStride = 0;
    bool halo2D = false;
    int64_t haloHStrideBytes = 0;
    int64_t haloWStrideBytes = 0;
    int64_t haloWRoundsVal = 0;
};

// ---------------------------------------------------------------------------
// Shared utilities (defined in flowtransfer_common.cpp). Promoted from the
// former file-local statics; kept in namespace blueprint_sched to avoid clashes.
// ---------------------------------------------------------------------------
int traceToFuncArgIndex(Value v);
int traceFlowConfigToFuncArgIndex(dfscheblueprint::FlowConfigOp flowConfig);

// lookupSymbolOp is a template so its definition must live in the header.
template <typename OpTy> OpTy lookupSymbolOp(Operation *rootOp, SymbolRefAttr target) {
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

dfscheblueprint::TileGroupOp lookupTileGroup(Operation *rootOp, SymbolRefAttr target);
dfscheblueprint::FlowConfigOp lookupFlowConfig(Operation *rootOp, SymbolRefAttr target);
dfscheblueprint::DataSliceOp lookupDataSlice(Operation *rootOp, SymbolRefAttr target);
bool hasDSKernelReceiver(Operation *rootOp, StringRef kernelName);
Operation *getModuleOp(Operation *rootOp);
void generateDSKernelReceiver(ConversionPatternRewriter &rewriter, Location loc, Operation *insertBeforeOp,
                              StringRef kernelName, RankedTensorType tensorType, int64_t bufferLen,
                              uint32_t basePacketId, int64_t coreChannel, uint32_t flowIndex);
SmallVector<OpFoldResult> toOpFoldResult(ArrayRef<int64_t> values, OpBuilder &b);
LogicalResult preprocessConstantToMemref(Operation *topLevel, std::shared_ptr<BlueprintPassState> state);
Value resolveMemrefForView(Value viewValue, const BlueprintPassState &state);
bool isFullConnectAuto(ModuleOp moduleOp);
TilingClassification classifyTiling(ModuleOp moduleOp);
bool isNOuterPolicy(ModuleOp moduleOp);
OutputTileDescriptor buildOutputTileDescriptor(const BlueprintPassState &passState, MemRefType memrefType,
                                               int64_t numCoreTiles, ModuleOp moduleOp, int64_t ooElementSizeBytes);

// ---------------------------------------------------------------------------
// FlowLoweringCtx — trunk state shared across the extracted host/kernel helpers.
// Only variables that cross the major section boundaries live here; section-local
// temporaries stay inside their owning method.
// ---------------------------------------------------------------------------
struct FlowTransferConversion; // fwd

struct FlowLoweringCtx {
    // Non-owning references into matchAndRewrite.
    ConversionPatternRewriter &rewriter;
    Location loc;
    dfscheblueprint::FlowTransferOp op;

    FlowLoweringCtx(ConversionPatternRewriter &rw, Location l, dfscheblueprint::FlowTransferOp o)
        : rewriter(rw), loc(l), op(o) {}

    // --- prologue: roles + memref resolve ---
    dfscheblueprint::FlowConfigOp shimFlowConfig = nullptr;
    dfscheblueprint::FlowConfigOp coreFlowConfig = nullptr;
    bool shimIsSender = false;
    dfscheblueprint::TileGroupOp shimTileGroup = nullptr;
    dfscheblueprint::TileGroupOp coreTileGroup = nullptr;
    uint32_t basePacketId = 0;
    uint32_t flowIndex = 0;
    Value viewValue;
    Type viewType;
    tensor::ExtractSliceOp partExtractSlice;
    RankedTensorType shimTensorType;
    Value flowRootMemref;
    Value partitionSubview;
    MemRefType memrefType;
    Value ddrBuffer;

    // --- shim tile + DMA params ---
    int64_t shimCol = 0, shimRow = 0;
    dfschedule::DeclareTileOp shimTileOp;
    int64_t shimChannel = 0;
    StringRef dmaDirection;
    StringRef ioOperation;
    int64_t bufferLen = 1;
    int64_t elementSizeBytesShim = 1;
    StringRef transferType;
    int64_t shimBdLen = 0;
    int32_t dataId = -1;
    bool isHaloSlab = false;
    bool kAccumHaloSlab = false;
    int64_t kAccumL2Rounds = 0, kAccumL2Step = 0, kAccumRowPitchElems = 0;
    int64_t l2CorePerRoundLen = 0;
    int64_t numCoreTiles = 1;
    int64_t perTileShimLen = 0;
    int32_t shimBdIdVal = -1;
    arith::ConstantOp shimBdIdConst;
    ArrayAttr shimDimStrides;
    ArrayAttr shimDimWraps;

    // --- shim BD (OOO / non-OOO) ---
    bool isManyToOne = false;
    bool useOOO = false;
    SmallVector<int32_t> shimPerTileBdIds;
    Value lastShimBdHandle;
    SmallVector<Value> shimBdHandles;
    int64_t ooElementSizeBytes = 1;
    int64_t ooFullPartitionElements = 0;
    int64_t ooPerCoreElements = 0;
    int64_t ooPingPongSize = 1;
    int64_t ooNumIterations = 0;
    int64_t perRoundBytes = 0;
    int32_t shimIterStepSize = 0;
    int32_t shimIterWrap = 0;
    bool usedMRounds3D = false;
    int64_t oooMRounds = 1;
    int32_t oooIterStepSize = 0;
    int32_t oooIterWrap = 0;
    int64_t perTileStrideFromDims = 0;
    ArrayAttr perTileDimStrides = nullptr;
    ArrayAttr perTileDimWraps = nullptr;
    OutputTileDescriptor outDesc;
    dfschedule::ConfigCreateIoOp createIoOp;

    // --- core tiles ---
    ArrayAttr coreTilesAttr;
    SmallVector<Value> coreTiles;
    int64_t coreChannel = 0;
    dfscheblueprint::bp_direction coreDmaDir = dfscheblueprint::bp_direction::S2MM;
    StringRef coreDmaDirection;
    StringRef coreIoOperation;
    std::optional<ArrayAttr> sliceSymbolsOpt;
    int64_t flowPingL1Offset = 0;
    int64_t flowPongL1Offset = 0;
    bool flowAddrsValid = false;
    bool isInput = false;
    int funcArgIdx = -1;
    int dirIdx = 0;
    SmallVector<Attribute> tileConfigDicts;
    SmallVector<DeferredCoreStartIo> deferredCoreStartIos;
    int tileIndex = 0;

    // --- kernel_config finalize + schedule ---
    SmallVector<Attribute> kernelConfigSymbols;
    SmallVector<Attribute> calleeAttrs;
    SmallVector<Attribute> computeKernelAttrs;
    TilingClassification classification;
    bool needsOuterLoop = false;
    bool fullConnect = true;
    int64_t haloL2RoundsForLoop = 0;
    int64_t haloKRoundsForLoop = 0;
    dfschedule::GetBdIdOp getBdIdOp;
};

// Per-core-tile locals, carried across the extracted kernel sub-methods so the
// per-tile loop body stays byte-identical while each sub-method is < 200 lines.
struct CoreTileCtx {
    int64_t col = 0, row = 0;
    dfschedule::DeclareTileOp coreTileOp;
    int64_t bufferSize = 0;
    int64_t elementSizeBytes = 1;
    int64_t perTileSize = 0;
    int64_t bufferOffset = 0;
    int64_t fullPartitionElements = 0;
    int64_t perCoreElements = 0;
    int64_t perCorePerKRound = 0;
    int ppDepth = 2;
    int64_t pingPongBufferSize = 1;
    bool hostSpatialHaloPort = false;
    bool kSplitSlabPort = false;
    int64_t numIterations = 0;
    int acquireLockId = 0;
    int releaseLockId = 0;
    // per-tile subview / BD emission
    MemRefType shapedPerTileType;
    Value perTileToken;
    int64_t pingL1Offset = 0;
    int64_t pongL1Offset = 0;
    int64_t coreBdLen = 0;
    bool isOutputFlow = false;
    int bdAcquireLockId = 0;
    int bdReleaseLockId = 0;
    bool coreBdEnablePacket = false;
    int32_t coreBdPacketId = 0;
    int32_t coreOooBdId = -1;
    Value firstCoreBdHandle;
};

// ---------------------------------------------------------------------------
// The conversion pattern. Extracted helper member methods are declared here and
// defined across flowtransfer_host.cpp / flowtransfer_kernel.cpp.
// ---------------------------------------------------------------------------
struct FlowTransferConversion : public OpConversionPattern<dfscheblueprint::FlowTransferOp> {
    std::shared_ptr<ResourceMgr> resourceMgr;
    std::shared_ptr<BlueprintPassState> passState;
    double bufferRatio;
    int64_t maxPingPongBytes;
    // Buffer index mapping keyed by data_id (see original comment).
    mutable std::unordered_map<int32_t, int> dataIdToInputIdx;
    mutable std::unordered_map<int32_t, int> dataIdToOutputIdx;
    mutable int nextInputIdx = 0;
    mutable int nextOutputIdx = 0;

    FlowTransferConversion(MLIRContext *ctx, std::shared_ptr<ResourceMgr> mgr,
                           std::shared_ptr<BlueprintPassState> state, double ratio, int64_t maxPPBytes)
        : OpConversionPattern<dfscheblueprint::FlowTransferOp>(ctx), resourceMgr(std::move(mgr)),
          passState(std::move(state)), bufferRatio(ratio), maxPingPongBytes(maxPPBytes) {}

    LogicalResult matchAndRewrite(dfscheblueprint::FlowTransferOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override;

    // --- host helpers (flowtransfer_host.cpp) ---
    void emitShimTileAndParams(FlowLoweringCtx &c) const;
    LogicalResult computeShimBdParams(FlowLoweringCtx &c) const;
    void emitShimBdOoo(FlowLoweringCtx &c) const;
    void emitShimBdNonOoo(FlowLoweringCtx &c) const;
    void classifyScheduleMode(FlowLoweringCtx &c) const;
    LogicalResult emitScheduleMultipleInput(FlowLoweringCtx &c) const;
    // Sub-helper of emitScheduleMultipleInput: pure offset/repeat parameter
    // computation (no op emission). Fills MultipleInputOffsetParams.
    void computeMultipleInputOffsetParams(FlowLoweringCtx &c, ModuleOp moduleOp, bool nOuterPolicy,
                                          struct MultipleInputOffsetParams &p) const;
    void emitScheduleOooOutput(FlowLoweringCtx &c) const;
    void emitScheduleStraightLine(FlowLoweringCtx &c) const;

    // --- kernel helpers (flowtransfer_kernel.cpp) ---
    LogicalResult emitCoreTileConfigs(FlowLoweringCtx &c) const;
    LogicalResult emitCoreTileParams(FlowLoweringCtx &c, CoreTileCtx &t) const;
    // Sub-helper of emitCoreTileParams: clamp buffer + compute numIterations
    // (halo / K-round / K-split) + validate pp_depth=1. No op emission.
    LogicalResult computeCoreIterations(FlowLoweringCtx &c, CoreTileCtx &t) const;
    LogicalResult emitCoreBufferDma(FlowLoweringCtx &c, CoreTileCtx &t) const;
    // Sub-helper of emitCoreBufferDma: CoreMemAllocator ping/pong address
    // allocation (first tile only) + per-tile ping/pong L1 offset assignment.
    void emitCoreBufferAlloc(FlowLoweringCtx &c, CoreTileCtx &t) const;
    void emitCoreSingleBufferBd(FlowLoweringCtx &c, CoreTileCtx &t) const;
    void emitCorePingPongBd(FlowLoweringCtx &c, CoreTileCtx &t) const;
    void finalizeKernelConfig(FlowLoweringCtx &c) const;
};

} // namespace blueprint_sched

#endif // __FLOWTRANSFER_INTERNAL_H__
