/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "passdfscheduleprovenancemap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace mlir;
using namespace dfschedule;

namespace mlir {

// ---------------------------------------------------------------------------
// Simple JSON writer
// ---------------------------------------------------------------------------
class DfscheJsonWriter {
    llvm::raw_ostream &os;
    int indentLevel = 0;
    bool needsComma = false;

    void writeIndent() {
        for (int i = 0; i < indentLevel; ++i)
            os << "  ";
    }

    void writeCommaIfNeeded() {
        if (needsComma)
            os << ",";
        os << "\n";
    }

  public:
    explicit DfscheJsonWriter(llvm::raw_ostream &os) : os(os) {}

    void beginObject() {
        writeCommaIfNeeded();
        writeIndent();
        os << "{";
        indentLevel++;
        needsComma = false;
    }

    void endObject() {
        os << "\n";
        indentLevel--;
        writeIndent();
        os << "}";
        needsComma = true;
    }

    void beginArray(StringRef key) {
        writeCommaIfNeeded();
        writeIndent();
        os << "\"" << key << "\": [";
        indentLevel++;
        needsComma = false;
    }

    void endArray() {
        os << "\n";
        indentLevel--;
        writeIndent();
        os << "]";
        needsComma = true;
    }

    void keyValue(StringRef k, int64_t v) {
        writeCommaIfNeeded();
        writeIndent();
        os << "\"" << k << "\": " << v;
        needsComma = true;
    }

    void keyValue(StringRef k, StringRef v) {
        writeCommaIfNeeded();
        writeIndent();
        os << "\"" << k << "\": \"" << v << "\"";
        needsComma = true;
    }

    void keyValueBool(StringRef k, bool v) {
        writeCommaIfNeeded();
        writeIndent();
        os << "\"" << k << "\": " << (v ? "true" : "false");
        needsComma = true;
    }

    void intInArray(int64_t v) {
        writeCommaIfNeeded();
        writeIndent();
        os << v;
        needsComma = true;
    }

    void beginRoot() {
        os << "{";
        indentLevel++;
        needsComma = false;
    }

    void endRoot() { os << "\n}\n"; }
};

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

struct BdConfig {
    int bdIdConst = -1;
    int64_t bufferOffset = 0;
    int len = 0;
    bool enablePacket = false;
    int packetId = 0;
    int nextBd = 0;
    int acquireLockId = -1;
    int acquireLockVal = 0;
    int releaseLockId = -1;
    int releaseLockVal = 0;
    int outOfOrderBdId = -1;
    std::vector<int> dimStrides;
    std::vector<int> dimWraps;
    int iterStepSize = 0;
    int iterWrap = 0;
    int tileCol = -1;
    int tileRow = -1;
};

struct IoConfig {
    int channel = -1;
    std::string direction;
    std::string ioOperation;
    bool enableOutOfOrder = false;
    int tileCol = -1;
    int tileRow = -1;
    std::vector<BdConfig> bdChain;
};

struct StartIoEntry {
    int flowIndex = -1;
    int repeatCount = 1;
    bool insideScfFor = false;
    int64_t loopLb = 0;
    int64_t loopUb = 0;
    int tileCol = -1;
    int tileRow = -1;
    // Copy of IoConfig for this start_io (no pointers)
    IoConfig ioConfig;
    bool hasIoConfig = false;
};

struct KernelConfigEntry {
    std::string symName;
    int flowIndex = -1;
    int packetId = 0;
    int dmaChannel = 0;
    int bufferMode = 0;
    int numBuffers = 0;
    int acquireLockId = -1;
    int releaseLockId = -1;
    int bufferOffset = 0;
    int bufferSize = 0;
    int elementSize = 0;
    int numIterations = 0;
    int tileIndex = -1;
};

struct LoadKernelGroupInfo {
    std::string callee;
    std::vector<std::pair<int, int>> tiles;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static int getConstI32(Value v) {
    if (auto constOp = v.getDefiningOp<arith::ConstantIntOp>())
        return constOp.value();
    if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
        if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
            return intAttr.getInt();
    }
    return -1;
}

static std::pair<int, int> resolveTile(Value tileVal) {
    if (auto declOp = tileVal.getDefiningOp<DeclareTileOp>())
        return {declOp.getCol(), declOp.getRow()};
    return {-1, -1};
}

static int64_t resolveBufferOffset(Value bufVal) {
    if (auto bindOp = bufVal.getDefiningOp<BindCoreBufferOp>())
        return bindOp.getOffset();
    return 0;
}

static std::vector<BdConfig> collectBdChain(Value bdHandleVal) {
    std::vector<BdConfig> chain;
    Value current = bdHandleVal;
    while (current) {
        auto bdOp = current.getDefiningOp<ConfigDmaBdOp>();
        if (!bdOp)
            break;

        BdConfig bd;
        bd.bdIdConst = getConstI32(bdOp.getBdId());
        bd.bufferOffset = resolveBufferOffset(bdOp.getBuffer());
        bd.len = bdOp.getLen();
        bd.enablePacket = bdOp.getEnablePacket();
        bd.packetId = bdOp.getPacketId();
        bd.nextBd = bdOp.getNextBd();
        bd.acquireLockId = bdOp.getAcquireLockId();
        bd.acquireLockVal = bdOp.getAcquireLockVal();
        bd.releaseLockId = bdOp.getReleaseLockId();
        bd.releaseLockVal = bdOp.getReleaseLockVal();
        bd.outOfOrderBdId = bdOp.getOutOfOrderBdId();

        if (auto strides = bdOp.getDimStrides()) {
            for (auto s : strides->getAsRange<IntegerAttr>())
                bd.dimStrides.push_back(s.getInt());
        }
        if (auto wraps = bdOp.getDimWraps()) {
            for (auto w : wraps->getAsRange<IntegerAttr>())
                bd.dimWraps.push_back(w.getInt());
        }
        bd.iterStepSize = bdOp.getIterStepSize();
        bd.iterWrap = bdOp.getIterWrap();

        auto [col, row] = resolveTile(bdOp.getTile());
        bd.tileCol = col;
        bd.tileRow = row;

        chain.push_back(bd);

        if (bdOp.getLinkedBd())
            current = bdOp.getLinkedBd();
        else
            current = Value();
    }
    std::reverse(chain.begin(), chain.end());
    return chain;
}

static IoConfig buildIoConfig(ConfigCreateIoOp createIoOp) {
    IoConfig io;
    io.channel = createIoOp.getChannel();
    io.direction = createIoOp.getDirection().str();
    io.ioOperation = createIoOp.getIoOperation().str();
    io.enableOutOfOrder = createIoOp.getEnableOutOfOrder();
    auto [col, row] = resolveTile(createIoOp.getTile());
    io.tileCol = col;
    io.tileRow = row;
    io.bdChain = collectBdChain(createIoOp.getBdConfig());
    return io;
}

static std::string tileStr(int col, int row) { return "(" + std::to_string(col) + "," + std::to_string(row) + ")"; }

// ---------------------------------------------------------------------------
// Invariant checks
// ---------------------------------------------------------------------------
struct InvariantCheck {
    std::string check;
    std::string tile;
    std::string channel;
    std::string result;
};

static InvariantCheck checkPingPongChain(const IoConfig &io) {
    InvariantCheck chk;
    chk.check = "ping-pong-chain";
    chk.tile = tileStr(io.tileCol, io.tileRow);
    chk.channel = io.direction + " ch" + std::to_string(io.channel);

    if (io.bdChain.size() < 2) {
        chk.result = "SKIP: single BD (no ping-pong)";
        return chk;
    }
    bool cycleOk = true;
    for (size_t i = 0; i < io.bdChain.size(); i++) {
        int expectedNextId = (i + 1 < io.bdChain.size()) ? io.bdChain[i + 1].bdIdConst : io.bdChain[0].bdIdConst;
        if (io.bdChain[i].nextBd != expectedNextId)
            cycleOk = false;
    }
    if (cycleOk) {
        std::string ids;
        for (size_t i = 0; i < io.bdChain.size(); i++) {
            if (i > 0)
                ids += "->";
            ids += "BD" + std::to_string(io.bdChain[i].bdIdConst);
        }
        ids += "->BD" + std::to_string(io.bdChain[0].bdIdConst);
        chk.result = "OK: " + ids + " cycle";
    } else {
        chk.result = "WARN: next_bd chain may not form cycle";
    }
    return chk;
}

static InvariantCheck checkLockSymmetry(const IoConfig &io) {
    InvariantCheck chk;
    chk.check = "lock-symmetry";
    chk.tile = tileStr(io.tileCol, io.tileRow);
    chk.channel = io.direction + " ch" + std::to_string(io.channel);

    if (io.bdChain.empty()) {
        chk.result = "SKIP: no BDs";
        return chk;
    }

    int acqId = io.bdChain[0].acquireLockId;
    int relId = io.bdChain[0].releaseLockId;
    int acqVal = io.bdChain[0].acquireLockVal;
    int relVal = io.bdChain[0].releaseLockVal;
    bool consistent = true;
    for (const auto &bd : io.bdChain) {
        if (bd.acquireLockId != acqId || bd.releaseLockId != relId || bd.acquireLockVal != acqVal ||
            bd.releaseLockVal != relVal) {
            consistent = false;
            break;
        }
    }
    if (consistent) {
        chk.result = "OK: acq lock" + std::to_string(acqId) + "(" + std::to_string(acqVal) + "), rel lock" +
                     std::to_string(relId) + "(" + std::to_string(relVal) + ")";
    } else {
        chk.result = "WARN: inconsistent lock IDs/values across BD chain";
    }
    return chk;
}

// ---------------------------------------------------------------------------
// Main pass implementation
// ---------------------------------------------------------------------------
void DfscheduleProvenanceMapPass::runOnOperation() {
    auto op = getOperation();
    auto module = dyn_cast<ModuleOp>(op);
    if (!module) {
        // Try walking parent
        op->walk([&](ModuleOp mod) { module = mod; });
        if (!module)
            return;
    }

    // === Extract module attributes ===
    int64_t tileM = 0, tileN = 0, effectiveK = 0, fullK = 0;
    int64_t kRounds = 0, mRounds = 0, nRounds = 0;
    if (auto attr = module->getAttrOfType<IntegerAttr>("routing.tile_m"))
        tileM = attr.getInt();
    if (auto attr = module->getAttrOfType<IntegerAttr>("routing.tile_n"))
        tileN = attr.getInt();
    if (auto attr = module->getAttrOfType<IntegerAttr>("routing.effective_k"))
        effectiveK = attr.getInt();
    if (auto attr = module->getAttrOfType<IntegerAttr>("routing.full_k"))
        fullK = attr.getInt();
    if (auto attr = module->getAttrOfType<IntegerAttr>("routing.k_rounds"))
        kRounds = attr.getInt();
    if (auto attr = module->getAttrOfType<IntegerAttr>("routing.m_rounds"))
        mRounds = attr.getInt();
    if (auto attr = module->getAttrOfType<IntegerAttr>("routing.n_rounds"))
        nRounds = attr.getInt();

    // === Find host block ===
    HostBlockOp hostBlock;
    module->walk([&](HostBlockOp hb) { hostBlock = hb; });
    if (!hostBlock) {
        std::cout << "[DfscheduleProvenanceMapPass] No dfschedule.host block found, skipping.\n";
        return;
    }

    // === Collect all create_io as IoConfig values, keyed by Operation* ===
    std::map<Operation *, IoConfig> ioMap;
    hostBlock->walk([&](ConfigCreateIoOp createIoOp) { ioMap[createIoOp.getOperation()] = buildIoConfig(createIoOp); });

    // === Collect all start_io entries ===
    // Walk the host block body directly; handle scf.for nesting
    std::vector<StartIoEntry> startIoEntries;

    auto processStartIo = [&](StartIoOp startOp, bool inLoop, int64_t loopLb, int64_t loopUb) {
        StartIoEntry entry;
        entry.flowIndex = startOp.getFlowIndex();
        entry.repeatCount = startOp.getRepeatCount();
        entry.insideScfFor = inLoop;
        entry.loopLb = loopLb;
        entry.loopUb = loopUb;

        auto ioHandleOp = startOp.getIoHandle().getDefiningOp();
        if (ioHandleOp) {
            auto it = ioMap.find(ioHandleOp);
            if (it != ioMap.end()) {
                entry.tileCol = it->second.tileCol;
                entry.tileRow = it->second.tileRow;
                entry.ioConfig = it->second; // copy, not pointer
                entry.hasIoConfig = true;
            }
        }
        startIoEntries.push_back(entry);
    };

    Block &hostBody = hostBlock.getBody().front();
    for (auto &bodyOp : hostBody.getOperations()) {
        if (auto startOp = dyn_cast<StartIoOp>(&bodyOp)) {
            processStartIo(startOp, false, 0, 0);
        } else if (auto forOp = dyn_cast<scf::ForOp>(&bodyOp)) {
            int64_t lb = 0, ub = 0;
            if (auto lbConst = forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>())
                lb = lbConst.value();
            if (auto ubConst = forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>())
                ub = ubConst.value();
            for (auto &innerOp : forOp.getBody()->getOperations()) {
                if (auto startOp = dyn_cast<StartIoOp>(&innerOp))
                    processStartIo(startOp, true, lb, ub);
            }
        }
    }

    // === Build per-tile DMA channel list from ioMap ===
    // Use a stable structure: tile -> vector<IoConfig>
    struct TileData {
        int col = -1, row = -1;
        std::string type;
        std::vector<IoConfig> dmaChannels;
    };
    std::map<std::pair<int, int>, TileData> tileDataMap;

    // Initialize from declaretile
    hostBlock->walk([&](DeclareTileOp declOp) {
        auto key = std::make_pair((int)declOp.getCol(), (int)declOp.getRow());
        if (tileDataMap.find(key) == tileDataMap.end()) {
            TileData td;
            td.col = declOp.getCol();
            td.row = declOp.getRow();
            td.type = (td.row == 0) ? "shim" : "core";
            tileDataMap[key] = td;
        }
    });

    // Add DMA channels (only from top-level create_io, not loop-internal ones,
    // which are dynamic shim BDs — handle those via start_io entries)
    for (auto &[opPtr, io] : ioMap) {
        auto key = std::make_pair(io.tileCol, io.tileRow);
        // Ensure tile exists
        if (tileDataMap.find(key) == tileDataMap.end()) {
            TileData td;
            td.col = io.tileCol;
            td.row = io.tileRow;
            td.type = (td.row == 0) ? "shim" : "core";
            tileDataMap[key] = td;
        }
        tileDataMap[key].dmaChannels.push_back(io);
    }

    // === Collect kernel configs ===
    std::vector<KernelConfigEntry> kernelConfigs;
    hostBlock->walk([&](DeclareKernelConfigOp kcOp) {
        auto tileConfigsAttr = kcOp.getTileConfigs();
        for (auto cfgAttr : tileConfigsAttr) {
            auto dict = dyn_cast<DictionaryAttr>(cfgAttr);
            if (!dict)
                continue;

            KernelConfigEntry kc;
            kc.symName = kcOp.getSymName().str();
            if (auto a = dict.getAs<IntegerAttr>("flow_index"))
                kc.flowIndex = a.getInt();
            if (auto a = dict.getAs<IntegerAttr>("packet_id"))
                kc.packetId = a.getInt();
            if (auto a = dict.getAs<IntegerAttr>("dma_channel"))
                kc.dmaChannel = a.getInt();
            if (auto a = dict.getAs<IntegerAttr>("buffer_mode"))
                kc.bufferMode = a.getInt();
            if (auto a = dict.getAs<IntegerAttr>("num_buffers"))
                kc.numBuffers = a.getInt();
            if (auto a = dict.getAs<IntegerAttr>("acquire_lock_id"))
                kc.acquireLockId = a.getInt();
            if (auto a = dict.getAs<IntegerAttr>("release_lock_id"))
                kc.releaseLockId = a.getInt();
            if (auto a = dict.getAs<IntegerAttr>("buffer_offset"))
                kc.bufferOffset = a.getInt();
            if (auto a = dict.getAs<IntegerAttr>("buffer_size"))
                kc.bufferSize = a.getInt();
            if (auto a = dict.getAs<IntegerAttr>("element_size"))
                kc.elementSize = a.getInt();
            if (auto a = dict.getAs<IntegerAttr>("num_iterations"))
                kc.numIterations = a.getInt();
            if (auto a = dict.getAs<IntegerAttr>("tile_index"))
                kc.tileIndex = a.getInt();

            kernelConfigs.push_back(kc);
        }
    });

    // === Collect load_kernel_group ===
    LoadKernelGroupInfo loadKgInfo;
    hostBlock->walk([&](LoadKernelGroupOp lgOp) {
        auto calleeArr = lgOp.getCallee();
        if (!calleeArr.empty())
            loadKgInfo.callee = cast<FlatSymbolRefAttr>(calleeArr[0]).getValue().str();
        for (auto tile : lgOp.getTiles()) {
            auto [c, r] = resolveTile(tile);
            loadKgInfo.tiles.push_back({c, r});
        }
    });

    // === Invariant checks ===
    std::vector<InvariantCheck> invariantChecks;
    for (auto &[key, td] : tileDataMap) {
        for (auto &io : td.dmaChannels) {
            invariantChecks.push_back(checkPingPongChain(io));
            invariantChecks.push_back(checkLockSymmetry(io));
        }
    }

    // === Write JSON ===
    std::string jsonPath = outputDir + "/dfscheduleprovenancemap.json";
    std::error_code ec;
    llvm::raw_fd_ostream fileOs(jsonPath, ec, llvm::sys::fs::OF_None);
    if (ec) {
        llvm::errs() << "Failed to write dfschedule provenance map to " << jsonPath << ": " << ec.message() << "\n";
        return;
    }

    DfscheJsonWriter jw(fileOs);
    jw.beginRoot();

    jw.keyValue("version", 1);

    // partition origin (absolute physical start column); phys_col = col + startcol
    if (startCol >= 0)
        jw.keyValue("startcol", startCol);

    // compiler aie-gen (raw --aie-version value); Python tools map it to debug offset-map
    if (!aieGen.empty())
        jw.keyValue("aie_gen", StringRef(aieGen));

    // module_attrs
    {
        jw.beginArray("module_attrs");
        jw.beginObject();
        jw.keyValue("tile_m", tileM);
        jw.keyValue("tile_n", tileN);
        jw.keyValue("effective_k", effectiveK);
        jw.keyValue("full_k", fullK);
        jw.keyValue("k_rounds", kRounds);
        jw.keyValue("m_rounds", mRounds);
        jw.keyValue("n_rounds", nRounds);
        jw.endObject();
        jw.endArray();
    }

    // tiles
    {
        jw.beginArray("tiles");
        std::vector<std::pair<int, int>> sortedKeys;
        for (auto &[k, _] : tileDataMap)
            sortedKeys.push_back(k);
        std::sort(sortedKeys.begin(), sortedKeys.end());

        for (auto &key : sortedKeys) {
            auto &td = tileDataMap[key];
            if (td.dmaChannels.empty())
                continue;

            jw.beginObject();
            jw.keyValue("col", td.col);
            jw.keyValue("row", td.row);
            jw.keyValue("type", StringRef(td.type));

            jw.beginArray("dma_channels");
            for (size_t chIdx = 0; chIdx < td.dmaChannels.size(); chIdx++) {
                auto &io = td.dmaChannels[chIdx];
                jw.beginObject();
                jw.keyValue("channel", io.channel);
                jw.keyValue("direction", StringRef(io.direction));
                jw.keyValueBool("enable_out_of_order", io.enableOutOfOrder);

                // Find matching start_io for flow_index / repeat_count
                int flowIndex = -1;
                int repeatCount = 1;
                bool inScfFor = false;
                int64_t loopLb = 0, loopUb = 0;
                for (auto &entry : startIoEntries) {
                    if (!entry.hasIoConfig)
                        continue;
                    // Match by tile + channel + direction
                    if (entry.ioConfig.tileCol == io.tileCol && entry.ioConfig.tileRow == io.tileRow &&
                        entry.ioConfig.channel == io.channel && entry.ioConfig.direction == io.direction) {
                        flowIndex = entry.flowIndex;
                        repeatCount = entry.repeatCount;
                        inScfFor = entry.insideScfFor;
                        loopLb = entry.loopLb;
                        loopUb = entry.loopUb;
                        break;
                    }
                }
                if (flowIndex >= 0)
                    jw.keyValue("flow_index", flowIndex);

                // bd_chain
                jw.beginArray("bd_chain");
                for (auto &bd : io.bdChain) {
                    jw.beginObject();
                    if (bd.bdIdConst >= 0)
                        jw.keyValue("bd_id", bd.bdIdConst);
                    else
                        jw.keyValue("bd_id", StringRef("runtime"));
                    jw.keyValue("buffer_offset", bd.bufferOffset);
                    jw.keyValue("len", bd.len);
                    jw.keyValueBool("enable_packet", bd.enablePacket);
                    jw.keyValue("packet_id", bd.packetId);
                    jw.keyValue("next_bd", bd.nextBd);

                    jw.beginArray("acquire_lock");
                    jw.beginObject();
                    jw.keyValue("id", bd.acquireLockId);
                    jw.keyValue("val", bd.acquireLockVal);
                    jw.endObject();
                    jw.endArray();

                    jw.beginArray("release_lock");
                    jw.beginObject();
                    jw.keyValue("id", bd.releaseLockId);
                    jw.keyValue("val", bd.releaseLockVal);
                    jw.endObject();
                    jw.endArray();

                    if (bd.outOfOrderBdId >= 0)
                        jw.keyValue("out_of_order_bd_id", bd.outOfOrderBdId);

                    if (!bd.dimStrides.empty()) {
                        jw.beginArray("dim_strides");
                        for (auto s : bd.dimStrides)
                            jw.intInArray(s);
                        jw.endArray();
                    }
                    if (!bd.dimWraps.empty()) {
                        jw.beginArray("dim_wraps");
                        for (auto w : bd.dimWraps)
                            jw.intInArray(w);
                        jw.endArray();
                    }
                    if (bd.iterStepSize > 0)
                        jw.keyValue("iter_step_size", bd.iterStepSize);
                    if (bd.iterWrap > 0)
                        jw.keyValue("iter_wrap", bd.iterWrap);

                    jw.endObject();
                }
                jw.endArray(); // bd_chain

                // start_io
                jw.beginArray("start_io");
                jw.beginObject();
                jw.keyValue("repeat_count", repeatCount);
                if (inScfFor) {
                    jw.keyValueBool("inside_scf_for", true);
                    std::string loopRange = std::to_string(loopLb) + ".." + std::to_string(loopUb);
                    jw.keyValue("loop_range", StringRef(loopRange));
                }
                jw.endObject();
                jw.endArray();

                // contract
                {
                    std::string contract = io.direction + " ch" + std::to_string(io.channel) + ": ";
                    if (io.bdChain.size() >= 2)
                        contract += "ping-pong ";
                    contract += (io.ioOperation == "SEND") ? "send" : "receive";
                    if (!io.bdChain.empty()) {
                        contract += ", " + std::to_string(io.bdChain[0].len) + "B each";
                        if (io.bdChain[0].enablePacket)
                            contract += ", pkt_id=" + std::to_string(io.bdChain[0].packetId);
                        if (io.bdChain[0].outOfOrderBdId >= 0)
                            contract += ", ooo_bd=" + std::to_string(io.bdChain[0].outOfOrderBdId);
                        contract += ", lock " + std::to_string(io.bdChain[0].acquireLockId) + "/" +
                                    std::to_string(io.bdChain[0].releaseLockId);
                    }
                    if (repeatCount > 1)
                        contract += ", repeat=" + std::to_string(repeatCount);
                    jw.keyValue("contract", StringRef(contract));
                }

                jw.endObject(); // dma_channel
            }
            jw.endArray(); // dma_channels

            jw.endObject(); // tile
        }
        jw.endArray(); // tiles
    }

    // kernel_configs
    {
        jw.beginArray("kernel_configs");
        for (auto &kc : kernelConfigs) {
            jw.beginObject();
            jw.keyValue("sym_name", StringRef(kc.symName));
            jw.keyValue("flow_index", kc.flowIndex);
            jw.keyValue("packet_id", kc.packetId);
            jw.keyValue("dma_channel", kc.dmaChannel);
            jw.keyValue("buffer_mode", kc.bufferMode);
            jw.keyValue("num_buffers", kc.numBuffers);
            jw.keyValue("acquire_lock_id", kc.acquireLockId);
            jw.keyValue("release_lock_id", kc.releaseLockId);
            jw.keyValue("buffer_offset", kc.bufferOffset);
            jw.keyValue("buffer_size", kc.bufferSize);
            jw.keyValue("element_size", kc.elementSize);
            jw.keyValue("num_iterations", kc.numIterations);
            jw.keyValue("tile_index", kc.tileIndex);
            jw.endObject();
        }
        jw.endArray();
    }

    // load_kernel_group
    if (!loadKgInfo.callee.empty()) {
        jw.beginArray("load_kernel_group");
        jw.beginObject();
        jw.keyValue("callee", StringRef(loadKgInfo.callee));
        jw.beginArray("tiles");
        for (auto &[c, r] : loadKgInfo.tiles) {
            jw.beginObject();
            jw.keyValue("col", c);
            jw.keyValue("row", r);
            jw.endObject();
        }
        jw.endArray();
        jw.endObject();
        jw.endArray();
    }

    // flow_summary
    {
        std::map<int, std::vector<const StartIoEntry *>> flowGroups;
        for (auto &entry : startIoEntries)
            flowGroups[entry.flowIndex].push_back(&entry);

        jw.beginArray("flow_summary");
        for (auto &[fi, entries] : flowGroups) {
            jw.beginObject();
            jw.keyValue("flow_index", fi);

            std::string dir = "unknown";
            if (!entries.empty() && entries[0]->hasIoConfig) {
                dir = (entries[0]->ioConfig.direction == "S2MM") ? "input" : "output";
            }
            jw.keyValue("direction", StringRef(dir));

            jw.beginArray("entries");
            for (auto *e : entries) {
                jw.beginObject();
                jw.keyValue("tile_col", e->tileCol);
                jw.keyValue("tile_row", e->tileRow);
                jw.keyValue("repeat_count", e->repeatCount);
                if (e->insideScfFor) {
                    jw.keyValueBool("inside_scf_for", true);
                    std::string lr = std::to_string(e->loopLb) + ".." + std::to_string(e->loopUb);
                    jw.keyValue("loop_range", StringRef(lr));
                }
                if (e->hasIoConfig) {
                    jw.keyValue("channel", e->ioConfig.channel);
                    jw.keyValue("io_direction", StringRef(e->ioConfig.direction));
                    if (!e->ioConfig.bdChain.empty())
                        jw.keyValue("bd_len", e->ioConfig.bdChain[0].len);
                }
                jw.endObject();
            }
            jw.endArray();

            jw.endObject();
        }
        jw.endArray();
    }

    // invariant_checks
    {
        jw.beginArray("invariant_checks");
        for (auto &chk : invariantChecks) {
            jw.beginObject();
            jw.keyValue("check", StringRef(chk.check));
            jw.keyValue("tile", StringRef(chk.tile));
            jw.keyValue("channel", StringRef(chk.channel));
            jw.keyValue("result", StringRef(chk.result));
            jw.endObject();
        }
        jw.endArray();
    }

    jw.endRoot();
    fileOs.close();

    std::cout << "Dfschedule provenance map written to " << jsonPath << std::endl;
}

} // namespace mlir
