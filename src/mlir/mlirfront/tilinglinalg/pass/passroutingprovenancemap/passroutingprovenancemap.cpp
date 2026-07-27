/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "passroutingprovenancemap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include <string>
#include <vector>

using namespace mlir;
using namespace routinghw;

namespace mlir {

// ---------------------------------------------------------------------------
// Simple JSON writer (no external JSON library dependency).
// Mirrors the writer used by the dmaphop / dfschedule provenance passes.
// ---------------------------------------------------------------------------
class JsonWriter {
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
    explicit JsonWriter(llvm::raw_ostream &os) : os(os) {}

    void beginObject() {
        writeCommaIfNeeded();
        writeIndent();
        os << "{";
        indentLevel++;
        needsComma = false;
    }

    void beginObjectInline() {
        if (needsComma)
            os << ",";
        os << "\n";
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

    void key(StringRef k) {
        writeCommaIfNeeded();
        writeIndent();
        os << "\"" << k << "\": ";
        needsComma = false;
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

    void stringInArray(StringRef v) {
        writeCommaIfNeeded();
        writeIndent();
        os << "\"" << v << "\"";
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
// Generic attribute readers (never assume a specific generated accessor name).
// ---------------------------------------------------------------------------
static int64_t getIntAttr(Operation *op, StringRef name, int64_t dflt = -1) {
    if (auto a = op->getAttrOfType<IntegerAttr>(name))
        return a.getInt();
    return dflt;
}

static std::string getStrAttr(Operation *op, StringRef name, StringRef dflt = "") {
    if (auto a = op->getAttrOfType<StringAttr>(name))
        return a.getValue().str();
    return dflt.str();
}

// ---------------------------------------------------------------------------
// A physical tile (resolved from a tilecreate / ioshimtilecreate handle).
// ---------------------------------------------------------------------------
struct TileInfo {
    int64_t col = -1;
    int64_t row = -1;
    std::string kind = "unknown"; // "core" | "shim" | "unknown"
    std::string comments;
    // shim-only extras
    bool isShim = false;
    int64_t ioid = -1;
    int64_t dmaDirection = -1;
    int64_t channelUsed = -1;
};

// Resolve an i32 tile handle SSA value to its (col,row) + tile kind by looking
// at the routinghw.tilecreate / routinghw.ioshimtilecreate op that defines it.
static TileInfo resolveTile(Value tileHandle) {
    TileInfo info;
    if (!tileHandle)
        return info;
    Operation *defOp = tileHandle.getDefiningOp();
    if (!defOp)
        return info;
    if (auto shim = dyn_cast<IOShimTileCreate>(defOp)) {
        info.col = getIntAttr(shim, "col");
        info.row = getIntAttr(shim, "row");
        info.comments = getStrAttr(shim, "comments");
        info.kind = "shim";
        info.isShim = true;
        info.ioid = getIntAttr(shim, "IOID");
        info.dmaDirection = getIntAttr(shim, "dmadirection");
        info.channelUsed = getIntAttr(shim, "channelused");
    } else if (auto core = dyn_cast<TileCreate>(defOp)) {
        info.col = getIntAttr(core, "col");
        info.row = getIntAttr(core, "row");
        info.comments = getStrAttr(core, "comments");
        info.kind = "core";
    }
    return info;
}

static void writeTileRef(JsonWriter &jw, StringRef key, const TileInfo &t) {
    jw.key(key);
    jw.beginObject();
    jw.keyValue("col", t.col);
    jw.keyValue("row", t.row);
    jw.keyValue("type", StringRef(t.kind));
    jw.endObject();
}

// Write a {dir, idx, ...} port sub-object.
static void writePort(JsonWriter &jw, StringRef key, StringRef dir, int64_t idx) {
    jw.key(key);
    jw.beginObject();
    jw.keyValue("dir", dir);
    jw.keyValue("idx", idx);
    jw.endObject();
}

// ---------------------------------------------------------------------------
// Emit one stream-switch / enable connection op as a JSON object. Returns true
// if the op was recognized as a routing connection op.
// ---------------------------------------------------------------------------
static bool writeConnectionOp(JsonWriter &jw, Operation *op) {
    if (auto c = dyn_cast<ConnectStreamSingleSwitchPort>(op)) {
        TileInfo t = resolveTile(op->getOperand(0));
        jw.beginObjectInline();
        jw.keyValue("kind", "circuit_connect");
        writeTileRef(jw, "tile", t);
        writePort(jw, "slave", getStrAttr(op, "slaveportdirection"), getIntAttr(op, "slaveportidx"));
        writePort(jw, "master", getStrAttr(op, "masterportdirection"), getIntAttr(op, "masterportidx"));
        jw.endObject();
        return true;
    }
    if (auto c = dyn_cast<ConnectStreamPktSwitchPort>(op)) {
        TileInfo t = resolveTile(op->getOperand(0));
        jw.beginObjectInline();
        jw.keyValue("kind", "packet_connect");
        writeTileRef(jw, "tile", t);
        // receiving slave slot
        jw.key("recv_slave");
        jw.beginObject();
        jw.keyValue("dir", getStrAttr(op, "receiveslavedirection"));
        jw.keyValue("idx", getIntAttr(op, "receiveslaveportidx"));
        jw.keyValue("pktid", getIntAttr(op, "receiveslavepktid"));
        jw.keyValue("pkttype", getIntAttr(op, "receiveslavepkttype"));
        jw.endObject();
        // local DMA slot
        jw.key("local_dma");
        jw.beginObject();
        jw.keyValue("dir", getStrAttr(op, "localdmadirection"));
        jw.keyValue("idx", getIntAttr(op, "localdmaportidx"));
        jw.keyValue("pktid", getIntAttr(op, "localdmapktid"));
        jw.keyValue("pkttype", getIntAttr(op, "localdmapkttype"));
        jw.endObject();
        // forwarding master port
        writePort(jw, "forward_master", getStrAttr(op, "forwardmasterdirection"),
                  getIntAttr(op, "forwardmasterportidx"));
        bool preserve = false;
        if (auto a = op->getAttrOfType<BoolAttr>("preserveheader"))
            preserve = a.getValue();
        jw.keyValueBool("preserve_header", preserve);
        jw.endObject();
        return true;
    }
    if (auto c = dyn_cast<EnableExtToAieShimPort>(op)) {
        TileInfo t = resolveTile(op->getOperand(0));
        jw.beginObjectInline();
        jw.keyValue("kind", "shim_ext_to_aie");
        writeTileRef(jw, "tile", t);
        writePort(jw, "port", getStrAttr(op, "portdirection"), getIntAttr(op, "portidx"));
        jw.endObject();
        return true;
    }
    if (auto c = dyn_cast<EnableAieToExtShimPort>(op)) {
        TileInfo t = resolveTile(op->getOperand(0));
        jw.beginObjectInline();
        jw.keyValue("kind", "shim_aie_to_ext");
        writeTileRef(jw, "tile", t);
        writePort(jw, "port", getStrAttr(op, "portdirection"), getIntAttr(op, "portidx"));
        jw.endObject();
        return true;
    }
    if (auto c = dyn_cast<ConnectStreamSwitchPort>(op)) {
        // Multi-tile circuit connection: src side + dst side.
        TileInfo src = resolveTile(op->getOperand(1));
        TileInfo dst = resolveTile(op->getOperand(2));
        jw.beginObjectInline();
        jw.keyValue("kind", "circuit_connect_pair");
        writeTileRef(jw, "src_tile", src);
        writePort(jw, "src_slave", getStrAttr(op, "srcslaveport"), getIntAttr(op, "srcslaveportidx"));
        writePort(jw, "src_master", getStrAttr(op, "srcmasterport"), getIntAttr(op, "srcmasterportidx"));
        writeTileRef(jw, "dst_tile", dst);
        writePort(jw, "dst_slave", getStrAttr(op, "dstslaveport"), getIntAttr(op, "dstslaveportidx"));
        writePort(jw, "dst_master", getStrAttr(op, "dstmasterport"), getIntAttr(op, "dstmasterportidx"));
        jw.endObject();
        return true;
    }
    if (auto c = dyn_cast<CreateShimStreamSwitchPort>(op)) {
        TileInfo t = resolveTile(op->getOperand(0));
        jw.beginObjectInline();
        jw.keyValue("kind", "shim_stream_switch_port");
        writeTileRef(jw, "tile", t);
        jw.key("shim_master");
        jw.beginObject();
        jw.keyValue("port", getStrAttr(op, "shimmasterport"));
        jw.keyValue("idx", getIntAttr(op, "shimmasterportidx"));
        jw.keyValue("type", getIntAttr(op, "shimmasterporttype"));
        jw.endObject();
        jw.endObject();
        return true;
    }
    return false;
}

// Is this op a routinghw connection/enable op (i.e. it programs a stream switch)?
static bool isRoutingConnectionOp(Operation *op) {
    return isa<ConnectStreamSingleSwitchPort, ConnectStreamPktSwitchPort, EnableExtToAieShimPort,
               EnableAieToExtShimPort, ConnectStreamSwitchPort, CreateShimStreamSwitchPort>(op);
}

// ---------------------------------------------------------------------------
// Emit all tiles declared inside a routing group body (deduplicated by
// col/row/kind so the same physical tile isn't listed many times).
// ---------------------------------------------------------------------------
static void writeGroupTiles(JsonWriter &jw, Region &body) {
    // Dedup by (col,row,isShim) so the same physical tile isn't listed twice.
    std::vector<std::tuple<int64_t, int64_t, bool>> seenKeys;
    auto alreadySeen = [&](int64_t c, int64_t r, bool shim) {
        for (auto &k : seenKeys)
            if (std::get<0>(k) == c && std::get<1>(k) == r && std::get<2>(k) == shim)
                return true;
        seenKeys.emplace_back(c, r, shim);
        return false;
    };

    jw.beginArray("tiles");
    body.walk([&](Operation *op) {
        bool isShim = false;
        if (isa<IOShimTileCreate>(op))
            isShim = true;
        else if (!isa<TileCreate>(op))
            return;

        // Read attributes into fresh locals right before writing (no cross-scope
        // storage of StringRef/std::string).
        int64_t col = getIntAttr(op, "col");
        int64_t row = getIntAttr(op, "row");
        if (alreadySeen(col, row, isShim))
            return;

        std::string comments = getStrAttr(op, "comments");

        jw.beginObjectInline();
        jw.keyValue("col", col);
        jw.keyValue("row", row);
        jw.keyValue("type", isShim ? StringRef("shim") : StringRef("core"));
        if (!comments.empty())
            jw.keyValue("comments", StringRef(comments));
        if (isShim) {
            jw.keyValue("ioid", getIntAttr(op, "IOID"));
            jw.keyValue("dma_direction", getIntAttr(op, "dmadirection"));
            jw.keyValue("channel_used", getIntAttr(op, "channelused"));
        }
        jw.endObject();
    });
    jw.endArray();
}

// ---------------------------------------------------------------------------
// Main pass implementation
// ---------------------------------------------------------------------------
void RoutingProvenanceMapPass::runOnOperation() {
    Operation *topOp = getOperation();
    auto moduleOp = dyn_cast<ModuleOp>(topOp);
    if (!moduleOp) {
        topOp->emitError("RoutingProvenanceMapPass requires a ModuleOp");
        signalPassFailure();
        return;
    }

    // Determine output path and ensure directory exists.
    std::string outPath;
    if (!outputDir.empty()) {
        if (std::error_code EC = llvm::sys::fs::create_directories(outputDir)) {
            llvm::errs() << "RoutingProvenanceMapPass: failed to create directory " << outputDir << ": " << EC.message()
                         << "\n";
            outPath = "routingprovenancemap.json";
        } else {
            outPath = outputDir + "/routingprovenancemap.json";
        }
    } else {
        outPath = "routingprovenancemap.json";
    }

    std::error_code ec;
    llvm::raw_fd_ostream fileOs(outPath, ec, llvm::sys::fs::OF_None);
    if (ec) {
        llvm::errs() << "RoutingProvenanceMapPass: failed to open " << outPath << ": " << ec.message() << "\n";
        signalPassFailure();
        return;
    }

    JsonWriter jw(fileOs);
    jw.beginRoot();

    jw.keyValue("version", (int64_t)1);
    // partition origin (absolute physical start column); phys_col = col + startcol
    if (startCol >= 0)
        jw.keyValue("startcol", (int64_t)startCol);
    if (!aieGen.empty())
        jw.keyValue("aie_gen", StringRef(aieGen));

    // Module attributes (mirrors the other provenance maps for cross-file joins).
    auto getI64 = [&](StringRef n) -> int64_t {
        if (auto a = moduleOp->getAttrOfType<IntegerAttr>(n))
            return a.getInt();
        return 0;
    };
    jw.key("module_attrs");
    jw.beginObject();
    jw.keyValue("tile_m", getI64("routing.tile_m"));
    jw.keyValue("tile_n", getI64("routing.tile_n"));
    jw.keyValue("tile_rows", getI64("routing.tile_rows"));
    jw.keyValue("tile_cols", getI64("routing.tile_cols"));
    jw.keyValue("effective_k", getI64("routing.effective_k"));
    jw.keyValue("full_k", getI64("routing.full_k"));
    jw.keyValue("k_rounds", getI64("routing.k_rounds"));
    jw.keyValue("m_rounds", getI64("routing.m_rounds"));
    jw.keyValue("n_rounds", getI64("routing.n_rounds"));
    jw.endObject();

    // Walk every routing group (routing.RoutingCreate). Each group carries a
    // Memo ("col"/"row"), a scf_idx (the split index), the physical tiles it
    // touches, and the ordered list of stream-switch connections it programs.
    int totalConnections = 0;
    int capturedConnections = 0;

    jw.beginArray("routing_groups");
    int groupCounter = 0;
    moduleOp->walk([&](routing::RoutingCreate rc) {
        Region &body = rc.getBody();

        jw.beginObjectInline();
        jw.keyValue("id", std::string("group_") + std::to_string(groupCounter++));
        jw.keyValue("memo", getStrAttr(rc, "Memo"));

        // Resolve scf_idx from the constant feeding operand 0.
        int64_t scfIdx = -1;
        if (rc->getNumOperands() > 0) {
            if (auto cst = dyn_cast_or_null<arith::ConstantOp>(rc->getOperand(0).getDefiningOp())) {
                if (auto ia = dyn_cast<IntegerAttr>(cst.getValue()))
                    scfIdx = ia.getInt();
            }
        }
        jw.keyValue("scf_idx", scfIdx);

        // Emit the IOID of the first shim in this group.
        // Read directly from the group's tile list instead of a separate walk,
        // because the same isa<IOShimTileCreate> pattern is proven to work there.
        int64_t groupIoid = -1;
        for (Block &blk : body) {
            for (Operation &innerOp : blk) {
                if (isa<IOShimTileCreate>(&innerOp)) {
                    groupIoid = getIntAttr(&innerOp, "IOID");
                    break;
                }
            }
            if (groupIoid >= 0)
                break;
        }
        jw.keyValue("ioid", groupIoid);

        // Tiles declared inside this group.
        writeGroupTiles(jw, body);

        // Ordered stream-switch connections.
        jw.beginArray("connections");
        body.walk([&](Operation *op) {
            if (!isRoutingConnectionOp(op))
                return;
            totalConnections++;
            if (writeConnectionOp(jw, op))
                capturedConnections++;
        });
        jw.endArray();

        jw.endObject();
    });
    jw.endArray();

    jw.endRoot();
    fileOs.close();

    // Safety net: warn if any connection op was missed (should not happen).
    if (totalConnections != capturedConnections) {
        llvm::errs() << "RoutingProvenanceMapPass: captured " << capturedConnections << " of " << totalConnections
                     << " connection ops\n";
    }
    std::cout << "Routing provenance map written to " << outPath << " (" << groupCounter << " groups, "
              << capturedConnections << " connections)" << std::endl;
}

} // namespace mlir
