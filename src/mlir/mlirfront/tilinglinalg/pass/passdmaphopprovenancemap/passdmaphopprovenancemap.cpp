/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "passdmaphopprovenancemap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include <string>
#include <vector>

using namespace mlir;
using namespace dmaphop;
using namespace routing;

namespace mlir {

// ---------------------------------------------------------------------------
// Simple JSON writer (no external JSON library dependency)
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

    // Write a raw value after key() has been called
    void rawValue(int64_t v) {
        os << v;
        needsComma = true;
    }

    void rawStringValue(StringRef v) {
        os << "\"" << v << "\"";
        needsComma = true;
    }

    void stringInArray(StringRef v) {
        writeCommaIfNeeded();
        writeIndent();
        os << "\"" << v << "\"";
        needsComma = true;
    }

    void intInArray(int64_t v) {
        writeCommaIfNeeded();
        writeIndent();
        os << v;
        needsComma = true;
    }

    // Begin root (no comma before first object)
    void beginRoot() {
        os << "{";
        indentLevel++;
        needsComma = false;
    }

    void endRoot() { os << "\n}\n"; }
};

// ---------------------------------------------------------------------------
// Helper: get element byte width from MLIR Type
// ---------------------------------------------------------------------------
static int64_t getElementBytes(mlir::Type elemType) {
    if (auto intType = dyn_cast<IntegerType>(elemType))
        return (intType.getWidth() + 7) / 8;
    if (elemType.isF32())
        return 4;
    if (elemType.isF16() || elemType.isBF16())
        return 2;
    if (elemType.isF64())
        return 8;
    return 1; // fallback
}

// ---------------------------------------------------------------------------
// Helper: get type name string
// ---------------------------------------------------------------------------
static std::string getTypeStr(mlir::Type elemType) {
    if (auto intType = dyn_cast<IntegerType>(elemType))
        return "i" + std::to_string(intType.getWidth());
    if (elemType.isF32())
        return "f32";
    if (elemType.isF16())
        return "f16";
    if (elemType.isBF16())
        return "bf16";
    if (elemType.isF64())
        return "f64";
    return "unknown";
}

// ---------------------------------------------------------------------------
// Walk-based symbol lookup helpers (robust across nested SymbolTable scopes)
// ---------------------------------------------------------------------------

/// Find a dmaphop::port with the given sym_name by walking from contextOp upward.
static dmaphop::port findPortByName(Operation *contextOp, StringRef name) {
    dmaphop::port result = nullptr;
    // Walk the parent region of the contextOp first, then the enclosing module
    Operation *searchRoot = contextOp->getParentOfType<ModuleOp>();
    if (!searchRoot)
        searchRoot = contextOp->getParentOp();
    if (!searchRoot)
        return nullptr;
    searchRoot->walk([&](dmaphop::port op) {
        if (op.getSymName() == name) {
            result = op;
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
    });
    return result;
}

/// Find a dmaphop::consumer with the given sym_name.
static dmaphop::consumer findConsumerByName(Operation *contextOp, StringRef name) {
    dmaphop::consumer result = nullptr;
    Operation *searchRoot = contextOp->getParentOfType<ModuleOp>();
    if (!searchRoot)
        searchRoot = contextOp->getParentOp();
    if (!searchRoot)
        return nullptr;
    searchRoot->walk([&](dmaphop::consumer op) {
        if (op.getSymName() == name) {
            result = op;
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
    });
    return result;
}

/// Find a dmaphop::producer with the given sym_name.
static dmaphop::producer findProducerByName(Operation *contextOp, StringRef name) {
    dmaphop::producer result = nullptr;
    Operation *searchRoot = contextOp->getParentOfType<ModuleOp>();
    if (!searchRoot)
        searchRoot = contextOp->getParentOp();
    if (!searchRoot)
        return nullptr;
    searchRoot->walk([&](dmaphop::producer op) {
        if (op.getSymName() == name) {
            result = op;
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
    });
    return result;
}

/// Resolve a symbol ref from create_path producers/consumers to the underlying port op.
static dmaphop::port resolvePort(Operation *contextOp, FlatSymbolRefAttr symRef) {
    StringRef name = symRef.getValue();
    // Try direct port lookup
    if (auto portOp = findPortByName(contextOp, name))
        return portOp;
    // Try consumer indirection
    if (auto consumerOp = findConsumerByName(contextOp, name)) {
        return findPortByName(contextOp, consumerOp.getFrom());
    }
    // Try producer indirection
    if (auto producerOp = findProducerByName(contextOp, name)) {
        return findPortByName(contextOp, producerOp.getTp());
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Struct to hold tile info extracted from a port op
// ---------------------------------------------------------------------------
struct TileInfo {
    int64_t col = -1;
    int64_t row = -1;
    std::string type; // "shim", "core", "mem"
};

static TileInfo getTileInfo(dmaphop::port portOp) {
    TileInfo info;
    if (!portOp)
        return info;
    auto tileValue = portOp.getTile();
    auto tileOp = dyn_cast_or_null<dmaphop::tile>(tileValue.getDefiningOp());
    if (!tileOp)
        return info;
    info.col = tileOp.getCol();
    info.row = tileOp.getRow();
    info.type = tileOp.getTiletype().str();
    return info;
}

// ---------------------------------------------------------------------------
// Struct for a hop in the path
// ---------------------------------------------------------------------------
struct HopInfo {
    std::string fromPortSym;
    TileInfo fromTile;
    std::string toPortSym;
    TileInfo toTile;
};

// ---------------------------------------------------------------------------
// Struct for consumer/producer endpoint
// ---------------------------------------------------------------------------
struct EndpointInfo {
    std::string sym;     // consumer/producer symbol name
    std::string portSym; // referenced port symbol
    TileInfo tile;
    int64_t dmaPort = -1;
    int32_t pktId = -1; // only for producers with dmapktid
    int64_t channel = -1;
};

// ---------------------------------------------------------------------------
// Struct for partition info
// ---------------------------------------------------------------------------
struct PartitionInfo {
    int64_t splitdim = -1;
    int64_t splitnum = -1;
    int64_t index = -1;
    std::string hwAxisOwner;
    std::string replicateOn;
};

// ---------------------------------------------------------------------------
// Struct for a complete communication path entry
// ---------------------------------------------------------------------------
struct CommPathEntry {
    std::string id;
    std::string direction; // "push" or "pull"

    // Data info
    std::vector<int64_t> tensorShape;
    std::string elementType;
    int64_t elementBytes = 0;
    int64_t totalBytes = 0;

    // Partition info (from routingextract_data -> partitiontensor)
    PartitionInfo partition;

    // For push: single producer (shim), multiple consumers (core)
    // For pull: multiple producers (core), single consumer (shim)
    EndpointInfo shimEndpoint;
    std::vector<EndpointInfo> coreEndpoints;

    // Hop chain
    std::vector<HopInfo> hops;

    // Invariants
    std::vector<std::string> invariants;
};

// ---------------------------------------------------------------------------
// Helper: try to trace back through routingextract_data -> partitiontensor
// ---------------------------------------------------------------------------
static PartitionInfo getPartitionInfo(Value dataVal) {
    PartitionInfo info;

    // For push: data operand may be routingextract_data result
    Operation *defOp = dataVal.getDefiningOp();
    if (!defOp)
        return info;

    // Check if it's routingextract_data
    if (defOp->getName().getStringRef() == "routing.routingextract_data") {
        // Get index operand (second operand)
        if (defOp->getNumOperands() >= 2) {
            Value idxVal = defOp->getOperand(1);
            // Direct constant
            if (auto constOp = dyn_cast_or_null<arith::ConstantOp>(idxVal.getDefiningOp())) {
                if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
                    info.index = intAttr.getInt();
            }
            // Block argument from RoutingCreate (scf_idx is passed as operand)
            else if (auto blockArg = dyn_cast<BlockArgument>(idxVal)) {
                // The RoutingCreate op passes scf_idx as its operand,
                // and the block receives it as %arg3. Trace back.
                auto parentOp = blockArg.getOwner()->getParentOp();
                if (parentOp && parentOp->getName().getStringRef() == "routing.RoutingCreate") {
                    // The scf_idx is the first operand of RoutingCreate
                    if (parentOp->getNumOperands() > 0) {
                        Value scfIdx = parentOp->getOperand(0);
                        if (auto constOp2 = dyn_cast_or_null<arith::ConstantOp>(scfIdx.getDefiningOp())) {
                            if (auto intAttr2 = dyn_cast<IntegerAttr>(constOp2.getValue()))
                                info.index = intAttr2.getInt();
                        }
                    }
                }
            }
        }
        // Get the partitiontensor from first operand
        if (defOp->getNumOperands() >= 1) {
            Value ptVal = defOp->getOperand(0);
            Operation *ptOp = ptVal.getDefiningOp();
            if (ptOp && ptOp->getName().getStringRef() == "routing.partitiontensor") {
                if (auto attr = ptOp->getAttrOfType<IntegerAttr>("splitdim"))
                    info.splitdim = attr.getInt();
                if (auto attr = ptOp->getAttrOfType<IntegerAttr>("splitnum"))
                    info.splitnum = attr.getInt();
                if (auto attr = ptOp->getAttrOfType<StringAttr>("hw_axis_owner"))
                    info.hwAxisOwner = attr.getValue().str();
                if (auto attr = ptOp->getAttrOfType<StringAttr>("replicate_on"))
                    info.replicateOn = attr.getValue().str();
            }
        }
    }

    return info;
}

// ---------------------------------------------------------------------------
// Helper: format port location string
// ---------------------------------------------------------------------------
static std::string fmtPortLoc(StringRef portSym, const TileInfo &tile) {
    return (portSym + "(" + std::to_string(tile.col) + "," + std::to_string(tile.row) + ")").str();
}

// ---------------------------------------------------------------------------
// Write a single TileInfo as JSON
// ---------------------------------------------------------------------------
static void writeTileJson(JsonWriter &jw, const TileInfo &tile) {
    jw.keyValue("col", tile.col);
    jw.keyValue("row", tile.row);
    jw.keyValue("type", tile.type);
}

// ---------------------------------------------------------------------------
// Process a push op
// ---------------------------------------------------------------------------
static CommPathEntry processPushOp(dmaphop::push pushOp, int &counter) {
    CommPathEntry entry;
    entry.id = "push_" + std::to_string(counter++);
    entry.direction = "push";

    // Get data tensor info
    auto dataType = dyn_cast<RankedTensorType>(pushOp.getData().getType());
    if (dataType) {
        for (auto dim : dataType.getShape())
            entry.tensorShape.push_back(dim);
        entry.elementType = getTypeStr(dataType.getElementType());
        entry.elementBytes = getElementBytes(dataType.getElementType());
        entry.totalBytes = entry.elementBytes;
        for (auto dim : entry.tensorShape)
            entry.totalBytes *= dim;
    }

    // Get partition info from data operand
    entry.partition = getPartitionInfo(pushOp.getData());

    // Get path
    auto pathOp = dyn_cast_or_null<dmaphop::create_path>(pushOp.getPath().getDefiningOp());
    if (!pathOp)
        return entry;

    // Helper to process a single symbol ref for shim endpoint
    auto processProducerSym = [&](FlatSymbolRefAttr symRef) {
        auto portOp = resolvePort(pushOp, symRef);
        if (!portOp)
            return;
        entry.shimEndpoint.sym = symRef.getValue().str();
        entry.shimEndpoint.portSym = portOp.getSymName().str();
        entry.shimEndpoint.tile = getTileInfo(portOp);
        entry.shimEndpoint.channel = portOp.getDirectionChannel().value_or(-1);
    };

    // Helper to process a single symbol ref for consumer endpoint
    auto processConsumerSym = [&](FlatSymbolRefAttr symRef) {
        EndpointInfo ep;
        ep.sym = symRef.getValue().str();
        if (auto consumerOp = findConsumerByName(pushOp, symRef.getValue())) {
            ep.dmaPort = consumerOp.getDmaPort();
            if (auto portOp = findPortByName(pushOp, consumerOp.getFrom())) {
                ep.portSym = portOp.getSymName().str();
                ep.tile = getTileInfo(portOp);
                ep.channel = portOp.getDirectionChannel().value_or(-1);
            }
        } else if (auto portOp = findPortByName(pushOp, symRef.getValue())) {
            ep.portSym = portOp.getSymName().str();
            ep.tile = getTileInfo(portOp);
            ep.channel = portOp.getDirectionChannel().value_or(-1);
        }
        entry.coreEndpoints.push_back(ep);
    };

    // Extract producer (shim) from create_path's producers array
    // Handles both flat [sym, sym] and nested [[sym, sym]] formats
    auto producersAttr = pathOp.getProducers();
    if (auto arrayAttr = dyn_cast<ArrayAttr>(producersAttr)) {
        for (auto attr : arrayAttr) {
            if (auto symRef = dyn_cast<FlatSymbolRefAttr>(attr)) {
                processProducerSym(symRef);
            } else if (auto innerArray = dyn_cast<ArrayAttr>(attr)) {
                for (auto innerAttr : innerArray) {
                    if (auto symRef = dyn_cast<FlatSymbolRefAttr>(innerAttr))
                        processProducerSym(symRef);
                }
            }
        }
    }

    // Extract consumers (core tiles) from create_path's consumers array
    auto consumersAttr = pathOp.getConsumers();
    if (auto arrayAttr = dyn_cast<ArrayAttr>(consumersAttr)) {
        for (auto attr : arrayAttr) {
            if (auto symRef = dyn_cast<FlatSymbolRefAttr>(attr)) {
                processConsumerSym(symRef);
            } else if (auto innerArray = dyn_cast<ArrayAttr>(attr)) {
                for (auto innerAttr : innerArray) {
                    if (auto symRef = dyn_cast<FlatSymbolRefAttr>(innerAttr))
                        processConsumerSym(symRef);
                }
            }
        }
    }

    // Extract hops
    for (auto hopVal : pathOp.getHops()) {
        auto hopOp = dyn_cast_or_null<dmaphop::create_hop>(hopVal.getDefiningOp());
        if (!hopOp)
            continue;

        HopInfo hop;
        if (auto srcPortOp = dyn_cast_or_null<dmaphop::port>(hopOp.getSource().getDefiningOp())) {
            hop.fromPortSym = srcPortOp.getSymName().str();
            hop.fromTile = getTileInfo(srcPortOp);
        }
        if (auto dstPortOp = dyn_cast_or_null<dmaphop::port>(hopOp.getDestination().getDefiningOp())) {
            hop.toPortSym = dstPortOp.getSymName().str();
            hop.toTile = getTileInfo(dstPortOp);
        }
        entry.hops.push_back(hop);
    }

    // Compute invariants
    entry.invariants.push_back("producer total bytes (" + std::to_string(entry.totalBytes) + ") broadcast to " +
                               std::to_string(entry.coreEndpoints.size()) + " consumers");

    if (!entry.coreEndpoints.empty()) {
        bool allSameDma = true;
        int64_t firstDma = entry.coreEndpoints[0].dmaPort;
        for (auto &ep : entry.coreEndpoints) {
            if (ep.dmaPort != firstDma) {
                allSameDma = false;
                break;
            }
        }
        if (allSameDma)
            entry.invariants.push_back("all consumer dma_ports match (" + std::to_string(firstDma) + ")");
    }

    entry.invariants.push_back("hop chain length: " + std::to_string(entry.hops.size()));

    return entry;
}

// ---------------------------------------------------------------------------
// Process a pull op
// ---------------------------------------------------------------------------
static CommPathEntry processPullOp(dmaphop::pull pullOp, int &counter) {
    CommPathEntry entry;
    entry.id = "pull_" + std::to_string(counter++);
    entry.direction = "pull";

    // Get data tensor info
    auto dataType = dyn_cast<RankedTensorType>(pullOp.getData().getType());
    if (dataType) {
        for (auto dim : dataType.getShape())
            entry.tensorShape.push_back(dim);
        entry.elementType = getTypeStr(dataType.getElementType());
        entry.elementBytes = getElementBytes(dataType.getElementType());
        entry.totalBytes = entry.elementBytes;
        for (auto dim : entry.tensorShape)
            entry.totalBytes *= dim;
    }

    // Get partition info
    entry.partition = getPartitionInfo(pullOp.getData());

    // Get path
    auto pathOp = dyn_cast_or_null<dmaphop::create_path>(pullOp.getPath().getDefiningOp());
    if (!pathOp)
        return entry;

    // Helper to process a single symbol ref for shim endpoint (consumer in pull)
    auto processShimConsumerSym = [&](FlatSymbolRefAttr symRef) {
        auto portOp = resolvePort(pullOp, symRef);
        if (!portOp)
            return;
        entry.shimEndpoint.sym = symRef.getValue().str();
        entry.shimEndpoint.portSym = portOp.getSymName().str();
        entry.shimEndpoint.tile = getTileInfo(portOp);
        entry.shimEndpoint.channel = portOp.getDirectionChannel().value_or(-1);
    };

    // Helper to process a single symbol ref for core producer endpoint
    auto processCoreProducerSym = [&](FlatSymbolRefAttr symRef) {
        EndpointInfo ep;
        ep.sym = symRef.getValue().str();
        if (auto producerOp = findProducerByName(pullOp, symRef.getValue())) {
            ep.dmaPort = producerOp.getDmaPort();
            if (auto portOp = findPortByName(pullOp, producerOp.getTp())) {
                ep.portSym = portOp.getSymName().str();
                ep.tile = getTileInfo(portOp);
                ep.channel = portOp.getDirectionChannel().value_or(-1);
                if (auto pktId = portOp.getDmapktid())
                    ep.pktId = static_cast<int32_t>(*pktId);
            }
        } else if (auto portOp = findPortByName(pullOp, symRef.getValue())) {
            ep.portSym = portOp.getSymName().str();
            ep.tile = getTileInfo(portOp);
            ep.channel = portOp.getDirectionChannel().value_or(-1);
            if (auto pktId = portOp.getDmapktid())
                ep.pktId = static_cast<int32_t>(*pktId);
        }
        entry.coreEndpoints.push_back(ep);
    };

    // Extract consumer (shim) from create_path's consumers array
    // Handles both flat [sym] and nested [[sym]] formats
    auto consumersAttr = pathOp.getConsumers();
    if (auto arrayAttr = dyn_cast<ArrayAttr>(consumersAttr)) {
        for (auto attr : arrayAttr) {
            if (auto symRef = dyn_cast<FlatSymbolRefAttr>(attr)) {
                processShimConsumerSym(symRef);
            } else if (auto innerArray = dyn_cast<ArrayAttr>(attr)) {
                for (auto innerAttr : innerArray) {
                    if (auto symRef = dyn_cast<FlatSymbolRefAttr>(innerAttr))
                        processShimConsumerSym(symRef);
                }
            }
        }
    }

    // Extract producers (core tiles) from create_path's producers array
    auto producersAttr = pathOp.getProducers();
    if (auto arrayAttr = dyn_cast<ArrayAttr>(producersAttr)) {
        for (auto attr : arrayAttr) {
            if (auto symRef = dyn_cast<FlatSymbolRefAttr>(attr)) {
                processCoreProducerSym(symRef);
            } else if (auto innerArray = dyn_cast<ArrayAttr>(attr)) {
                for (auto innerAttr : innerArray) {
                    if (auto symRef = dyn_cast<FlatSymbolRefAttr>(innerAttr))
                        processCoreProducerSym(symRef);
                }
            }
        }
    }

    // Extract hops
    for (auto hopVal : pathOp.getHops()) {
        auto hopOp = dyn_cast_or_null<dmaphop::create_hop>(hopVal.getDefiningOp());
        if (!hopOp)
            continue;

        HopInfo hop;
        if (auto srcPortOp = dyn_cast_or_null<dmaphop::port>(hopOp.getSource().getDefiningOp())) {
            hop.fromPortSym = srcPortOp.getSymName().str();
            hop.fromTile = getTileInfo(srcPortOp);
        }
        if (auto dstPortOp = dyn_cast_or_null<dmaphop::port>(hopOp.getDestination().getDefiningOp())) {
            hop.toPortSym = dstPortOp.getSymName().str();
            hop.toTile = getTileInfo(dstPortOp);
        }
        entry.hops.push_back(hop);
    }

    // Get per-producer buffer sizes from producer_buffers
    std::vector<int64_t> producerBufferSizes;
    for (auto buf : pullOp.getProducerBuffers()) {
        if (auto bufType = dyn_cast<RankedTensorType>(buf.getType())) {
            int64_t bytes = getElementBytes(bufType.getElementType());
            for (auto dim : bufType.getShape())
                bytes *= dim;
            producerBufferSizes.push_back(bytes);
        }
    }

    // Compute invariants
    if (!producerBufferSizes.empty()) {
        int64_t totalProduced = 0;
        for (auto s : producerBufferSizes)
            totalProduced += s;
        entry.invariants.push_back("sum of producer buffer bytes (" + std::to_string(totalProduced) +
                                   ") gathered into total (" + std::to_string(entry.totalBytes) + ")");
    }

    if (!entry.coreEndpoints.empty()) {
        bool allSameDma = true;
        int64_t firstDma = entry.coreEndpoints[0].dmaPort;
        for (auto &ep : entry.coreEndpoints) {
            if (ep.dmaPort != firstDma) {
                allSameDma = false;
                break;
            }
        }
        if (allSameDma)
            entry.invariants.push_back("all producer dma_ports match (" + std::to_string(firstDma) + ")");

        bool hasPktIds = entry.coreEndpoints[0].pktId >= 0;
        if (hasPktIds) {
            std::string pktList;
            for (size_t i = 0; i < entry.coreEndpoints.size(); ++i) {
                if (i > 0)
                    pktList += ", ";
                pktList += std::to_string(entry.coreEndpoints[i].pktId);
            }
            entry.invariants.push_back("packet IDs: [" + pktList + "]");
        }
    }

    entry.invariants.push_back("hop chain length: " + std::to_string(entry.hops.size()));

    return entry;
}

// ---------------------------------------------------------------------------
// Write a CommPathEntry as JSON
// ---------------------------------------------------------------------------
static void writeCommPathJson(JsonWriter &jw, const CommPathEntry &entry) {
    jw.beginObjectInline();

    jw.keyValue("id", entry.id);
    jw.keyValue("direction", entry.direction);

    // Data section
    jw.key("data");
    jw.beginObject();
    {
        jw.beginArray("tensor_shape");
        for (size_t i = 0; i < entry.tensorShape.size(); ++i)
            jw.intInArray(entry.tensorShape[i]);
        jw.endArray();
        jw.keyValue("element_type", entry.elementType);
        jw.keyValue("element_bytes", entry.elementBytes);
        jw.keyValue("total_bytes", entry.totalBytes);

        if (entry.partition.splitdim >= 0) {
            jw.key("partition_info");
            jw.beginObject();
            jw.keyValue("splitdim", entry.partition.splitdim);
            jw.keyValue("splitnum", entry.partition.splitnum);
            jw.keyValue("index", entry.partition.index);
            if (!entry.partition.hwAxisOwner.empty())
                jw.keyValue("hw_axis_owner", entry.partition.hwAxisOwner);
            if (!entry.partition.replicateOn.empty())
                jw.keyValue("replicate_on", entry.partition.replicateOn);
            jw.endObject();
        }
    }
    jw.endObject();

    // Stages section
    jw.beginArray("stages");
    {
        if (entry.direction == "push") {
            // Stage 1: Producer (shim)
            jw.beginObjectInline();
            jw.keyValue("role", "producer");
            jw.key("tile");
            jw.beginObject();
            writeTileJson(jw, entry.shimEndpoint.tile);
            jw.endObject();
            jw.keyValue("port_sym", entry.shimEndpoint.portSym);
            jw.keyValue("channel", entry.shimEndpoint.channel);
            jw.keyValue("contract", "produce via MM2S from DDR");
            jw.keyValue("config_ref", std::string("dmaphop.port @") + entry.shimEndpoint.portSym);
            jw.endObject();

            // Stage 2: Channel (hops)
            jw.beginObjectInline();
            jw.keyValue("role", "channel");
            jw.beginArray("hops");
            for (auto &hop : entry.hops) {
                jw.beginObjectInline();
                jw.keyValue("from", fmtPortLoc(hop.fromPortSym, hop.fromTile));
                jw.keyValue("to", fmtPortLoc(hop.toPortSym, hop.toTile));
                jw.endObject();
            }
            jw.endArray();
            jw.keyValue("contract",
                        std::string("lossless delivery via ") + std::to_string(entry.hops.size()) + "-hop chain");
            jw.keyValue("config_ref", "dmaphop.create_path");
            jw.endObject();

            // Stage 3: Consumers (core tiles)
            jw.beginObjectInline();
            jw.keyValue("role", "consumer");
            jw.beginArray("tiles");
            for (auto &ep : entry.coreEndpoints) {
                jw.beginObjectInline();
                writeTileJson(jw, ep.tile);
                jw.keyValue("port_sym", ep.portSym);
                jw.keyValue("consumer_sym", ep.sym);
                jw.keyValue("dma_port", ep.dmaPort);
                jw.endObject();
            }
            jw.endArray();
            jw.keyValue("contract", "receive via S2MM, consume in compute kernel");
            if (!entry.coreEndpoints.empty()) {
                std::string ref = "dmaphop.consumer @" + entry.coreEndpoints.front().sym;
                if (entry.coreEndpoints.size() > 1)
                    ref += ".." + entry.coreEndpoints.back().sym;
                jw.keyValue("config_ref", ref);
            }
            jw.endObject();

        } else {
            // Pull direction: producers are core tiles, consumer is shim

            // Stage 1: Producers (core tiles)
            jw.beginObjectInline();
            jw.keyValue("role", "producer");
            jw.beginArray("tiles");
            for (auto &ep : entry.coreEndpoints) {
                jw.beginObjectInline();
                writeTileJson(jw, ep.tile);
                jw.keyValue("port_sym", ep.portSym);
                jw.keyValue("producer_sym", ep.sym);
                jw.keyValue("dma_port", ep.dmaPort);
                if (ep.pktId >= 0)
                    jw.keyValue("pkt_id", ep.pktId);
                jw.endObject();
            }
            jw.endArray();
            jw.keyValue("contract", "each core produces partial output via MM2S");
            if (!entry.coreEndpoints.empty()) {
                std::string ref = "dmaphop.producer @" + entry.coreEndpoints.front().sym;
                if (entry.coreEndpoints.size() > 1)
                    ref += ".." + entry.coreEndpoints.back().sym;
                jw.keyValue("config_ref", ref);
            }
            jw.endObject();

            // Stage 2: Channel (hops)
            jw.beginObjectInline();
            jw.keyValue("role", "channel");
            jw.beginArray("hops");
            for (auto &hop : entry.hops) {
                jw.beginObjectInline();
                jw.keyValue("from", fmtPortLoc(hop.fromPortSym, hop.fromTile));
                jw.keyValue("to", fmtPortLoc(hop.toPortSym, hop.toTile));
                jw.endObject();
            }
            jw.endArray();
            jw.keyValue("contract", std::string("gather via ") + std::to_string(entry.hops.size()) + "-hop chain");
            jw.keyValue("config_ref", "dmaphop.create_path");
            jw.endObject();

            // Stage 3: Consumer (shim)
            jw.beginObjectInline();
            jw.keyValue("role", "consumer");
            jw.key("tile");
            jw.beginObject();
            writeTileJson(jw, entry.shimEndpoint.tile);
            jw.endObject();
            jw.keyValue("port_sym", entry.shimEndpoint.portSym);
            jw.keyValue("channel", entry.shimEndpoint.channel);
            jw.keyValue("contract", "receive gathered output via S2MM to DDR");
            jw.keyValue("config_ref", std::string("dmaphop.port @") + entry.shimEndpoint.portSym);
            jw.endObject();
        }
    }
    jw.endArray();

    // Invariants
    jw.beginArray("invariants");
    for (auto &inv : entry.invariants)
        jw.stringInArray(inv);
    jw.endArray();

    jw.endObject();
}

// ---------------------------------------------------------------------------
// Main pass implementation
// ---------------------------------------------------------------------------
void DmaphopProvenanceMapPass::runOnOperation() {
    Operation *topOp = getOperation();
    auto moduleOp = dyn_cast<ModuleOp>(topOp);
    if (!moduleOp) {
        topOp->emitError("DmaphopProvenanceMapPass requires a ModuleOp");
        signalPassFailure();
        return;
    }

    // Determine output path and ensure directory exists
    std::string outPath;
    if (!outputDir.empty()) {
        // Create the output directory if it doesn't exist yet
        if (std::error_code EC = llvm::sys::fs::create_directories(outputDir)) {
            llvm::errs() << "DmaphopProvenanceMapPass: failed to create directory " << outputDir << ": " << EC.message()
                         << "\n";
            // Non-fatal: fall back to current directory
            outPath = "dmaphopprovenacemap.json";
        } else {
            outPath = outputDir + "/dmaphopprovenacemap.json";
        }
    } else {
        outPath = "dmaphopprovenacemap.json";
    }

    // 1. Extract module attributes
    struct ModuleAttrs {
        int64_t tile_m = 0, tile_n = 0;
        int64_t tile_rows = 0, tile_cols = 0;
        int64_t effective_k = 0, full_k = 0;
        int64_t k_rounds = 0, m_rounds = 0, n_rounds = 0;
    } modAttrs;

    auto getI64Attr = [&](StringRef name) -> int64_t {
        if (auto attr = moduleOp->getAttrOfType<IntegerAttr>(name))
            return attr.getInt();
        return 0;
    };

    modAttrs.tile_m = getI64Attr("routing.tile_m");
    modAttrs.tile_n = getI64Attr("routing.tile_n");
    modAttrs.tile_rows = getI64Attr("routing.tile_rows");
    modAttrs.tile_cols = getI64Attr("routing.tile_cols");
    modAttrs.effective_k = getI64Attr("routing.effective_k");
    modAttrs.full_k = getI64Attr("routing.full_k");
    modAttrs.k_rounds = getI64Attr("routing.k_rounds");
    modAttrs.m_rounds = getI64Attr("routing.m_rounds");
    modAttrs.n_rounds = getI64Attr("routing.n_rounds");

    // 2. Walk all push/pull ops and collect communication paths
    std::vector<CommPathEntry> entries;
    int pushCounter = 0;
    int pullCounter = 0;

    moduleOp->walk([&](dmaphop::push pushOp) { entries.push_back(processPushOp(pushOp, pushCounter)); });

    moduleOp->walk([&](dmaphop::pull pullOp) { entries.push_back(processPullOp(pullOp, pullCounter)); });

    // 3. Write JSON
    std::error_code ec;
    llvm::raw_fd_ostream fileOs(outPath, ec, llvm::sys::fs::OF_None);
    if (ec) {
        llvm::errs() << "DmaphopProvenanceMapPass: failed to open " << outPath << ": " << ec.message() << "\n";
        signalPassFailure();
        return;
    }

    JsonWriter jw(fileOs);
    jw.beginRoot();

    jw.keyValue("version", (int64_t)1);

    // partition origin (absolute physical start column); phys_col = col + startcol
    if (startCol >= 0)
        jw.keyValue("startcol", (int64_t)startCol);

    // compiler aie-gen (raw --aie-version value); Python tools map it to debug offset-map
    if (!aieGen.empty())
        jw.keyValue("aie_gen", StringRef(aieGen));

    // Module attributes
    jw.key("module_attrs");
    jw.beginObject();
    jw.keyValue("tile_m", modAttrs.tile_m);
    jw.keyValue("tile_n", modAttrs.tile_n);
    jw.keyValue("tile_rows", modAttrs.tile_rows);
    jw.keyValue("tile_cols", modAttrs.tile_cols);
    jw.keyValue("effective_k", modAttrs.effective_k);
    jw.keyValue("full_k", modAttrs.full_k);
    jw.keyValue("k_rounds", modAttrs.k_rounds);
    jw.keyValue("m_rounds", modAttrs.m_rounds);
    jw.keyValue("n_rounds", modAttrs.n_rounds);
    jw.endObject();

    // Communication paths
    jw.beginArray("communication_paths");
    for (auto &entry : entries) {
        writeCommPathJson(jw, entry);
    }
    jw.endArray();

    jw.endRoot();
    fileOs.close();

    std::cout << "Provenance map written to " << outPath << " (" << entries.size() << " paths)" << std::endl;
}

} // namespace mlir
