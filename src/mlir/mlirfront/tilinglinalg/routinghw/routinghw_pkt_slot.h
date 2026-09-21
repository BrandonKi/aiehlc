/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef ROUTINGHW_PKT_SLOT_H
#define ROUTINGHW_PKT_SLOT_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"

namespace routinghw {
namespace pktslot {

constexpr int32_t kRecvSlaveMask = 0;
constexpr int32_t kRecvSlaveMsel = 0;
constexpr int32_t kRecvSlaveArbiter = 0;
constexpr int32_t kRecvSlaveSlot = 0;

constexpr int32_t kLocalDmaMask = 0x1f;
constexpr int32_t kLocalDmaMsel = 0;
constexpr int32_t kLocalDmaArbiter = 0;
constexpr int32_t kLocalDmaSlot = 0;

constexpr int32_t kForwardMasterArbiter = 0;
constexpr int32_t kForwardMasterMselEn = 1;

struct RecvSlaveSlotCfg {
    int32_t mask;
    int32_t msel;
    int32_t arbiter;
    int32_t slot;
};

struct LocalDmaSlotCfg {
    int32_t mask;
    int32_t msel;
    int32_t arbiter;
    int32_t slot;
};

struct ForwardMasterCfg {
    int32_t arbiter;
    int32_t mselEn;
};

inline int32_t getI32Attr(Operation *op, llvm::StringRef name, int32_t dflt) {
    if (auto a = op->getAttrOfType<mlir::IntegerAttr>(name))
        return static_cast<int32_t>(a.getInt());
    return dflt;
}

inline RecvSlaveSlotCfg recvSlaveFromOp(Operation *op) {
    return {getI32Attr(op, "receiveslavemask", kRecvSlaveMask), getI32Attr(op, "receiveslavemsel", kRecvSlaveMsel),
            getI32Attr(op, "receiveslavearbiter", kRecvSlaveArbiter),
            getI32Attr(op, "receiveslaveslot", kRecvSlaveSlot)};
}

inline LocalDmaSlotCfg localDmaFromOp(Operation *op) {
    return {getI32Attr(op, "localdmamask", kLocalDmaMask), getI32Attr(op, "localdmamsel", kLocalDmaMsel),
            getI32Attr(op, "localdmaarbiter", kLocalDmaArbiter), getI32Attr(op, "localdmaslot", kLocalDmaSlot)};
}

inline ForwardMasterCfg forwardMasterFromOp(Operation *op) {
    return {getI32Attr(op, "forwardmasterarbiter", kForwardMasterArbiter),
            getI32Attr(op, "forwardmastermselen", kForwardMasterMselEn)};
}

template <typename BuilderT>
inline void stampPktSlotAttrs(BuilderT &builder, llvm::SmallVectorImpl<mlir::NamedAttribute> &attrs) {
    attrs.emplace_back(builder.getStringAttr("receiveslavemask"), builder.getI32IntegerAttr(kRecvSlaveMask));
    attrs.emplace_back(builder.getStringAttr("receiveslavemsel"), builder.getI32IntegerAttr(kRecvSlaveMsel));
    attrs.emplace_back(builder.getStringAttr("receiveslavearbiter"), builder.getI32IntegerAttr(kRecvSlaveArbiter));
    attrs.emplace_back(builder.getStringAttr("receiveslaveslot"), builder.getI32IntegerAttr(kRecvSlaveSlot));
    attrs.emplace_back(builder.getStringAttr("localdmamask"), builder.getI32IntegerAttr(kLocalDmaMask));
    attrs.emplace_back(builder.getStringAttr("localdmamsel"), builder.getI32IntegerAttr(kLocalDmaMsel));
    attrs.emplace_back(builder.getStringAttr("localdmaarbiter"), builder.getI32IntegerAttr(kLocalDmaArbiter));
    attrs.emplace_back(builder.getStringAttr("localdmaslot"), builder.getI32IntegerAttr(kLocalDmaSlot));
    attrs.emplace_back(builder.getStringAttr("forwardmasterarbiter"), builder.getI32IntegerAttr(kForwardMasterArbiter));
    attrs.emplace_back(builder.getStringAttr("forwardmastermselen"), builder.getI32IntegerAttr(kForwardMasterMselEn));
}

template <typename BuilderT> struct PktSlotAttrValues {
    mlir::IntegerAttr recvMask;
    mlir::IntegerAttr recvMsel;
    mlir::IntegerAttr recvArbiter;
    mlir::IntegerAttr recvSlot;
    mlir::IntegerAttr dmaMask;
    mlir::IntegerAttr dmaMsel;
    mlir::IntegerAttr dmaArbiter;
    mlir::IntegerAttr dmaSlot;
    mlir::IntegerAttr fwdArbiter;
    mlir::IntegerAttr fwdMselEn;
};

template <typename BuilderT> inline PktSlotAttrValues<BuilderT> makePktSlotAttrs(BuilderT &builder) {
    return {builder.getI32IntegerAttr(kRecvSlaveMask),        builder.getI32IntegerAttr(kRecvSlaveMsel),
            builder.getI32IntegerAttr(kRecvSlaveArbiter),     builder.getI32IntegerAttr(kRecvSlaveSlot),
            builder.getI32IntegerAttr(kLocalDmaMask),         builder.getI32IntegerAttr(kLocalDmaMsel),
            builder.getI32IntegerAttr(kLocalDmaArbiter),      builder.getI32IntegerAttr(kLocalDmaSlot),
            builder.getI32IntegerAttr(kForwardMasterArbiter), builder.getI32IntegerAttr(kForwardMasterMselEn)};
}

template <typename BuilderT>
inline PktSlotAttrValues<BuilderT> makePktSlotAttrs(BuilderT &builder, int32_t recvSlot, int32_t arbiter) {
    return {builder.getI32IntegerAttr(kRecvSlaveMask),    builder.getI32IntegerAttr(kRecvSlaveMsel),
            builder.getI32IntegerAttr(arbiter),           builder.getI32IntegerAttr(recvSlot),
            builder.getI32IntegerAttr(kLocalDmaMask),     builder.getI32IntegerAttr(kLocalDmaMsel),
            builder.getI32IntegerAttr(arbiter),           builder.getI32IntegerAttr(kLocalDmaSlot),
            builder.getI32IntegerAttr(arbiter),           builder.getI32IntegerAttr(kForwardMasterMselEn)};
}

} // namespace pktslot
} // namespace routinghw

#define ROUTINGHW_PKT_SLOT_ATTRS(slotAttrs)                                                                            \
    (slotAttrs).recvMask, (slotAttrs).recvMsel, (slotAttrs).recvArbiter, (slotAttrs).recvSlot, (slotAttrs).dmaMask,    \
        (slotAttrs).dmaMsel, (slotAttrs).dmaArbiter, (slotAttrs).dmaSlot, (slotAttrs).fwdArbiter,                      \
        (slotAttrs).fwdMselEn

#endif
