/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "passwaitmerge.h"
#include "dfschedulemanager.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace dfschedule;

namespace {

/// Check whether an scf.for loop body contains a dfschedule.schedule.wait op.
static bool forBodyContainsWait(scf::ForOp forOp) {
    bool found = false;
    forOp.getBody()->walk([&](dfschedule::ScheduleWaitOp) {
        found = true;
        return WalkResult::interrupt();
    });
    return found;
}

/// Check if two scf.for ops have the same loop bounds (lb, ub, step).
/// Bounds match if they refer to the same SSA value.
static bool sameBounds(scf::ForOp a, scf::ForOp b) {
    return a.getLowerBound() == b.getLowerBound() && a.getUpperBound() == b.getUpperBound() &&
           a.getStep() == b.getStep();
}

/// Collect consecutive runs of scf.for ops that:
///   1. All contain a schedule.wait in their body
///   2. Share the same loop bounds (lb, ub, step)
///   3. Have no iter_args (no loop-carried values)
/// Returns groups of >=2 loops to merge.
static SmallVector<SmallVector<scf::ForOp>> collectMergeGroups(Block &block) {
    SmallVector<SmallVector<scf::ForOp>> groups;
    SmallVector<scf::ForOp> currentRun;

    for (Operation &op : block) {
        auto forOp = dyn_cast<scf::ForOp>(&op);
        if (forOp && forOp.getInitArgs().empty() && forBodyContainsWait(forOp)) {
            if (currentRun.empty()) {
                currentRun.push_back(forOp);
            } else {
                if (sameBounds(currentRun.front(), forOp)) {
                    currentRun.push_back(forOp);
                } else {
                    // Bounds changed — flush current run
                    if (currentRun.size() >= 2)
                        groups.push_back(std::move(currentRun));
                    currentRun.clear();
                    currentRun.push_back(forOp);
                }
            }
        } else {
            // Non-for or for without wait — flush current run
            if (currentRun.size() >= 2)
                groups.push_back(std::move(currentRun));
            currentRun.clear();
        }
    }
    // Final flush
    if (currentRun.size() >= 2)
        groups.push_back(std::move(currentRun));

    return groups;
}

/// Merge a group of scf.for loops into a single scf.for.
/// The merged body contains all ops from each original loop body (with
/// the induction variable remapped), and all schedule.wait ops moved to
/// the end of the merged body.
static void mergeForGroup(SmallVector<scf::ForOp> &group, OpBuilder &builder) {
    scf::ForOp first = group.front();

    // Create the merged for loop at the position of the first loop
    builder.setInsertionPoint(first);
    auto mergedFor =
        builder.create<scf::ForOp>(first.getLoc(), first.getLowerBound(), first.getUpperBound(), first.getStep());

    Block *mergedBody = mergedFor.getBody();
    // The merged for body has a yield terminator already.
    // We insert ops before the yield.
    Operation *yieldOp = mergedBody->getTerminator();
    builder.setInsertionPoint(yieldOp);

    // Collect all wait ops to move to end
    SmallVector<Operation *> waitOps;

    for (scf::ForOp forOp : group) {
        Block *srcBody = forOp.getBody();
        Value srcIV = forOp.getInductionVar();
        Value mergedIV = mergedFor.getInductionVar();

        // Clone each op from source body into merged body,
        // remapping the induction variable.
        IRMapping mapping;
        mapping.map(srcIV, mergedIV);

        for (Operation &op : *srcBody) {
            if (op.hasTrait<OpTrait::IsTerminator>())
                continue; // skip scf.yield

            Operation *cloned = builder.clone(op, mapping);

            // Track cloned wait ops to reposition later
            if (isa<dfschedule::ScheduleWaitOp>(cloned)) {
                waitOps.push_back(cloned);
            }
        }
    }

    // Move all wait ops to the end of the merged body (before yield)
    for (Operation *waitOp : waitOps) {
        waitOp->moveBefore(yieldOp);
    }

    // Erase the original for loops
    for (scf::ForOp forOp : group) {
        forOp.erase();
    }
}

/// Process a single block: find and merge consecutive for-loops with waits.
static int processBlock(Block &block) {
    auto groups = collectMergeGroups(block);
    if (groups.empty())
        return 0;

    OpBuilder builder(block.getParentOp()->getContext());
    int totalMerged = 0;
    for (auto &group : groups) {
        llvm::errs() << "[WaitMerge] Merging " << group.size() << " for-loops with bounds ["
                     << group.front().getLowerBound() << " to " << group.front().getUpperBound() << " step "
                     << group.front().getStep() << "]\n";
        mergeForGroup(group, builder);
        totalMerged += group.size();
    }
    return totalMerged;
}

} // namespace

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

void mlir::WaitMergePass::runOnOperation() {
    ModuleOp moduleOp = getOperation();

    int totalMerged = 0;
    int blocksProcessed = 0;

    // Process dfschedule.host blocks (post-canonicalize IR)
    moduleOp.walk([&](dfschedule::HostBlockOp hostOp) {
        Block &block = hostOp.getBody().front();
        totalMerged += processBlock(block);
        blocksProcessed++;
    });

    // Also handle routing.RoutingCreate blocks (pre-canonicalize IR)
    if (blocksProcessed == 0) {
        moduleOp.walk([&](Operation *op) {
            if (op->getName().getStringRef() != "routing.RoutingCreate")
                return;
            for (Region &region : op->getRegions()) {
                for (Block &block : region) {
                    totalMerged += processBlock(block);
                    blocksProcessed++;
                }
            }
        });
    }

    llvm::errs() << "[WaitMerge] Processed " << blocksProcessed << " blocks, merged " << totalMerged << " for-loops\n";
}
