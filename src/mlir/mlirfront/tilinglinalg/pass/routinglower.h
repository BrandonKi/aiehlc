/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef __ROUTING_LOWER__
#define __ROUTING_LOWER__
#include "common.h"
#include "routingmanager.h"
#include "routinghwmanager.h"
#include "routing/routingpath.h"
#include "routing/routingtopology.h"
using namespace mlir;
class RoutingLowerPass : public PassWrapper<RoutingLowerPass, OperationPass<>> {
public:
    RoutingLowerPass(RoutingTopology& rtopology);
private:
    void runOnOperation() override;
    void getDependentDialects(DialectRegistry &registry) const;
    RoutingTopology& rtopology_;
};
#endif //