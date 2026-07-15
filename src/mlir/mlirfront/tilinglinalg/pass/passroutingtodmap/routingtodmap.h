/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef __ROUTING_TODMAP__
#define __ROUTING_TODMAP__
#include "common.h"
#include "dmapmanager.h"
#include "routingmanager.h"
#include "routing/routingpath.h"
#include "routing/routingtopology.h"
using namespace mlir;
class RoutingToDmapPass : public PassWrapper<RoutingToDmapPass, OperationPass<>> {
public:
    RoutingToDmapPass(RoutingTopology& rtopology);
private:
    void runOnOperation() override;
    void getDependentDialects(DialectRegistry &registry) const;
    RoutingTopology& rtopology_;
};
#endif //