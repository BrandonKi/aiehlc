/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//this pass will lower the ir into aie driver api, so this pass
//is a assemly codegen.
#include "common.h"
#include "routinghwmanager.h"
#include "routingmanager.h"
#include "routing/routingtopology.h"
using namespace mlir;
class RoutingHWLowerPass : public PassWrapper<RoutingHWLowerPass, OperationPass<>> {
public:
    RoutingHWLowerPass(RoutingTopology& rtopology):rtopology_(rtopology){};
private:
    void runOnOperation() override;
    RoutingTopology& rtopology_;
};