/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef __ROUTING_DEADARG_CLEAN__
#define __ROUTING_DEADARG_CLEAN__
#include "common.h"
using namespace mlir;
class RoutingDeadArgPass : public PassWrapper<RoutingDeadArgPass, OperationPass<ModuleOp>> {
public:
    RoutingDeadArgPass(){};
private:
    void runOnOperation() override;
};
#endif //