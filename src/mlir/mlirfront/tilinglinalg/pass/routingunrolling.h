/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef __ROUTING_UNROLLING_LOWER__
#define __ROUTING_UNROLLING_LOWER__
#include "common.h"
using namespace mlir;
class RoutingUnrollingLowerPass : public PassWrapper<RoutingUnrollingLowerPass, OperationPass<>> {
public:
    RoutingUnrollingLowerPass();
private:
    void runOnOperation() override;
};
#endif //