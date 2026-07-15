/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#ifndef __ROUTING_HW_VERIFY__
#define __ROUTING_HW_VERIFY__
#include "mlir/Pass/Pass.h"
#include "routinghwmanager.h"
#include "routingmanager.h"
using namespace mlir;
class RoutingHWVerifyPass : public PassWrapper<RoutingHWVerifyPass, OperationPass<>> {
  public:
    RoutingHWVerifyPass() = default;

  private:
    void runOnOperation() override;
};
#endif // __ROUTING_HW_VERIFY__
