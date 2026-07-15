/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include <iostream>
#include "dmapmanager.h"
int main(int argc, char* argv[]) {
    MLIRContext ctx;
    dmapmanager mtest;
    mtest.loaddialect(&ctx);
    mtest.ops_test(&ctx);
    std::cout << "main" <<std::endl;
    return 0;
}