/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include <iostream>
#include "dfschedulemanager.h"

int main(int argc, char* argv[]) {
    MLIRContext ctx;
    dfschedulemanager hoptest;
    
    // Load the dfschedulehop dialect
    hoptest.loaddialect(&ctx);
    
    // Run the cache-and-forward broadcast test
    hoptest.ops_test(&ctx);
    
    std::cout << "Completed dfschedulehop dialect test" << std::endl;
    return 0;
}