/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/
#include <iostream>
#include "routinghwmanager.h"
#include "routinghwlower.h"
#include "routingmanager.h"
#include "routinglower.h"
#include "routingunrolling.h"
#include "mlir/Conversion/SCFToEmitC/SCFToEmitC.h"
//#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IRPrinter/IRPrintingPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"


#include "routingdeadargclean.h"

#include "routingconstantfold.h"

int main(int argc, char* argv[]) {
    MLIRContext ctx;
    
    routingmanager mtest;
    routinghwmanager mtesthw;
    mtesthw.loaddialect(&ctx);
    mtest.loaddialect(&ctx);

    ctx.getOrLoadDialect<arith::ArithDialect>();
    
    //auto module1 = mtest.createroutingfunc(&ctx,1);
    auto module1 = mtest.ops_testNew(&ctx,1);
    module1.dump();
    //auto module2 = mtesthw.ops_test(&ctx);
    std::cout << "main" <<std::endl;
    
    mlir::PassManager pm(&ctx);;
    RoutingTopology rtopology("Gen2");
    
    pm.addPass(mlir::createPrintIRPass());
    pm.addPass(std::make_unique<RoutingUnrollingLowerPass>());
    pm.addPass(mlir::createPrintIRPass());
    pm.addPass(std::make_unique<RoutingLowerPass>(rtopology));
    pm.addPass(mlir::createPrintIRPass());
    pm.addPass(std::make_unique<RoutingHWLowerPass>(rtopology));
    pm.addPass(mlir::createPrintIRPass());
    //remove dead arg
    pm.addPass(std::make_unique<RoutingDeadArgPass>());
    pm.addPass(mlir::createPrintIRPass());
    pm.addPass(std::make_unique<RoutingConstantFoldPass>());
    pm.addPass(mlir::createPrintIRPass());
    //remove the dead code
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createPrintIRPass());

    //remove dead arg
    //pm.addPass(mlir::createConvertSCFToEmitCPass());
    (void)pm.run(module1);
    //module1.dump();
/*
    mlir::PassManager pm2(&ctx);;
    pm2.addPass(std::make_unique<RoutingHWLowerPass>(rtopology));
    (void)pm2.run(module1);
    module1.dump();
  */
//conver emitc into c code  
    mlir::LogicalResult result = mlir::emitc::translateToCpp(module1, llvm::outs());
    return 0;
}