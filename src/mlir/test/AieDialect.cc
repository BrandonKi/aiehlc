/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "AieDialect.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <string>

#include "aiedialect.cc.inc"
#define GET_OP_CLASSES
#include "aieop.cc.inc"
// using namespace mlir::aie;
using namespace mlir;

void AieADialect::initialize() {
	  addOperations< 
#define GET_OP_LIST
#include "aieop.cc.inc"
		        >();

	llvm::StringRef name = "name";
	TypeID id;
	//Dialect dl(name, NULL, id);
}

void dumpmlir() {
	 mlir::MLIRContext context;
	   // Load our Dialect in this MLIR Context.
	 context.getOrLoadDialect<mlir::AieADialect>();
	 ModuleOp module = ModuleOp::create(UnknownLoc::get(&context));
	 OpBuilder builder(&context);
	 builder.setInsertionPointToEnd(module.getBody());
	 //FuncOp func = FuncOp::create(builder.getUnknownLoc(), "kernelName", builder.getFunctionType({}, {}));
	 //module.push_back(func);
	 //Block *block = func.addEntryBlock();
	 //builder.setInsertionPointToStart(block);
	 auto elementType = IntegerType::get(&context, 64); 

	 // prepare the mlir::value for loadkernel parameter
	 std::vector<double> data = {0};
	 std::vector<int64_t> dims = {1};
	 auto dataType = mlir::RankedTensorType::get(dims, builder.getF64Type());
  	 auto denseAttr = DenseElementsAttr::get(dataType, llvm::ArrayRef(data));
	 auto constantOp = builder.create<ConstantOp>(UnknownLoc::get(&context), denseAttr);
	 mlir::Value value = constantOp.getResult();

	 builder.create<LoadKernelOp>(builder.getUnknownLoc(), value,value);
	 module.dump();
}
/*
int main(int argc, char* argv[]) {
	dumpmlir();
	return 1;
}
*/
