/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include <iostream>
//#include "llvm/IR/IRBuilder.h"
//#include "llvm/IR/Function.h"
//#include "llvm/IR/Module.h"
//#include "mlir/IR/Attributes.h"
//#include "mlir/IR/Function.h"
//#include "mlir/IR/Module.h"
/*
#include "mlir/IR/Operation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/StorageUniquer.h"
#include "MLIRLoadKernelLexer.h"
#include "mlir/IR/MLIRContext.h"
#include "MLIRLoadKernelParser.h"
#include "MLIRLoadKernelListener.h"
*/
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
//#include "mlir/IR/Function.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Builders.h"
//#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include "MLIRLoadKernelLexer.h"
#include "MLIRLoadKernelParser.h"
#include "MLIRLoadKernelBaseListener.h"

#include <ctype.h>
using namespace antlr4;
//using namespace llvm;
using namespace mlir;
class MyListener : public MLIRLoadKernelBaseListener {
public:
	void generatemlir() {
		MLIRContext context;
	/*
		mlir::FuncOp func = mlir::FuncOp::create(UnknownLoc::get(&context), "loadkernel",
				FunctionType::get({MemRefType::get({}, context),
					MemRefType::get({2}, IntegerType::get(32, context))},
					{}, &context));

		Block *block = func.addEntryBlock();

		OpBuilder builder = OpBuilder::atBlockBegin(block);

		Value array = builder.create<ConstantOp>(
				UnknownLoc::get(&context), DenseElementsAttr::get<IntegerAttr>(
					VectorType::get(2, IntegerType::get(32, context)),
					{IntegerAttr::get(IntegerType::get(32, context), 1),
					IntegerAttr::get(IntegerType::get(32, context), 2)}));

		builder.create<CallOp>(UnknownLoc::get(&context), "loadkernel",
				ArrayRef<Type>{MemRefType::get({}, context),
				MemRefType::get({2}, IntegerType::get(32, context))},
				ValueRange{block->getArgument(0), array});

		func.print(llvm::outs());
		*/
	}
    void enterLoadKernelExpression(MLIRLoadKernelParser::LoadKernelExpressionContext *ctx) override {
        std::cout << "Found loadkernel expression:" << std::endl;
        std::cout << "ID: " << ctx->ID()->getText() << std::endl;

        auto matrixContext = ctx->matrix();
        std::cout << "Matrix:" << std::endl;
        for (auto rowContext : matrixContext->row()) {
            std::cout << "{ " << rowContext->INT(0)->getText() << ", " << rowContext->INT(1)->getText() << " }" << std::endl;
        }
    	mlir::MLIRContext mc;

        std::cout << std::endl;
	generatemlir();
    }
};
/*
int main() {
    ANTLRInputStream input("loadkernel(aa, {{1, 2}, {2, 2}})");
    MLIRLoadKernelLexer lexer(&input);
    CommonTokenStream tokens(&lexer);
    MLIRLoadKernelParser parser(&tokens);

    MyListener listener;
    tree::ParseTreeWalker::DEFAULT.walk(&listener, parser.start());

    return 0;
}
*/
#include "MLIRLoadKernelLexer.h"
#include "MLIRLoadKernelParser.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"


using namespace antlr4;
//using namespace MLIR;
/*
class MLIRLKListener : public MLIRLoadKernelBaseListener {
public:
    MLIRLKListener(MLIRContext &context) : builder(&context) {}

    void enterLoadKernelExpression(MLIRLoadKernelParser::LoadKernelExpressionContext *ctx) override {
        auto id = ctx->ID()->getText();
        auto matrixContext = ctx->matrix();

        //builder.createLoadKernel(id, matrixContext);
    }

    ModuleOp getModule() {
        return moduleOp;
    }

private:
    OpBuilder builder;
    ModuleOp moduleOp;
};
*/
ModuleOp generateMLIR(const std::string &input) {
    //MLIRContext context;
    ANTLRInputStream antlrInput(input);
    MLIRLoadKernelLexer lexer(&antlrInput);
    CommonTokenStream tokens(&lexer);
    MLIRLoadKernelParser parser(&tokens);
    auto tree = parser.start();
    //llvm::report_fatal_error("",1);
    //llvm::errs() << "Error: Something went wrong!\n";
    //llvm::StdThreadPool ls;
    std::cout << "mc " << std::endl;
    mlir::MLIRContext mc;
    //llvm::report_fatal_error("",1);
    MyListener listener;
    tree::ParseTreeWalker::DEFAULT.walk(&listener, tree);
		return ModuleOp();
    //return listener.getModule();
}
#include "mlir/IR/Dialect.h"
//#include "mlir/IR/Function.h"
//#include "mlir/IR/OpDefinition.h"

using namespace mlir;
/*
class ToyDialect : public Dialect {
public:
  ToyDialect(MLIRContext *context) : Dialect(::llvm::StringLiteral("toy"), context, ::mlir::TypeID::get<ToyDialect>()) {
  }
};

class ToyPrintOp : public Op<ToyPrintOp> {
public:
  using Op::Op;

  static ParseResult parse(OpAsmParser &parser, OperationState &result) {
    return success();
  }

  void print(OpAsmPrinter &p) {
  }
};

ToyDialect* dd = new ToyDialect((MLIRContext *)NULL);

//*/

void dumpmlir();
int main() {
    //mlir::registerAsmPrinterCLOptions();
    //mlir::registerMLIRContextCLOptions();
    std::string input = "loadkernel(aa, {1, 2})";
    ModuleOp mlirModule = generateMLIR(input);

    dumpmlir();
    //mlirModule.dump();
    return 0;
}

