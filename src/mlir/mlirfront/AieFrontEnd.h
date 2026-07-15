/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef __AIEFRONTEND__
#define __AIEFRONTEND__
#include <iostream>
#include "AieDialect.h"
// for clang
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Instructions.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/AST/Decl.h"

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
//#include "MLIRLoadKernelLexer.h"
//#include "MLIRLoadKernelParser.h"
//#include "MLIRLoadKernelBaseListener.h"

#include <ctype.h>
using namespace mlir;
using namespace clang;
using namespace clang::tooling;
void dumpmlir();
//ModuleOp generateMLIR(const std::string &input);
void generateMLIR(const std::string &input);

void EmitLoadKernel(int x, int y);
void EmitWindowOperation();
void TestParserOperation(std::string file_path);
void TestPrinterParser(std::string file_path);

void TraverseMLIR(std::string file_path);
void generateWrapperCode(std::string file_path);

void CreateLoadKernel(int x, int y, std::string name = "");
void CreateWindowOperation(int arg_dir, int arg_size, long arg_pingaddr, long arg_pongaddr,
       int arg_pinglockid, int arg_ponglockid, std::string name);
void CreateKernelObjectOperation(int NumInputArgs = 1, int NumOutputArgs = 1);
void CreateTileKernelObjectOperation(int, int, std::string);

// Structure to hold streaming configuration
struct StreamingConfig {
    bool enabled;
    uint32_t chunk_size; // in elements
    uint32_t num_chunks;
};

class AieFrontEnd {
public:
	AieFrontEnd();
	void set_llvm_aie(bool);
	std::string dumpir();
	void createKernelFunction(std::vector<std::string> params);
    void createKernelDefinitionOp(FunctionDecl *f, clang::Rewriter *Rewriter, clang::SourceLocation linestart,
                                  bool isStreaming = false);
    mlir::Value createKernelWindowOp(ParmVarDecl * param,
                                     std::string kernelFuncName,
                                     OpBuilder &builder,
                                     long unsigned pingaddress,
                                     long unsigned pongaddress,
                                     unsigned pinglockid,
                                     unsigned ponglockid,
                                     clang::Rewriter* Rewriter,
                                     clang::SourceLocation linestart,
                                     int size,
                                     int direction);
	mlir::Value createKernelWindowOp(OpBuilder &builder,
                                     int arg_dir,
                                     int arg_size,
                                     long arg_pingaddr,
                                     long arg_pongaddr,
                                     int arg_pinglockid,
                                     int arg_ponglockid,
                                     std::string name);
	
	mlir::Value createKernelFuncName(OpBuilder &builder, std::string name);
	mlir::Value createKernelFileName(OpBuilder &builder, std::string name);

	void Parser(std::string file_path);
  void RunPass(std::string file_path);
	std::unordered_map<std::string, ::mlir::aie::CreateKernelObjectOp> kernelfunctionToKernelObjectMap;
    std::unordered_map<std::string, StreamingConfig> kernelStreamingConfig;

  private:
	mlir::OpBuilder mbuilder;
	mlir::MLIRContext mlirContext;
	mlir::OwningOpRef<mlir::ModuleOp> module;
};
#endif//__AIEFRONTEND__
