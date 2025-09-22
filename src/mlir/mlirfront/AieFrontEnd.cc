/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

#include <iostream>
#include <string>
#include <sstream>
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
#include "mlir/IR/BuiltinTypes.h"
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

#include "AieFrontEnd.h"
#include "AieDialect.h"


#include "mlir/IR/Dialect.h"
//#include "MLIRLoadKernelLexer.h"
//#include "MLIRLoadKernelParser.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "mlirpass/aiehybrid.h"

bool use_llvm_aie = false;

//using namespace antlr4;
//using namespace llvm;
using namespace mlir;
using namespace mlir::aie;

/*
class MyListener : public MLIRLoadKernelBaseListener {
public:
    void enterLoadKernelExpression(MLIRLoadKernelParser::LoadKernelExpressionContext *ctx) override {
        std::cout << "Found loadkernel expression:" << std::endl;
        std::cout << "ID: " << ctx->ID()->getText() << std::endl;

        mlir::MLIRContext mc;

        auto matrixContext = ctx->matrix();
        std::cout << "Matrix:" << std::endl;
        for (auto rowContext : matrixContext->row()) {
            std::cout << "{ " << rowContext->INT(0)->getText() << ", " << rowContext->INT(1)->getText() << " }" << std::endl;
                        std::string name = ctx->ID()->getText();
            int x = std::stoi(std::string(rowContext->INT(0)->getText()));
            int y = std::stoi(std::string(rowContext->INT(1)->getText()));
            //EmitLoadKernel(x, y);
                        CreateLoadKernel(x, y, name);
        }
        std::cout << std::endl;
    }
};
*/
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
//ModuleOp generateMLIR(const std::string &input) {
void generateMLIR(const std::string &input) {
    //MLIRContext context;
    /*
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
    //MyListener listener;
    //tree::ParseTreeWalker::DEFAULT.walk(&listener, tree);
        return ;//ModuleOp();
    //return listener.getModule();
    */
}
//#include "mlir/IR/Function.h"
//#include "mlir/IR/OpDefinition.h"


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
AieFrontEnd::AieFrontEnd() :mbuilder(&mlirContext) {
    mlirContext.getOrLoadDialect<mlir::aie::AieADialect>();
    module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&mlirContext));
    mbuilder.setInsertionPointToStart(module->getBody());
    return;
}

void AieFrontEnd::set_llvm_aie(bool use) {
    use_llvm_aie = use;
}

mlir::Value AieFrontEnd::createKernelFuncName(OpBuilder &builder, std::string name)
{
    mlir::StringAttr nameAttr = mlir::StringAttr::get(&mlirContext,name);
    auto typeinfo = mlir::aie::KernelFuncNameType::get(&mlirContext, nameAttr);
    auto nameop = builder.create<mlir::aie::CreateKernelFuncNameOp>(builder.getUnknownLoc(),
                                                                                            typeinfo,
                                                                                            nameAttr);
    return nameop;
}

mlir::Value AieFrontEnd::createKernelFileName(OpBuilder &builder, std::string name)
{
    mlir::StringAttr nameAttr = mlir::StringAttr::get(&mlirContext,name);
    auto typeinfo = mlir::aie::KernelFileNameType::get(&mlirContext, nameAttr);
    auto nameop = builder.create<mlir::aie::CreateKernelFileNameOp>(builder.getUnknownLoc(),
                                                                                            typeinfo,
                                                                                            nameAttr);
    return nameop;
}

mlir::Value AieFrontEnd::createKernelWindowOp(OpBuilder &builder,
                                              int arg_dir,
                                              int arg_size, 
                                              long arg_pingaddr, 
                                              long arg_pongaddr, 
                                              int arg_pinglockid, 
                                              int arg_ponglockid, 
                                              std::string name)
{
    //mlir::MLIRContext context;
    //context.getOrLoadDialect<AieADialect>();

    //ModuleOp module = ModuleOp::create(UnknownLoc::get(&context));
    //OpBuilder builder(&context);
    //builder.setInsertionPointToEnd(module.getBody());

    auto i32dataType = mlir::IntegerType::get(&mlirContext, 32);
    auto i64dataType = mlir::IntegerType::get(&mlirContext, 64);

    auto attr_direction = mlir::IntegerAttr::get(i32dataType, arg_dir);
    auto attr_size = mlir::IntegerAttr::get(i32dataType, arg_size);
    auto attr_pingaddr = mlir::IntegerAttr::get(i64dataType, arg_pingaddr);
    auto attr_pongaddr = mlir::IntegerAttr::get(i64dataType, arg_pongaddr);
    auto attr_pinglockid = mlir::IntegerAttr::get(i32dataType, arg_pinglockid);
    auto attr_ponglockid = mlir::IntegerAttr::get(i32dataType, arg_ponglockid);

    mlir::StringAttr nameAttr = builder.getStringAttr(name);

    mlir::Type typeinfo = mlir::aie::WindowType::get(&mlirContext, 
            arg_dir, 
            arg_size, 
            arg_pingaddr, 
            arg_pongaddr, 
            arg_pinglockid, 
            arg_ponglockid, 
            nameAttr);

    auto windowOp = builder.create<mlir::aie::CreateWindowOp>(builder.getUnknownLoc(), 
            typeinfo,
            attr_direction, 
            attr_size, 
            attr_pingaddr, 
            attr_pongaddr, 
            attr_pinglockid, 
            attr_ponglockid, 
            nameAttr);
    //module.dump();
    return windowOp;
}

mlir::Value AieFrontEnd::createKernelWindowOp(ParmVarDecl * param,
                                                std::string kernelFuncName,
                                                OpBuilder &builder,
                                                long unsigned pingaddress,
                                                long unsigned pongaddress,
                                                unsigned pinglockid,
                                                unsigned ponglockid,
                                                clang::Rewriter* Rewriter,
                                                clang::SourceLocation linestart,
                                                int size,
                                                int direction) {
    std::string typeName = param->getType().getAsString();
    StringRef name = param->getNameAsString();
    // FIXME only works with window type API
    if (typeName.find("input_window") != std::string::npos) {
        direction = 0;
    } else if (typeName.find("output_window")!= std::string::npos) {
        direction = 1;
    }
    int windowSize = size * sizeof(int);
  return createKernelWindowOp(builder,
                                    direction,
                                    windowSize,
                                    pingaddress,
                                    pongaddress,
                                    pinglockid,
                                    ponglockid,
                                    name.str());
}
void AieFrontEnd::createKernelFunction(std::vector<std::string> params) {
    // std::cout << "--------------AieFrontEnd::createKernelFunction-----------" << std::endl;
    for (auto x: params) {
        std::cout << x << std::endl;
    }
    // std::cout << "--------------AieFrontEnd::createKernelFunction----END-------" << std::endl;
    return;
}
void AieFrontEnd::createKernelDefinitionOp(FunctionDecl *f,
                                           clang::Rewriter* Rewriter,
                                           clang::SourceLocation linestart) {
    llvm::SmallVector<mlir::Value, 4> windowOps;
    llvm::SmallVector<WindowType> windowOpTypes;
    std::string kernelFuncName = f->getNameInfo().getName().getAsString();
    // std::cout << " kernelFuncNamexx is " << kernelFuncName << std::endl;
    // Assume you have a way to determine the kernel symbol, input args, and output args
    int numInputArgs = 0; // Placeholder: calculate based on function's input
    int numOutputArgs = 0; // Placeholder: calculate based on function's output
    int64_t pingaddress = 0x1000; //start address
    int32_t pingLockID = 0;
    int32_t pongLockID = 1;
    int32_t paramNum = f->getNumParams();
    uint32_t setpingaddress = 0;

    auto kname = createKernelFuncName(mbuilder, kernelFuncName);
    auto fname = createKernelFileName(mbuilder, "./" + kernelFuncName + ".c");
    std::vector<std::string> window_types = {};

    for (int i = 0; i < paramNum; i++) {
        ParmVarDecl* param = f->getParamDecl(i);
        window_types.push_back(param->getType().getAsString());
        //xchess seems have a logic issue, we need to reserve 4 time large buffer
        //for 512 that means the local buffer is 32 int32 length
        int defaultSize = 512;
        //get the windows size
        auto getvalue = [&](std::string str) {
                    std::istringstream istr(str);
                    std::string tmp;
                    std::getline(istr, tmp, ':');
                    istr >> tmp;
                    return tmp;
        };
        for (auto attr: param->attrs()) {
            if (attr->getKind() == clang::attr::Kind::Annotate) {
                auto annotateAttr= static_cast<clang::AnnotateAttr*>(attr);
                std::string annotation = annotateAttr->getAnnotation().str();
                if (annotation.find("size_hint") != std::string::npos) {
                    int csize = 0;
                    auto nstr = getvalue(annotation);
                     csize = stol(nstr, nullptr, 0);
                    defaultSize = csize ? csize : defaultSize;
                } else if (annotation.find("mem_address") != std::string::npos) {
                    auto nstr = getvalue(annotation);
                    setpingaddress = stol(nstr, nullptr, 0);
                    pingaddress = setpingaddress ? setpingaddress : pingaddress;
                }
            }
        }
        //the xchess have a logic to ask 4 time space to storage the data, not sure whether that is bug
        //enclose is a workground to handle compile complain
        defaultSize = defaultSize * 4;
        int64_t pongaddress = pingaddress  + defaultSize *sizeof(int);
        auto windowOp = createKernelWindowOp(param,
                                            kernelFuncName,
                                            mbuilder,
                                            pingaddress,
                                            pongaddress,
                                            pingLockID,
                                            pongLockID,
                                            Rewriter,
                                            linestart,
                                            defaultSize,
                                            i); // FIXME works for kernel(input, output)
        if (windowOp) {
            windowOps.push_back(windowOp);
            windowOpTypes.push_back(dyn_cast<aie::WindowType>(windowOp.getType()));
        }
        pingaddress = pongaddress + (defaultSize * sizeof(int));
        pingLockID = pongLockID + 1;
        pongLockID = pingLockID + 1;
        std::string typeName = param->getType().getAsString();
        // FIXME only works with window type API
        if (typeName.find("input_window") != std::string::npos) {
            numInputArgs += 1;
        } else if (typeName.find("output_window") != std::string::npos) {
            numOutputArgs += 1;
        }
    }

    ArrayRef<WindowType> typeArrayRef(windowOpTypes);
    mlir::ValueRange windowOpsRange(windowOps);

    mlir::Type result_type = mlir::aie::KernelType::get(&mlirContext, numInputArgs, numOutputArgs,
                        dyn_cast<aie::KernelFuncNameType>(kname.getType()),
                        dyn_cast<aie::KernelFileNameType>(fname.getType()),
                        typeArrayRef); 
    std::string custom_type = window_types[0];
    // FIXME temporary, remove the type if using the window type API
    if (custom_type.find("input_window") != std::string::npos || custom_type.find("output_window") != std::string::npos) {
        custom_type = "";
    }
    mlir::aie::CreateKernelObjectOp kernelObject = mbuilder.create<mlir::aie::CreateKernelObjectOp>(
            mbuilder.getUnknownLoc(), result_type, numInputArgs, numOutputArgs, kname,fname, window_types[0], window_types[1], windowOps);
    kernelfunctionToKernelObjectMap[f->getName().str()] = kernelObject;
    return;
}

std::string AieFrontEnd::dumpir() {
    std::string output;
    llvm::raw_string_ostream os(output);
    module->print(os);
    // std::cout << output << std::endl;
    return output;
}

void AieFrontEnd::RunPass(std::string file_path) {
    mlir::MLIRContext ctx;
    ctx.getOrLoadDialect<AieADialect>();

    llvm::SourceMgr sourceMgr;
    auto fo = llvm::MemoryBuffer::getFile(file_path.c_str());

  if (auto err = fo.getError()) {
        std::cerr << "file open error" << std::endl;
        return;
    }

    sourceMgr.AddNewSourceBuffer(std::move(*fo), llvm::SMLoc());
    auto module = mlir::parseSourceFile(sourceMgr, &ctx);

    if (!module) {
        std::cerr << "module error" <<std::endl;
        return;
    }

    auto pm = PassManager::on<mlir::ModuleOp>(&ctx);
      pm.addPass(std::make_unique<HybridPass>());
    if (failed(pm.run(*module))) {
        std::cerr << "pm.run failed " << std::endl;
    }
}

void AieFrontEnd::Parser(std::string file_path) {
    // std::cout << "Parser" << std::endl;
    mlir::MLIRContext context;
    llvm::SourceMgr sourceMgr;
    context.getOrLoadDialect<AieADialect>();

    auto foe = llvm::MemoryBuffer::getFile(file_path.c_str());

    if (auto err = foe.getError()) {
        std::cerr << "file error" << std::endl;
        return;
    }
  sourceMgr.AddNewSourceBuffer(std::move(*foe), llvm::SMLoc());
    auto module = mlir::parseSourceFile(sourceMgr, &context);

    if (!module) {
        llvm::errs() << "get error module\n";
        return;
    }
    mlir::Operation *op = module.get();
    op->walk([&](mlir::Operation* op) {
            std::cout << op->getName().getStringRef().str() << std::endl;
            //op->emitRemark() << "Visiting operation: " << op->getName();
    });
}


void TestPrinterParser(std::string file_path){
    std::cout << "**********Testing Printer Parser***********\n";
    TestParserOperation(file_path);
}
