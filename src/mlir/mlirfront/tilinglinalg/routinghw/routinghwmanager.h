/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef __ROUTINGHW__
#define __ROUTINGHW__
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/TypeSwitch.h"

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

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
// fieldparser
#include "mlir/IR/DialectImplementation.h"
// llvm dialect
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
// #include "mlir/Dialect/LLVM/Transforms/Passes.h"

#include "routinghwdialect.h.inc"

using namespace mlir;
using namespace routinghw;

#define GET_ATTRDEF_CLASSES
#include "routinghwattr.h.inc"
#undef GET_ATTRDEF_CLASSES

#define GET_TYPEDEF_CLASSES
#include "routinghwtype.h.inc"
#undef GET_TYPEDEF_CLASSES

#define GET_OP_CLASSES
#include "routinghwop.h.inc"
#undef GET_OP_CLASSEST

class routinghwmanager{
public:
    routinghwmanager(){};
    ModuleOp ops_test(MLIRContext* ctx) ;
    mlir::func::FuncOp createroutingfunc(MLIRContext* ctx) ;
    static void loaddialect(MLIRContext* ctx);
};
#endif //__ROUTINGHW__