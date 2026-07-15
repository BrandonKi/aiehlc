/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef MLIR_AIELINK_DIALECT_H_
#define MLIR_AIELINK_DIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/TypeSwitch.h"

/// Include the auto-generated header file containing the declaration of the toy
/// dialect.
#include "mlirlinkrelocate/aielinkdialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "mlirlinkrelocate/aielinktype.h.inc"
/// Include the auto-generated header file containing the declarations of the
/// toy operations.
#define GET_OP_CLASSES
#include "mlirlinkrelocate/aielinkop.h.inc"

#endif // MLIR_AIELINK_DIALECT_H_
