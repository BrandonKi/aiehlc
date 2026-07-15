/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
#include "routingmanager.h"
#include "tilinglinalg_pipeline.h"
#include <iostream>
#include <map>

#define GET_TYPEDEF_CLASSES
#define GET_ATTRDEF_CLASSES
#define GET_OP_CLASSES
#define GET_OP_DEFS
// #define GET_OP_LIST
#include "routinginterface.cc.inc"
#include "routingdialect.cc.inc"
#include "routingattr.cc.inc"
#include "routingtype.cc.inc"

#include "routingop.cc.inc"
// #undef GET_OP_LIST
#undef GET_OP_DEFS

#undef GET_OP_CLASSES
#undef GET_ATTRDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

void routingdialect::initialize()  { 
    addOperations<
    #define GET_OP_LIST
    #include "routingop.cc.inc"
        >();
    // 如果有 Attr / Type：addAttributes<...>(); addTypes<...>();
     addAttributes<
    #define GET_ATTRDEF_LIST
    #include "routingattr.cc.inc"
    >();

        // 3. Types
    addTypes<
    #define GET_TYPEDEF_LIST
    #include "routingtype.cc.inc"
    >();
}
// implment interface api
int64_t tilearrayType::getTileBase() const {
    return 100;
}

// implement creattilearrayop result assembly
LogicalResult createtilearrayOp::inferReturnTypes(mlir::MLIRContext* ctx, 
    std::optional<mlir::Location>, 
    mlir::ValueRange, 
    mlir::DictionaryAttr args, 
    mlir::OpaqueProperties, 
    mlir::RegionRange, 
    llvm::SmallVectorImpl<mlir::Type>& infer) {
        //args is arguments
        //auto items = args.get("items").dyn_cast_or_null<TileRangeArrayAttr>();
        //if (!items) return failure();
    TileRangeAttr ta = TileRangeAttr::get(ctx, -1,-1);
    llvm::SmallVector<TileRangeAttr,4> slist;
    slist.push_back(ta);
    infer.push_back(tilearrayType::get(ctx,slist));

    return success();

}

LogicalResult createdataio::inferReturnTypes(mlir::MLIRContext* ctx, 
    std::optional<mlir::Location>, 
    mlir::ValueRange, 
    mlir::DictionaryAttr args, 
    mlir::OpaqueProperties, 
    mlir::RegionRange, 
    llvm::SmallVectorImpl<mlir::Type>& infer) {
        //args is arguments
        auto items = args.get("iotype").dyn_cast_or_null<StringAttr>();
        auto items2 = args.get("direction").dyn_cast_or_null<StringAttr>();
        if (!items||!items) return failure();
        infer.push_back(dataioType::get(ctx,items, items2));

    return success();

}

// In RoutingOps.cpp

// This is the C++ implementation for the printer.
void routing::RoutingCreate::print(OpAsmPrinter &p) {
  // `p` is the printer object. The `<<` operator prints literal strings.
  //p << " on_row ";

  // Use printAttribute to print attributes. Angle brackets are just literals.
  p << "<";
  p << "Memo = \"";
  p << (getMemo());
  p << "\">";

  // Use printOperand for SSA values, and print its type.
  p << " ( scf_idx = ";
  p.printOperand(getScfIdx());
  p << " : ";
  p.printType(getScfIdx().getType());
  p << ")";

  // Use printOptionalAttrDict to print any attributes we haven't
  // explicitly printed. This is good practice for forward compatibility.
  // We need to tell it which attributes we already handled.
  //p.printOptionalAttrDict(this->getAttrs(), /*elidedAttrs=*/{"device_row"});

  // Print the result type.
  p << " -> ";
  p.printType(getResult().getType());
  
  // Use printRegion to print the region.
  p.printRegion(getBody(), /*printEntryBlockArgs=*/true,
                              /*printBlockTerminators=*/true);
}

// In RoutingOps.cpp

// This is the C++ implementation for the parser.
// Format (matches printer): <Memo = "str"> ( scf_idx = %val : type) -> type { region }
ParseResult RoutingCreate::parse(OpAsmParser &parser, OperationState &result) {
  // --- 1. Parse <Memo = "string"> ---
  if (parser.parseLess())
    return failure();

  // Parse key-value pairs inside angle brackets
  while (true) {
    // Check for closing >
    if (succeeded(parser.parseOptionalGreater()))
      break;

    StringRef attrName;
    if (parser.parseKeyword(&attrName) || parser.parseEqual())
      return failure();

    if (attrName == "Memo") {
      StringAttr memoAttr;
      if (parser.parseAttribute(memoAttr, "Memo", result.attributes))
        return failure();
    } else {
      // Unknown attribute inside <>, try to parse as generic attribute
      Attribute genericAttr;
      if (parser.parseAttribute(genericAttr, attrName, result.attributes))
        return failure();
    }
    parser.parseOptionalComma();
  }

  // --- 2. Parse ( scf_idx = %operand : type ) ---
  OpAsmParser::UnresolvedOperand inputOperand;
  Type inputType;
  if (parser.parseLParen() ||
      parser.parseKeyword("scf_idx") || parser.parseEqual() ||
      parser.parseOperand(inputOperand) ||
      parser.parseColon() ||
      parser.parseType(inputType) ||
      parser.parseRParen())
    return failure();

  // Parse any optional attributes
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // --- 3. Parse -> result_type ---
  Type resultType;
  if (parser.parseArrow() || parser.parseType(resultType))
    return failure();

  // --- 4. Parse the region ---
  Region *body = result.addRegion();
  llvm::SmallVector<OpAsmParser::Argument> regionArgs;
  OpAsmParser::Argument arg;
  arg.type = inputType;
  regionArgs.push_back(arg);
  if (parser.parseRegion(*body, regionArgs))
    return failure();

  // --- 5. Populate the OperationState ---
  result.addTypes(resultType);
  if (parser.resolveOperand(inputOperand, inputType, result.operands))
    return failure();

  return success();
}

// LevelAttr custom printer: prints
//   <base = N, total = N, slice = N, step = N, rounds = N[, slice_tiling = #routing.level<...>]>
// The nested slice_tiling level is printed with its full `#routing.level<...>`
// mnemonic (via operator<<), unlike the auto struct(params) form which strips it.
void routing::LevelAttr::print(mlir::AsmPrinter &printer) const {
    printer << "<base = " << getBase() << ", total = " << getTotal() << ", slice = " << getSlice()
            << ", step = " << getStep() << ", rounds = " << getRounds();
    if (auto sliceTiling = getSliceTiling())
        printer << ", slice_tiling = " << sliceTiling;
    printer << ">";
}

// LevelAttr custom parser: mirror of print. slice_tiling is optional (leaf level
// omits it). Nested level is read back via parseAttribute (full mnemonic form).
mlir::Attribute routing::LevelAttr::parse(mlir::AsmParser &parser, mlir::Type) {
    int64_t base, total, slice, step, rounds;
    if (parser.parseLess() || parser.parseKeyword("base") || parser.parseEqual() || parser.parseInteger(base) ||
        parser.parseComma() || parser.parseKeyword("total") || parser.parseEqual() || parser.parseInteger(total) ||
        parser.parseComma() || parser.parseKeyword("slice") || parser.parseEqual() || parser.parseInteger(slice) ||
        parser.parseComma() || parser.parseKeyword("step") || parser.parseEqual() || parser.parseInteger(step) ||
        parser.parseComma() || parser.parseKeyword("rounds") || parser.parseEqual() || parser.parseInteger(rounds))
        return {};
    routing::LevelAttr sliceTiling;
    if (succeeded(parser.parseOptionalComma())) {
        if (parser.parseKeyword("slice_tiling") || parser.parseEqual() || parser.parseAttribute(sliceTiling))
            return {};
    }
    if (parser.parseGreater())
        return {};
    return routing::LevelAttr::get(parser.getContext(), base, total, slice, step, rounds, sliceTiling);
}

// DimAttr custom printer: prints <outer = #routing.level<...>> with the full
// nested-level mnemonic (auto struct(params) would strip the prefix).
void routing::DimAttr::print(mlir::AsmPrinter &printer) const { printer << "<outer = " << getOuter() << ">"; }

// DimAttr custom parser: reads <outer = #routing.level<...>>.
mlir::Attribute routing::DimAttr::parse(mlir::AsmParser &parser, mlir::Type) {
    routing::LevelAttr outer;
    if (parser.parseLess() || parser.parseKeyword("outer") || parser.parseEqual() || parser.parseAttribute(outer) ||
        parser.parseGreater())
        return {};
    return routing::DimAttr::get(parser.getContext(), outer);
}

// TilingAttr custom printer: prints <d0 = #routing.dim<...>, d1 = ...>.
// The leading mnemonic (#routing.tiling) is emitted by the dialect printer.
void routing::TilingAttr::print(mlir::AsmPrinter &printer) const {
    printer << "<";
    auto dims = getDims();
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i)
            printer << ", ";
        printer << "d" << i << " = " << dims[i];
    }
    printer << ">";
}

// TilingAttr custom parser: reads <dN = #routing.dim<...>, ...> into the dims
// array (the dN index label is positional and re-derived on print).
mlir::Attribute routing::TilingAttr::parse(mlir::AsmParser &parser, mlir::Type) {
    if (parser.parseLess())
        return {};
    llvm::SmallVector<routing::DimAttr> dims;
    if (succeeded(parser.parseOptionalGreater()))
        return routing::TilingAttr::get(parser.getContext(), dims);
    do {
        StringRef label;
        if (parser.parseKeyword(&label) || parser.parseEqual())
            return {};
        routing::DimAttr dim;
        if (parser.parseAttribute(dim))
            return {};
        dims.push_back(dim);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseGreater())
        return {};
    return routing::TilingAttr::get(parser.getContext(), dims);
}

// partitiontensorOp printer
void routing::partitiontensor::print(OpAsmPrinter &printer) {
    printer << " " << getTensor() << " : " << getTensor().getType();
    printer << " {\n  partition = " << getPartitionAttr();
    if (getTiling())
        printer << ",\n  tiling = " << getTilingAttr();
    printer << "\n} -> " << getOutput().getType();
}

// partitiontensorOp parser
ParseResult routing::partitiontensor::parse(OpAsmParser &parser, OperationState &result) {
    OpAsmParser::UnresolvedOperand tensorOperand;
    Type tensorType;

    if (parser.parseOperand(tensorOperand) || parser.parseColonType(tensorType))
        return failure();

    if (parser.parseLBrace())
        return failure();

    if (parser.parseKeyword("partition") || parser.parseEqual())
        return failure();
    routing::PartitionAttr partitionAttr;
    if (parser.parseAttribute(partitionAttr))
        return failure();
    result.addAttribute("partition", partitionAttr);

    if (succeeded(parser.parseOptionalComma())) {
        if (parser.parseKeyword("tiling") || parser.parseEqual())
            return failure();
        routing::TilingAttr tilingAttr;
        if (parser.parseAttribute(tilingAttr))
            return failure();
        result.addAttribute("tiling", tilingAttr);
    }

    if (parser.parseRBrace())
        return failure();

    if (parser.parseArrow())
        return failure();

    Type resultType;
    if (parser.parseType(resultType))
        return failure();

    result.addTypes(resultType);
    if (parser.resolveOperand(tensorOperand, tensorType, result.operands))
        return failure();

    return success();
}

// createhwmesh printer
// Format: routing.routingcreatehwmesh row = <row>, col = <col> [partition = <startCol>, <endCol>, <startRow>, <endRow>]
// -> type
void routing::createhwmesh::print(OpAsmPrinter &p) {
    p << " row = " << getRow() << ", col = " << getCol();
    if (getStartCol()) {
        p << " partition = " << *getStartCol() << ", " << *getEndCol() << ", " << *getStartRow() << ", "
          << *getEndRow();
    }
    p.printOptionalAttrDict(getOperation()->getAttrs(),
                            /*elidedAttrs=*/{"row", "col", "startCol", "endCol", "startRow", "endRow"});
    p << " -> " << getOutput().getType();
}

// createhwmesh parser
ParseResult routing::createhwmesh::parse(OpAsmParser &parser, OperationState &result) {
    int64_t row, col;
    if (parser.parseKeyword("row") || parser.parseEqual() || parser.parseInteger(row) || parser.parseComma() ||
        parser.parseKeyword("col") || parser.parseEqual() || parser.parseInteger(col))
        return failure();

    result.addAttribute("row", parser.getBuilder().getI64IntegerAttr(row));
    result.addAttribute("col", parser.getBuilder().getI64IntegerAttr(col));

    // Parse optional partition = startCol, endCol, startRow, endRow
    if (succeeded(parser.parseOptionalKeyword("partition"))) {
        int64_t startCol, endCol, startRow, endRow;
        if (parser.parseEqual() || parser.parseInteger(startCol) || parser.parseComma() ||
            parser.parseInteger(endCol) || parser.parseComma() || parser.parseInteger(startRow) ||
            parser.parseComma() || parser.parseInteger(endRow))
            return failure();
        result.addAttribute("startCol", parser.getBuilder().getI64IntegerAttr(startCol));
        result.addAttribute("endCol", parser.getBuilder().getI64IntegerAttr(endCol));
        result.addAttribute("startRow", parser.getBuilder().getI64IntegerAttr(startRow));
        result.addAttribute("endRow", parser.getBuilder().getI64IntegerAttr(endRow));
    }

    if (parser.parseOptionalAttrDict(result.attributes))
        return failure();

    Type resultType;
    if (parser.parseArrow() || parser.parseType(resultType))
        return failure();

    result.addTypes(resultType);
    return success();
}

//routing class
void routingmanager::type_interface_test(MLIRContext* ctx) {
        //ctx->getOrLoadDialect<routing::routingdialect>();
        TileRangeAttr tarrayattr = TileRangeAttr::get(ctx, 1, 1);
        llvm::SmallVector<TileRangeAttr,4> slist;
        slist.push_back(tarrayattr);
        tilearrayType ttype = tilearrayType::get(ctx, slist);
        if (auto tt = dyn_cast<TileInterface>(ttype)) {
            llvm::outs() << tt.getTileBase() << "\n";
        }
        auto len = ttype.getItems().size();
        llvm::outs() << len << " is the length\n"; 
    }
ModuleOp routingmanager::ops_test(MLIRContext* ctx, int totalN) {
    OpBuilder builder(ctx);
    mlir::ModuleOp m = ModuleOp::create(builder.getUnknownLoc());
    auto func = createroutingfunc(ctx, totalN);
    m.push_back(func);
    auto functype = builder.getFunctionType({},{});
    mlir::func::FuncOp main = builder.create<func::FuncOp>(builder.getUnknownLoc(), "main", functype);
    auto block = main.addEntryBlock();
    builder.setInsertionPointToEnd(block);
    //Value cnum   = builder.create<arith::ConstantIndexOp>(location, 1);
    // cnum and rnum is not the ARRAY/hw shape, it means reserve how many row , how many col used to do broadcast
    Value cnum = builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), 1);
    Value rnum = builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), 8);
    //Value total = builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(),16);
    auto callop = builder.create<mlir::func::CallOp>(builder.getUnknownLoc(), func, ValueRange({cnum, rnum}));
    auto retop = builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    m.push_back(main);
    llvm::errs() << m;
    return m;
}

ModuleOp routingmanager::ops_testNew(MLIRContext *ctx, int totalN, std::string routingname) {
    const int hwrowused = 2, hwcolused = 2;
    OpBuilder builder(ctx);
    mlir::ModuleOp m = ModuleOp::create(builder.getUnknownLoc());
    auto functype = builder.getFunctionType({},{});

    if (routingname.empty()) {
        routingname = "main";
    }

    mlir::func::FuncOp main = builder.create<func::FuncOp>(builder.getUnknownLoc(), routingname, functype);

    auto block = main.addEntryBlock();
    builder.setInsertionPointToEnd(block);
    auto mesh = builder.create<createhwmesh>(builder.getUnknownLoc(), hwrowused, hwcolused,
                                             /*startCol=*/mlir::IntegerAttr{}, /*endCol=*/mlir::IntegerAttr{},
                                             /*startRow=*/mlir::IntegerAttr{}, /*endRow=*/mlir::IntegerAttr{});

    // Helper to create a schedule tensor with dense init data
    auto createTensor = [&](SmallVector<int64_t> shapeVec, int64_t startVal) -> Value {
        SmallVector<Attribute> shape;
        for (int64_t v : shapeVec)
            shape.push_back(builder.getI64IntegerAttr(v));
        ArrayAttr vals = builder.getArrayAttr(shape);
        IntegerAttr dimnum = builder.getI64IntegerAttr(shapeVec.size());
        auto tensorType = RankedTensorType::get(shapeVec, builder.getI8Type());

        SmallVector<llvm::APInt> initValues;
        int64_t totalElements = 1;
        for (auto dim : shapeVec)
            totalElements *= dim;
        for (int64_t i = 0; i < totalElements; ++i)
            initValues.push_back(llvm::APInt(8, startVal + i));

        auto denseAttr = DenseElementsAttr::get(tensorType, initValues);
        auto initConstant = builder.create<arith::ConstantOp>(builder.getUnknownLoc(), tensorType, denseAttr);
        auto tensor = builder.create<createscheduletensor>(builder.getUnknownLoc(), tensorType,
                                                           initConstant.getResult(), vals, dimnum);
        return tensor;
    };

    // GEMM: C[16x16] = A[16x16] * B[16x16]
    // A (input)  — start from 1
    // B (input)  — start from 2 (must differ from A to prevent constant folding/merge)
    // C (output) — zeroed out (result buffer)
    Value tensorA = createTensor({16, 16}, 1);
    Value tensorB = createTensor({16, 16}, 2);
    Value tensorC = createTensor({16, 16}, 0);

    createroutingfuncGEMM(builder, ctx, mesh, tensorA, tensorB, tensorC, hwrowused, "row");

    auto retop = builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    m.push_back(main);
    llvm::errs() << m;
    return m;
}

void routingmanager::createroutingfuncGEMM(OpBuilder &builder, MLIRContext *ctx, Value mesh, Value tensorA,
                                           Value tensorB, Value tensorC, uint32_t hwsplitnum, std::string splitAxis) {
    auto location = builder.getUnknownLoc();
    auto exec = builder.create<scf::ExecuteRegionOp>(location, /*result types*/ TypeRange{});
    exec->setAttr("routing_memo", builder.getStringAttr(splitAxis));
    {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&exec.getRegion().emplaceBlock());

        // One partitionmesh for the entire GEMM
        auto partmesh = builder.create<partitionmesh>(location, mesh, hwsplitnum, splitAxis);

        int splitdimn = splitAxis == "row" ? 0 : 1;

        // Get tensor types
        auto tensorTypeA = tensorA.getType().cast<RankedTensorType>();
        auto tensorTypeB = tensorB.getType().cast<RankedTensorType>();
        auto tensorTypeC = tensorC.getType().cast<RankedTensorType>();

        // Partition tensor A (input): split by row, each tile group gets its rows
        auto partTensorA = builder.create<partitiontensor>(
            location, tensorTypeA, tensorA,
            routing::PartitionAttr::get(ctx, hwsplitnum, splitdimn, splitAxis, "col", ""), routing::TilingAttr{});

        // Partition tensor B (input): replicated to all tile groups (broadcast)
        // splitnum=1 means no split — full tensor goes to each group
        // hw_axis_owner="" means not owned by any axis, replicate_on=splitAxis means broadcast along split axis
        auto partTensorB = builder.create<partitiontensor>(
            location, tensorTypeB, tensorB, routing::PartitionAttr::get(ctx, 1, splitdimn, "", splitAxis, ""),
            routing::TilingAttr{});

        // Partition tensor C (output): split by row, each tile group produces its rows
        auto partTensorC = builder.create<partitiontensor>(
            location, tensorTypeC, tensorC,
            routing::PartitionAttr::get(ctx, hwsplitnum, splitdimn, splitAxis, "col", ""), routing::TilingAttr{});

        // Calculate split tensor shapes
        SmallVector<int64_t> splitShapeA(tensorTypeA.getShape());
        SmallVector<int64_t> splitShapeC(tensorTypeC.getShape());
        if (splitdimn == 0) {
            if (splitShapeA[0] != ShapedType::kDynamic)
                splitShapeA[0] /= hwsplitnum;
            if (splitShapeC[0] != ShapedType::kDynamic)
                splitShapeC[0] /= hwsplitnum;
        } else {
            if (splitShapeA[1] != ShapedType::kDynamic)
                splitShapeA[1] /= hwsplitnum;
            if (splitShapeC[1] != ShapedType::kDynamic)
                splitShapeC[1] /= hwsplitnum;
        }
        auto splitTensorTypeA = RankedTensorType::get(splitShapeA, tensorTypeA.getElementType());
        // B is not split (splitnum=1), extract returns full tensor
        auto splitTensorTypeB = tensorTypeB;
        auto splitTensorTypeC = RankedTensorType::get(splitShapeC, tensorTypeC.getElementType());

        Value lb = builder.create<arith::ConstantIndexOp>(location, 0);
        Value ub = builder.create<arith::ConstantIndexOp>(location, hwsplitnum);
        Value step = builder.create<arith::ConstantIndexOp>(location, 1);

        auto scf = builder.create<mlir::scf::ForOp>(location, lb, ub, step);
        {
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(scf.getBody());
            auto memo = builder.getStringAttr(splitAxis);
            mlir::Value scf_idx = scf.getInductionVar();
            Value idx = builder.create<arith::IndexCastOp>(location, builder.getI32Type(), scf_idx);

            auto routingcreateOp = builder.create<routing::RoutingCreate>(
                location, idx, memo, [&](OpBuilder &b, Location bodyLoc, Value sidx) {
                    auto tilelist = b.create<extract_tiles>(location, partmesh, sidx);

                    // --- Input A: split slice per tile group ---
                    auto sliceA = b.create<extract_data>(location, splitTensorTypeA, partTensorA, sidx);
                    auto hwioA = b.create<createhwiowithtarget>(location, tilelist, "input", "mem2");
                    b.create<movedatabyio>(location, sliceA, hwioA);

                    // --- Input B: broadcast full tensor to each tile group ---
                    auto sliceB = b.create<extract_data>(location, splitTensorTypeB, partTensorB, sidx);
                    auto hwioB = b.create<createhwiowithtarget>(location, tilelist, "input", "mem2");
                    b.create<movedatabyio>(location, sliceB, hwioB);

                    // --- Output C: gather from tile group ---
                    auto sliceC = b.create<extract_data>(location, splitTensorTypeC, partTensorC, sidx);
                    auto gatherC = b.create<routinggatherout>(location, sliceC.getType(), tilelist, sliceC);
                    auto hwioC = b.create<createhwiowithtarget>(location, tilelist, "output", "mem2");
                    b.create<movedatabyio>(location, gatherC, hwioC);

                    b.create<routing::YieldOp>(location);
                });
        }

        builder.create<scf::YieldOp>(location);
    };
}

void routingmanager::loaddialect(MLIRContext* ctx) {
    ctx->getOrLoadDialect<mlir::func::FuncDialect>();
    ctx->getOrLoadDialect<routing::routingdialect>();
    ctx->getOrLoadDialect<mlir::scf::SCFDialect>();
    ctx->getOrLoadDialect<mlir::arith::ArithDialect>();
}
mlir::func::FuncOp routingmanager::createroutingfunc(MLIRContext* ctx, int totalN, bool purefunc) {
        OpBuilder builder(ctx);
        auto location = builder.getUnknownLoc();
        //ModuleOp module= ModuleOp::create(builder.getUnknownLoc());
        IndexType itype= builder.getIndexType();
        mlir::FunctionType ftype = builder.getFunctionType({itype,itype},{});
        func::FuncOp func = builder.create<func::FuncOp>(builder.getUnknownLoc(), "createroute", ftype);
        Block* eb = func.addEntryBlock();
        builder.setInsertionPointToStart(eb);

        auto getconstant = [&](Value value) -> int {
            if (auto definingOp = value.getDefiningOp()) {
                if (auto constantOp = dyn_cast<arith::ConstantIntOp>(definingOp)) {
                    if (auto intAttr = constantOp.getValue().dyn_cast<mlir::IntegerAttr>()) {
                        return intAttr.getInt();
                    }
                }
            }
            return 0;
        };

        Value row = eb->getArgument(0);
        Value col = eb->getArgument(1);
        //Value total_col = eb->getArgument(2);

        Value lb = builder.create<arith::ConstantIndexOp>(location, 0);
        Value ub = builder.create<arith::ConstantIndexOp>(location, totalN);
        Value step = builder.create<arith::ConstantIndexOp>(location,1);
  // ──────────────────────────────
  // 2. 创建 scf.forall
  //    OpBuilder 会回调一个 lambda，让你往 region 里塞指令
  // ──────────────────────────────
        auto scf = builder.create<mlir::scf::ForOp>(location, lb, ub, step);
        {
            //use such format to fix the generic format print issue 
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(scf.getBody());
            Value cnum_i32, rnum_i32;
            //TileRangeArrayAttr tlist = TileRangeArrayAttr::get(ctx, slist);
            if (purefunc) {
                col   = builder.create<arith::ConstantIndexOp>(location, 1);
                row   = builder.create<arith::ConstantIndexOp>(location, 8);
            }
            cnum_i32 = builder.create<arith::IndexCastOp>(location,builder.getI32Type(), col);
            rnum_i32 = builder.create<arith::IndexCastOp>(location, builder.getI32Type(), row);

            /*
            //create mesh
            auto mesh = builder.create<createhwmesh>(builder.getUnknownLoc(),  rnum_i32, cnum_i32);

            auto patitionmesh = builder.create<partitionmesh>(builder.getUnknownLoc(),  mesh, cnum_i32, "row");
            //------------
            //fixme should use real subview
            Value subview = builder.create<arith::ConstantIntOp>(builder.getUnknownLoc(), 1, 32);
            SmallVector<Attribute> shape;
            for (int64_t v : {10, 20})
            shape.push_back(builder.getI64IntegerAttr(v));
            ArrayAttr vals = builder.getArrayAttr(shape);  // satisfies I64ArrayAttr
            // 3) I64Attr ($dim).
            IntegerAttr dimnum = builder.getI64IntegerAttr(2);
            auto tensor = builder.create<createscheduletensor>(builder.getUnknownLoc(),  subview, vals, dimnum);
            //
            auto hw_row_number = rnum_i32;
            IntegerAttr splitdim = builder.getI64IntegerAttr(0);//dim 0 is

            mlir::StringAttr hw_axis_owner=builder.getStringAttr("row");
            mlir::StringAttr replicate_on=builder.getStringAttr("col");
            mlir::StringAttr single_tile_owner=builder.getStringAttr("");
            auto rowtensor = builder.create<partitiontensor>(builder.getUnknownLoc(),
                    tensor, routing::PartitionAttr::get(ctx, hw_row_number, 0, "row", "col", ""),
            routing::TilingAttr{});
            //extract data
            auto edata = builder.create<extract_data>(builder.getUnknownLoc(), rowtensor, cnum_i32);
            auto emeshtile = builder.create<extract_tiles>(builder.getUnknownLoc(), patitionmesh, cnum_i32);
            //extract tile
            */
            //

            auto tilearray = builder.create<createtilearrayOp>(builder.getUnknownLoc(), rnum_i32, cnum_i32);
            auto io = builder.create<createdataio>(builder.getUnknownLoc(), "mem", "input");
            
          
            auto output = builder.getI32Type();
            auto op3 = builder.create<creatbroadcast>(builder.getUnknownLoc(), io.getResult(), tilearray.getResult());
            auto result = tilearray.getResult().getType().dyn_cast<tilearrayType>();

            llvm::outs() << "result count is "<< result.getItems().size() <<" \n";
           
        }
         
         //without return the print will go into generic as some verify failed.
         auto retop = builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());

       
        return func;
}

void routingmanager::createroutingfuncByDim(OpBuilder& builder, MLIRContext* ctx, bool binput, Value mesh, Value tensor,
                                           uint32_t hwsplitnum, std::string splitAxis) {
        auto location = builder.getUnknownLoc();
        auto tensorhwaxisowner = splitAxis;
        // no region creatation
        //
        auto exec = builder.create<scf::ExecuteRegionOp>(builder.getUnknownLoc(), /*result types*/TypeRange{});
        exec->setAttr("routing_memo", builder.getStringAttr(splitAxis));
        //Block *body = builder.createBlock(&exec.getRegion());
        {
                    OpBuilder::InsertionGuard guard(builder);
                    //fix the upper scope return go inside this region issue
                    builder.setInsertionPointToStart(&exec.getRegion().emplaceBlock());
                ///*               
                    auto patitionmesh = builder.create<partitionmesh>(builder.getUnknownLoc(),  mesh, hwsplitnum, splitAxis);
                    int splitdimn= splitAxis == "row" ? 0 : 1;
                    IntegerAttr splitdim = builder.getI64IntegerAttr(splitdimn);//dim 0 is 
                    if (splitdimn == 0) {
                        assert(splitAxis == "row" && "splitdim 0 must be row");
                    } else {
                        assert(splitAxis == "col" && "splitdim 1 must be col");
                    }
                    // Get tensor type from input tensor for partitiontensor result
                    auto tensorType = tensor.getType().cast<RankedTensorType>();
                    auto rowtensor = builder.create<partitiontensor>(
                        builder.getUnknownLoc(), tensorType, tensor,
                        routing::PartitionAttr::get(ctx, hwsplitnum, 0, tensorhwaxisowner, "col", ""),
                        routing::TilingAttr{});

                    // Calculate split tensor type
                    SmallVector<int64_t> splitShape(tensorType.getShape());
                    if (splitdimn== 0) {
                        if (splitShape[0] != ShapedType::kDynamic) splitShape[0] /= hwsplitnum;
                    } else {
                        if (splitShape[1] != ShapedType::kDynamic) splitShape[1] /= hwsplitnum;
                    }
                    auto splitTensorType = RankedTensorType::get(splitShape, tensorType.getElementType());

                    Value lb = builder.create<arith::ConstantIndexOp>(location, 0);
                    Value ub = builder.create<arith::ConstantIndexOp>(location, hwsplitnum);
                    Value step = builder.create<arith::ConstantIndexOp>(location,1);
                    // create RoutingCreate region op
                    // inside region
                    ///*
                    auto scf = builder.create<mlir::scf::ForOp>(location, lb, ub, step);
                    { 
                        OpBuilder::InsertionGuard guard(builder);
                        builder.setInsertionPointToStart(scf.getBody());
                        auto memo = builder.getStringAttr(splitAxis);
                        mlir::Value scf_idx = scf.getInductionVar();
                        
                        Value idx = builder.create<arith::IndexCastOp>(builder.getUnknownLoc(),builder.getI32Type(), scf_idx);
                        
                        auto routingcreateOp = builder.create<routing::RoutingCreate>(builder.getUnknownLoc(), idx, memo, [&](OpBuilder &builder1, Location bodyLoc,Value sidx) { 
                            //use such format to fix the generic format print issue 
                            ///*
                            // extract_data returns the split tensor type
                            auto slicetensor = builder1.create<extract_data>(builder1.getUnknownLoc(), splitTensorType, rowtensor, sidx);
                            auto tilelist = builder1.create<extract_tiles>(builder1.getUnknownLoc(), patitionmesh, sidx);
                            
                            if (binput) {
                                auto hwio = builder1.create<createhwiowithtarget>(builder1.getUnknownLoc(), tilelist, "input", "mem2");
                                auto datamov = builder1.create<movedatabyio>(builder1.getUnknownLoc(), slicetensor, hwio);
                            } else {
                                // routinggatherout now takes AnyTensor and returns AnyTensor
                                auto gatherdata = builder1.create<routinggatherout>(builder1.getUnknownLoc(), slicetensor.getType(), tilelist, slicetensor);
                                auto hwio = builder1.create<createhwiowithtarget>(builder1.getUnknownLoc(), tilelist, "output", "mem2");
                                auto datamov = builder1.create<movedatabyio>(builder1.getUnknownLoc(), gatherdata, hwio);
                            }
                            //*/
                            builder1.create<routing::YieldOp>(builder1.getUnknownLoc());
                            
                        });
                        
                        //extract tile
                    }
                    //*/

                    // use routing.yield to return value finish the yield
                 builder.create<scf::YieldOp>(builder.getUnknownLoc());
        };
       
        return ;//func;
}
// ---------------------------------------------------------------------------
// SplitModel::fromSpatialTag
// ---------------------------------------------------------------------------

TensorSplitDesc SplitModel::fromSpatialTag(const std::string &tag, bool isInput) {
    // splitDim always defaults to 0 (row-based tensor split).
    // The spatial tag controls hwAxisOwner and replicateOn (mesh partitioning),
    // NOT which tensor dimension to split.
    if (tag == "row_broadcast_in")
        return {0, "row", "col", "broadcast", "default", 2, 4096};
    if (tag == "col_broadcast_in")
        return {0, "col", "row", "broadcast", "default", 2, 4096};
    if (tag == "tiled_in")
        return {0, "row", "", "scatter", "default", 2, 4096};
    if (tag == "row_major_out")
        return {0, "row", "col", "gather", "ltor", 2, 4096};
    if (tag == "col_major_out")
        return {0, "col", "row", "gather", "rtol", 2, 4096};
    if (tag == "row_reduce_out")
        return {0, "row", "", "scatter", "default", 2, 4096};
    // default: backward compat
    return isInput ? TensorSplitDesc{0, "row", "col", "broadcast", "default", 2, 4096}
                   : TensorSplitDesc{0, "row", "col", "gather", "ltor", 2, 4096};
}

// ---------------------------------------------------------------------------
// SplitModel::fromPolicyFields — struct field-based lookup
// ---------------------------------------------------------------------------

TensorSplitDesc SplitModel::fromPolicyFields(int pattern, int distribution, int mergeOrder, int pingPong, bool isInput,
                                             int maxBufferBytes, int layoutTransform) {
    // Map enum values to strings
    static const char *patternStr[] = {"broadcast", "scatter", "multicast", "gather"};
    static const char *flowStr[] = {"default", "ltor", "rtol"};

    std::string pat = (pattern >= 0 && pattern <= 3) ? patternStr[pattern] : "broadcast";
    std::string flw = (mergeOrder >= 0 && mergeOrder <= 2) ? flowStr[mergeOrder] : "default";

    // Determine hwAxisOwner and replicateOn from distribution + pattern
    std::string hwAxis, replOn;
    if (distribution == 0) { // Row
        hwAxis = "row";
        replOn = (pattern == 1) ? "" : "col"; // Scatter has no replicateOn
    } else if (distribution == 1) {           // Col
        hwAxis = "col";
        replOn = (pattern == 1) ? "" : "row";
    } else { // Grid
        hwAxis = "row";
        replOn = "";
    }

    // Map layoutTransform enum to string
    std::string ltStr;
    if (layoutTransform == 1)
        ltStr = "dma_shuffle";
    else if (layoutTransform == 2)
        ltStr = "core_shuffle";

    return {0, hwAxis, replOn, pat, flw, pingPong, maxBufferBytes, ltStr};
}

// ---------------------------------------------------------------------------
// createroutingfuncBySplitModel — generalized routing driven by SplitModel
// Creates separate scf.execute_region per mesh axis. Tensors sharing
// the same hwAxisOwner (or assigned to an axis via replicateOn when
// hwAxisOwner is empty) are grouped into one execute_region.
// ---------------------------------------------------------------------------

void routingmanager::createroutingfuncBySplitModel(OpBuilder &builder, MLIRContext *ctx, Value mesh,
                                                   const std::vector<Value> &tensors,
                                                   const std::vector<bool> &isInputFlags, int meshRows, int meshCols,
                                                   const SplitModel &splitModel) {

    auto location = builder.getUnknownLoc();

    // ── Step 1: Group tensors by effective axis ──
    // axis -> [(tensorIndex, isInput)]
    // Tensors with empty hwAxisOwner but non-empty replicateOn are
    // broadcast-only; assign them to the replicateOn axis group.
    struct TensorEntry {
        unsigned index;
        bool isInput;
    };
    std::map<std::string, std::vector<TensorEntry>> axisGroups;

    for (unsigned i = 0; i < tensors.size(); ++i) {
        const auto &split = splitModel.tensorSplits[i];
        std::string effectiveAxis;
        if (!split.hwAxisOwner.empty()) {
            effectiveAxis = split.hwAxisOwner;
        } else if (!split.replicateOn.empty()) {
            // Broadcast-only tensor: assign to the replicateOn axis group
            effectiveAxis = split.replicateOn;
        } else {
            // Fallback: default to "row"
            effectiveAxis = "row";
        }
        axisGroups[effectiveAxis].push_back({i, isInputFlags[i]});
    }

    // ── Step 2: For each axis group, create a separate execute_region ──
    for (auto &[axis, entries] : axisGroups) {
        uint32_t axisSplitNum = (axis == "row") ? meshRows : meshCols;

        auto exec = builder.create<scf::ExecuteRegionOp>(location, TypeRange{});
        exec->setAttr("routing_memo", builder.getStringAttr(axis));
        {
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(&exec.getRegion().emplaceBlock());

            // partitionmesh for this axis
            auto partmesh = builder.create<partitionmesh>(location, mesh, axisSplitNum, axis);

            // Pre-compute partitiontensor and split types for tensors in this group
            struct TensorInfo {
                Value partTensor;
                RankedTensorType splitTensorType;
                uint32_t splitnum;
                bool isInput;
            };
            std::vector<TensorInfo> tensorInfos;

            for (auto &entry : entries) {
                const auto &split = splitModel.tensorSplits[entry.index];
                auto tensorType = tensors[entry.index].getType().cast<RankedTensorType>();

                // Determine splitnum within this axis group
                uint32_t splitnum;
                if (!split.hwAxisOwner.empty()) {
                    // This tensor is owned by this axis — split by axisSplitNum
                    splitnum = axisSplitNum;
                } else {
                    // Broadcast-only tensor (empty hwAxisOwner): splitnum=1
                    // Full tensor is replicated to each group
                    splitnum = 1;
                }

                auto partTensor = builder.create<partitiontensor>(
                    location, tensorType, tensors[entry.index],
                    routing::PartitionAttr::get(ctx, splitnum, split.splitDim, split.hwAxisOwner, split.replicateOn,
                                                ""),
                    routing::TilingAttr{});

                // Spatial-halo: when this tensor uses overlapping halo distribution,
                // each tile-row owns `haloSlice` rows along splitDim (instead of an even
                // dimSize/splitnum slice) and consecutive tile-rows advance by `haloStep`.
                // Build a grouped #routing.tiling<...> descriptor (one #routing.dim per
                // tensor dim) and attach it on the partitiontensor op so
                // DmaphopTodfscheblueprintPass can compute the overlapping per-tile DDR
                // offset. (Round-trips via the op's custom printer/parser.)
                bool isHalo = (split.haloMode == 1 && splitnum > 1 && split.haloSlice > 0);
                if (isHalo) {
                    // Two-level (nested) halo: each L1 tile slice (haloSlice rows) is
                    // further chunked on-core into haloL2Rounds temporal rounds of
                    // haloL2Slice rows advancing by haloL2Step → the row dim's nested
                    // (inner) #routing.level. Opt-in: only emitted when haloL2Rounds > 1.
                    int32_t l2Slice = 0, l2Step = 0, l2Rounds = 1;
                    if (split.haloL2Rounds > 1 && split.haloL2Slice > 0) {
                        l2Slice = split.haloL2Slice;
                        l2Step = split.haloL2Step;
                        l2Rounds = split.haloL2Rounds;
                    }
                    // K-contraction accumulate split (independent of the H/row L2):
                    // the K dim is chunked into kRounds on-core accumulate rounds of
                    // kSlice elements advancing by kStep → the other dim's outer
                    // #routing.level. Opt-in: only populated when kAccumRounds > 1.
                    int32_t kSlice = 0, kStep = 0, kRounds = 1;
                    if (split.kAccumRounds > 1 && split.kAccumSlice > 0) {
                        kSlice = split.kAccumSlice;
                        kStep = split.kAccumStep;
                        kRounds = split.kAccumRounds;
                    }

                    auto *ctx2 = builder.getContext();
                    auto shape = tensorType.getShape();
                    int sd = (split.splitDim == 0) ? 0 : 1; // HW-split (row) tensor dim
                    int otherDim = 1 - sd;                  // K-accum dim

                    // Row (HW-split) dim: outer L1 level + optional nested L2 slice_tiling.
                    routing::LevelAttr rowSliceTiling;
                    if (l2Rounds > 1) {
                        rowSliceTiling = routing::LevelAttr::get(
                            ctx2, /*base=*/split.haloSlice, /*total=*/(int64_t)l2Slice * l2Rounds,
                            /*slice=*/l2Slice, /*step=*/l2Step, /*rounds=*/l2Rounds,
                            /*slice_tiling=*/routing::LevelAttr{});
                    }
                    auto rowOuter = routing::LevelAttr::get(
                        ctx2, /*base=*/shape[sd], /*total=*/(int64_t)split.haloSlice * splitnum,
                        /*slice=*/split.haloSlice, /*step=*/split.haloStep, /*rounds=*/(int64_t)splitnum,
                        /*slice_tiling=*/rowSliceTiling);
                    auto rowDim = routing::DimAttr::get(ctx2, rowOuter);

                    // K-accum dim: single outer level (no slice_tiling). When there is no
                    // K-accum split (kRounds<=1) this degenerates to a full-dim single slice.
                    int64_t kSl = (kRounds > 1) ? kSlice : shape[otherDim];
                    int64_t kSt = (kRounds > 1) ? kStep : shape[otherDim];
                    int64_t kRn = (kRounds > 1) ? kRounds : 1;
                    auto colOuter = routing::LevelAttr::get(ctx2, /*base=*/shape[otherDim], /*total=*/kSl * kRn,
                                                            /*slice=*/kSl, /*step=*/kSt, /*rounds=*/kRn,
                                                            /*slice_tiling=*/routing::LevelAttr{});
                    auto colDim = routing::DimAttr::get(ctx2, colOuter);

                    // Order DimAttrs by tensor dimension index (d0, d1, ...).
                    llvm::SmallVector<routing::DimAttr> dims(2);
                    dims[sd] = rowDim;
                    dims[otherDim] = colDim;
                    partTensor.setTilingAttr(routing::TilingAttr::get(ctx2, dims));
                }

                // Calculate split tensor shape
                SmallVector<int64_t> splitShape(tensorType.getShape());
                if (splitnum > 1) {
                    int sd = (split.splitDim == 0) ? 0 : 1;
                    if (splitShape[sd] != ShapedType::kDynamic) {
                        if (isHalo)
                            splitShape[sd] = split.haloSlice; // overlapping slice (e.g. 61)
                        else
                            splitShape[sd] /= splitnum; // even split
                    }
                }
                auto splitTensorType =
                    (splitnum > 1) ? RankedTensorType::get(splitShape, tensorType.getElementType()) : tensorType;

                tensorInfos.push_back({partTensor, splitTensorType, splitnum, entry.isInput});
            }

            // scf.for loop over axisSplitNum
            Value lb = builder.create<arith::ConstantIndexOp>(location, 0);
            Value ub = builder.create<arith::ConstantIndexOp>(location, axisSplitNum);
            Value step = builder.create<arith::ConstantIndexOp>(location, 1);

            auto scf = builder.create<mlir::scf::ForOp>(location, lb, ub, step);
            {
                OpBuilder::InsertionGuard guard(builder);
                builder.setInsertionPointToStart(scf.getBody());
                auto memo = builder.getStringAttr(axis);
                mlir::Value scf_idx = scf.getInductionVar();
                Value idx = builder.create<arith::IndexCastOp>(location, builder.getI32Type(), scf_idx);

                auto routingcreateOp = builder.create<routing::RoutingCreate>(
                    location, idx, memo, [&](OpBuilder &b, Location bodyLoc, Value sidx) {
                        auto tilelist = b.create<extract_tiles>(location, partmesh, sidx);

                        for (auto &info : tensorInfos) {
                            auto sliceTensor =
                                b.create<extract_data>(location, info.splitTensorType, info.partTensor, sidx);

                            if (info.isInput) {
                                auto hwio = b.create<createhwiowithtarget>(location, tilelist, "input", "mem2");
                                b.create<movedatabyio>(location, sliceTensor, hwio);
                            } else {
                                auto gatherData =
                                    b.create<routinggatherout>(location, sliceTensor.getType(), tilelist, sliceTensor);
                                auto hwio = b.create<createhwiowithtarget>(location, tilelist, "output", "mem2");
                                b.create<movedatabyio>(location, gatherData, hwio);
                            }
                        }

                        b.create<routing::YieldOp>(location);
                    });
            }

            builder.create<scf::YieldOp>(location);
        }
    }
}