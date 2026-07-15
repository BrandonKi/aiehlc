/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

#ifndef AIE_ATTRS
#define AIE_ATTRS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "mlir/IR/OpBase.td" // This line is crucial
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/IR/AttrTypeBase.td"

def routingdialect : Dialect {
	let name = "routing";
	let cppNamespace = "routing";
	dag discardableAttrs = (ins);
	let hasOperationAttrVerify = 0;
	let useDefaultTypePrinterParser = 1;
	let useDefaultAttributePrinterParser = 1;
	let usePropertiesForAttributes = 0;
}

class routingop<string mnemonic, list<Trait> traits = []> : Op<routingdialect, mnemonic, traits>;
//class routingop<string name, list<Trait>  traits=[]> : Op<routingdialect, name, traits> {}
class routingtype<string name, list<Trait> traits=[]> : TypeDef<routingdialect, name, traits> {}
class routingattr<string name> : AttrDef<routingdialect, name> {}

def TileRangeAttr : routingattr<"TileRange"> {
  let mnemonic = "tilerangeattr";
  let cppNamespace = "routing";
  let parameters = (ins
    "int64_t":$rowNum,
    "int64_t":$colNum
  );

  let assemblyFormat = "`{` $rowNum `,` $colNum `}`";
}

def TileRangeArrayAttr : routingattr<"TileRangeArray"> {
  let mnemonic = "TileRangeArrayAttr";
  let summary = "a tile array";
  let cppNamespace = "routing";
  let parameters = (ins
    OptionalArrayRefParameter<"TileRangeAttr">:$items
  );
  let assemblyFormat = "`{` $items `}`";
}

def HaloAttr : routingattr<"Halo"> {
  let mnemonic = "halo";
  let summary = "spatial overlapping (halo) tile distribution descriptor";
  let cppNamespace = "routing";
  let parameters = (ins
    "int32_t":$sliceSize,   // L1 rows each tile owns (overlapping slice)
    "int32_t":$step,        // L1 stride between consecutive tile slices
    "int32_t":$overlap,     // sliceSize - step (shared halo rows)
    // L2 (nested) halo; defaults => no nested level (legacy single-level path)
    DefaultValuedParameter<"int32_t", "0">:$l2Slice,
    DefaultValuedParameter<"int32_t", "0">:$l2Step,
    DefaultValuedParameter<"int32_t", "1">:$l2Rounds,
    // K-contraction accumulate split (independent of the H/row L2 halo above);
    // defaults => no K-accum split (struct-format omits default values on print)
    DefaultValuedParameter<"int32_t", "0">:$kSlice,
    DefaultValuedParameter<"int32_t", "0">:$kStep,
    DefaultValuedParameter<"int32_t", "1">:$kRounds
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

// Recursive per-dimension tiling level. Each level describes one tiling
// granularity over a tensor dimension; `slice_tiling` (optional) nests the next
// (finer/on-core) level. A null `slice_tiling` marks a leaf level. Custom
// assembly format prints the nested level with its full `#routing.level<...>`
// mnemonic (the auto struct(params) form would strip the prefix).
def LevelAttr : routingattr<"Level"> {
  let mnemonic = "level";
  let summary = "recursive per-dimension tiling level descriptor";
  let cppNamespace = "routing";
  let parameters = (ins
    "int64_t":$base,    // full extent this level partitions
    "int64_t":$total,   // slice * rounds (covered extent, incl. overlap)
    "int64_t":$slice,   // elements each step owns
    "int64_t":$step,    // stride between consecutive steps
    "int64_t":$rounds,  // number of steps at this level
    OptionalParameter<"routing::LevelAttr">:$slice_tiling // nested finer level (null => leaf)
  );
  let hasCustomAssemblyFormat = 1;
}

// One tensor dimension's tiling: a single outer level (which may nest a finer
// slice_tiling level). Custom assembly so `outer` prints the full
// `#routing.level<...>` mnemonic rather than the stripped struct form.
def DimAttr : routingattr<"Dim"> {
  let mnemonic = "dim";
  let summary = "per-dimension tiling descriptor";
  let cppNamespace = "routing";
  let parameters = (ins
    "routing::LevelAttr":$outer
  );
  let hasCustomAssemblyFormat = 1;
}

// Whole-tensor tiling: one DimAttr per tensor dimension, printed as d0/d1/...
def TilingAttr : routingattr<"Tiling"> {
  let mnemonic = "tiling";
  let summary = "per-tensor tiling descriptor (one DimAttr per dim)";
  let cppNamespace = "routing";
  let parameters = (ins
    OptionalArrayRefParameter<"routing::DimAttr">:$dims
  );
  let hasCustomAssemblyFormat = 1;
}

// Mesh-partition descriptor (the former scalar partitiontensor args).
def PartitionAttr : routingattr<"Partition"> {
  let mnemonic = "partition";
  let summary = "tensor mesh-partition descriptor";
  let cppNamespace = "routing";
  let parameters = (ins
    "int64_t":$splitnum,
    "int64_t":$splitdim,
    StringRefParameter<>:$hwAxisOwner,
    StringRefParameter<>:$replicateOn,
    StringRefParameter<>:$singleTileOwner
  );
  let assemblyFormat = "`<` struct(params) `>`";
}
#endif
