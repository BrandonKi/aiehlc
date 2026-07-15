/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

//Provide a definition of the 'aie' dialect in the ODS framework so that we
// can define our operations.
#ifndef AIE_OPS
#define AIE_OPS

include "aietypes.td"
// Provide a definition of the 'aie' dialect in the ODS framework so that we
// can define our operations.
// Base class for aie dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Aie_Op<string mnemonic, list<Trait> traits = []> :
    Op<Aie_ADialect, mnemonic, traits>;

def Aie_LoadKernelOp: Aie_Op<"loadkernel"> {
  let summary = "aie op";
  let arguments = (ins I32:$lhs, I32:$rhs);
  let results = (outs);
  let description = [{
    "This is my custom operation."
  }];
  let assemblyFormat = "$lhs `,` $rhs `,` attr-dict";
}
def ConstantOp : Aie_Op<"constant", [Pure]> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = aie.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                        : tensor<2x3xf64>
    ```
  }];
  // The constant operation takes an attribute as the only input.
  let arguments = (ins F64ElementsAttr:$value);
  // The constant operation returns a single value of TensorType.
  let results = (outs F64Tensor);
  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 0;
  // Add custom build methods for the constant operation. These method populates
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<ConstantOp>(...)`.
  let builders = [
    // Build a constant with a given constant tensor value.
    OpBuilder<(ins "DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,
    // Build a constant with a given constant floating-point value.
    OpBuilder<(ins "double":$value)>
  ];
  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 0;
}


def ConstantInteger32Op: Aie_Op<"constant_integer32_op", [Pure]> {
  let summary = "Custom operation with int32 attribute and i32 output";
  let description = [{
    An operation that takes an int32 attribute input and produces an
    i32 value as a result.
  }];
  let arguments = (ins I32Attr:$inputAttr);
  let results = (outs I32:$outputValue);
  let builders = [ OpBuilder<(ins "IntegerAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>
  ];
  let assemblyFormat = [{ $inputAttr `:` type($outputValue) attr-dict }];
}

def ConstantInteger64Op: Aie_Op<"constant_integer64_op", [Pure]> {
  let summary = "Custom operation with int32 attribute and i32 output";
  let description = [{
    An operation that takes an int32 attribute input and produces an
    i32 value as a result.
  }];
  let arguments = (ins I64Attr:$inputAttr);
  let results = (outs I64:$outputValue);
  let builders = [ OpBuilder<(ins "IntegerAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>
  ];
  let assemblyFormat = [{ $inputAttr `:` type($outputValue) attr-dict}];
}

/*
NOTE: The CreateWindowOp does not need custom builder as the default builder
from the tablegen matches the default order.
Check build methods in aieop.h.inc
*/
def Aie_CreateWindowOp: Aie_Op<"create_window", [Pure]> {
  let summary = "Compute a custom operation that produces an Aie_WindowType";
  let description = [{
    This operation computes the specified Aie_WindowType 
  }];
  let arguments = (ins I32Attr:$direction, I32Attr:$size, I64Attr:$pingaddr, 
    I64Attr:$pongaddr, I32Attr:$pinglockid, I32Attr:$ponglockid, StrAttr:$name);
  let results = (outs Aie_WindowType:$result);
  let assemblyFormat =  [{ $direction `,` $size `,` $pingaddr `,` $pongaddr `,` $pinglockid `,`
    $ponglockid `,` $name attr-dict `:` type(results) }];
}

def Aie_CreateKernelFuncNameOp : Aie_Op<"create_kernelfuncname_op", [Pure]> {
  let summary = "Create kernel name object";
  let description = [{
    create the kernel name op
  }];
  let arguments = (ins
    StrAttr:$name
  );
  let results = (outs Aie_KernelFuncNameType:$result);
  //let builders = [OpBuilder<(ins "Type":$result_type, "StringAttr":$name), [{
  // build($_builder, $state, result_type, name);
  //}] >];

  let assemblyFormat = [{
    `{` $name `}` `(` `)` attr-dict `:` type(results)
  }];
}

def Aie_CreateKernelFileNameOp : Aie_Op<"create_kernelfilename_op", [Pure]> {
  let summary = "Create kernel file name object";
  let description = [{
    create the kernel file name op
  }];
  let arguments = (ins
    StrAttr:$name
  );
  let results = (outs Aie_KernelFileNameType:$result);
  //let builders = [OpBuilder<(ins "Type":$result_type, "StringAttr":$name), [{
  //	build($_builder, $state, result_type, name);
  //}] >];

  let assemblyFormat = [{
    `{` $name `}` `(` `)` attr-dict `:` type(results)
  }];
}


/*
NOTE: ArrayRef<Value> is not supported in default build methods from tablegen, so creating
custom build method with ValueRange.
*/
def Aie_CreateKernelObjectOp : Aie_Op<"create_kernelobject_op", [Pure]> {
  let summary = "Creates a kernel object";
  let description = [{
    This operation creates a kernel object with the number of input and output arguments,
    and a variadic list of windows for the arguments.
  }];
  let arguments = (ins
    I32Attr:$numInputArgs,
    I32Attr:$numOutputArgs,
    Aie_KernelFuncNameType:$kname,
    Aie_KernelFileNameType:$fname,
    StrAttr:$in_param_type,
    StrAttr:$out_param_type,
    Variadic<AnyType>:$arguments
  );
  let results = (outs Aie_KernelObjectType:$result);
  let builders = [OpBuilder<( ins "Type":$result_type, "IntegerAttr":$numInputArgs, "IntegerAttr":$numOutputArgs,
                  "KernelFuncNameType":$kname, "KernelFileNameType":$fname, "std::string":$in_param_type, "std::string":$out_param_type, "ValueRange":$arguments), [{
      ValueRange args = ValueRange(arguments);
      build($_builder, $_state, result_type, numInputArgs, numOutputArgs,kname, fname, in_param_type, out_param_type, arguments);// args);
    }]
  >];
  //when use AnyType, we still need to inferred what the type of the Anytype, in this case we use functiona-type, the operands represend
  // all arguments, and functional-type(intput, output) means the input parameter is same with the output type, then the mlir can know
  // the anytype for kname is kernelfuncname and ethe fname is kernenlFilenametype etc
  let assemblyFormat = [{ 
    `{` $numInputArgs `,` $numOutputArgs  `}` attr-dict `(` $kname `,` $fname `,` $in_param_type `,` $out_param_type `,` $arguments `)` `:` functional-type(operands, results)

  }];
}

/*
NOTE: The CreateTileKernelObjectOp does not need custom builder as the default builder
from the tablegen matches the default order.
Check build methods in aieop.h.inc
*/
def Aie_CreateTileKernelObjectOp : Aie_Op<"create_tilekernelobject_op", [Pure]> {
  let summary = "Creates a Tile kernel object";
  let description = [{
    This operation creates a tile kernel object with row, column and kernel source.
  }];
  let arguments = (ins 
    I32: $row,
    I32: $col,
    StrAttr: $kernels
  );
  let results = (outs Aie_TileKernelObjectType:$result);
  
  let assemblyFormat = [{
    $row `,` $col `,` $kernels attr-dict `:` type(results)
  }];
}

/*
NOTE: ArrayRef<Value> is not supported in default build methods from tablegen, so creating
custom build method with ValueRange.
*/
def Aie_KernelDefinition : Aie_Op<"kernel_definition_op", [Pure]> {
  let summary = "Defines a kernel with a variable number of windows (window type).";
  let arguments = (ins Variadic<AnyType>:$arguments);
  let results = (outs);
  let builders = [OpBuilder<( ins "Type":$result_type, "ArrayRef<Value>":$arguments), [{
      ValueRange args = ValueRange(arguments);
      build($_builder, $_state, result_type, args);
    }]
  >];
  let assemblyFormat = [{ 
    `(` $arguments `)` attr-dict `:` functional-type($arguments, results)
  }];
}

// Change AnyType to the Correct Type.
def Aie_KernelCall : Aie_Op<"kernel_call_op", [Pure]> {
  let summary = "Represents a call to an AIE kernel with location and arguments";
  let arguments = (ins 
    I32:$col, 
    I32:$row, 
    Variadic<AnyType>:$arguments);
  let results = (outs);
  let builders = [OpBuilder<( ins "Type":$result_type, "Value":$col, "Value":$row, 
    "ArrayRef<Value>":$arguments), [{
      ValueRange args = ValueRange(arguments);
      build($_builder, $_state, result_type, col, row, args);
    }]
  >];
  let assemblyFormat = [{ 
    $col `,` $row `(` $arguments `)` attr-dict `:` functional-type($arguments, results)
  }];
}

def Aie_AllocOp : Aie_Op<"alloc_op", [Pure]> {
  let summary = "Allocates a buffer on the Host and returns a memref to the allocated memory.";
  let description = [{
    This operation allocates a buffer of the specified size in AIE Host Memory. 
    The result is a memref pointing to the allocated memory. The type of the 
    memref (including its shape and element type) must be specified when 
    creating this operation.
  }];
  let arguments = (ins I64:$size);
  let results = (outs AnyMemRef:$result);

  let assemblyFormat = "`(` $size `)` attr-dict `:` type($result)";
}

// Change AnyType to a correct Type for implicit constraint
def Aie_LoadKernel : Aie_Op<"load_kernel_op", [Pure]> {
  let summary = "Represents a call to an AIE kernel with location and arguments";
  let arguments = (ins
    StrAttr:$name,
    I32Attr:$col, 
    I32Attr:$row,
    Variadic<AnyType>:$arguments);
  let results = (outs);
  let assemblyFormat = [{ 
    $name `,` $col `,` $row `(` $arguments `)` attr-dict `:` functional-type($arguments, results)
  }];
}


/*
COPIED FROM ARUN - NEED TO VERFIY THE PURPOSE AND CHANGES
*/
def Aie_MoveData : Aie_Op<"move_data_op", [Pure]> {
  let summary = "Moves Data from one tile to another.";
  let arguments = (ins 
    Aie_RoutingInstanceType:$routingInstance, 
    Aie_LocType:$source,
    AnyType:$sourceObject,
    I32:$dataSize, 
    AnyType:$destinationObject, 
    Aie_LocType:$destination
  );
  let results = (outs);
}

def Aie_Route : Aie_Op<"route_op", [Pure]> {
  let summary = "Finds route between two AIE tiles."; 
  let arguments = (ins 
    Aie_RoutingInstanceType:$routingInstance,
    Aie_RoutingConstraintType:$routeConstraints,
    Aie_LocType:$source,
    Aie_LocType:$destination);
  let results = (outs);
}

def Aie_DeRoute : Aie_Op<"de_route_op", [Pure]> {
  let summary = "Removes the Route between two tiles";
  let arguments = (ins 
    Aie_RoutingInstanceType:$routingInstance,
    Aie_LocType:$source,
    Aie_LocType:$destination,
    I1:$shouldModifyCoreConfig);
  let results = (outs);
}

def Aie_RoutesReveal : Aie_Op<"routes_reveal", [Pure]> {
  let summary = "...";
  let arguments = (ins 
    Aie_RoutingInstanceType:$routingInstance, 
    Aie_LocType:$source, 
    Aie_LocType:$destination);
  let results = (outs);
}

def Aie_RouteDmaWait : Aie_Op<"route_dmawait_op", [Pure]> {
  let summary = "...";
  let arguments = (ins 
    Aie_RoutingInstanceType:$routingInstance, 
    Aie_LocType:$tile, 
    I1:$isS2MM);
  let results = (outs);
}

def Aie_CoreWait : Aie_Op<"core_wait_op", [Pure]> {
  let summary = "...";
  let arguments = (ins 
    Aie_RoutingInstanceType:$routingInstance, 
    Aie_LocType:$tile);
  let results = (outs);
}

def Aie_Run : Aie_Op<"run_op", [Pure]> {
  let summary = "...";
  let arguments = (ins 
    Aie_RoutingInstanceType:$routingInstance, 
    Aie_LocType:$tile);
  let results = (outs);
}

def Aie_SetCoreExecute  : Aie_Op<"set_core_execute_op", [Pure]> {
  let summary = "...";
  let arguments = (ins 
    Aie_RoutingInstanceType:$routingInstance, 
    Aie_LocType:$tile, 
    I1:$isExecute);
  let results = (outs);
}

def Aie_ConfigHostEdgeConstraints : Aie_Op<"config_host_edge_constraints_op", [Pure]> {
  let summary = "...";
  let arguments = (ins 
    Aie_RoutingInstanceType:$routingInstance,
    Aie_HostAieConstraintType:$userResCon);
  let results = (outs);
}

def Aie_ResetHostEdgeConstraints : Aie_Op<"reset_host_edge_constraints_op", [Pure]> {
  let summary = "...";
  let arguments = (ins 
    Aie_RoutingInstanceType:$routingInstance, 
    Aie_LocType:$tile);
  let results = (outs);
}

def Aie_InitRoutingHandler : Aie_Op<"init_routing_handler", [Pure]> {
  let summary = "...";
  let arguments = (ins Aie_DeviceInstanceType:$devInst);
  let results = (outs Aie_RoutingInstanceType:$instance);
}

def Aie_RoutingInstanceFree : Aie_Op<"routing_instance_free_op", [Pure]> {
  let summary = "...";
  let arguments = (ins Aie_RoutingInstanceType:$routingInstance);
  let results = (outs);
}


#endif
