/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

//Provide a definition of the 'aie' dialect in the ODS framework so that we
// can define our operations.
#ifndef AIELINK_OPS
#define AIELINK_OPS

include "aielinktypes.td"
class AieLink_Op<string mnemonic, list<Trait> traits = []> :
    Op<AieLinkDialect, mnemonic, traits>;

def AieLink_AllocMemBankOp: AieLink_Op<"allocmembank"> {
	let summary = "alloc mem bank";
	let arguments = (ins
    I32Attr:$locationdirect, // 0 south, 1 west, 2 north, 3 east
		StrAttr:$locationtext,
    I32Attr:$baseaddr,
    I32Attr:$memtotalsize,
    I32Attr:$islocal
	);
	let results = (outs Aie_L1MemBankType:$result);
	let description = [{
    "This is a operation to alloc mem bank op."
  }];
  let hasCustomAssemblyFormat = 1;
  /*
  let builders = [OpBuilder<( ins "Type":$result_type, 
																	"IntegerAttr":$locationdirect, 
																	"StringAttr":$locationtext, 
																	"IntegerAttr":$baseaddr, 
																	"IntegerAttr":$memtotalsize, 
																	"IntegerAttr":$islocal), [{
      build($_builder, $_state, result_type, locationdirect,locationtext,baseaddr,memtotalsize,islocal);// args);
    }]
  >];
  */
  let assemblyFormat = [{$locationdirect `,` 
											  $locationtext `,` 
											  $baseaddr `,` 
							          $memtotalsize `,` 
									      $islocal `,` attr-dict 
                        `:` functional-type(operands,results)}];
}

def AieLink_AllocMemBuffer: AieLink_Op<"allocmembuffer"> {
	let summary = "alloc mem buffer";
	let arguments = (ins
    Aie_L1MemBankType:$banktype, // 0 south, 1 west, 2 north, 3 east
    I32Attr:$memaddr,
    I32Attr:$memsize
	);
	let results = (outs Aie_MemBufferType:$result);
	let description = [{
    "This is a operation to alloc mem bank op."
  }];
  /*
  let builders = [OpBuilder<( ins "Type":$result_type, 
																	"Aie_L1MemBankType":$banktype, 
																	"StringAttr":$locationtext, 
																	"IntegerAttr":$baseaddr, 
																	"IntegerAttr":$memtotalsize, 
																	"IntegerAttr":$islocal), [{
      build($_builder, $_state, result_type, locationdirect,locationtext,baseaddr,memtotalsize,islocal);// args);
    }]
  >];
  */
  let assemblyFormat = [{$banktype `,` 
											  $memaddr `,` 
											  $memsize `,` attr-dict 
                        `:` functional-type(operands,results)}];
}

#endif
