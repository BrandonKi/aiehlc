/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

include "testtype.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
def Test_createtilearrayOp : testop<"testcreatetilearray", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
	let summary = "createtilearray";
	let cppNamespace = "test";
	let description = "create a tile array";
	let arguments = ( ins
		I32:$rownum,
		I32:$colnum
	);
	let results = (outs
		tilearray:$output
	);
	let assemblyFormat = "$rownum $colnum `:` type($rownum) type($colnum) attr-dict `->` type($output)";
}

def Test_createdataio : testop<"testcreatedataio", [DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
	let summary = "createdataio";
	let cppNamespace = "test";
	let description = "create data io";

	let arguments = (ins
		StrAttr:$iotype,
		StrAttr:$direction
	);
	let results = (outs
		dataio:$output
	);
	let assemblyFormat = " $iotype `,` $direction attr-dict `->` type($output)";
}

def Test_creatbroadcast : testop<"testcreatebroadcast"> {
	let summary = "createbroadcast";
	let cppNamespace = "test";
	let description = "create a broadcast";

	let arguments = (ins
		dataio:$io,
		tilearray:$tilearray);

	let results = (outs
		I32:$output);
	let assemblyFormat = "$io `,` $tilearray `:` type($io) `,` type($tilearray)  attr-dict`->` type($output)";
}


