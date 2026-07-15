/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

#ifndef AIE_TYPES
#define AIE_TYPES
include "routingattr.td"
include "routinginterface.td"
def tilearray : routingtype<"tilearray", [DeclareTypeInterfaceMethods<TileInterface>]> {
	let mnemonic = "tilearray";
	let summary = "this a tile array";
	let parameters = (
		ins
		OptionalArrayRefParameter<"TileRangeAttr">:$items
	);

	let assemblyFormat = [{ $items }];
		
}
def dataio : routingtype<"dataio"> {
	let mnemonic = "dataio";
	let summary = "the io used to receive or output data";
	let parameters = (ins
		"StringAttr": $iotype,
		"StringAttr": $direction
	);
	let assemblyFormat = [{ $iotype `,` $direction}];
}
#endif
