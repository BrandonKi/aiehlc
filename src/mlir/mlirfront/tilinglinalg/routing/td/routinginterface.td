/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

include "mlir/IR/OpBase.td"

def TileInterface: TypeInterface<"TileInterface"> {
	let cppNamespace = "routing";
	let description = "base addr compute";
	let methods = [
		InterfaceMethod<"get base addrss", "int64_t", "getTileBase">
	];
}
