/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

include "routinghwattr.td"

def TileArrayHandleType :RoutingHWType<"tilearrayhandle"> {
    let mnemonic = "tilearrayhandle";
    let parameters= (
        ins
        "tilearrayhandleAttr": $handle
    );
    let assemblyFormat = [{$handle}];
}
