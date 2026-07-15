/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef __ROUTING_CONSTRUCT__
#define __ROUTING_CONSTRUCT__
typedef MATRIX_ID_TYPE uint16_t;
typedef struct _ROWCOL{
    MATRIX_ID_TYPE row;
    MATRIX_ID_TYPE col;
} ROWCOL;
class routingconstruct {
public:
    routingconstruct();
private:
    std::vector<std::vector<MATRIX_ID_TYPE>> matrix;
    std::unordered_map<ROWCOL, std::shared_ptr<routingtile>>;
};
#endif //__ROUTING_CONSTRUCT__