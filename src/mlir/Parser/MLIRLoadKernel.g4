/******************************************************************************
* Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

// MLIRLoadKernel.g4

grammar MLIRLoadKernel;

start: loadKernelExpression;

loadKernelExpression: 'loadkernel' '(' ID ',' matrix ')';

matrix: '{{' row (',' row)* '}}';
row: '{' INT ',' INT '}';

ID: [a-zA-Z_][a-zA-Z0-9_]*;
INT: [0-9]+;

WS: [ \t\r\n]+ -> skip;

