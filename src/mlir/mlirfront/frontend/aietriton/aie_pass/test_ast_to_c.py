#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""
Unit test for the AST -> KernelOps -> C pipeline.

Usage:
    # AST -> KernelOps only (no C++ .so needed):
    cd src/mlir/mlirfront && python -m aietriton.aie_pass.test_ast_to_c

    # Full pipeline including C generation (requires built _aietriton_core.so):
    cd src/mlir/mlirfront && python -m aietriton.aie_pass.test_ast_to_c --full
"""

import ast
import sys
import textwrap

# ---------------------------------------------------------------------------
# Inline test kernel (mirrors triton_matmul.py::matmul_simple)
# ---------------------------------------------------------------------------
MATMUL_SIMPLE_SOURCE = textwrap.dedent("""\
    def matmul_simple(
        a_ptr,
        b_ptr,
        c_ptr,
        M, N, K,
        BLOCK_M: tl.constexpr = 8,
        BLOCK_N: tl.constexpr = 16,
        BLOCK_K: tl.constexpr = 16,
    ):
        tile_row = tl.program_id(axis=0)
        tile_col = tl.program_id(axis=1)

        row_start = tile_row * BLOCK_M
        col_start = tile_col * BLOCK_N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

        for k_start in range(0, K, BLOCK_K):
            a_block = tl.load(
                a_ptr + (row_start + tl.arange(0, BLOCK_M)[:, None]) * K
                      + (k_start  + tl.arange(0, BLOCK_K)[None, :])
            )
            b_block = tl.load(
                b_ptr + (k_start  + tl.arange(0, BLOCK_K)[:, None]) * N
                      + (col_start + tl.arange(0, BLOCK_N)[None, :])
            )
            acc += tl.dot(a_block, b_block)

        result = acc.to(tl.int8)
        tl.store(c_ptr + (row_start + tl.arange(0, BLOCK_M)[:, None]) * N
                       + (col_start + tl.arange(0, BLOCK_N)[None, :]), result)
""")


def _classify_params_test(func_def):
    """Simplified param classification for testing."""
    tensors, scalars, constexprs = [], [], []
    for p in func_def.args.args:
        if p.annotation and isinstance(p.annotation, ast.Attribute):
            if p.annotation.attr == "constexpr":
                constexprs.append(p.arg)
                continue
        if p.arg.endswith("_ptr") or p.arg in ("A", "B", "C", "a_ptr", "b_ptr", "c_ptr"):
            tensors.append(p.arg)
        else:
            scalars.append(p.arg)
    return tensors, scalars, constexprs


def test_ast_to_kernel_ops():
    """Test Phase 1: Python AST -> KernelOp list."""
    from aietriton.aie_pass.ast_to_kernelops import ast_to_kernel_ops

    tree = ast.parse(MATMUL_SIMPLE_SOURCE)
    func_def = tree.body[0]
    assert isinstance(func_def, ast.FunctionDef)

    tensors, scalars, constexprs = _classify_params_test(func_def)

    kwargs = {"BLOCK_M": 8, "BLOCK_N": 16, "BLOCK_K": 16, "M": 16, "N": 16, "K": 16}
    kernel_ops = ast_to_kernel_ops(func_def, tensors, constexprs, kwargs)

    print("=== KernelOps ===")
    for op in kernel_ops:
        print(f"  {op}")

    # Verify expected ops
    op_types = [op["op"] for op in kernel_ops]

    assert "get_coreid" in op_types, f"Missing get_coreid, got: {op_types}"
    assert "for_range" in op_types, f"Missing for_range, got: {op_types}"
    assert "acquire_input" in op_types, f"Missing acquire_input, got: {op_types}"
    assert "gemm_body" in op_types, f"Missing gemm_body, got: {op_types}"
    assert "acquire_output" in op_types, f"Missing acquire_output, got: {op_types}"
    assert "release_output" in op_types, f"Missing release_output, got: {op_types}"
    assert "end_for" in op_types, f"Missing end_for, got: {op_types}"

    # Verify for_range trip count
    for_op = next(op for op in kernel_ops if op["op"] == "for_range")
    # K=16, BLOCK_K=16 -> trip_count = 1
    assert for_op["trip_count"] == 1, f"Expected trip_count=1, got {for_op['trip_count']}"

    # Verify gemm_body dimensions
    gemm_op = next(op for op in kernel_ops if op["op"] == "gemm_body")
    assert gemm_op["m"] == 8, f"Expected m=8, got {gemm_op['m']}"
    assert gemm_op["n"] == 16, f"Expected n=16, got {gemm_op['n']}"
    assert gemm_op["k"] == 16, f"Expected k=16, got {gemm_op['k']}"

    # Verify window indices
    input_ops = [op for op in kernel_ops if op["op"] == "acquire_input"]
    assert len(input_ops) == 2, f"Expected 2 acquire_input ops, got {len(input_ops)}"
    assert input_ops[0]["window_idx"] == 0
    assert input_ops[1]["window_idx"] == 1

    output_ops = [op for op in kernel_ops if op["op"] == "acquire_output"]
    assert len(output_ops) == 1, f"Expected 1 acquire_output op, got {len(output_ops)}"
    assert output_ops[0]["window_idx"] == 0

    print("\n=== AST -> KernelOps: PASS ===\n")
    return kernel_ops


def test_full_pipeline(kernel_ops):
    """Test Phase 2: KernelOps -> pybind11 -> MLIR EmitC -> C string."""
    try:
        from aietriton import _aietriton_core
    except ImportError:
        print("SKIP: _aietriton_core.so not built (run 'make' in build/ first)")
        return

    c_code = _aietriton_core.build_kernel_body(
        "matmul_simple",  # kernel name
        "int8",           # element type
        2,                # num input windows
        1,                # num output windows
        kernel_ops
    )

    print("=== Generated C code ===")
    print(c_code)

    # Verify key patterns in generated C
    assert "void matmul_simple(" in c_code, "Missing function signature"
    assert "input_window_int8 *window_in_0" in c_code, "Missing input window 0"
    assert "input_window_int8 *window_in_1" in c_code, "Missing input window 1"
    assert "output_window_int8 *window_out_0" in c_code, "Missing output window 0"
    assert "acquire_input_window(window_in_0)" in c_code, "Missing acquire_input_window 0"
    assert "acquire_input_window(window_in_1)" in c_code, "Missing acquire_input_window 1"
    assert "acquire_output_window(window_out_0)" in c_code, "Missing acquire_output_window 0"
    assert "for (int i = 0; i < 8; i++)" in c_code, "Missing GEMM i-loop"
    assert "for (int j = 0; j < 16; j++)" in c_code, "Missing GEMM j-loop"
    assert "for (int kk = 0; kk < 16; kk++)" in c_code, "Missing GEMM k-loop"
    assert "(int8_t)sum" in c_code, "Missing saturation cast"
    assert "release_input_window" in c_code, "Missing release_input_window"
    assert "release_output_window" in c_code, "Missing release_output_window"
    assert "get_coreid()" in c_code, "Missing get_coreid"

    print("\n=== Full pipeline: PASS ===\n")


def main():
    full = "--full" in sys.argv

    print("--- Test: AST -> KernelOps ---")
    kernel_ops = test_ast_to_kernel_ops()

    if full:
        print("--- Test: Full Pipeline (KernelOps -> C) ---")
        test_full_pipeline(kernel_ops)

    print("All tests passed.")


if __name__ == "__main__":
    main()
