###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################

import ast
import inspect
import textwrap
import numpy as np


def compile_and_run(fn, name, grid, args, kwargs):
    """Main entry: parse kernel AST, extract params, call C++ pipeline."""
    mesh_rows, mesh_cols = grid

    # --- Phase 1: Extract tensor specs from runtime args ---
    source = textwrap.dedent(inspect.getsource(fn))
    tree = ast.parse(source)
    kernel_def = _find_kernel_def(tree, name)

    # Classify parameters by AST analysis
    param_names = [p.arg for p in kernel_def.args.args]
    tensor_params, scalar_params, constexpr_params = _classify_params(
        kernel_def, param_names
    )

    # Build TensorParam list from runtime numpy arrays
    tensor_specs = []
    num_input_windows = 0
    num_output_windows = 0
    element_type = "int8"  # default
    for i, pname in enumerate(tensor_params):
        arr = args[i]  # numpy array
        shape = list(arr.shape)
        bits = _numpy_dtype_to_bits(arr.dtype)
        is_input = _is_input_tensor(kernel_def, pname)
        tensor_specs.append((shape, bits, is_input))
        if is_input:
            num_input_windows += 1
        else:
            num_output_windows += 1
        # Derive element type from first tensor's dtype
        if i == 0:
            element_type = _bits_to_element_type(bits)

    # --- Phase 2: Extract kernel body as C code string ---
    kernel_body = _extract_kernel_body(
        kernel_def, param_names, tensor_params, constexpr_params, kwargs,
        element_type, num_input_windows, num_output_windows, name
    )

    # --- Phase 3: Call C++ pipeline via pybind11 ---
    from . import _aietriton_core

    output_dir = "./worklocal"
    success = _aietriton_core.run_aie_pipeline(
        mesh_rows, mesh_cols, tensor_specs, output_dir, kernel_body, name
    )
    if not success:
        raise RuntimeError(f"TilingLinalgPipeline failed for kernel '{name}'")


def _find_kernel_def(tree, name):
    """Find FunctionDef node matching kernel name."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise ValueError(f"Kernel function '{name}' not found")


def _classify_params(func_def, param_names):
    """Classify function params into tensor/scalar/constexpr."""
    tensors, scalars, constexprs = [], [], []
    for p in func_def.args.args:
        if p.annotation and _is_constexpr_annotation(p.annotation):
            constexprs.append(p.arg)
        elif p.arg.endswith("_ptr") or p.arg in (
            "A",
            "B",
            "C",
            "a_ptr",
            "b_ptr",
            "c_ptr",
        ):
            tensors.append(p.arg)
        else:
            scalars.append(p.arg)
    return tensors, scalars, constexprs


def _is_constexpr_annotation(node):
    """Check if annotation is tl.constexpr."""
    if isinstance(node, ast.Attribute):
        return node.attr == "constexpr"
    return False


def _is_input_tensor(func_def, param_name):
    """AST walk: if param appears as target of tl.store -> output, else input."""
    for node in ast.walk(func_def):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "store":
                # tl.store(c_ptr + ..., result) -- first arg references the output ptr
                if node.args and _references_name(node.args[0], param_name):
                    return False
    return True


def _references_name(node, name):
    """Check if AST node references a given variable name."""
    if isinstance(node, ast.Name) and node.id == name:
        return True
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id == name:
            return True
    return False


def _numpy_dtype_to_bits(dtype):
    """Convert numpy dtype to bit width."""
    return dtype.itemsize * 8


def _bits_to_element_type(bits):
    """Convert bit width to element type string."""
    if bits == 8:
        return "int8"
    elif bits == 16:
        return "int16"
    elif bits == 32:
        return "int32"
    return "int8"


def _extract_kernel_body(func_def, param_names, tensor_params, constexpr_params,
                         kwargs, element_type, num_input_windows,
                         num_output_windows, kernel_name):
    """Translate Python kernel AST -> C via KernelOps -> pybind11 -> MLIR EmitC."""
    from .aie_pass.ast_to_kernelops import ast_to_kernel_ops

    # Phase 1: Python AST -> KernelOp list
    kernel_ops = ast_to_kernel_ops(
        func_def, tensor_params, constexpr_params, kwargs
    )

    if not kernel_ops:
        return ""  # fallback to auto-gen if AST produced no ops

    # Phase 2: KernelOps -> pybind11 -> MLIR EmitC -> C string
    from . import _aietriton_core

    c_code = _aietriton_core.build_kernel_body(
        kernel_name, element_type,
        num_input_windows, num_output_windows,
        kernel_ops
    )

    return c_code
