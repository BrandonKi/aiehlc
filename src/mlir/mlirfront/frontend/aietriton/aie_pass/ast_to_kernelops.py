###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################

"""
Python AST -> KernelOp list translation.

Walks the AST of an @aie_triton.jit function body and produces a flat list
of typed dicts (KernelOps) that cross the pybind11 boundary to the C++
KernelBodyEmitter.

KernelOp schema:
    {"op": "get_coreid"}
    {"op": "for_range", "trip_count": N}
    {"op": "acquire_input", "window_idx": N, "var": "inN"}
    {"op": "acquire_output", "window_idx": N, "var": "outN"}
    {"op": "gemm_body", "m": M, "n": N, "k": K}
    {"op": "release_input", "window_idx": N}
    {"op": "release_output", "window_idx": N}
    {"op": "end_for"}
"""

import ast


def ast_to_kernel_ops(func_def, tensor_params, constexpr_params, kwargs):
    """Walk @aie_triton.jit function AST -> list of KernelOp dicts.

    Args:
        func_def: ast.FunctionDef node of the kernel function
        tensor_params: list of tensor parameter names (e.g. ["a_ptr", "b_ptr", "c_ptr"])
        constexpr_params: list of constexpr parameter names (e.g. ["BLOCK_M", "BLOCK_N", "BLOCK_K"])
        kwargs: dict of keyword arguments passed at call site (constexpr values)

    Returns:
        list of KernelOp dicts
    """
    walker = _ASTWalker(func_def, tensor_params, constexpr_params, kwargs)
    return walker.walk()


class _ASTWalker:
    """Walks an @aie_triton.jit function AST and produces KernelOp list."""

    def __init__(self, func_def, tensor_params, constexpr_params, kwargs):
        self.func_def = func_def
        self.tensor_params = tensor_params
        self.constexpr_params = constexpr_params
        self.kwargs = kwargs

        # Build tensor -> window_idx maps.
        # Input tensors: those not targeted by tl.store
        # Output tensors: those targeted by tl.store
        self.input_tensors = []
        self.output_tensors = []
        for pname in tensor_params:
            if self._is_store_target(pname):
                self.output_tensors.append(pname)
            else:
                self.input_tensors.append(pname)

        # Map tensor name -> (direction, window_idx)
        self.tensor_window_map = {}
        for i, name in enumerate(self.input_tensors):
            self.tensor_window_map[name] = ("input", i)
        for i, name in enumerate(self.output_tensors):
            self.tensor_window_map[name] = ("output", i)

        # Resolve constexpr values from kwargs and defaults
        self.constexpr_values = {}
        for p in func_def.args.args:
            if p.arg in constexpr_params:
                if p.arg in kwargs:
                    self.constexpr_values[p.arg] = kwargs[p.arg]
                elif p.default is not None:
                    # Try to get default from the function def
                    pass  # handled below

        # Resolve defaults for constexpr params
        defaults = func_def.args.defaults
        args = func_def.args.args
        num_defaults = len(defaults)
        num_args = len(args)
        for i, default in enumerate(defaults):
            arg_idx = num_args - num_defaults + i
            arg_name = args[arg_idx].arg
            if arg_name in constexpr_params and arg_name not in self.constexpr_values:
                if isinstance(default, ast.Constant):
                    self.constexpr_values[arg_name] = default.value

        # Track accumulator variable names
        self.accumulator_vars = set()

        # Dedup: only emit one get_coreid (C code extracts both row and col)
        self._coreid_emitted = False

        # Track which loads/stores have been emitted in current for body
        self.ops = []

    def walk(self):
        """Walk the function body and return KernelOp list."""
        self.ops = []
        self._walk_stmts(self.func_def.body)
        return self.ops

    def _walk_stmts(self, stmts):
        """Walk a list of statements."""
        for stmt in stmts:
            self._visit_stmt(stmt)

    def _visit_stmt(self, stmt):
        """Dispatch to handler based on statement type."""
        if isinstance(stmt, ast.Assign):
            self._visit_assign(stmt)
        elif isinstance(stmt, ast.AugAssign):
            self._visit_aug_assign(stmt)
        elif isinstance(stmt, ast.For):
            self._visit_for(stmt)
        elif isinstance(stmt, ast.Expr):
            self._visit_expr(stmt)

    def _visit_assign(self, node):
        """Handle assignment: x = expr."""
        value = node.value

        # tl.program_id(...) -> get_coreid (emit once for all axes)
        if self._is_tl_call(value, "program_id"):
            if not self._coreid_emitted:
                self.ops.append({"op": "get_coreid"})
                self._coreid_emitted = True
            return

        # tl.zeros(...) -> track accumulator, no op emitted
        if self._is_tl_call(value, "zeros"):
            if isinstance(node.targets[0], ast.Name):
                self.accumulator_vars.add(node.targets[0].id)
            return

        # tl.load(...) -> acquire_input
        if self._is_tl_call(value, "load"):
            self._emit_load(value)
            return

        # tl.dot(a, b) -> gemm_body
        if self._is_tl_call(value, "dot"):
            self._emit_gemm_body()
            return

        # acc.to(tl.int8) -> no separate op (saturate folded into gemm_body)
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
            if value.func.attr == "to":
                return

        # tl.make_block_ptr(...), tl.advance(...) -> tracked but no op
        if self._is_tl_call(value, "make_block_ptr"):
            return
        if self._is_tl_call(value, "advance"):
            return

        # Simple arithmetic (row_start = tile_row * BLOCK_M) -> no op
        # c_offsets = (...) -> no op

    def _visit_aug_assign(self, node):
        """Handle augmented assignment: acc += tl.dot(a, b)."""
        if self._is_tl_call(node.value, "dot"):
            self._emit_gemm_body()

    def _visit_for(self, node):
        """Handle for loop: for k in range(0, K, BLOCK_K)."""
        trip_count = self._compute_trip_count(node)
        self.ops.append({"op": "for_range", "trip_count": trip_count})

        # Track which inputs we acquire inside the loop
        acquired_inputs = []
        acquired_outputs = []

        # Walk loop body
        for stmt in node.body:
            before_len = len(self.ops)
            self._visit_stmt(stmt)
            # Track what was acquired
            for op in self.ops[before_len:]:
                if op["op"] == "acquire_input":
                    acquired_inputs.append(op["window_idx"])
                elif op["op"] == "acquire_output":
                    acquired_outputs.append(op["window_idx"])

        # Emit releases for everything acquired inside the loop
        for idx in acquired_inputs:
            self.ops.append({"op": "release_input", "window_idx": idx})
        for idx in acquired_outputs:
            self.ops.append({"op": "release_output", "window_idx": idx})

        self.ops.append({"op": "end_for"})

    def _visit_expr(self, node):
        """Handle expression statement: tl.store(ptr, val)."""
        if isinstance(node.value, ast.Call):
            call = node.value
            if self._is_tl_call(call, "store"):
                self._emit_store(call)

    def _emit_load(self, call_node):
        """Emit acquire_input from tl.load(ptr + ...)."""
        if not call_node.args:
            return

        ptr_arg = call_node.args[0]
        ptr_name = self._get_ptr_base(ptr_arg)

        if ptr_name and ptr_name in self.tensor_window_map:
            direction, idx = self.tensor_window_map[ptr_name]
            if direction == "input":
                self.ops.append({
                    "op": "acquire_input",
                    "window_idx": idx,
                    "var": f"in{idx}",
                })

    def _emit_store(self, call_node):
        """Emit acquire_output + release_output from tl.store(ptr, val)."""
        if not call_node.args:
            return

        ptr_arg = call_node.args[0]
        ptr_name = self._get_ptr_base(ptr_arg)

        if ptr_name and ptr_name in self.tensor_window_map:
            direction, idx = self.tensor_window_map[ptr_name]
            if direction == "output":
                self.ops.append({
                    "op": "acquire_output",
                    "window_idx": idx,
                    "var": f"out{idx}",
                })
                self.ops.append({
                    "op": "release_output",
                    "window_idx": idx,
                })

    def _emit_gemm_body(self):
        """Emit gemm_body op with dimensions from constexpr params."""
        m = self._resolve_constexpr("BLOCK_M", 8)
        n = self._resolve_constexpr("BLOCK_N", 8)
        k = self._resolve_constexpr("BLOCK_K", 8)
        self.ops.append({"op": "gemm_body", "m": m, "n": n, "k": k})

    def _compute_trip_count(self, for_node):
        """Compute trip count from for k in range(start, stop, step)."""
        if not isinstance(for_node.iter, ast.Call):
            return 2  # default

        call = for_node.iter
        func = call.func
        if not (isinstance(func, ast.Name) and func.id == "range"):
            return 2

        args = call.args
        if len(args) == 3:
            start = self._eval_expr(args[0])
            stop = self._eval_expr(args[1])
            step = self._eval_expr(args[2])
            if start is not None and stop is not None and step is not None and step != 0:
                return (stop - start) // step
        elif len(args) == 2:
            start = self._eval_expr(args[0])
            stop = self._eval_expr(args[1])
            if start is not None and stop is not None:
                return stop - start
        elif len(args) == 1:
            stop = self._eval_expr(args[0])
            if stop is not None:
                return stop

        return 2  # fallback

    def _eval_expr(self, node):
        """Evaluate a simple constant or constexpr reference."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in self.constexpr_values:
                return self.constexpr_values[node.id]
            # Try to resolve from kwargs
            if node.id in self.kwargs:
                return self.kwargs[node.id]
        return None

    def _get_ptr_base(self, node):
        """Walk BinOp tree to find the ast.Name that is a tensor parameter.

        E.g., a_ptr + row_offsets * K + k_offsets -> "a_ptr"
        """
        if isinstance(node, ast.Name):
            if node.id in self.tensor_window_map:
                return node.id
        if isinstance(node, ast.BinOp):
            left = self._get_ptr_base(node.left)
            if left:
                return left
            return self._get_ptr_base(node.right)
        # For tl.load(block_ptr) where block_ptr was assigned from tl.make_block_ptr(base=a_ptr, ...)
        # We don't track block_ptr aliases here - the simple case handles direct ptr references
        return None

    def _is_tl_call(self, node, func_name):
        """Check if node is a call to tl.<func_name>(...)."""
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        # tl.func_name(...)
        if isinstance(func, ast.Attribute) and func.attr == func_name:
            if isinstance(func.value, ast.Name) and func.value.id == "tl":
                return True
        return False

    def _is_store_target(self, param_name):
        """Check if param appears as first arg of tl.store(param, ...)."""
        for node in ast.walk(self.func_def):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "store":
                    if isinstance(func.value, ast.Name) and func.value.id == "tl":
                        if node.args and self._references_name(node.args[0], param_name):
                            return True
        return False

    def _references_name(self, node, name):
        """Check if AST node references a given variable name."""
        if isinstance(node, ast.Name) and node.id == name:
            return True
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id == name:
                return True
        return False

    def _resolve_constexpr(self, name, default):
        """Resolve a constexpr value, falling back to default."""
        if name in self.constexpr_values:
            return self.constexpr_values[name]
        if name in self.kwargs:
            return self.kwargs[name]
        return default
