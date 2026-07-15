###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################

from . import language
from ._compiler import compile_and_run


class _Mesh:
    def __init__(self, rows, cols):
        self.rows, self.cols = rows, cols


def mesh(rows, cols):
    return _Mesh(rows, cols)


def set_device(device_id):
    pass


def synchronize():
    pass


def jit(fn):
    """Decorator: marks a function as an AIE kernel.
    Actual compilation happens at launch time (kernel[grid](...) call).
    """
    return _JitKernel(fn)


class _JitKernel:
    def __init__(self, fn):
        self._fn = fn
        self._name = fn.__name__

    def __getitem__(self, grid):
        """kernel[grid] returns a launcher."""
        return _KernelLauncher(self._fn, self._name, grid)


class _KernelLauncher:
    def __init__(self, fn, name, grid):
        self._fn, self._name, self._grid = fn, name, grid

    def __call__(self, *args, **kwargs):
        """kernel[grid](A, B, C, M, N, K, BLOCK_M=8, ...) triggers compilation."""
        compile_and_run(self._fn, self._name, self._grid, args, kwargs)
