###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################

# Type stubs used by @aie_triton.jit kernels (parsed via AST, not executed)


class constexpr:
    pass


class _DType:
    def __init__(self, name, bits):
        self.name, self.bits = name, bits


int8 = _DType("int8", 8)
int16 = _DType("int16", 16)
int32 = _DType("int32", 32)
float32 = _DType("float32", 32)


def program_id(axis):
    return 0


def zeros(shape, dtype):
    return None


def load(ptr):
    return None


def store(ptr, val):
    pass


def dot(a, b):
    return None


def arange(start, end):
    return None


def make_block_ptr(**kw):
    return None


def advance(ptr, off):
    return None
