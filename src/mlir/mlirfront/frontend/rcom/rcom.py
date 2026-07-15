#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""rcom -- ROCm/HIP matrix-multiply front end for AIEHLC.

Reads a canonical ROCm/HIP int8 GEMM source, recognizes it as a GEMM, and
emits a CUDA-style ``.cc`` + ``.h`` (the same dialect as
``example/tileprogram/ccode/simplematmul2.cc``). The generated ``.cc`` is then
fed to the unchanged ``script/aiehlc.sh`` pipeline to produce host + kernel
ELFs.

    matmul_hip.cpp  --(rcom.py)-->  <name>.cc + <name>.h  --(aiehlc.sh)-->  aout/main.elf

Design notes:
  * Kernel body is *recognized + templated*, not translated line-for-line. rcom
    detects the HIP kernel is a GEMM and emits the proven AIE matmul template
    (cache-A / stream-B) from simplematmul2.cc.
  * int8 only in v1. Default dims M=N=K=256 on a 4x4 mesh (the HW-validated
    config). M/N/K and mesh are overridable; deviations emit a warning because
    only the proven config is HW-validated.
"""

import argparse
import os
import re
import subprocess
import sys

# The one HW-validated configuration. Deviating from it emits a warning.
PROVEN_M = 256
PROVEN_N = 256
PROVEN_K = 256
PROVEN_ROWS = 4
PROVEN_COLS = 4


class ParseError(Exception):
    """Raised when the HIP source is not a recognizable int8 GEMM."""


# ---------------------------------------------------------------------------
# HIP parser
# ---------------------------------------------------------------------------

def _strip_comments(src):
    """Remove // and /* */ comments so regex scans don't trip on them."""
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.DOTALL)
    src = re.sub(r"//[^\n]*", " ", src)
    return src


def _extract_body(src, sig_end):
    """Return the balanced-brace body starting at the first '{' after sig_end."""
    start = src.find("{", sig_end)
    if start < 0:
        raise ParseError("kernel has no body")
    depth = 0
    for i in range(start, len(src)):
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return src[start + 1:i]
    raise ParseError("unbalanced braces in kernel body")


def _classify_dtype(param_type):
    """Map a C pointer param element type to an rcom element type or None."""
    t = param_type.replace("const", "").replace("*", "").strip()
    t = re.sub(r"\s+", " ", t)
    if t in ("int8_t", "signed char", "char"):
        return "int8"
    return None


def parse_hip(src):
    """Parse a canonical HIP GEMM. Returns a dict with kernel metadata.

    Keys: name, dtype, inputs (2 ptr names), output (ptr name), M, N, K.
    Raises ParseError with a clear message if not a recognizable int8 GEMM.
    """
    clean = _strip_comments(src)

    # 1. Find the __global__ kernel signature.
    m = re.search(r"__global__\s+void\s+(\w+)\s*\(([^)]*)\)", clean)
    if not m:
        raise ParseError("no '__global__ void NAME(...)' kernel found")
    name = m.group(1)
    param_str = m.group(2)
    body = _extract_body(clean, m.end())

    # 2. Split params; identify pointer params and their element types.
    params = [p.strip() for p in param_str.split(",") if p.strip()]
    ptr_names = []
    ptr_types = {}
    for p in params:
        if "*" not in p:
            continue
        # last identifier token is the name
        pname_m = re.search(r"(\w+)\s*$", p)
        if not pname_m:
            continue
        pname = pname_m.group(1)
        ptype = p[:pname_m.start()]
        ptr_names.append(pname)
        ptr_types[pname] = ptype
    if len(ptr_names) < 3:
        raise ParseError(
            "expected >= 3 pointer params (A, B, C); got %d" % len(ptr_names))

    # 3. dtype: every pointer param must be int8 in v1.
    dtype = None
    for pname in ptr_names:
        dt = _classify_dtype(ptr_types[pname])
        if dt is None:
            raise ParseError(
                "unsupported element type for '%s' (v1 supports int8 only): '%s'"
                % (pname, ptr_types[pname].strip()))
        dtype = dt

    # 4. Direction: the pointer written via NAME[...] = ... is the output.
    assigned = set()
    for am in re.finditer(r"(\w+)\s*\[[^\]]*\]\s*=(?!=)", body):
        assigned.add(am.group(1))
    outputs = [p for p in ptr_names if p in assigned]
    if len(outputs) == 1:
        output = outputs[0]
    else:
        # Fallback: 3rd pointer param is the output.
        output = ptr_names[2]
    inputs = [p for p in ptr_names if p != output]
    if len(inputs) < 2:
        raise ParseError("could not identify two input pointers")
    inputs = inputs[:2]

    # 5. GEMM validation: require a MAC (sum += A[..] * B[..]) or C = A*B.
    has_output_assign = bool(re.search(re.escape(output) + r"\s*\[", body))
    has_mac = bool(re.search(r"\+=\s*[^;]*\*", body)) or bool(
        re.search(r"\w+\s*\[[^\]]*\]\s*\*\s*\w+\s*\[", body))
    has_k_loop = bool(re.search(r"for\s*\([^)]*\bk\b", body))
    if not (has_output_assign and has_mac and has_k_loop):
        raise ParseError(
            "kernel '%s' is not a recognized GEMM "
            "(need output[...] = ..., a MAC 'sum += a*b', and a K loop)" % name)

    # 6. Dims M, N, K from #define or const int (CLI overrides applied later).
    dims = {}
    for key in ("M", "N", "K"):
        dm = re.search(r"#define\s+" + key + r"\s+(\d+)", clean)
        if not dm:
            dm = re.search(r"const\s+int\s+" + key + r"\s*=\s*(\d+)", clean)
        if dm:
            dims[key] = int(dm.group(1))

    return {
        "name": name,
        "dtype": dtype,
        "inputs": inputs,
        "output": output,
        "M": dims.get("M"),
        "N": dims.get("N"),
        "K": dims.get("K"),
    }


# ---------------------------------------------------------------------------
# Tiling heuristics
# ---------------------------------------------------------------------------

def compute_tiling(M, N, K):
    """Return (m_tile, n_tile, k_chunk) for the given dims.

    Reproduces the proven simplematmul2 config exactly for M=N=K=256
    (m_tile=n_tile=16, k_chunk=64). For other dims uses documented heuristics:
    tile_size ~= 16 (falls back to the full dim when not divisible), and a
    K-chunk chosen so K % chunk == 0.
    """
    m_tile = 16 if M % 16 == 0 else M
    n_tile = 16 if N % 16 == 0 else N
    if K % 4 == 0:
        k_chunk = K // 4
    else:
        k_chunk = K
        for c in range(min(64, K), 0, -1):
            if K % c == 0:
                k_chunk = c
                break
    return m_tile, n_tile, k_chunk


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

def _subst(template, mapping):
    for k, v in mapping.items():
        template = template.replace("@@%s@@" % k, str(v))
    return template


_H_TEMPLATE = r"""/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * GENERATED by rcom (src/tool/frontend/rcom.py) from a ROCm/HIP GEMM source.
 * Do not edit by hand; regenerate from the HIP input instead.
 *
 * GEMM: C[MxN] = A[MxK] * B^T[NxK], int8
 * Deployed on a HW_ROWS x HW_COLS AIE tile mesh.
 ******************************************************************************/
#include "xil_cache.h"
#include "xiltimer.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// GEMM dimensions (from HIP source / rcom CLI overrides)
#define M @@M@@
#define K @@K@@
#define N @@N@@

// HW mesh dimensions (number of AIE tile rows and columns)
#define HW_ROWS @@HW_ROWS@@
#define HW_COLS @@HW_COLS@@

static int verify_matmul(const int8_t *A, const int8_t *B, const int8_t *C);

// Pure scalar matmul: C_ref[M][N] = A[M][K] * B^T[N][K]
static void scalar_matmul(int8_t *C_ref, const int8_t *A, const int8_t *B) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int16_t sum = 0;
            for (int k = 0; k < K; k++)
                sum += (int16_t)A[i * K + k] * (int16_t)B[j * K + k];
            if (sum > 127)
                sum = 127;
            else if (sum < -128)
                sum = -128;
            C_ref[i * N + j] = (int8_t)sum;
        }
    }
}

// Verify AIE output C against CPU reference
static int verify_matmul(const int8_t *A, const int8_t *B, const int8_t *C) {
    const int dataprintsize = 64;
    int mismatches = 0;
    int8_t C_ref[M * N];
    printf("before scalar_matmul\n");
    XTime t_start, t_end;
    XTime_GetTime(&t_start);
    scalar_matmul(C_ref, A, B);
    XTime_GetTime(&t_end);
    double elapsed_ms = 1.0 * (t_end - t_start) / COUNTS_PER_SECOND * 1000.0;
    printf("after scalar_matmul\n");
    printf("scalar_matmul time: %.3f ms\n", elapsed_ms);
    for (int i = 0; i < M * N; i++) {
        if (C[i] != C_ref[i]) {
            printf("MISMATCH C[%d]: got %d, expected %d\n", i, C[i], C_ref[i]);
            mismatches++;
            if (mismatches > 128)
                break;
        }
    }
    printf("\nA [%dx%d]:\n", M, K);
    for (int i = 0; i < (M > dataprintsize ? dataprintsize : M); i++) {
        printf("  [");
        for (int j = 0; j < (K > dataprintsize ? dataprintsize : K); j++) {
            printf("%4d", A[i * K + j]);
            if (j < K - 1)
                printf(",");
        }
        printf("]\n");
    }
    printf("\nB [%dx%d]:\n", K, N);
    for (int i = 0; i < (K > dataprintsize ? dataprintsize : K); i++) {
        printf("  [");
        for (int j = 0; j < (N > dataprintsize ? dataprintsize : N); j++) {
            printf("%4d", B[i * N + j]);
            if (j < N - 1)
                printf(",");
        }
        printf("]\n");
    }
    printf("\nC [%dx%d]:\n", M, N);
    for (int i = 0; i < (M > dataprintsize ? dataprintsize : M); i++) {
        printf("  [");
        for (int j = 0; j < (N > dataprintsize ? dataprintsize : N); j++) {
            printf("%4d", C[i * N + j]);
            if (j < N - 1)
                printf(",");
        }
        printf("]\n");
    }
    printf("\nC_ref [%dx%d]:\n", M, N);
    for (int i = 0; i < (M > dataprintsize ? dataprintsize : M); i++) {
        printf("  [");
        for (int j = 0; j < (N > dataprintsize ? dataprintsize : N); j++) {
            printf("%4d", C_ref[i * N + j]);
            if (j < N - 1)
                printf(",");
        }
        printf("]\n");
    }
    int total_elements = M * N;
    if (mismatches == 0)
        printf("PASS: all %d elements match.\n", total_elements);
    else
        printf("FAIL: %d mismatches out of %d.\n", mismatches, total_elements);
    return mismatches;
}
"""


_CC_TEMPLATE = r"""/******************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * GENERATED by rcom (src/tool/frontend/rcom.py) from a ROCm/HIP GEMM source.
 * Do not edit by hand; regenerate from the HIP input instead.
 *
 * AIE Programming Model - Matrix Multiplication (Parameterized Kernel API).
 * The HIP thread-indexed GEMM body is recognized and replaced with the proven
 * AIE "cache-A / stream-B" matmul template.
 ******************************************************************************/
#include "@@HEADER@@"
#pragma aie_debug_level(2 | AIE_DEBUG_FLAG_DISABLE_PARTITIONTEARDOWN)
// Composition-based spatial spaces: a generic SpatialPolicy composed with a
// PER-PORT 2D iteration space. Each port describes its OWN matrix via d1/d2:
//   win_a A=[M,K] -> d1 = M-tile,  d2 = K-chunk
//   win_b B=[N,K] -> d1 = N-tile,  d2 = K-chunk
//   win_c C=[M,N] -> d1 = M-tile,  d2 = N-tile
constexpr aie::GemmSpace RowBA = {
    .policy = {.map = {.act = aie::Pattern::Broadcast, .layout = aie::Layout::Row},
               .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
               .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{4096}}},
    .d1 = {.fullsize = M, .tile_size = @@MTILE@@, .stride = @@MTILE@@},  // A: M-tile
    .d2 = {.fullsize = K, .tile_size = @@KCHUNK@@, .stride = @@KCHUNK@@}}; // A: K chunk
constexpr aie::GemmSpace ColBB = {.policy = {.map = {.wgt = aie::Pattern::Broadcast, .layout = aie::Layout::Col},
                                             .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
                                             .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{4096}}},
                                  .d1 = {.fullsize = N, .tile_size = @@NTILE@@, .stride = @@NTILE@@},  // B: N-tile
                                  .d2 = {.fullsize = K, .tile_size = @@KCHUNK@@, .stride = @@KCHUNK@@}}; // B: K chunk
constexpr aie::GemmSpace LtoR_Merge = {
    .policy = {.map = {.layout = aie::Layout::Row, .merge_order = aie::Flow::LeftToRight},
               .mat = {.pad = aie::PadMaterialize::DDR, .im2col = aie::Im2col::None},
               .sched = {.pp_depth = 2, .l1_budget = aie::Bytes{4096}}},
    .d1 = {.fullsize = M, .tile_size = @@MTILE@@, .stride = @@MTILE@@},  // C: M-tile
    .d2 = {.fullsize = N, .tile_size = @@NTILE@@, .stride = @@NTILE@@}}; // C: N-tile
#define DEBUG_OUTPUT_ORDER 1
constexpr aie::GlobalPolicy @@POLICY@@ = {.fullconnect_auto = 1};
__global__(@@POLICY@@) void @@NAME@@(aie::port<input_window_int8 *, RowBA> win_a,
                                      aie::port<input_window_int8 *, ColBB> win_b,
                                      aie::port<output_window_int8 *, LtoR_Merge> win_c) {

    // Compiler-resolved tiling parameters
    const int tile_rows = aie::get_tile_rows();
    const int tile_cols = aie::get_tile_cols();
    const int eff_k = aie::get_effective_k();            // K chunk size per k-round
    const int k_rounds = aie::get_k_rounds();            // number of K-accumulation rounds
    const int num_a_rounds = aie::get_num_rounds(win_a); // DMA rounds per k-round for A
    const int num_b_rounds = aie::get_num_rounds(win_b); // DMA rounds per k-round for B
    const int num_c_rounds = aie::get_num_rounds(win_c);
    const int buf_sz_a = aie::get_buffer_size(win_a);
    const int buf_sz_b = aie::get_buffer_size(win_b);
    const int buf_sz_c = aie::get_buffer_size(win_c);

    // Spatial sub-tile iteration counts (per-port: A->M rounds, B->N rounds)
    const int m_rounds = aie::get_spatial_multiple_rounds(win_a);
    const int n_rounds = aie::get_spatial_multiple_rounds(win_b);

    // Derived per-round sizes (using effective_k, not full k_dim)
    const int rows_per_round = buf_sz_a / eff_k;
    const int cols_per_round = buf_sz_b / eff_k;

#if DEBUG_OUTPUT_ORDER
    unsigned coreid = get_coreid();
    int col = coreid >> 16;
    int row = coreid & 0x1F;
    int8_t tag = (int8_t)((row & 0x7) | ((col & 0x7) << 3));
    klog("DEBUG", 3);
#endif

    // Local buffers - accum/local_out hold one M-sub-tile strip (tile_rows x data_cols)
    int8_t all_A[tile_rows * eff_k];
    int16_t accum[tile_rows * tile_cols];
    int8_t local_out[tile_rows * tile_cols];

    // ===== M sub-tile loop: each mr gets fresh A data across all k_rounds =====
    for (int mr = 0; mr < m_rounds * n_rounds; mr++) {

        klog("MR  ", (int32_t)mr);
        // Zero accumulators for this M sub-tile
        for (int i = 0; i < tile_rows * tile_cols; i++)
            accum[i] = 0;

        // ===== K-round loop: accumulate partial products =====
        for (int kr = 0; kr < k_rounds; kr++) {
            klog("KRA ", (int32_t)kr);
            // --- Phase 1: Receive and cache A chunk for this (mr, kr) ---
            for (int ra = 0; ra < num_a_rounds; ra++) {
                int8_t *A_ptr = (int8_t *)acquire_input_window(win_a);
                for (int i = 0; i < buf_sz_a; i++) {
                    all_A[ra * buf_sz_a + i] = A_ptr[i];
                }
#if DEBUG_OUTPUT_ORDER
                for (int l = 0; l < (buf_sz_a < 8 ? buf_sz_a : 8); l++) {
                    klog("A   ", (int32_t)A_ptr[l]);
                }
#endif
                release_input_window(win_a);
            }

            for (int rb = 0; rb < num_b_rounds; rb++) {
                int8_t *B_ptr = (int8_t *)acquire_input_window(win_b);
                for (int i = 0; i < tile_rows; i++) {
                    for (int j = 0; j < cols_per_round; j++) {
                        int16_t sum = 0;
                        for (int k = 0; k < eff_k; k++) {
                            sum += (int16_t)all_A[i * eff_k + k] * (int16_t)B_ptr[j * eff_k + k];
                        }
                        accum[i * tile_cols + rb * cols_per_round + j] += sum;
                    }
                }

#if DEBUG_OUTPUT_ORDER
                klog("B0  ", (int32_t)B_ptr[0]);
#endif

                release_input_window(win_b);
            }
        } // end k_rounds

        // ===== Saturate accumulators to int8 for this M sub-tile =====
        for (int i = 0; i < tile_rows * tile_cols; i++) {
            int16_t val = accum[i];
            if (val > 127)
                val = 127;
            else if (val < -128)
                val = -128;
            local_out[i] = (int8_t)val;
        }
        for (int rc = 0; rc < num_c_rounds; rc++) {
            int8_t *out = (int8_t *)acquire_output_window(win_c);
            const int rows_per_c_round = buf_sz_c / tile_cols;
            for (int i = 0; i < rows_per_c_round; i++) {
                for (int j = 0; j < tile_cols; j++) {
                    out[i * tile_cols + j] = local_out[rc * buf_sz_c + i * tile_cols + j];
                }
            }
#if DEBUG_OUTPUT_ORDER
            klog("C0 ", (int32_t)out[0]);
#endif

            release_output_window(win_c);
        }
    } // end mr
}

// HOST
int main() {
    printf("=== Matrix Multiply with Data Caching on AIE %dx%d Mesh ===\n", HW_ROWS, HW_COLS);
    printf("    C[%dx%d] = A[%dx%d] * B^T[%dx%d], int8\n", M, N, M, K, K, N);
    aieSetDevice(0);
    aieArray device;
    aieMesh mesh = device.partition({0, @@ENDCOL@@, 0, 6}, HW_ROWS, HW_COLS);
    int8_t *A = (int8_t *)device.alloc(M * K * sizeof(int8_t) * 4);
    int8_t *B = (int8_t *)device.alloc(K * N * sizeof(int8_t) * 4);
    int8_t *C = (int8_t *)device.alloc(M * N * sizeof(int8_t) * 4);
    for (int i = 0; i < M * K; i++)
        A[i] = (int8_t)((i % 7) - 3);
    for (int i = 0; i < K * N; i++)
        B[i] = (int8_t)((i % 5) - 2);
    for (int i = 0; i < M * N; i++)
        C[i] = 0;
    @@NAME@@<<<mesh>>>(A, B, C, M, N, K);
    int result = verify_matmul(A, B, C);
    device.free(A);
    device.free(B);
    device.free(C);
    return result;
}
"""


def emit_h(cfg):
    return _subst(_H_TEMPLATE, {
        "M": cfg["M"], "K": cfg["K"], "N": cfg["N"],
        "HW_ROWS": cfg["rows"], "HW_COLS": cfg["cols"],
    })


def emit_cc(cfg):
    return _subst(_CC_TEMPLATE, {
        "HEADER": cfg["header"],
        "NAME": cfg["name"],
        "POLICY": cfg["name"] + "_policy",
        "MTILE": cfg["m_tile"],
        "NTILE": cfg["n_tile"],
        "KCHUNK": cfg["k_chunk"],
        "ENDCOL": cfg["cols"] - 1,
    })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_mesh(mesh_str):
    m = re.match(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$", mesh_str)
    if not m:
        raise argparse.ArgumentTypeError("mesh must be RxC, e.g. 4x4")
    return int(m.group(1)), int(m.group(2))


def build_config(args):
    """Parse the HIP source and merge CLI overrides into a generation config."""
    with open(args.hip_src, "r") as f:
        src = f.read()
    parsed = parse_hip(src)

    M = args.M if args.M else parsed["M"]
    N = args.N if args.N else parsed["N"]
    K = args.K if args.K else parsed["K"]
    if M is None or N is None or K is None:
        missing = [d for d, v in (("M", M), ("N", N), ("K", K)) if v is None]
        raise ParseError(
            "could not determine dims %s from HIP source; pass --M/--N/--K"
            % ",".join(missing))

    rows, cols = args.mesh
    m_tile, n_tile, k_chunk = compute_tiling(M, N, K)

    if (M, N, K, rows, cols) != (PROVEN_M, PROVEN_N, PROVEN_K, PROVEN_ROWS, PROVEN_COLS):
        sys.stderr.write(
            "[rcom] WARNING: config M=%d N=%d K=%d mesh=%dx%d deviates from the "
            "HW-validated %dx%dx%d / %dx%d path; tiling is heuristic and untested "
            "on HW.\n" % (M, N, K, rows, cols,
                          PROVEN_M, PROVEN_N, PROVEN_K, PROVEN_ROWS, PROVEN_COLS))

    name = args.name
    return {
        "name": name,
        "header": name + ".h",
        "dtype": parsed["dtype"],
        "M": M, "N": N, "K": K,
        "rows": rows, "cols": cols,
        "m_tile": m_tile, "n_tile": n_tile, "k_chunk": k_chunk,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="rcom.py",
        description="Compile a ROCm/HIP int8 GEMM into AIE host+kernel ELFs.")
    ap.add_argument("hip_src", help="path to the HIP GEMM source (.cpp)")
    ap.add_argument("--name", default="matmul",
                    help="generated kernel/function name (default: matmul)")
    ap.add_argument("--out", default="./gen",
                    help="output directory for <name>.cc/.h (default: ./gen)")
    ap.add_argument("--mesh", type=_parse_mesh, default=(4, 4),
                    help="AIE tile mesh RxC (default: 4x4)")
    ap.add_argument("--M", type=int, help="override M")
    ap.add_argument("--N", type=int, help="override N")
    ap.add_argument("--K", type=int, help="override K")
    ap.add_argument("--aie-version", default="5",
                    help="AIE version passed to aiehlc.sh (default: 5)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--emit-only", action="store_true", default=True,
                      help="only generate <name>.cc/.h (default)")
    mode.add_argument("--build", action="store_true",
                      help="generate then invoke script/aiehlc.sh to build ELFs")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.hip_src):
        sys.stderr.write("[rcom] error: no such file: %s\n" % args.hip_src)
        return 2

    try:
        cfg = build_config(args)
    except ParseError as e:
        sys.stderr.write("[rcom] error: %s\n" % e)
        return 1

    os.makedirs(args.out, exist_ok=True)
    cc_path = os.path.join(args.out, cfg["name"] + ".cc")
    h_path = os.path.join(args.out, cfg["name"] + ".h")
    with open(cc_path, "w") as f:
        f.write(emit_cc(cfg))
    with open(h_path, "w") as f:
        f.write(emit_h(cfg))
    sys.stderr.write("[rcom] generated %s\n[rcom] generated %s\n" % (cc_path, h_path))

    if args.build:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
        aiehlc_sh = os.path.join(repo_root, "script", "aiehlc.sh")
        cmd = "source %s --aie-version %s --runtime-source-file %s" % (
            aiehlc_sh, args.aie_version, os.path.abspath(cc_path))
        sys.stderr.write("[rcom] building: %s\n" % cmd)
        rc = subprocess.call(["bash", "-c", cmd], cwd=repo_root)
        return rc

    return 0


if __name__ == "__main__":
    sys.exit(main())
