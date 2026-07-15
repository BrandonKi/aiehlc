#!/usr/bin/env bash
###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
#
# rcom.sh -- ROCm/HIP GEMM front end wrapper.
#
# Generates a CUDA-style <name>.cc/.h from a HIP GEMM source (via rcom.py),
# then hands the generated .cc to the unchanged aiehlc.sh pipeline to build
# host + kernel ELFs (aout/main.elf).
#
# Usage:
#   source script/rcom.sh --rocm-source-file <hip> [--aie-version 5] \
#          [--mesh RxC] [--name matmul] [--out DIR]
#
usage() {
    echo "Usage: source $0 --rocm-source-file <hip> [--aie-version 5] [--mesh RxC] [--name NAME] [--out DIR]"
    return 1
}

main() {
    local rocm_source_file=""
    local aie_version="5"
    local mesh="4x4"
    local name="matmul"
    local out=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -help|--help)
                usage
                return 0
                ;;
            --rocm-source-file)
                rocm_source_file="$2"
                shift 2
                ;;
            --aie-version)
                aie_version="$2"
                shift 2
                ;;
            --mesh)
                mesh="$2"
                shift 2
                ;;
            --name)
                name="$2"
                shift 2
                ;;
            --out)
                out="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                usage
                return 1
                ;;
        esac
    done

    if [ -z "$rocm_source_file" ]; then
        echo "Error: --rocm-source-file is required."
        usage
        return 1
    fi

    local SCRIPT_DIR
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local REPO_ROOT="${SCRIPT_DIR}/.."
    local RCOM_PY="${REPO_ROOT}/src/tool/frontend/rcom.py"

    # NOTE: keep the generated files OUTSIDE aout/ -- aiehlc.sh runs
    # `rm -rf $(pwd)/aout/` at startup and would delete them otherwise.
    if [ -z "$out" ]; then
        out="${REPO_ROOT}/example/tileprogram/rocm/gen"
    fi

    echo "[rcom.sh] Generating AIE source from ${rocm_source_file}..."
    python3 "$RCOM_PY" "$rocm_source_file" \
        --name "$name" --mesh "$mesh" --out "$out" --emit-only
    local RCOM_RC=$?
    if [ $RCOM_RC -ne 0 ]; then
        echo "[rcom.sh] Error: rcom.py failed (exit code $RCOM_RC)."
        return $RCOM_RC
    fi

    local gen_cc="${out}/${name}.cc"
    if [ ! -f "$gen_cc" ]; then
        echo "[rcom.sh] Error: expected generated file not found: $gen_cc"
        return 1
    fi

    echo "[rcom.sh] Building via aiehlc.sh..."
    source "${SCRIPT_DIR}/aiehlc.sh" --aie-version "$aie_version" --runtime-source-file "$gen_cc"
}

main "$@"
