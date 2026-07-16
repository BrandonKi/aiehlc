#!/usr/bin/env bash
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: MIT

set -e

SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SIM_DIR}/../.." && pwd)"

HOST_SRC=""
KERNEL_OBJS=""
KERNEL_NAMES=""
USE_LLVM_AIE=0
AIE_GEN=2
SHIM_COL=3
KERNEL_COL=4
KERNEL_ROW=1
WORK_DIR=""
STUB_ALL=0
STUB_TILES=""
NOC_ALL=0
SKIP_BUILD=0
SKIP_WORK=0
SIM_OPTS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host-src)      HOST_SRC="$2";       shift 2 ;;
        --kernel-objs)   KERNEL_OBJS="$2";           shift 2 ;;
        --kernel-names)  KERNEL_NAMES="$2";          shift 2 ;;
        --use-llvm-aie)  USE_LLVM_AIE=1;             shift ;;
        --aie-gen)       AIE_GEN="$2";        shift 2 ;;
        --work-dir)      WORK_DIR="$2";       shift 2 ;;
        --stub-all)      STUB_ALL=1;          shift ;;
        --stub-tiles)    STUB_TILES="$2";     shift 2 ;;
        --noc-all)       NOC_ALL=1;           shift ;;
        --skip-build)    SKIP_BUILD=1;        shift ;;
        --skip-work)     SKIP_WORK=1;         shift ;;
        --sim-opts)      SIM_OPTS="$2";       shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "$HOST_SRC" ]; then
    echo "Error: --host-src is required"
    return 1 2>/dev/null || exit 1
fi

case "$AIE_GEN" in
    1) AIE_ARCH_NUM=10; AIE_ARCH_STR="aie"    ;;
    2) AIE_ARCH_NUM=20; AIE_ARCH_STR="aie-ml" ;;
    5) AIE_ARCH_NUM=22; AIE_ARCH_STR="aie2ps" ;;
    *)
        echo "Error: --aie-gen must be 1, 2, or 5 (got: $AIE_GEN)"
        return 1 2>/dev/null || exit 1
        ;;
esac

if [ -z "$WORK_DIR" ]; then
    WORK_DIR="${SIM_DIR}/Work_gen${AIE_GEN}"
fi

HOST_SRC_ABS="$(cd "$(dirname "$HOST_SRC")" && pwd)/$(basename "$HOST_SRC")"
PS_SO="${SIM_DIR}/build/aiehlc_ps.so"
STUB_ELF="${SIM_DIR}/build/stub_kernel_build/stub_kernel"

if [ -z "${XILINX_VITIS}" ]; then
    source "${REPO_ROOT}/script/setup.sh" --path-set-only
fi

if [ "$SKIP_BUILD" -eq 0 ]; then
    echo "=== [1/3] Building PS shared library ==="
    echo "  AIE gen: ${AIE_GEN} (__AIE_ARCH__=${AIE_ARCH_NUM}, aiearch=${AIE_ARCH_STR})"
    make -C "${SIM_DIR}" \
        AIEHLC_HOST_SRC="${HOST_SRC_ABS}" \
        AIE_ARCH="${AIE_ARCH_NUM}" \
        ${KERNEL_OBJS:+KERNEL_OBJS="${KERNEL_OBJS}"} \
        ${KERNEL_NAMES:+KERNEL_NAMES="${KERNEL_NAMES}"} \
        ${KERNEL_NAMES:+KERNEL_FUNCNAME="$(echo "${KERNEL_NAMES}" | cut -d' ' -f1)"} \
        USE_LLVM_AIE="${USE_LLVM_AIE}"
    if [ "$STUB_ALL" -eq 1 ] || [ -n "$STUB_TILES" ]; then
        echo "  Building stub kernel..."
        make -C "${SIM_DIR}" \
            AIEHLC_HOST_SRC="${HOST_SRC_ABS}" \
            AIE_ARCH="${AIE_ARCH_NUM}" \
            build_stub
    fi
else
    echo "=== [1/3] Skipping build (--skip-build) ==="
fi

if [ "$SKIP_WORK" -eq 0 ]; then
    echo "=== [2/3] Generating Work/ package ==="
    if [ "$STUB_ALL" -eq 1 ]; then
        echo "  Mode: stub-all"
    else
        echo "  Shim col: ${SHIM_COL}  Kernel tile: col=${KERNEL_COL} row=${KERNEL_ROW}"
    fi
    bash "${SIM_DIR}/gen_work_package.sh" \
        --work-dir      "${WORK_DIR}" \
        --ps-so         "${PS_SO}" \
        --aie-arch      "${AIE_ARCH_STR}" \
        --shim-col      "${SHIM_COL}" \
        --kernel-col    "${KERNEL_COL}" \
        --kernel-row "${KERNEL_ROW}" \
        --stub-elf      "${STUB_ELF}" \
        $([ "$STUB_ALL" -eq 1 ] && echo --stub-all) \
        ${STUB_TILES:+--stub-tiles "$STUB_TILES"} \
        $([ "$NOC_ALL"  -eq 1 ] && echo --noc-all)
else
    echo "=== [2/3] Skipping Work/ generation (--skip-work) ==="
fi

echo "=== [3/3] Launching aiesimulator ==="
echo "  Work dir: ${WORK_DIR}"
echo ""

export AIE_WORK_DIR="${WORK_DIR}"

if [ ! -x "${REPO_ROOT}/aiehlc_aiesimulator" ]; then
    bash "${SIM_DIR}/gen_aiesimulator.sh" "${REPO_ROOT}/aiehlc_aiesimulator"
fi

"${REPO_ROOT}/aiehlc_aiesimulator" \
    --pkg-dir "${WORK_DIR}" \
    --display-run-interval=100000 \
    ${SIM_OPTS}
