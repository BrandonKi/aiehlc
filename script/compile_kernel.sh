#!/bin/bash
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Minimal standalone AIE-2PS kernel compilation script
# Compiles kernel.cc (which includes compute_kernel.cc) to kernel ELF + kernel.o
#
# Usage: compile_kernel.sh [func_name]
#   func_name  - kernel function name for binary symbols (default: compute_kernel)
#                Produces kernel.o with _binary_kernel_<func_name>_{start,end,size}
###############################################################################

# Optional first argument: kernel function name (default: compute_kernel)
KERNEL_FUNC_NAME="${1:-compute_kernel}"

# Save the script directory BEFORE sourcing anything
# Use a unique name to avoid conflicts with setup.sh
KERNEL_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Repo root: script lives in script/ -> one level up.
AIEHLC_DIR="$(cd "${KERNEL_SCRIPT_DIR}/.." && pwd)"

# WORKLOCAL_DIR comes from the caller (default: current dir). All generated
# kernel sources (kernel.cc, aieml.prx) and build artifacts live here.
WORKLOCAL_DIR="${WORKLOCAL_DIR:-$(pwd)}"
BUILD_DIR="${WORKLOCAL_DIR}/build"

# Source setup.sh to set XILINX_VITIS and other environment variables
if [ -f "${AIEHLC_DIR}/script/setup.sh" ]; then
    echo "Sourcing Vitis environment from ${AIEHLC_DIR}/script/setup.sh..."
    # Save current directory to restore after sourcing
    SAVED_PWD="$(pwd)"
    source "${AIEHLC_DIR}/script/setup.sh" --path-set-only
    # Restore directory
    cd "${SAVED_PWD}"
    echo "✓ Environment loaded: XILINX_VITIS=${XILINX_VITIS}"
else
    echo "Warning: setup.sh not found at ${AIEHLC_DIR}/script/setup.sh"
    echo "Checking if XILINX_VITIS is already set..."
fi

# Check XILINX_VITIS is set
if [ -z "$XILINX_VITIS" ]; then
    echo "Error: XILINX_VITIS environment variable not set"
    echo "Please source Vitis settings: source /path/to/Vitis/settings64.sh"
    exit 1
fi

# Clean previous kernel build artifacts--to fix the _main missing error
# Only remove kernel.o and kernel ELF (not kernel_*.o from multi-kernel builds)
rm -f "${BUILD_DIR}"/chesswork/kernel* "${BUILD_DIR}"/kernel.o "${BUILD_DIR}"/kernel 2>/dev/null || true

echo "Kernel func name: ${KERNEL_FUNC_NAME}"

# Use generated BCF/PRX if available (from tilinglinalg pipeline), fall back to stock
if [ -f "${WORKLOCAL_DIR}/aieml.prx" ]; then
    PRX_FILE="${WORKLOCAL_DIR}/aieml.prx"
    echo "Using generated PRX: ${PRX_FILE}"
else
    PRX_FILE="aie2ps.prx"
    echo "Using stock PRX: ${PRX_FILE}"
fi

# Build from WORKLOCAL_DIR so the relative --kernel-cc ./kernel.cc resolves there.
cd "${WORKLOCAL_DIR}"
_KC_PLATFORM="${PLATFORM:-baremetal}"
source ${AIEHLC_DIR}/script/kc.sh --kernel-cc ./kernel.cc --func-name "${KERNEL_FUNC_NAME}" --aie-version 5 --platform "${_KC_PLATFORM}" --debug-output --output-dir $BUILD_DIR --prx "${PRX_FILE}"
