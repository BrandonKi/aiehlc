#!/usr/bin/env bash
###############################################################################
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
#
# Environment Setup Verification Script
#
# Validates that all required tools, paths, and environment variables are
# correctly configured before building or running AIEHLC.
#
# Usage:
#   source script/verify_env.sh            # Full check
#   source script/verify_env.sh --quick    # Skip slow checks (BSP, aie-rt)
#
# Based on recurring environment issues from git history:
#   b7c10e9 - Docker timezone broke aiecompiler
#   8c19102 - Local AIEngine path not working
#   37844b5 - Scripts failing from different directories
#   7ecf617 - Kernel source not found
#   149da36 - Vitis and llvm-aie version mismatch
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AIEHLC_DIR="${SCRIPT_DIR}/.."

PASS_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0
QUICK_MODE=0

if [[ "${1:-}" == "--quick" ]]; then
    QUICK_MODE=1
fi

pass() {
    echo "  [PASS] $1"
    ((PASS_COUNT++))
}

warn() {
    echo "  [WARN] $1"
    ((WARN_COUNT++))
}

fail() {
    echo "  [FAIL] $1"
    ((FAIL_COUNT++))
}

echo "=== AIEHLC Environment Verification ==="
echo ""

# --- 1. XILINX_VITIS ---
echo "--- Vitis Installation ---"

if [ -n "$XILINX_VITIS" ]; then
    if [ -d "$XILINX_VITIS" ]; then
        pass "XILINX_VITIS=$XILINX_VITIS"
        if command -v aiecompiler &>/dev/null; then
            pass "aiecompiler found: $(which aiecompiler)"
        else
            warn "aiecompiler not in PATH (may need: source \$XILINX_VITIS/../settings64.sh)"
        fi
        if command -v xchesscc &>/dev/null; then
            pass "xchesscc found: $(which xchesscc)"
        else
            warn "xchesscc not in PATH (needed for kernel compilation)"
        fi
    else
        fail "XILINX_VITIS=$XILINX_VITIS does not exist"
    fi
else
    if command -v aiecompiler &>/dev/null; then
        warn "XILINX_VITIS not set but aiecompiler found at $(which aiecompiler)"
    else
        fail "XILINX_VITIS not set and aiecompiler not found"
    fi
fi
echo ""

# --- 2. LLVM ---
echo "--- LLVM Installation ---"

if [ -n "$LLVM_INSTALL_DIR" ]; then
    if [ -d "$LLVM_INSTALL_DIR" ]; then
        pass "LLVM_INSTALL_DIR=$LLVM_INSTALL_DIR"
    else
        warn "LLVM_INSTALL_DIR=$LLVM_INSTALL_DIR does not exist (may be a stale path)"
    fi
else
    warn "LLVM_INSTALL_DIR not set (needed for aiehlc build)"
fi

LLVM_AIE="$AIEHLC_DIR/thirdparty/llvm-aie"
if [ -d "$LLVM_AIE" ]; then
    pass "llvm-aie found at $LLVM_AIE"
else
    warn "llvm-aie not found at $LLVM_AIE (run setup.sh --enable-llvmaie)"
fi
echo ""

# --- 3. Cross-compiler toolchain ---
echo "--- Cross-Compiler Toolchain ---"

if [ -n "$XILINX_VITIS" ]; then
    BAREMETAL_GCC="$XILINX_VITIS/gnu/aarch64/lin/aarch64-none/bin/aarch64-none-elf-g++"
    LINUX_GCC="$XILINX_VITIS/gnu/aarch64/lin/aarch64-linux/bin/aarch64-linux-gnu-g++"

    if [ -f "$BAREMETAL_GCC" ]; then
        pass "Baremetal g++ found"
    else
        warn "Baremetal g++ not found at $BAREMETAL_GCC"
    fi

    if [ -f "$LINUX_GCC" ]; then
        pass "Linux g++ found"
    else
        warn "Linux g++ not found at $LINUX_GCC"
    fi
else
    warn "Skipping cross-compiler check (XILINX_VITIS not set)"
fi
echo ""

# --- 4. AIE Driver (aie-rt) ---
echo "--- AIE Driver (aie-rt) ---"

AIE_DRIVER_DIR="$AIEHLC_DIR/thirdparty/alib"
if [ -d "$AIE_DRIVER_DIR/aie-rt" ]; then
    pass "aie-rt directory exists"
    if [ -f "$AIE_DRIVER_DIR/aie-rt/driver/src/stream_switch/xaie_ss.c" ]; then
        pass "aie-rt driver source found"
    else
        warn "aie-rt driver source incomplete"
    fi
else
    if [[ "$QUICK_MODE" -eq 0 ]]; then
        fail "aie-rt not found at $AIE_DRIVER_DIR/aie-rt (run setup.sh)"
    else
        warn "aie-rt not found (skipped in quick mode)"
    fi
fi
echo ""

# --- 5. BSP directories ---
echo "--- BSP Directories ---"

if [[ "$QUICK_MODE" -eq 0 ]]; then
    ARCH_DIR="$AIEHLC_DIR/thirdparty/arch"
    for proc in psv_cortexa72_0 psv_cortexr5_0 cortexa78_0; do
        if [ -d "$ARCH_DIR/$proc/workspace" ]; then
            pass "BSP for $proc exists"
        else
            warn "BSP for $proc not found (run setup.sh)"
        fi
    done
else
    warn "BSP check skipped (quick mode)"
fi
echo ""

# --- 6. Docker timezone (if in Docker) ---
echo "--- Environment Sanity ---"

if [ -f "/.dockerenv" ]; then
    TZ_CHECK=$(date +%Z 2>/dev/null)
    if [ "$TZ_CHECK" = "UTC" ] || [ -z "$TZ_CHECK" ]; then
        warn "Docker detected with UTC/unset timezone (commit b7c10e9: wrong TZ breaks aiecompiler)"
        warn "Fix: Set TZ in Dockerfile or docker run: -e TZ=America/Los_Angeles"
    else
        pass "Docker timezone: $TZ_CHECK"
    fi
else
    pass "Not running in Docker (timezone check N/A)"
fi

# Check locale
if locale 2>/dev/null | grep -q "UTF-8\|utf-8"; then
    pass "Locale supports UTF-8"
else
    warn "Locale may not support UTF-8 (can cause string processing issues)"
fi
echo ""

# --- 7. Build directory ---
echo "--- Build Directories ---"

UNITEST_BUILD="$AIEHLC_DIR/src/mlir/mlirfront/tilinglinalg/pass/unitest/build"
if [ -d "$UNITEST_BUILD" ]; then
    if [ -f "$UNITEST_BUILD/test" ]; then
        pass "unitest binary exists at $UNITEST_BUILD/test"
    else
        warn "unitest build dir exists but binary not found (run: cd build && cmake .. && make -j4)"
    fi
else
    warn "unitest build directory not found (create with: mkdir build && cd build && cmake ..)"
fi

WORKLOCAL="$AIEHLC_DIR/src/mlir/mlirfront/tilinglinalg/pass/unitest/worklocal"
if [ -d "$WORKLOCAL" ]; then
    pass "worklocal directory exists"
else
    warn "worklocal directory not found (will be created by test run)"
fi
echo ""

# --- 8. PAL/Board test environment (optional) ---
echo "--- PAL/Board Test (optional) ---"

if [ -n "$PALIP" ]; then
    pass "PALIP=$PALIP"
else
    warn "PALIP not set (needed for HW board tests)"
fi

if [ -n "$BOARDNAME" ]; then
    pass "BOARDNAME=$BOARDNAME"
else
    warn "BOARDNAME not set (needed for HW board tests)"
fi

if [ -n "$USERNAME" ]; then
    pass "USERNAME=$USERNAME"
else
    warn "USERNAME not set (needed for board SSH access)"
fi
echo ""

# --- Summary ---
echo "=== Summary ==="
echo "  PASS: $PASS_COUNT"
echo "  WARN: $WARN_COUNT"
echo "  FAIL: $FAIL_COUNT"
echo ""

if [ "$FAIL_COUNT" -gt 0 ]; then
    echo "  STATUS: FAILED -- fix the FAIL items before building"
    echo "  Hint: Run 'source script/setup.sh' to configure the environment"
elif [ "$WARN_COUNT" -gt 0 ]; then
    echo "  STATUS: WARNINGS -- build may work but some features may be unavailable"
else
    echo "  STATUS: ALL CLEAR -- environment is fully configured"
fi
echo ""
