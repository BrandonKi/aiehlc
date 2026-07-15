#!/usr/bin/env bash
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
# Verify host ELF: run on HW via apppaltest.py and check console for success/errors.
# Bundled in hostcodegen skill; resolves repo root from skill path.
# Usage: verify_host.sh [--compile] [elf_path]
#   --compile   Compile host first (run hostcompile.sh from worklocal).
#   elf_path    ELF to run (default: worklocal/build/host from repo root).
# Requires: USERNAME, PALIP, BOARDNAME (or script/test/envlocal.sh).
# Exit: 0 if verify passed (no AIE ERROR / Invalid Tile; runtime teardown seen), 1 otherwise.
###############################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# From skill scripts/ go up to repo root: scripts -> hostcodegen -> skills -> .cursor -> repo
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
DEFAULT_ELF="${REPO_ROOT}/src/mlir/mlirfront/tilinglinalg/pass/unitest/worklocal/build/host"
WORKLOCAL_DIR="${REPO_ROOT}/src/mlir/mlirfront/tilinglinalg/pass/unitest/worklocal"
TEST_DIR="${REPO_ROOT}/script/test"

DO_COMPILE=0
ELF_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --compile)
            DO_COMPILE=1
            shift
            ;;
        *)
            ELF_PATH="$1"
            shift
            ;;
    esac
done

if [[ -z "$ELF_PATH" ]]; then
    ELF_PATH="$DEFAULT_ELF"
fi

# Resolve to absolute path
ELF_PATH="$(cd "$(dirname "$ELF_PATH")" 2>/dev/null && pwd)/$(basename "$ELF_PATH")" || true
if [[ ! -f "$ELF_PATH" ]]; then
    echo "Error: ELF not found: $ELF_PATH"
    echo "Run with --compile to build first, or pass a valid path."
    exit 1
fi

# Optional: source env for USERNAME, PALIP, BOARDNAME
if [[ -f "${TEST_DIR}/envlocal.sh" ]]; then
    export PATH_SET_ONLY="${PATH_SET_ONLY:-1}"
    source "${TEST_DIR}/envlocal.sh" 2>/dev/null || true
fi

if [[ "$DO_COMPILE" -eq 1 ]]; then
    echo ">>> Compiling host..."
    if [[ -f "${WORKLOCAL_DIR}/host.cc" ]]; then
        (cd "$REPO_ROOT" && source script/setup.sh --path-set-only 2>/dev/null; cd "$WORKLOCAL_DIR" && source hostcompile.sh) || { echo "Error: hostcompile failed"; exit 1; }
        ELF_PATH="$DEFAULT_ELF"
        ELF_PATH="$(cd "$(dirname "$ELF_PATH")" && pwd)/$(basename "$ELF_PATH")"
    else
        echo "Error: host.cc not found in worklocal. Run unitest (e.g. ./test) first."
        exit 1
    fi
fi

LOG_FILE="${TEST_DIR}/.verify_host_console.log"
echo ">>> Running host on HW (ELF=$ELF_PATH), logging to $LOG_FILE"
python3 "${TEST_DIR}/apppaltest.py" "$ELF_PATH" 2>&1 | tee "$LOG_FILE" || true

# Verify: fail if console contains AIE error / invalid tile; pass if runtime teardown seen
FAIL_MARKERS="AIE ERROR|Invalid Tile Type|Cannot find Tile Type"
PASS_MARKERS="device_teardown done|device_init OK"

if grep -qE "$FAIL_MARKERS" "$LOG_FILE" 2>/dev/null; then
    echo ""
    echo "VERIFY FAIL: Console contains error markers ($FAIL_MARKERS)"
    grep -E "$FAIL_MARKERS" "$LOG_FILE" | head -5
    exit 1
fi

if grep -qE "$PASS_MARKERS" "$LOG_FILE" 2>/dev/null; then
    echo ""
    echo "VERIFY PASS: Runtime messages found in console; no AIE/Invalid Tile errors."
    exit 0
fi

echo ""
echo "VERIFY INCONCLUSIVE: No failure markers found, but no success markers ($PASS_MARKERS) in log. Check $LOG_FILE"
exit 1
