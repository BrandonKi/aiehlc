#!/usr/bin/env bash
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
###############################################################################
# toolenv.sh - Source this from ~/.bashrc to activate aiehlc shell tools
#
# What it does:
#   1. Runs "source setup.sh --path-set-only" (PATH, XILINX_VITIS, LLVM_AIE, …)
#   2. Adds script/ and script/tool/ to PATH
#   3. Sources all *.sh tool files in this directory (shell functions)
###############################################################################

_AIEHLC_TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_AIEHLC_SCRIPT_DIR="$(cd "$_AIEHLC_TOOL_DIR/.." && pwd)"

# --- Run setup.sh --path-set-only (sets XILINX_VITIS, LLVM_AIE_PATH, etc.) --
if [ -f "$_AIEHLC_SCRIPT_DIR/setup.sh" ]; then
    source "$_AIEHLC_SCRIPT_DIR/setup.sh" --path-set-only
fi

# --- Add script/ and script/tool/ to PATH --------------------------------
for _d in "$_AIEHLC_SCRIPT_DIR" "$_AIEHLC_TOOL_DIR"; do
    case ":$PATH:" in
        *":$_d:"*) ;;  # already in PATH
        *) export PATH="$_d:$PATH" ;;
    esac
done
unset _d

# --- Source every tool definition (skip this file itself) -----------------
for _tool_file in "$_AIEHLC_TOOL_DIR"/*.sh; do
    [ "$_tool_file" = "$_AIEHLC_TOOL_DIR/toolenv.sh" ] && continue
    # shellcheck disable=SC1090
    source "$_tool_file"
done
unset _tool_file _AIEHLC_TOOL_DIR _AIEHLC_SCRIPT_DIR
