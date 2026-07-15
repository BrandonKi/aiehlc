#!/usr/bin/env python3
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Parse host_canonicalized dataflow from either a generated host.cc (C code)
or an MLIR IR dump (logdf) and visualize the data connections from source
tensor through partitioning, slicing, DMA BD config, IO, to start_io.

Accepts both file formats (auto-detected):
  - host.cc  : generated C code with __Runtime_* calls
  - logdf    : MLIR IR dump containing dfschedule.host @host_canonicalized

Usage:
    python host_canonicalized_analysis.py path/to/host.cc              # html + serve (IR view)
    python host_canonicalized_analysis.py path/to/host.cc --view api   # API call graph
    python host_canonicalized_analysis.py path/to/logdf                # MLIR IR dump
    python host_canonicalized_analysis.py path/to/host.cc -m text      # text tree
    python host_canonicalized_analysis.py path/to/host.cc -m html -o out.html
    python host_canonicalized_analysis.py path/to/host.cc --no-serve
"""

import argparse
import http.server
import re
import socket
import sys
import webbrowser
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SSAOp:
    """A single SSA operation parsed from the MLIR IR."""
    result: str
    op_type: str
    operands: List[str]
    attrs: Dict[str, str]
    result_type: str
    raw: str
    line_num: int


@dataclass
class TreeNode:
    """A node in the dataflow tree for visualization."""
    ssa_name: str
    op: SSAOp
    label: str
    detail: str
    children: List["TreeNode"] = field(default_factory=list)
    node_type: str = ""


# ---------------------------------------------------------------------------
# 1. Block extraction
# ---------------------------------------------------------------------------

def extract_host_canonicalized(text: str) -> List[Tuple[str, int]]:
    """Extract lines inside ``dfschedule.host @host_canonicalized { ... }``
    after the ScheduleCanonicalizePass header.

    Returns a list of (line_text, 1-based line_number) tuples for the inner
    content of the block (excluding the opening/closing braces).
    """
    lines = text.split("\n")

    in_section = False
    collecting = False
    depth = 0
    content: List[Tuple[str, int]] = []

    for i, line in enumerate(lines):
        if "ScheduleCanonicalizePass" in line:
            in_section = True
            continue

        if in_section and not collecting:
            if "dfschedule.host @host_canonicalized" in line and "{" in line:
                collecting = True
                depth = 1
                continue

        if collecting:
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                break
            content.append((line, i + 1))

    return content


# ---------------------------------------------------------------------------
# 2. Multi-line merge & SSA parser
# ---------------------------------------------------------------------------

def _merge_multiline_ops(
    content: List[Tuple[str, int]],
) -> List[Tuple[str, int]]:
    """Join multi-line ops (attribute blocks spanning several lines) into
    single logical lines.  Returns (merged_text, first_line_number)."""
    merged: List[Tuple[str, int]] = []
    cur_text = ""
    cur_line = 0
    brace_depth = 0

    for raw, lno in content:
        stripped = raw.strip()
        if not stripped:
            continue

        if cur_text == "":
            cur_text = stripped
            cur_line = lno
            brace_depth = stripped.count("{") - stripped.count("}")
        else:
            cur_text += " " + stripped
            brace_depth += stripped.count("{") - stripped.count("}")

        if brace_depth <= 0:
            merged.append((cur_text, cur_line))
            cur_text = ""
            brace_depth = 0

    if cur_text:
        merged.append((cur_text, cur_line))

    return merged


def _parse_ssa_op(line: str, line_num: int) -> Optional[SSAOp]:
    """Parse a single merged MLIR operation line into an SSAOp."""
    rest = line.strip()
    if not rest or rest in ("}", "};"):
        return None

    # --- result name ---
    result_name = ""
    m = re.match(r"%(\w+)\s*=\s*(.*)", rest)
    if m:
        result_name = m.group(1)
        rest = m.group(2)

    # --- op name (first dotted identifier) ---
    m_op = re.match(r"([\w]+(?:\.[\w]+)*)", rest)
    if not m_op:
        return None
    op_type = m_op.group(1)

    # --- operands: every %name in the line except the result itself ---
    all_refs = re.findall(r"%(\w+)", line)
    operands: List[str] = []
    skipped_result = False
    for ref in all_refs:
        if ref == result_name and not skipped_result:
            skipped_result = True
            continue
        operands.append(ref)

    # --- attributes from { ... } blocks ---
    attrs: Dict[str, str] = {}
    for am in re.finditer(r"\{([^}]*)\}", line):
        attr_text = am.group(1)
        for kv in re.finditer(
            r"(\w+)\s*=\s*(\"[^\"]*\"|true|false|-?\d+(?:\.\d+)?(?:\s*:\s*\w+)?)",
            attr_text,
        ):
            key = kv.group(1)
            val = kv.group(2).strip().strip('"')
            if " : " in val:
                val = val.split(":")[0].strip()
            attrs[key] = val

    # --- result type ---
    result_type = ""
    tm = re.search(r"->\s*(!?[\w<>.,\s]+)\s*$", line)
    if tm:
        result_type = tm.group(1).strip()
    else:
        tm2 = re.search(r":\s*(!?[\w<>.,!\s]+)\s*$", line)
        if tm2:
            result_type = tm2.group(1).strip()

    return SSAOp(
        result=result_name,
        op_type=op_type,
        operands=operands,
        attrs=attrs,
        result_type=result_type,
        raw=line,
        line_num=line_num,
    )


def parse_mlir_block(text: str) -> List[SSAOp]:
    """Parse the host_canonicalized block from MLIR IR (logdf)."""
    content = extract_host_canonicalized(text)
    if not content:
        return []
    merged = _merge_multiline_ops(content)
    ops: List[SSAOp] = []
    for line, lno in merged:
        op = _parse_ssa_op(line, lno)
        if op:
            ops.append(op)
    return ops


# ---------------------------------------------------------------------------
# 2b. host.cc (C code) parser
# ---------------------------------------------------------------------------

def parse_host_cc(text: str) -> List[SSAOp]:
    """Parse the host_canonicalized() function from generated C code (host.cc).

    Traces variable assignments through the same chain:
    PartitionTensor init -> extract_slice -> buffer_arg -> dma_bd_config
    -> createio -> startio.
    """
    all_lines = text.split("\n")

    # Scope to host_canonicalized() function body
    func_start = -1
    func_end = len(all_lines)
    depth = 0
    for i, line in enumerate(all_lines):
        if re.search(r"\bhost_canonicalized\s*\([^)]*\)\s*\{", line):
            func_start = i + 1
            depth = 1
            continue
        if func_start >= 0:
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                func_end = i
                break
    if func_start < 0:
        func_start = 0

    lines = all_lines[func_start:func_end]
    line_offset = func_start
    ops: List[SSAOp] = []

    buf_arg_map: Dict[str, str] = {}  # void* vN -> PT var it came from
    static_arrays: Dict[str, List[int]] = {}  # array var -> list of int values

    # Pre-pass: collect static int64_t array declarations and multi-dim int arrays
    for line in lines:
        m = re.search(
            r"static\s+const\s+int64_t\s+(\w+)\[\]\s*=\s*\{([^}]+)\}",
            line.strip(),
        )
        if not m:
            m = re.search(
                r"static\s+(?:const\s+)?\w+\s+(\w+)(?:\[\d+\])+\s*=\s*\{([^}]+)\}",
                line.strip(),
            )
        if m:
            vals = [int(v.strip()) for v in m.group(2).split(",") if v.strip().lstrip('-').isdigit()]
            static_arrays[m.group(1)] = vals

    for i, line in enumerate(lines):
        lno = line_offset + i + 1
        stripped = line.strip()

        # --- __Runtime_malloc ---
        m = re.search(r"void\*\s+(\w+)\s*=\s*__Runtime_malloc\((\d+)\)", stripped)
        if m:
            ops.append(SSAOp(
                result=m.group(1), op_type="mem_allocate",
                operands=[],
                attrs={"alloc_size": m.group(2)},
                result_type="void*", raw=stripped, line_num=lno,
            ))
            continue

        # --- __Runtime_memcpy ---
        m = re.search(
            r"__Runtime_memcpy\(\s*(\w+),\s*(?:\(void\*\)\s*)?(\w+),\s*(\d+)\)",
            stripped,
        )
        if m:
            dst_var = m.group(1)
            src_var = m.group(2)
            for prev_op in reversed(ops):
                if prev_op.result == dst_var and prev_op.op_type == "mem_allocate":
                    prev_op.attrs["source_array"] = src_var
                    break
            continue

        # --- __runtime_buffer_offset ---
        m = re.search(
            r"void\*\s+(\w+)\s*=\s*__runtime_buffer_offset\((\w+),\s*(-?\d+)\)",
            stripped,
        )
        if m:
            ops.append(SSAOp(
                result=m.group(1), op_type="buffer_offset",
                operands=[m.group(2)],
                attrs={"offset": m.group(3)},
                result_type="void*", raw=stripped, line_num=lno,
            ))
            continue

        # --- __Runtime_init_PartitionTensor ---
        m = re.search(
            r"PartitionTensor\s+(\w+)\s*=\s*__Runtime_init_PartitionTensor\("
            r"([^,]+),\s*(\d+),\s*(\d+),\s*(\w+),\s*(\w+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(-?\d+)\)",
            stripped,
        )
        if m:
            hw_map = {"0": "row", "1": "col"}
            elem_size = int(m.group(3))
            orig_shape = static_arrays.get(m.group(5), [])
            part_shape = static_arrays.get(m.group(6), [])
            pt_attrs: Dict[str, str] = {
                "splitdim": m.group(7), "splitnum": m.group(8),
                "hw_axis_owner": hw_map.get(m.group(9), m.group(9)),
                "elem_size": str(elem_size),
            }
            if orig_shape:
                pt_attrs["orig_shape"] = "x".join(str(d) for d in orig_shape)
            if part_shape:
                pt_attrs["part_shape"] = "x".join(str(d) for d in part_shape)
            ops.append(SSAOp(
                result=m.group(1), op_type="partitiontensor",
                operands=[m.group(2).strip()],
                attrs=pt_attrs,
                result_type="PartitionTensor", raw=stripped, line_num=lno,
            ))
            continue

        # --- __Runtime_extract_slice_contiguous_2d ---
        m = re.search(
            r"PartitionTensor\s+(\w+)\s*=\s*__Runtime_extract_slice_contiguous_2d\("
            r"(\w+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)",
            stripped,
        )
        if m:
            off0, off1 = int(m.group(3)), int(m.group(4))
            sz0, sz1 = int(m.group(5)), int(m.group(6))
            ops.append(SSAOp(
                result=m.group(1), op_type="extract_slice",
                operands=[m.group(2)],
                attrs={"off0": str(off0), "off1": str(off1),
                       "size0": str(sz0), "size1": str(sz1)},
                result_type=f"PartitionTensor [{off0}:{off0+sz0}, {off1}:{off1+sz1}]",
                raw=stripped, line_num=lno,
            ))
            continue

        # --- __Runtime_extract_slice_strided_2d ---
        m = re.search(
            r"PartitionTensor\s+(\w+)\s*=\s*__Runtime_extract_slice_strided_2d\("
            r"\w+,\s*(\w+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)",
            stripped,
        )
        if m:
            off0, off1 = int(m.group(3)), int(m.group(4))
            sz0, sz1 = int(m.group(5)), int(m.group(6))
            ops.append(SSAOp(
                result=m.group(1), op_type="extract_slice",
                operands=[m.group(2)],
                attrs={"off0": str(off0), "off1": str(off1),
                       "size0": str(sz0), "size1": str(sz1)},
                result_type=f"PartitionTensor [{off0}:{off0+sz0}, {off1}:{off1+sz1}]",
                raw=stripped, line_num=lno,
            ))
            continue

        # --- XAie_TileLoc ---
        m = re.search(
            r"XAie_LocType\s+(\w+)\s*=\s*XAie_TileLoc\((\d+),\s*(\d+)\)",
            stripped,
        )
        if m:
            ops.append(SSAOp(
                result=m.group(1), op_type="declaretile",
                operands=[],
                attrs={"col": m.group(2), "row": m.group(3)},
                result_type="XAie_LocType", raw=stripped, line_num=lno,
            ))
            continue

        # --- __runtime_buffer_arg ---
        # Accepts both a plain variable name and a cast literal: (void*)N
        m = re.search(
            r"void\*\s+(\w+)\s*=\s*__runtime_buffer_arg\(\s*(?:\(void\*\)\s*)?(\w+)\s*\)",
            stripped,
        )
        if m:
            buf_arg_map[m.group(1)] = m.group(2)
            ops.append(SSAOp(
                result=m.group(1), op_type="declaretensor",
                operands=[m.group(2)],
                attrs={},
                result_type="void*", raw=stripped, line_num=lno,
            ))
            continue

        # --- __Runtime_dma_bd_config_multidim[_ooo] (22-arg: DevInst, tile, buf, bd_id, len, next_bd,
        #     enable_packet, packet_id, acq_lock_id, acq_lock_val, rel_lock_id, rel_lock_val, ooo_bd_id,
        #     num_dims, d0_stride, d0_wrap, d1_stride, d1_wrap, d2_stride, d2_wrap,
        #     iter_step_size, iter_wrap) ---
        m = re.search(
            r"XAie_DmaDesc\s+(\w+)\s*=\s*__Runtime_dma_bd_config_multidim(?:_ooo)?\("
            r"(\w+),\s*(\w+),\s*(\w+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),"
            r"\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),"
            r"\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),"
            r"\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),"
            r"\s*(-?\d+),\s*(-?\d+)\)",
            stripped,
        )
        if m:
            buf_var = m.group(4)
            source_pt = buf_arg_map.get(buf_var, buf_var)
            next_bd = m.group(7)
            if int(next_bd) < 0 or int(next_bd) > 65535:
                next_bd = "none"
            ooo_bd_id = m.group(14)
            if int(ooo_bd_id) < 0:
                ooo_bd_id = "none"
            num_dims = int(m.group(15))
            # Collect dimension strides/wraps based on num_dims
            dim_strides = [m.group(16)]
            dim_wraps = [m.group(17)]
            if num_dims >= 2:
                dim_strides.append(m.group(18))
                dim_wraps.append(m.group(19))
            if num_dims >= 3:
                dim_strides.append(m.group(20))
                dim_wraps.append(m.group(21))
            iter_step = m.group(22)
            iter_wrap = m.group(23)
            bd_attrs = {
                "bd_id": m.group(5),
                "len": m.group(6), "next_bd": next_bd,
                "enable_packet": "true" if int(m.group(8)) else "false",
                "packet_id": m.group(9),
                "acquire_lock_id": m.group(10),
                "acquire_lock_val": m.group(11),
                "release_lock_id": m.group(12),
                "release_lock_val": m.group(13),
                "ooo_bd_id": ooo_bd_id,
                "multidim": "true",
                "num_dims": str(num_dims),
                "dim_strides": ",".join(dim_strides),
                "dim_wraps": ",".join(dim_wraps),
                "iter_step": iter_step,
                "iter_wrap": iter_wrap,
            }
            ops.append(SSAOp(
                result=m.group(1), op_type="config.dma_bd",
                operands=[buf_var, m.group(3)],
                attrs=bd_attrs,
                result_type="XAie_DmaDesc", raw=stripped, line_num=lno,
            ))
            continue

        # --- __Runtime_dma_bd_config (13-arg: DevInst, tile, buf, bd_id, len, next_bd, enable_packet, packet_id, acq_lock_id, acq_lock_val, rel_lock_id, rel_lock_val, ooo_bd_id) ---
        m = re.search(
            r"XAie_DmaDesc\s+(\w+)\s*=\s*__Runtime_dma_bd_config\("
            r"(\w+),\s*(\w+),\s*(\w+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),"
            r"\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),"
            r"\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\)",
            stripped,
        )
        if m:
            buf_var = m.group(4)
            source_pt = buf_arg_map.get(buf_var, buf_var)
            next_bd = m.group(7)
            if int(next_bd) < 0 or int(next_bd) > 65535:
                next_bd = "none"
            ooo_bd_id = m.group(14)
            if int(ooo_bd_id) < 0:
                ooo_bd_id = "none"
            ops.append(SSAOp(
                result=m.group(1), op_type="config.dma_bd",
                operands=[buf_var, m.group(3)],
                attrs={
                    "bd_id": m.group(5),
                    "len": m.group(6), "next_bd": next_bd,
                    "enable_packet": "true" if int(m.group(8)) else "false",
                    "packet_id": m.group(9),
                    "acquire_lock_id": m.group(10),
                    "acquire_lock_val": m.group(11),
                    "release_lock_id": m.group(12),
                    "release_lock_val": m.group(13),
                    "ooo_bd_id": ooo_bd_id,
                },
                result_type="XAie_DmaDesc", raw=stripped, line_num=lno,
            ))
            continue

        # --- __Runtime_dma_bd_config (10-arg fallback: DevInst, tile, buf, bd_id, len, next_bd, enable_packet, packet_id, acq_lock_id, acq_lock_val) ---
        m = re.search(
            r"XAie_DmaDesc\s+(\w+)\s*=\s*__Runtime_dma_bd_config\("
            r"(\w+),\s*(\w+),\s*(\w+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),"
            r"\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\)",
            stripped,
        )
        if m:
            buf_var = m.group(4)
            next_bd = m.group(7)
            if int(next_bd) < 0 or int(next_bd) > 65535:
                next_bd = "none"
            ops.append(SSAOp(
                result=m.group(1), op_type="config.dma_bd",
                operands=[buf_var, m.group(3)],
                attrs={
                    "bd_id": m.group(5),
                    "len": m.group(6), "next_bd": next_bd,
                    "enable_packet": "true" if int(m.group(8)) else "false",
                    "packet_id": m.group(9),
                },
                result_type="XAie_DmaDesc", raw=stripped, line_num=lno,
            ))
            continue

        # --- __Runtime_dma_createio_4 ---
        m = re.search(
            r"io\s+(\w+)\s*=\s*__Runtime_dma_createio_4\("
            r"(\w+),\s*(\w+),\s*(-?\d+),\s*(-?\d+)(?:,\s*(\w+))?\)",
            stripped,
        )
        if m:
            direction = m.group(6) or "?"
            if "S2MM" in direction:
                direction = "S2MM"
            elif "MM2S" in direction:
                direction = "MM2S"
            ops.append(SSAOp(
                result=m.group(1), op_type="config.create_io",
                operands=[m.group(3), m.group(2)],
                attrs={
                    "channel": m.group(4),
                    "start_bd": m.group(5),
                    "direction": direction,
                    "io_operation": "RECV" if "S2MM" in direction else "SEND",
                },
                result_type="io", raw=stripped, line_num=lno,
            ))
            # Also pick up direction from preceding comment
            if lno >= 2:
                prev = lines[i - 1].strip() if i > 0 else ""
                cm = re.search(r"direction=(\w+)", prev)
                if cm:
                    ops[-1].attrs["direction"] = cm.group(1)
                    ops[-1].attrs["io_operation"] = (
                        "RECV" if "S2MM" in cm.group(1) else "SEND"
                    )
            continue

        # --- __Runtime_startio (4-arg: DevInst, io, bd_id, repeat_count) ---
        m = re.search(
            r"ioevent\s+(\w+)\s*=\s*__Runtime_startio\(\w+,\s*(\w+),\s*(-?\d+),\s*(-?\d+)\)",
            stripped,
        )
        if not m:
            # Fallback: 2-arg legacy format (io, bd_id)
            m = re.search(
                r"ioevent\s+(\w+)\s*=\s*__Runtime_startio\((\w+),\s*(\d+)\)",
                stripped,
            )
        if m:
            repeat = m.group(4) if m.lastindex >= 4 else "1"
            ops.append(SSAOp(
                result=m.group(1), op_type="start_io",
                operands=[m.group(2)],
                attrs={"bd_id": m.group(3), "repeat_count": repeat},
                result_type="ioevent", raw=stripped, line_num=lno,
            ))
            continue

        # --- __Runtime_dma_channel_enable_ooo ---
        m = re.search(
            r"__Runtime_dma_channel_enable_ooo\(\s*(\w+),\s*(\w+),\s*(-?\d+),\s*(\w+)\)",
            stripped,
        )
        if m:
            tile_var = m.group(2)
            channel = m.group(3)
            direction = m.group(4)
            if "S2MM" in direction:
                direction = "S2MM"
            elif "MM2S" in direction:
                direction = "MM2S"
            ops.append(SSAOp(
                result="", op_type="ooo_enable",
                operands=[tile_var],
                attrs={
                    "channel": channel,
                    "direction": direction,
                },
                result_type="", raw=stripped, line_num=lno,
            ))
            continue

        # --- __Runtime_load_kernel_group ---
        m = re.search(
            r"kernel_group\s+(\w+)\s*=\s*__Runtime_load_kernel_group\w*",
            stripped,
        )
        if m:
            tile_ops = re.findall(r"\b(v\d+)\b", stripped)
            ops.append(SSAOp(
                result=m.group(1), op_type="load_kernel_group",
                operands=tile_ops[:-1] if tile_ops else [],
                attrs={},
                result_type="kernel_group", raw=stripped, line_num=lno,
            ))
            continue

        # --- __Runtime_launch_kernel_group ---
        m = re.search(
            r"event\s+(\w+)\s*=\s*__Runtime_launch_kernel_group\((\w+)\)",
            stripped,
        )
        if m:
            ops.append(SSAOp(
                result=m.group(1), op_type="launch_kernel_group",
                operands=[m.group(2)],
                attrs={},
                result_type="event", raw=stripped, line_num=lno,
            ))
            continue

        # --- __Runtime_wait ---
        m = re.search(r"__Runtime_wait\((\w+)\)", stripped)
        if m:
            ops.append(SSAOp(
                result="", op_type="wait",
                operands=[m.group(1)],
                attrs={},
                result_type="", raw=stripped, line_num=lno,
            ))
            continue

        # --- __Runtime_free ---
        m = re.search(r"__Runtime_free\((\w+)\)", stripped)
        if m:
            ops.append(SSAOp(
                result="", op_type="mem_free",
                operands=[m.group(1)],
                attrs={},
                result_type="", raw=stripped, line_num=lno,
            ))
            continue

        # --- XAie_LockSetValue ---
        m = re.search(
            r"XAie_LockSetValue\(\w+,\s*XAie_TileLoc\((\d+),\s*(\d+)\),\s*"
            r"XAie_LockInit\((\d+),\s*(-?\d+)\)\)",
            stripped,
        )
        if m:
            tile_var = ""
            for prev_op in reversed(ops):
                if (prev_op.op_type == "declaretile"
                        and prev_op.attrs.get("col") == m.group(1)
                        and prev_op.attrs.get("row") == m.group(2)):
                    tile_var = prev_op.result
                    break
            ops.append(SSAOp(
                result="", op_type="lock_init",
                operands=[tile_var] if tile_var else [],
                attrs={
                    "col": m.group(1), "row": m.group(2),
                    "lock_id": m.group(3), "init_value": m.group(4),
                },
                result_type="", raw=stripped, line_num=lno,
            ))
            continue

    # Post-pass: annotate create_io ops with OOO enable info
    ooo_ops = [op for op in ops if op.op_type == "ooo_enable"]
    for ooo_op in ooo_ops:
        ooo_tile_var = ooo_op.operands[0] if ooo_op.operands else ""
        ooo_ch = ooo_op.attrs.get("channel", "")
        ooo_dir = ooo_op.attrs.get("direction", "")
        for op in ops:
            if op.op_type == "config.create_io":
                io_tile_var = op.operands[1] if len(op.operands) >= 2 else ""
                if (io_tile_var == ooo_tile_var
                        and op.attrs.get("channel", "") == ooo_ch
                        and op.attrs.get("direction", "") == ooo_dir):
                    op.attrs["ooo_enabled"] = "true"

    return ops


# ---------------------------------------------------------------------------
# 2c. Format detection and unified parser
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_INPUT = _SCRIPT_DIR / "../../aout/worklocal/host.cc"


def detect_and_parse(text: str) -> Tuple[List[SSAOp], str]:
    """Auto-detect file format and parse. Returns (ops, format_name)."""
    if "dfschedule.host @host_canonicalized" in text:
        return parse_mlir_block(text), "logdf"
    if "__Runtime_dma_bd_config" in text or "host_canonicalized" in text:
        ops = parse_host_cc(text)
        _compute_byte_offsets(ops)
        return ops, "host.cc"
    return [], "unknown"


# ---------------------------------------------------------------------------
# 2d. Byte-offset computation
# ---------------------------------------------------------------------------

@dataclass
class _PtInfo:
    """Tracks PartitionTensor memory layout through the slice chain."""
    elem_size: int = 1
    shape: List[int] = field(default_factory=list)
    cumulative_offset: int = 0
    alloc_size: int = 0
    alloc_var: str = ""


def _compute_byte_offsets(ops: List[SSAOp]) -> None:
    """Annotate ops with byte_offset / byte_range attrs by tracing
    __Runtime_malloc -> buffer_offset -> declaretensor -> dma_bd (new API)
    or XAie_MemAllocate -> PartitionTensor -> extract_slice -> buffer_arg (old API)."""
    pt_info: Dict[str, _PtInfo] = {}
    alloc_for_vaddr: Dict[str, _PtInfo] = {}

    for op in ops:
        if op.op_type == "mem_allocate":
            sz = int(op.attrs.get("alloc_size", 0))
            pt_info[op.result] = _PtInfo(alloc_size=sz, alloc_var=op.result)

        elif op.op_type == "declare_data":
            mem_var = op.operands[0] if op.operands else ""
            parent = pt_info.get(mem_var)
            alloc_sz = int(op.attrs.get("alloc_size", 0))
            info = _PtInfo(
                alloc_size=parent.alloc_size if parent else alloc_sz,
                alloc_var=parent.alloc_var if parent else mem_var,
            )
            pt_info[op.result] = info
            alloc_for_vaddr[op.result] = info
            op.attrs["byte_offset"] = "0"
            op.attrs["byte_range"] = f"[0:{info.alloc_size})"

        elif op.op_type == "partitiontensor":
            data_var = op.operands[0] if op.operands else ""
            parent = pt_info.get(data_var)
            elem_size = int(op.attrs.get("elem_size", 1))
            shape_str = op.attrs.get("orig_shape", "")
            shape = [int(d) for d in shape_str.split("x")] if shape_str else []
            alloc_sz = parent.alloc_size if parent else 0
            alloc_var = parent.alloc_var if parent else data_var
            info = _PtInfo(
                elem_size=elem_size, shape=shape,
                cumulative_offset=0, alloc_size=alloc_sz,
                alloc_var=alloc_var,
            )
            pt_info[op.result] = info
            total = 1
            for d in shape:
                total *= d
            total *= elem_size
            op.attrs["byte_offset"] = "0"
            op.attrs["byte_range"] = f"[0:{total})"
            op.attrs["byte_len"] = str(total)

        elif op.op_type == "extract_slice":
            src_var = op.operands[0] if op.operands else ""
            parent = pt_info.get(src_var)
            if not parent:
                continue
            off0 = int(op.attrs.get("off0", 0))
            off1 = int(op.attrs.get("off1", 0))
            sz0 = int(op.attrs.get("size0", 0))
            sz1 = int(op.attrs.get("size1", 0))
            parent_dim1 = parent.shape[1] if len(parent.shape) > 1 else 0
            local_offset = (off0 * parent_dim1 + off1) * parent.elem_size
            cumul = parent.cumulative_offset + local_offset
            byte_len = sz0 * sz1 * parent.elem_size
            info = _PtInfo(
                elem_size=parent.elem_size, shape=[sz0, sz1],
                cumulative_offset=cumul, alloc_size=parent.alloc_size,
                alloc_var=parent.alloc_var,
            )
            pt_info[op.result] = info
            op.attrs["byte_offset"] = str(cumul)
            op.attrs["byte_range"] = f"[{cumul}:{cumul + byte_len})"
            op.attrs["byte_len"] = str(byte_len)

        elif op.op_type == "buffer_offset":
            base_var = op.operands[0] if op.operands else ""
            parent = pt_info.get(base_var)
            explicit_offset = int(op.attrs.get("offset", 0))
            parent_cumul = parent.cumulative_offset if parent else 0
            parent_alloc_size = parent.alloc_size if parent else 0
            parent_alloc_var = parent.alloc_var if parent else base_var
            parent_elem_size = parent.elem_size if parent else 1
            parent_shape = parent.shape if parent else []
            cumul = parent_cumul + explicit_offset
            remaining = parent_alloc_size - cumul if parent_alloc_size > cumul else 0
            info = _PtInfo(
                elem_size=parent_elem_size,
                shape=parent_shape,
                cumulative_offset=cumul,
                alloc_size=parent_alloc_size,
                alloc_var=parent_alloc_var,
            )
            pt_info[op.result] = info
            op.attrs["byte_offset"] = str(cumul)
            op.attrs["byte_range"] = f"[{cumul}:{cumul + remaining})"
            op.attrs["byte_len"] = str(remaining)

        elif op.op_type == "declaretensor":
            src_var = op.operands[0] if op.operands else ""
            parent = pt_info.get(src_var)
            if parent:
                op.attrs["byte_offset"] = str(parent.cumulative_offset)
                byte_len = 1
                for d in parent.shape:
                    byte_len *= d
                byte_len *= parent.elem_size
                op.attrs["byte_range"] = (
                    f"[{parent.cumulative_offset}"
                    f":{parent.cumulative_offset + byte_len})"
                )
                op.attrs["byte_len"] = str(byte_len)


# ---------------------------------------------------------------------------
# 3. Dataflow graph & tree
# ---------------------------------------------------------------------------

def _tile_type(row: int) -> str:
    if row == 0:
        return "Shim"
    if row <= 2:
        return "Mem"
    return "AIE"


def _resolve_tile(ssa: str, def_map: Dict[str, SSAOp]) -> str:
    op = def_map.get(ssa)
    if op and "declaretile" in op.op_type:
        col = op.attrs.get("col", "?")
        row = op.attrs.get("row", "?")
        try:
            tt = _tile_type(int(row))
        except ValueError:
            tt = "?"
        return f"tile({col},{row}) [{tt}]"
    return ssa


def _slice_info(op: SSAOp) -> str:
    # host.cc format: attrs have off0/size0
    if "off0" in op.attrs:
        o0, o1 = int(op.attrs.get("off0", 0)), int(op.attrs.get("off1", 0))
        s0, s1 = int(op.attrs.get("size0", 0)), int(op.attrs.get("size1", 0))
        return f"[{o0}:{o0 + s0}, {o1}:{o1 + s1}]"
    # MLIR format: from raw text
    m = re.search(r"\[(\d+),\s*(\d+)\]\s*\[(\d+),\s*(\d+)\]", op.raw)
    if m:
        o0, o1, s0, s1 = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        return f"[{o0}:{o0 + s0}, {o1}:{o1 + s1}]"
    return ""


def _result_type_pretty(op: SSAOp) -> str:
    m = re.search(r"to\s+(tensor<[^>]+>)", op.raw)
    if m:
        return m.group(1)
    m = re.search(r"->\s*(memref<[^>]+>|tensor<[^>]+>|![\w.]+)", op.raw)
    if m:
        return m.group(1)
    m = re.search(r":\s*(tensor<[^>]+>)", op.raw)
    if m:
        return m.group(1)
    return op.result_type


@dataclass
class BdLink:
    """A cross-reference between two DMA BD ops."""
    source: str
    target: str
    link_type: str  # "operand", "next_bd", or "self_loop"
    src_bd_id: str = "?"
    tgt_bd_id: str = "?"


def find_bd_links(
    ops: List[SSAOp], def_map: Dict[str, SSAOp],
) -> List[BdLink]:
    """Find BD-to-BD dependencies: operand refs and next_bd chains."""
    links: List[BdLink] = []
    seen = set()

    bd_ops = [op for op in ops if "config.dma_bd" in op.op_type and op.result]

    def _get_bd_id(op: SSAOp) -> str:
        bd = op.attrs.get("bd_id", "?")
        if bd in ("?", ""):
            if len(op.operands) >= 3:
                bd_op = def_map.get(op.operands[2])
                if bd_op:
                    cm = re.search(r"constant\s+(\d+)", bd_op.raw)
                    if cm:
                        bd = cm.group(1)
        return bd

    for op in bd_ops:
        for opr in op.operands:
            src_op = def_map.get(opr)
            if src_op and "config.dma_bd" in src_op.op_type:
                key = (op.result, opr)
                if key not in seen:
                    seen.add(key)
                    links.append(BdLink(source=opr, target=op.result,
                                        link_type="operand",
                                        src_bd_id=_get_bd_id(src_op),
                                        tgt_bd_id=_get_bd_id(op)))

    tile_bd_map: Dict[str, List[SSAOp]] = defaultdict(list)
    for op in bd_ops:
        tile_key = op.operands[1] if len(op.operands) >= 2 else ""
        if tile_key:
            tile_bd_map[tile_key].append(op)

    for tile_key, tile_bds in tile_bd_map.items():
        bd_id_to_var: Dict[str, str] = {}
        for op in tile_bds:
            bd_id = op.attrs.get("bd_id", "")
            if bd_id and bd_id not in ("?", "none"):
                bd_id_to_var[bd_id] = op.result

        for op in tile_bds:
            nxt = op.attrs.get("next_bd", "none")
            own_bd = op.attrs.get("bd_id", "")
            if own_bd in ("?", ""):
                own_bd = _get_bd_id(op)
            if nxt in ("none", "?", ""):
                continue
            if nxt == own_bd:
                key = (op.result, op.result, "self_loop")
                if key not in seen:
                    seen.add(key)
                    links.append(BdLink(source=op.result, target=op.result,
                                        link_type="self_loop",
                                        src_bd_id=own_bd, tgt_bd_id=own_bd))
                continue
            target_var = bd_id_to_var.get(nxt)
            if target_var and target_var != op.result:
                key = (op.result, target_var)
                if key not in seen:
                    seen.add(key)
                    links.append(BdLink(source=op.result, target=target_var,
                                        link_type="next_bd",
                                        src_bd_id=own_bd, tgt_bd_id=nxt))

    return links


_DATAFLOW_OPS = {
    "mem_allocate", "declare_data",
    "buffer_offset",
    "partitiontensor", "extract_slice", "declaretensor",
    "config.dma_bd", "config.create_io", "start_io",
    "lock_init",
}

_API_NAME_MAP = {
    "mem_allocate": "__Runtime_malloc",
    "buffer_offset": "__runtime_buffer_offset",
    "mem_free": "__Runtime_free",
    "declare_data": "XAie_MemGetVAddr",
    "partitiontensor": "__Runtime_init_PartitionTensor",
    "extract_slice": "__Runtime_extract_slice_*",
    "declaretensor": "__runtime_buffer_arg",
    "config.dma_bd": "__Runtime_dma_bd_config",
    "config.create_io": "__Runtime_dma_createio_4",
    "start_io": "__Runtime_startio",
    "load_kernel_group": "__Runtime_load_kernel_group",
    "launch_kernel_group": "__Runtime_launch_kernel_group",
    "wait": "__Runtime_wait",
    "declaretile": "XAie_TileLoc",
    "lock_init": "XAie_LockSetValue",
    "ooo_enable": "__Runtime_dma_channel_enable_ooo",
}


def _extract_api_call(raw: str) -> Tuple[str, str]:
    """Extract (function_name, args_string) from a raw C source line.

    Returns a truncated args string (max 60 chars) for readability.
    """
    m = re.search(r"(\w+)\s*\((.+)\)\s*;?\s*$", raw)
    if not m:
        return raw, ""
    func = m.group(1)
    args = m.group(2).strip()
    if len(args) > 60:
        args = args[:57] + "..."
    return func, args

_FIRST_OPERAND_OPS = {"config.dma_bd", "config.create_io", "start_io"}


def _build_subtree(
    op: SSAOp,
    def_map: Dict[str, SSAOp],
    use_map: Dict[str, List[SSAOp]],
    visited: set,
    view_mode: str = "ir",
) -> Optional[TreeNode]:
    key = op.result or id(op)
    if key in visited:
        return None
    visited.add(key)

    rtype = _result_type_pretty(op)

    # --- determine label / detail / node_type ---
    if "mem_allocate" in op.op_type:
        node_type = "declare_data"
        alloc_sz = op.attrs.get("alloc_size", "?")
        src = op.attrs.get("source_array", "")
        src_lbl = f" <- {src}" if src else ""
        label = f"__Runtime_malloc {op.result}{src_lbl}"
        detail = f"size={alloc_sz} bytes"
    elif "buffer_offset" in op.op_type:
        node_type = "extract_slice"
        base_var = op.operands[0] if op.operands else "?"
        off = op.attrs.get("offset", "0")
        bo = op.attrs.get("byte_offset", "")
        offset_lbl = f"  @byte {bo}" if bo else f"  +{off}B"
        label = f"buffer_offset {op.result}{offset_lbl}"
        detail = f"base={base_var} offset={off}B"
        br = op.attrs.get("byte_range", "")
        if br:
            detail += f"  byte_range={br}"
    elif "declare_data" in op.op_type:
        src = op.attrs.get("source_array", "")
        src_lbl = f" ({src})" if src else ""
        alloc_sz = op.attrs.get("alloc_size", "")
        alloc_lbl = f" [{alloc_sz}B]" if alloc_sz else ""
        node_type, label = "declare_data", f"declare_data {op.result}{src_lbl}{alloc_lbl}"
        br = op.attrs.get("byte_range", "")
        detail = f"byte_range={br}  {rtype}" if br else rtype
    elif "partitiontensor" in op.op_type:
        node_type = "partition"
        sn = op.attrs.get("splitnum", "?")
        sd = op.attrs.get("splitdim", "?")
        hw = op.attrs.get("hw_axis_owner", "?")
        oshape = op.attrs.get("orig_shape", "")
        pshape = op.attrs.get("part_shape", "")
        es = op.attrs.get("elem_size", "?")
        br = op.attrs.get("byte_range", "")
        label = f"partitiontensor {op.result}"
        detail = (f"shape={oshape} part={pshape} elem={es}B "
                  f"splitnum={sn} splitdim={sd} hw_axis={hw}")
        if br:
            detail += f"  byte_range={br}"
    elif "extract_slice" in op.op_type:
        node_type = "extract_slice"
        bo = op.attrs.get("byte_offset", "")
        br = op.attrs.get("byte_range", "")
        bl = op.attrs.get("byte_len", "")
        offset_lbl = f"  @byte {bo}" if bo else ""
        label = f"extract_slice {op.result} {_slice_info(op)}{offset_lbl}"
        detail = f"byte_range={br} len={bl}B" if br else rtype
    elif "declaretensor" in op.op_type:
        node_type = "declaretensor"
        bo = op.attrs.get("byte_offset", "")
        br = op.attrs.get("byte_range", "")
        offset_lbl = f"  @byte {bo}" if bo else ""
        label = f"declaretensor {op.result}{offset_lbl}"
        detail = f"byte_range={br}" if br else rtype
    elif "config.dma_bd" in op.op_type:
        node_type = "dma_bd"
        tile = _resolve_tile(op.operands[1], def_map) if len(op.operands) >= 2 else "?"
        bd_val = op.attrs.get("bd_id", "?")
        if bd_val == "?":
            if len(op.operands) >= 3:
                bd_op = def_map.get(op.operands[2])
                if bd_op:
                    cm = re.search(r"constant\s+(\d+)", bd_op.raw)
                    if cm:
                        bd_val = cm.group(1)
        ln = op.attrs.get("len", "?")
        pkt = op.attrs.get("packet_id", "?")
        nxt = op.attrs.get("next_bd", "?")
        if nxt == "4294967295":
            nxt = "none"
        ooo = op.attrs.get("ooo_bd_id", "none")
        en_pkt = op.attrs.get("enable_packet", "false")
        # Detect linked_bd: check if any operand is another dma_bd result
        linked_bd_src = ""
        for opr in op.operands:
            src_op = def_map.get(opr)
            if src_op and "config.dma_bd" in src_op.op_type:
                linked_bd_src = opr
                break
        label = f"dma_bd {op.result} @ {tile}"
        detail = f"bd={bd_val} len={ln} pkt={pkt} next_bd={nxt}"
        if ooo != "none":
            detail += f" ooo_target_bd={ooo}"
        if en_pkt == "true":
            detail += " [PKT]"
        acq_lk = op.attrs.get("acquire_lock_id", "")
        rel_lk = op.attrs.get("release_lock_id", "")
        if acq_lk and acq_lk not in ("-1", ""):
            acq_val = op.attrs.get("acquire_lock_val", "?")
            rel_val = op.attrs.get("release_lock_val", "?")
            detail += f" acq_lock={acq_lk}({acq_val}) rel_lock={rel_lk}({rel_val})"
        if linked_bd_src:
            detail += f" linked_bd=%{linked_bd_src}"
        if op.attrs.get("multidim") == "true":
            nd = op.attrs.get("num_dims", "?")
            ds = op.attrs.get("dim_strides", "?")
            dw = op.attrs.get("dim_wraps", "?")
            it_s = op.attrs.get("iter_step", "?")
            it_w = op.attrs.get("iter_wrap", "?")
            detail += f" [{nd}D strides=[{ds}] wraps=[{dw}] iter={it_s}/{it_w}]"
    elif "ooo_enable" in op.op_type:
        node_type = "other"
        tile = _resolve_tile(op.operands[0], def_map) if op.operands else "?"
        ch = op.attrs.get("channel", "?")
        d = op.attrs.get("direction", "?")
        label = f"OOO enable @ {tile}"
        detail = f"ch={ch} dir={d}"
    elif "create_io" in op.op_type:
        node_type = "create_io"
        tile = _resolve_tile(op.operands[1], def_map) if len(op.operands) >= 2 else "?"
        d = op.attrs.get("direction", "?")
        ch = op.attrs.get("channel", "?")
        io = op.attrs.get("io_operation", "?")
        start_bd = op.attrs.get("start_bd", "")
        ooo_en = op.attrs.get("ooo_enabled", "false")
        label = f"create_io {op.result} @ {tile}"
        detail = f"ch={ch} {d} {io}"
        if start_bd:
            detail += f" start_bd={start_bd}"
        if ooo_en == "true":
            detail += " [OOO]"
    elif "start_io" in op.op_type:
        node_type = "start_io"
        fi = op.attrs.get("flow_index", op.attrs.get("bd_id", "?"))
        rc = op.attrs.get("repeat_count", "1")
        label = f"start_io {op.result}"
        detail = f"target_bd={fi} repeat={rc}"
    elif "launch_kernel_group" in op.op_type:
        node_type = "kernel"
        label = f"launch_kernel_group {op.result}"
        detail = ""
    elif "load_kernel_group" in op.op_type:
        node_type = "kernel"
        label = f"load_kernel_group {op.result}"
        detail = ""
    elif "getbdid" in op.op_type:
        node_type = "getbdid"
        tile = _resolve_tile(op.operands[0], def_map) if op.operands else "?"
        label = f"getbdid {op.result} @ {tile}"
        detail = ""
    elif "wait" in op.op_type:
        node_type = "wait"
        label = "wait"
        detail = ", ".join(op.operands)
    elif "lock_init" in op.op_type:
        node_type = "other"
        tile = _resolve_tile(op.operands[0], def_map) if op.operands else "?"
        label = f"lock_init @ {tile}"
        lock_id = op.attrs.get("lock_id", "?")
        init_val = op.attrs.get("init_value", "?")
        detail = f"lock={lock_id} init_value={init_val}"
    else:
        node_type = "other"
        label = f"{op.op_type} {op.result}" if op.result else op.op_type
        detail = rtype

    # --- API view: override label/detail with actual C function names ---
    if view_mode == "api":
        func, args = _extract_api_call(op.raw)
        result_arrow = f" -> {op.result}" if op.result else ""
        label = f"{func}({args}){result_arrow}"
        br = op.attrs.get("byte_range", "")
        bo = op.attrs.get("byte_offset", "")
        offset_suffix = f"  | byte_range={br}" if br else ""
        raw_trunc = op.raw if len(op.raw) <= 100 else op.raw[:97] + "..."
        detail = f"{raw_trunc}{offset_suffix}"

    node = TreeNode(
        ssa_name=op.result,
        op=op,
        label=label,
        detail=detail,
        node_type=node_type,
    )

    # --- children: follow users in the dataflow chain ---
    if op.result and op.result in use_map:
        for user in use_map[op.result]:
            is_df = any(kw in user.op_type for kw in _DATAFLOW_OPS)
            if not is_df:
                continue
            if any(kw in user.op_type for kw in _FIRST_OPERAND_OPS):
                if not user.operands or user.operands[0] != op.result:
                    continue
            child = _build_subtree(user, def_map, use_map, visited, view_mode)
            if child:
                node.children.append(child)

    return node


def build_trees(
    ops: List[SSAOp], view_mode: str = "ir",
) -> Tuple[List[TreeNode], List[BdLink]]:
    """Build visualisation trees rooted at declare_data or partitiontensor ops.

    Returns (trees, bd_links) where bd_links are cross-references between
    dma_bd nodes (operand references and next_bd chains).
    """
    def_map: Dict[str, SSAOp] = {}
    use_map: Dict[str, List[SSAOp]] = defaultdict(list)

    for op in ops:
        if op.result:
            def_map[op.result] = op
        for operand in op.operands:
            use_map[operand].append(op)

    bd_links = find_bd_links(ops, def_map)

    trees: List[TreeNode] = []
    visited: set = set()

    # Prefer mem_allocate as root (chains: alloc -> buffer_offset -> declaretensor -> dma_bd -> ...)
    for op in ops:
        if op.op_type == "mem_allocate":
            t = _build_subtree(op, def_map, use_map, visited, view_mode)
            if t:
                trees.append(t)

    # Also root declaretensor ops whose operand is a literal (tile-local buffers,
    # e.g. __runtime_buffer_arg((void*)0)), not reachable from any DDR allocation.
    for op in ops:
        if op.op_type == "declaretensor":
            parent_var = op.operands[0] if op.operands else ""
            if parent_var not in def_map:  # literal, not an SSA result
                t = _build_subtree(op, def_map, use_map, visited, view_mode)
                if t:
                    trees.append(t)

    # Root buffer_offset ops whose base operand is a function argument (not in def_map).
    # This captures shim tile DMA chains that start from function args (e.g. v1, v2, v3)
    # rather than from mem_allocate or literal declaretensor ops.
    for op in ops:
        if op.op_type == "buffer_offset":
            base_var = op.operands[0] if op.operands else ""
            if base_var and base_var not in def_map:
                t = _build_subtree(op, def_map, use_map, visited, view_mode)
                if t:
                    trees.append(t)

    if not trees:
        for op in ops:
            if "declare_data" in op.op_type:
                t = _build_subtree(op, def_map, use_map, visited, view_mode)
                if t:
                    trees.append(t)

    if not trees:
        for op in ops:
            if "partitiontensor" in op.op_type:
                t = _build_subtree(op, def_map, use_map, visited, view_mode)
                if t:
                    trees.append(t)

    # Attach lock_init ops as siblings before the first dma_bd on the same tile
    if view_mode == "api":
        _attach_lock_init_nodes(trees, ops, def_map, view_mode)

    return trees, bd_links


def _attach_lock_init_nodes(
    trees: List[TreeNode],
    ops: List[SSAOp],
    def_map: Dict[str, SSAOp],
    view_mode: str,
) -> None:
    """Insert lock_init TreeNodes into the tree next to the first dma_bd
    for the same tile. Walks the tree and inserts lock nodes as siblings."""
    lock_ops = [op for op in ops if op.op_type == "lock_init"]
    if not lock_ops:
        return

    def _find_and_attach(node: TreeNode) -> None:
        new_children: List[TreeNode] = []
        for child in node.children:
            if child.op and "config.dma_bd" in child.op.op_type:
                tile_var = child.op.operands[1] if len(child.op.operands) >= 2 else ""
                tile_op = def_map.get(tile_var)
                if tile_op:
                    tc = tile_op.attrs.get("col", "")
                    tr = tile_op.attrs.get("row", "")
                    for lock in lock_ops:
                        if (lock.attrs.get("col") == tc
                                and lock.attrs.get("row") == tr):
                            func, args = _extract_api_call(lock.raw)
                            lk_node = TreeNode(
                                ssa_name="",
                                op=lock,
                                label=f"{func}({args})",
                                detail=lock.raw,
                                node_type="other",
                            )
                            if lk_node not in new_children:
                                new_children.append(lk_node)
                            lock_ops.remove(lock)
                            break
            new_children.append(child)
            _find_and_attach(child)
        node.children = new_children

    for tree in trees:
        _find_and_attach(tree)


# ---------------------------------------------------------------------------
# 4. Text renderer
# ---------------------------------------------------------------------------

def render_text(trees: List[TreeNode], out, view_mode: str = "ir") -> None:
    title = ("host_canonicalized API Call Graph" if view_mode == "api"
             else "host_canonicalized Dataflow Analysis")
    out.write("=" * 72 + "\n")
    out.write(f"  {title}\n")
    out.write("=" * 72 + "\n\n")
    for tree in trees:
        _text_node(tree, out, "", True)
    out.write("\n")


def _text_node(node: TreeNode, out, prefix: str, is_last: bool) -> None:
    connector = "└── " if is_last else "├── "
    out.write(f"{prefix}{connector}{node.label}\n")
    if node.detail:
        ext = "    " if is_last else "│   "
        out.write(f"{prefix}{ext}    {node.detail}\n")
    child_pfx = prefix + ("    " if is_last else "│   ")
    for i, ch in enumerate(node.children):
        _text_node(ch, out, child_pfx, i == len(node.children) - 1)


# ---------------------------------------------------------------------------
# 5. HTML renderer
# ---------------------------------------------------------------------------

_OP_COLORS = {
    "declare_data":  ("#E3F2FD", "#1565C0"),
    "partition":     ("#F3E5F5", "#7B1FA2"),
    "extract_slice": ("#E8F5E9", "#2E7D32"),
    "declaretensor": ("#FFF3E0", "#E65100"),
    "dma_bd":        ("#FCE4EC", "#C62828"),
    "dma_bd_linked": ("#F3E5F5", "#6A1B9A"),
    "dma_bd_nextbd": ("#FFF8E1", "#E65100"),
    "create_io":     ("#E0F7FA", "#00695C"),
    "start_io":      ("#FFF9C4", "#F57F17"),
    "kernel":        ("#EDE7F6", "#4527A0"),
    "getbdid":       ("#EFEBE9", "#4E342E"),
    "wait":          ("#ECEFF1", "#37474F"),
    "constant":      ("#F5F5F5", "#757575"),
    "other":         ("#FAFAFA", "#424242"),
}


_NODE_ID_CTR = [0]


def _node_html(node: TreeNode) -> str:
    _NODE_ID_CTR[0] += 1
    nid = _NODE_ID_CTR[0]
    bg, fg = _OP_COLORS.get(node.node_type, _OP_COLORS["other"])
    has_ch = " has-children" if node.children else ""
    detail = (
        f'<div class="node-detail">{_esc(node.detail)}</div>'
        if node.detail else ""
    )
    children = ""
    if node.children:
        items = "\n".join(_node_html(c) for c in node.children)
        children = f'<div class="children">{items}</div>'

    bd_attr = ""
    if node.node_type == "dma_bd" and node.ssa_name:
        bd_attr = f' data-bd-var="{node.ssa_name}"'

    return (
        f'<div class="tree-node{has_ch}" data-nid="{nid}" data-type="{node.node_type}"{bd_attr}>'
        f'<div class="node-card" style="background:{bg};border-left-color:{fg};color:{fg}">'
        f'<span class="node-label">{_esc(node.label)}</span>'
        f'{detail}'
        f'<span class="line-badge">L{node.op.line_num}</span>'
        f"</div>"
        f"{children}"
        f"</div>"
    )


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_html(trees: List[TreeNode], output_path: str,
                bd_links: Optional[List[BdLink]] = None,
                view_mode: str = "ir") -> str:
    _NODE_ID_CTR[0] = 0
    tree_items = "\n".join(_node_html(t) for t in trees)
    html_title = ("host_canonicalized API Call Graph" if view_mode == "api"
                  else "host_canonicalized Dataflow Analysis")

    bd_links = bd_links or []
    bd_links_json = ", ".join(
        f'{{"src": "{lk.source}", "tgt": "{lk.target}", "type": "{lk.link_type}",'
        f' "srcBd": "{lk.src_bd_id}", "tgtBd": "{lk.tgt_bd_id}"}}'
        for lk in bd_links
    )

    _LEGEND_LABELS = {
        "dma_bd_linked": "dma_bd (linked_bd)",
        "dma_bd_nextbd": "dma_bd (next_bd chain)",
    }
    legend_items = []
    for name, (bg, fg) in _OP_COLORS.items():
        if name in ("constant", "other"):
            continue
        display = _LEGEND_LABELS.get(name, name)
        legend_items.append(
            f'<div class="legend-item">'
            f'<div class="legend-swatch" style="background:{bg};border-color:{fg}"></div>'
            f"{display}</div>"
        )
    legend_html = "\n".join(legend_items)

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html_title}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f5f5f5;
    padding: 24px;
    color: #333;
}}
h1 {{
    text-align: center;
    margin-bottom: 12px;
    font-size: 20px;
    color: #333;
}}
.toolbar {{
    text-align: center;
    margin-bottom: 14px;
}}
.toolbar button {{
    padding: 5px 14px;
    margin: 0 4px;
    border: 1px solid #bbb;
    border-radius: 4px;
    background: #fff;
    cursor: pointer;
    font-size: 12px;
}}
.toolbar button:hover {{
    background: #e0e0e0;
}}

/* Legend */
.legend {{
    display: flex;
    gap: 14px;
    justify-content: center;
    margin-bottom: 18px;
    flex-wrap: wrap;
    font-size: 12px;
}}
.legend-item {{
    display: flex;
    align-items: center;
    gap: 4px;
}}
.legend-swatch {{
    width: 14px;
    height: 14px;
    border-radius: 3px;
    border: 2px solid;
}}

/* Scroll wrapper */
.tree-scroll {{
    overflow-x: auto;
    padding-bottom: 16px;
}}

/* Horizontal tree (left to right) */
.tree-container {{
    position: relative;
    display: inline-block;
    min-width: 100%;
}}
#connectors {{
    position: absolute;
    top: 0;
    left: 0;
    z-index: 10;
    pointer-events: none;
}}

.tree-node {{
    display: flex;
    flex-direction: row;
    align-items: center;
}}
.children {{
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-left: 48px;
    padding: 4px 0;
}}

/* Node card */
.node-card {{
    display: inline-block;
    padding: 7px 14px 7px 12px;
    border-radius: 6px;
    border-left: 5px solid;
    font-size: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,.1);
    transition: box-shadow .15s;
    position: relative;
    white-space: nowrap;
    flex-shrink: 0;
    cursor: default;
}}
.has-children > .node-card {{
    cursor: pointer;
}}
.node-card:hover {{
    box-shadow: 0 3px 12px rgba(0,0,0,.2);
}}
.node-label {{
    font-weight: 700;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
}}
.node-detail {{
    font-size: 11px;
    margin-top: 3px;
    opacity: .75;
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
}}
.line-badge {{
    font-size: 9px;
    position: absolute;
    top: 3px;
    right: 6px;
    opacity: .45;
}}

/* Collapse / expand */
.collapsed > .children {{
    display: none;
}}
.collapse-icon {{
    display: inline-block;
    font-size: 9px;
    margin-left: 6px;
    transition: transform .2s;
    opacity: .45;
}}
.collapsed > .node-card .collapse-icon {{
    transform: rotate(-90deg);
}}
</style>
</head>
<body>
<h1>{html_title}</h1>
<div class="toolbar">
    <button onclick="expandAll()">Expand All</button>
    <button onclick="collapseAll()">Collapse All</button>
</div>
<div class="legend">
{legend_html}
</div>
<div class="tree-scroll">
<div class="tree-container" id="treeContainer">
<svg id="connectors">
<defs>
<marker id="arrowOperand" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
  <polygon points="0 0, 8 3, 0 6" fill="#6A1B9A"/>
</marker>
<marker id="arrowNextBd" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
  <polygon points="0 0, 8 3, 0 6" fill="#E65100"/>
</marker>
<marker id="arrowSelfLoop" markerWidth="6" markerHeight="5" refX="5" refY="2.5" orient="auto">
  <polygon points="0 0, 6 2.5, 0 5" fill="#FF6F00"/>
</marker>
</defs>
</svg>
{tree_items}
</div>
</div>
<script>
var bdLinks = [{bd_links_json}];

/* --- collapse / expand --- */
document.querySelectorAll('.has-children > .node-card').forEach(function(card) {{
    var icon = document.createElement('span');
    icon.className = 'collapse-icon';
    icon.textContent = '\\u25B6';
    card.appendChild(icon);
    card.addEventListener('click', function() {{
        card.parentElement.classList.toggle('collapsed');
        redrawAll();
    }});
}});
function expandAll() {{
    document.querySelectorAll('.collapsed').forEach(function(n) {{
        n.classList.remove('collapsed');
    }});
    redrawAll();
}}
function collapseAll() {{
    document.querySelectorAll('.has-children').forEach(function(n) {{
        n.classList.add('collapsed');
    }});
    redrawAll();
}}

function redrawAll() {{
    drawConnectors();
    drawBdLinks();
}}

/* --- SVG curved connectors (tree edges) --- */
function drawConnectors() {{
    var svg = document.getElementById('connectors');
    var container = document.getElementById('treeContainer');
    var cRect = container.getBoundingClientRect();

    svg.setAttribute('width', container.scrollWidth);
    svg.setAttribute('height', container.scrollHeight);

    /* Remove only tree-edge paths, preserve defs and bd-link paths */
    svg.querySelectorAll('path.tree-edge').forEach(function(p) {{ p.remove(); }});

    document.querySelectorAll('.has-children:not(.collapsed)').forEach(function(node) {{
        var parentCard = node.querySelector(':scope > .node-card');
        var childCards = node.querySelectorAll(':scope > .children > .tree-node > .node-card');
        if (!parentCard || !childCards.length) return;

        var pR = parentCard.getBoundingClientRect();
        var px = pR.right - cRect.left;
        var py = pR.top + pR.height / 2 - cRect.top;

        childCards.forEach(function(cc) {{
            var cR = cc.getBoundingClientRect();
            var cx = cR.left - cRect.left;
            var cy = cR.top + cR.height / 2 - cRect.top;
            var dx = (cx - px);
            var cpx = px + dx * 0.5;

            var path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            path.classList.add('tree-edge');
            path.setAttribute('d',
                'M ' + px + ' ' + py +
                ' C ' + cpx + ' ' + py + ', ' + cpx + ' ' + cy + ', ' + cx + ' ' + cy);
            path.setAttribute('fill', 'none');
            path.setAttribute('stroke', '#b0b0b0');
            path.setAttribute('stroke-width', '2');
            svg.appendChild(path);
        }});
    }});
}}

/* --- SVG cross-link lines between dma_bd nodes --- */
function drawBdLinks() {{
    var svg = document.getElementById('connectors');
    var container = document.getElementById('treeContainer');
    var cRect = container.getBoundingClientRect();

    svg.setAttribute('width', container.scrollWidth);
    svg.setAttribute('height', container.scrollHeight);

    svg.querySelectorAll('.bd-link').forEach(function(p) {{ p.remove(); }});

    function addLabel(svg, cx, cy, text, bg, fg) {{
        var g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        g.classList.add('bd-link');
        var txt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        txt.setAttribute('x', cx);
        txt.setAttribute('y', cy + 4);
        txt.setAttribute('text-anchor', 'middle');
        txt.setAttribute('font-size', '10');
        txt.setAttribute('font-family', 'monospace');
        txt.setAttribute('fill', fg);
        txt.textContent = text;
        var bbox_w = text.length * 6.5 + 8;
        var bbox_h = 16;
        var rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', cx - bbox_w / 2);
        rect.setAttribute('y', cy - bbox_h / 2);
        rect.setAttribute('width', bbox_w);
        rect.setAttribute('height', bbox_h);
        rect.setAttribute('rx', '3');
        rect.setAttribute('fill', bg);
        rect.setAttribute('stroke', fg);
        rect.setAttribute('stroke-width', '1');
        rect.setAttribute('opacity', '0.92');
        g.appendChild(rect);
        g.appendChild(txt);
        svg.appendChild(g);
    }}

    var pairSeen = {{}};

    bdLinks.forEach(function(link) {{
        var srcEl = document.querySelector('[data-bd-var="' + link.src + '"] > .node-card');
        var tgtEl = document.querySelector('[data-bd-var="' + link.tgt + '"] > .node-card');
        if (!srcEl || !tgtEl) return;

        if (srcEl.offsetParent === null || tgtEl.offsetParent === null) return;

        var sR = srcEl.getBoundingClientRect();
        var tR = tgtEl.getBoundingClientRect();

        var path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.classList.add('bd-link');
        path.setAttribute('fill', 'none');

        if (link.type === 'self_loop') {{
            var x = sR.right - cRect.left;
            var y = sR.top + sR.height / 2 - cRect.top;
            var r = 20;
            path.setAttribute('d',
                'M ' + x + ' ' + (y - 10) +
                ' A ' + r + ' ' + r + ' 0 1 1 ' + x + ' ' + (y + 10));
            path.setAttribute('stroke', '#FF6F00');
            path.setAttribute('stroke-width', '2');
            path.setAttribute('marker-end', 'url(#arrowSelfLoop)');
            path.setAttribute('opacity', '0.85');
            svg.appendChild(path);
            var labelText = 'BD' + link.srcBd + '\\u2192BD' + link.tgtBd;
            addLabel(svg, x + r + 6, y, labelText, '#FFF3E0', '#E65100');
        }} else if (link.type === 'operand') {{
            /* operand link: arc touches left edge of cards, curves left */
            var sx = sR.left - cRect.left;
            var sy = sR.top + sR.height / 2 - cRect.top;
            var tx = tR.left - cRect.left;
            var ty = tR.top + tR.height / 2 - cRect.top;
            var dy = Math.abs(ty - sy);
            var r = Math.max(70, dy * 0.7);
            path.setAttribute('d',
                'M ' + sx + ' ' + sy +
                ' A ' + r + ' ' + r + ' 0 0 0 ' + tx + ' ' + ty);
            path.setAttribute('stroke', '#6A1B9A');
            path.setAttribute('stroke-width', '2.5');
            path.setAttribute('stroke-dasharray', '6,3');
            path.setAttribute('marker-end', 'url(#arrowOperand)');
            svg.appendChild(path);
            var labelText = 'BD' + link.srcBd + '\\u2192BD' + link.tgtBd;
            var labelX = Math.min(sx, tx) - r * 0.5;
            var labelY = (sy + ty) / 2;
            addLabel(svg, labelX, labelY, labelText, '#F3E5F5', '#6A1B9A');
        }} else {{
            /* next_bd chain: alternate sides for reverse pairs */
            var pairKey = [link.src, link.tgt].sort().join('::');
            var goRight = !pairSeen[pairKey];
            pairSeen[pairKey] = true;

            var sy = sR.top + sR.height / 2 - cRect.top;
            var ty = tR.top + tR.height / 2 - cRect.top;
            var dy = Math.abs(ty - sy);
            var r = Math.max(70, dy * 0.7);

            if (goRight) {{
                /* first link: touches right edge, curves right */
                var sx = sR.right - cRect.left;
                var tx = tR.right - cRect.left;
                path.setAttribute('d',
                    'M ' + sx + ' ' + sy +
                    ' A ' + r + ' ' + r + ' 0 0 1 ' + tx + ' ' + ty);
                svg.appendChild(path);
                var labelX = Math.max(sx, tx) + r * 0.5;
            }} else {{
                /* second link: touches left edge, curves left (mirrored) */
                var sx = sR.left - cRect.left;
                var tx = tR.left - cRect.left;
                path.setAttribute('d',
                    'M ' + sx + ' ' + sy +
                    ' A ' + r + ' ' + r + ' 0 0 0 ' + tx + ' ' + ty);
                svg.appendChild(path);
                var labelX = Math.min(sx, tx) - r * 0.5;
            }}
            path.setAttribute('stroke', '#E65100');
            path.setAttribute('stroke-width', '2');
            path.setAttribute('marker-end', 'url(#arrowNextBd)');
            var labelText = 'BD' + link.srcBd + '\\u2192BD' + link.tgtBd;
            var labelY = (sy + ty) / 2;
            addLabel(svg, labelX, labelY, labelText, '#FFF3E0', '#E65100');
        }}
    }});
}}

/* initial draw */
redrawAll();
window.addEventListener('resize', redrawAll);
</script>
</body>
</html>"""

    Path(output_path).write_text(html)
    print(f"HTML saved to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# 6. HTTP server
# ---------------------------------------------------------------------------

def serve_html(html_path: str, port: int) -> None:
    abs_path = Path(html_path).resolve()
    directory = str(abs_path.parent)
    filename = abs_path.name

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=directory, **kw)

        def log_message(self, fmt, *args):
            pass

    hostname = socket.gethostname()
    local_url = f"http://localhost:{port}/{filename}"
    remote_url = f"http://{hostname}:{port}/{filename}"

    print("Serving host_canonicalized dataflow visualization:")
    print(f"  Local:  {local_url}")
    print(f"  Remote: {remote_url}")
    print("  (Ctrl+C to stop)")

    server = http.server.HTTPServer(("0.0.0.0", port), Handler)
    try:
        webbrowser.open(local_url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.shutdown()


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze host_canonicalized dataflow from host.cc or MLIR IR "
            "dump (logdf). Auto-detects file format."
        ),
    )
    parser.add_argument(
        "input_file", nargs="?", default=None,
        help=f"Path to host.cc or logdf file (default: {_DEFAULT_INPUT.name} relative to repo root)",
    )
    parser.add_argument(
        "-m", "--mode", choices=["text", "html"], default=None,
        help="Output mode (default: html + serve)",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output file path (default: auto-named)",
    )
    parser.add_argument(
        "--no-serve", action="store_true",
        help="Generate HTML file only, do not start HTTP server",
    )
    parser.add_argument(
        "--port", type=int, default=8089,
        help="HTTP server port (default: 8089)",
    )
    parser.add_argument(
        "--view", choices=["ir", "api"], default="ir",
        help="View mode: ir (default, abstract dataflow) or api (C API function names)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file) if args.input_file else _DEFAULT_INPUT
    if not input_path.exists():
        print(f"File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    text = input_path.read_text()
    ops, fmt = detect_and_parse(text)
    if not ops:
        print(f"No host_canonicalized data found in {input_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsed {input_path} as {fmt} ({len(ops)} ops)")

    view_mode = args.view

    trees, bd_links = build_trees(ops, view_mode=view_mode)
    if not trees:
        print("No dataflow chains found.", file=sys.stderr)
        sys.exit(1)

    if bd_links:
        print(f"  Found {len(bd_links)} BD cross-link(s)")

    mode = args.mode or "html"

    if mode == "text":
        if args.output:
            with open(args.output, "w") as f:
                render_text(trees, f, view_mode=view_mode)
            print(f"Text report saved to {args.output}")
        else:
            render_text(trees, sys.stdout, view_mode=view_mode)

    elif mode == "html":
        out_path = args.output or "host_canonicalized_dataflow.html"
        render_html(trees, out_path, bd_links=bd_links, view_mode=view_mode)
        if not args.no_serve:
            serve_html(out_path, args.port)


if __name__ == "__main__":
    main()
