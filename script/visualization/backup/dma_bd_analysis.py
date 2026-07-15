#!/usr/bin/env python3
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Parse a generated host.cc file and visualize DMA Buffer Descriptor (BD)
configurations per tile, including ping-pong chaining, packet IDs, IO
channels, and buffer sources.

Usage:
    python dma_bd_analysis.py <host.cc>                        # text (default)
    python dma_bd_analysis.py <host.cc> -m png -o out.png      # PNG diagram
    python dma_bd_analysis.py <host.cc> -m html                # HTML file
    python dma_bd_analysis.py <host.cc> -m html --serve        # HTML + HTTP server
"""

import argparse
import http.server
import os
import re
import sys
import threading
import webbrowser
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LockInfo:
    lock_id: int
    init_value: Optional[int] = None       # from XAie_LockSetValue
    operate_value: Optional[int] = None    # from XAie_DmaSetLock acquire/release value

@dataclass
class BdConfig:
    var: str
    tile_var: str
    tile_loc: Tuple[int, int]  # (col, row)
    buf_var: str               # raw buffer pointer var
    buf_source: str            # PartitionTensor var that buf_var came from
    bd_id: int
    offset: int
    length: int
    next_bd: int               # -1 means no chaining
    enable_packet: bool
    packet_id: int
    acquire_lock_id: int = -1  # -1 means unset / no lock
    acquire_lock_val: int = -1 # DMA lock operate value for acquire
    release_lock_id: int = -1  # -1 means unset / no lock
    release_lock_val: int = -1 # DMA lock operate value for release

@dataclass
class IoConfig:
    var: str
    tile_var: str
    tile_loc: Tuple[int, int]
    bd_var: str
    channel_id: int
    direction: str  # "S2MM" or "MM2S"

@dataclass
class SliceInfo:
    var: str
    source_var: str
    off0: int
    off1: int
    size0: int
    size1: int

@dataclass
class TileDmaInfo:
    loc: Tuple[int, int]
    tile_type: str  # "Shim", "Mem", "AIE"
    bds: List[BdConfig] = field(default_factory=list)
    ios: List[IoConfig] = field(default_factory=list)
    locks: Dict[int, LockInfo] = field(default_factory=dict)  # lock_id -> LockInfo

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _tile_type(row: int) -> str:
    if row == 0:
        return "Shim"
    if row <= 2:
        return "Mem"
    return "AIE"

def parse_host_cc(path: str) -> Dict[Tuple[int, int], TileDmaInfo]:
    text = Path(path).read_text()

    # 1. Map variable names to tile locations
    tile_map: Dict[str, Tuple[int, int]] = {}
    for m in re.finditer(
        r"XAie_LocType\s+(\w+)\s*=\s*XAie_TileLoc\((\d+),\s*(\d+)\)", text
    ):
        tile_map[m.group(1)] = (int(m.group(2)), int(m.group(3)))

    # 2. Map __runtime_buffer_arg(vX) -> vX  (raw ptr var -> PT var)
    buf_source: Dict[str, str] = {}
    for m in re.finditer(
        r"void\*\s+(\w+)\s*=\s*__runtime_buffer_arg\((\w+)\)", text
    ):
        buf_source[m.group(1)] = m.group(2)

    # 3. Parse slice hierarchy for labelling
    slices: Dict[str, SliceInfo] = {}
    for m in re.finditer(
        r"PartitionTensor\s+(\w+)\s*=\s*__Runtime_extract_slice_contiguous_2d"
        r"\((\w+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)",
        text,
    ):
        slices[m.group(1)] = SliceInfo(
            var=m.group(1),
            source_var=m.group(2),
            off0=int(m.group(3)),
            off1=int(m.group(4)),
            size0=int(m.group(5)),
            size1=int(m.group(6)),
        )

    def _slice_label(var: str) -> str:
        if var in slices:
            s = slices[var]
            return f"{var}[{s.off0}:{s.off0+s.size0}, {s.off1}:{s.off1+s.size1}]"
        return var

    # 4. Parse lock IDs and values from DMA BD Config comments
    #    Format: /* DMA BD Config: ..., acquire_lock_id=Y, acquire_lock_val=V, release_lock_id=Z, release_lock_val=W */
    bd_comment_locks: Dict[str, Tuple[int, int, int, int]] = {}  # bd_var -> (acq_id, acq_val, rel_id, rel_val)
    bd_comment_pattern = re.compile(
        r"/\*\s*DMA BD Config:.*?"
        r"acquire_lock_id=(-?\d+).*?"
        r"acquire_lock_val=(-?\d+).*?"
        r"release_lock_id=(-?\d+).*?"
        r"release_lock_val=(-?\d+).*?\*/",
        re.DOTALL,
    )
    bd_comment_pattern_old = re.compile(
        r"/\*\s*DMA BD Config:.*?"
        r"acquire_lock_id=(-?\d+).*?"
        r"release_lock_id=(-?\d+).*?\*/",
        re.DOTALL,
    )
    bd_call_pattern = re.compile(
        r"XAie_DmaDesc\s+(\w+)\s*=\s*__Runtime_dma_bd_config\("
    )
    for cm in bd_comment_pattern.finditer(text):
        acq_id, acq_val = int(cm.group(1)), int(cm.group(2))
        rel_id, rel_val = int(cm.group(3)), int(cm.group(4))
        after = text[cm.end():]
        call_m = bd_call_pattern.search(after)
        if call_m:
            bd_comment_locks[call_m.group(1)] = (acq_id, acq_val, rel_id, rel_val)
    # Fallback: old comment format without lock_val
    for cm in bd_comment_pattern_old.finditer(text):
        acq_id, rel_id = int(cm.group(1)), int(cm.group(2))
        after = text[cm.end():]
        call_m = bd_call_pattern.search(after)
        if call_m and call_m.group(1) not in bd_comment_locks:
            bd_comment_locks[call_m.group(1)] = (acq_id, -1, rel_id, -1)

    # 5. Parse XAie_LockSetValue calls for lock init values
    #    Pattern A: XAie_LockSetValue(dev, XAie_TileLoc(col, row), XAie_LockInit(lock_id, value))
    #    Pattern B: XAie_LockSetValue(dev, tile_var, XAie_LockInit(lock_id, value))
    lock_init_map: Dict[Tuple[int, int], Dict[int, int]] = defaultdict(dict)
    for m in re.finditer(
        r"XAie_LockSetValue\(\s*\w+,\s*XAie_TileLoc\(\s*(\d+),\s*(\d+)\s*\),"
        r"\s*XAie_LockInit\(\s*(\d+),\s*(\d+)\s*\)\)",
        text,
    ):
        loc = (int(m.group(1)), int(m.group(2)))
        lock_id = int(m.group(3))
        init_val = int(m.group(4))
        lock_init_map[loc][lock_id] = init_val
    for m in re.finditer(
        r"XAie_LockSetValue\(\s*\w+,\s*(\w+),\s*XAie_LockInit\(\s*(\d+),\s*(\d+)\s*\)\)",
        text,
    ):
        tvar = m.group(1)
        if tvar.startswith("XAie_TileLoc"):
            continue
        loc = tile_map.get(tvar, (-1, -1))
        lock_id = int(m.group(2))
        init_val = int(m.group(3))
        lock_init_map[loc][lock_id] = init_val

    # 6. Parse XAie_DmaSetLock calls for lock operate values
    #    XAie_DmaSetLock(&desc, XAie_LockInit(lock_id, value), XAie_LockInit(lock_id, value))
    lock_operate_map: Dict[str, Tuple[int, int]] = {}  # bd_var -> (acq_val, rel_val)
    for m in re.finditer(
        r"XAie_DmaSetLock\(\s*&(\w+).*?"
        r"XAie_LockInit\(\s*(-?\d+),\s*(-?\d+)\s*\).*?"
        r"XAie_LockInit\(\s*(-?\d+),\s*(-?\d+)\s*\)",
        text,
        re.DOTALL,
    ):
        bd_var = m.group(1)
        lock_operate_map[bd_var] = (int(m.group(3)), int(m.group(5)))

    # 7. Parse DMA BD configs
    #    New 13-arg format: dev, tile, buf, bd_id, addr, len, next_bd, enable_pkt, pkt_id,
    #                       acq_lock_id, acq_lock_val, rel_lock_id, rel_lock_val
    bd_map: Dict[str, BdConfig] = {}
    bd_call_13 = re.compile(
        r"XAie_DmaDesc\s+(\w+)\s*=\s*__Runtime_dma_bd_config\("
        r"(\w+),\s*(\w+),\s*(\w+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),"
        r"\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),"
        r"\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\)"
    )
    bd_call_10 = re.compile(
        r"XAie_DmaDesc\s+(\w+)\s*=\s*__Runtime_dma_bd_config\("
        r"(\w+),\s*(\w+),\s*(\w+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),"
        r"\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\)"
    )
    for m in bd_call_13.finditer(text):
        tile_var = m.group(3)
        buf_ptr = m.group(4)
        bd_var = m.group(1)
        loc = tile_map.get(tile_var, (-1, -1))
        next_bd_raw = int(m.group(8))
        if next_bd_raw < 0 or next_bd_raw > 65535:
            next_bd_raw = -1

        bd = BdConfig(
            var=bd_var, tile_var=tile_var, tile_loc=loc,
            buf_var=buf_ptr,
            buf_source=_slice_label(buf_source.get(buf_ptr, buf_ptr)),
            bd_id=int(m.group(5)), offset=int(m.group(6)), length=int(m.group(7)),
            next_bd=next_bd_raw,
            enable_packet=int(m.group(9)) != 0, packet_id=int(m.group(10)),
            acquire_lock_id=int(m.group(11)), acquire_lock_val=int(m.group(12)),
            release_lock_id=int(m.group(13)), release_lock_val=int(m.group(14)),
        )
        bd_map[bd.var] = bd

    # Fallback: old 10-arg format (lock info from comments)
    for m in bd_call_10.finditer(text):
        bd_var = m.group(1)
        if bd_var in bd_map:
            continue
        tile_var = m.group(3)
        buf_ptr = m.group(4)
        loc = tile_map.get(tile_var, (-1, -1))
        next_bd_raw = int(m.group(8))
        if next_bd_raw < 0 or next_bd_raw > 65535:
            next_bd_raw = -1
        acq_lock, acq_val, rel_lock, rel_val = -1, -1, -1, -1
        if bd_var in bd_comment_locks:
            acq_lock, acq_val, rel_lock, rel_val = bd_comment_locks[bd_var]
        bd = BdConfig(
            var=bd_var, tile_var=tile_var, tile_loc=loc,
            buf_var=buf_ptr,
            buf_source=_slice_label(buf_source.get(buf_ptr, buf_ptr)),
            bd_id=int(m.group(5)), offset=int(m.group(6)), length=int(m.group(7)),
            next_bd=next_bd_raw,
            enable_packet=int(m.group(9)) != 0, packet_id=int(m.group(10)),
            acquire_lock_id=acq_lock, acquire_lock_val=acq_val,
            release_lock_id=rel_lock, release_lock_val=rel_val,
        )
        bd_map[bd.var] = bd

    # 8. Parse IO configs -- also extract direction from preceding comment
    io_comment = re.compile(
        r"/\*\s*Create IO:.*?direction=(\w+)\s*\*/", re.DOTALL
    )
    io_call = re.compile(
        r"io\s+(\w+)\s*=\s*__Runtime_dma_createio_4\("
        r"(\w+),\s*(\w+),\s*(-?\d+),\s*(-?\d+)\)"
    )
    io_configs: List[IoConfig] = []
    for cm in io_comment.finditer(text):
        direction = cm.group(1)
        after = text[cm.end():]
        call_m = io_call.search(after)
        if call_m:
            tile_var = call_m.group(2)
            loc = tile_map.get(tile_var, (-1, -1))
            io_configs.append(IoConfig(
                var=call_m.group(1),
                tile_var=tile_var,
                tile_loc=loc,
                bd_var=call_m.group(3),
                channel_id=int(call_m.group(4)),
                direction=direction,
            ))

    # 9. Assemble per-tile info
    tiles: Dict[Tuple[int, int], TileDmaInfo] = {}
    for bd in bd_map.values():
        loc = bd.tile_loc
        if loc not in tiles:
            tiles[loc] = TileDmaInfo(loc=loc, tile_type=_tile_type(loc[1]))
        tiles[loc].bds.append(bd)
    for io in io_configs:
        loc = io.tile_loc
        if loc not in tiles:
            tiles[loc] = TileDmaInfo(loc=loc, tile_type=_tile_type(loc[1]))
        tiles[loc].ios.append(io)

    for info in tiles.values():
        info.bds.sort(key=lambda b: b.bd_id)

    # 10. Populate per-tile lock info from BD lock IDs/values and init maps
    for info in tiles.values():
        loc = info.loc
        for bd in info.bds:
            # Register acquire lock
            if bd.acquire_lock_id >= 0:
                if bd.acquire_lock_id not in info.locks:
                    info.locks[bd.acquire_lock_id] = LockInfo(lock_id=bd.acquire_lock_id)
                info.locks[bd.acquire_lock_id].operate_value = bd.acquire_lock_val
            # Register release lock
            if bd.release_lock_id >= 0:
                if bd.release_lock_id not in info.locks:
                    info.locks[bd.release_lock_id] = LockInfo(lock_id=bd.release_lock_id)
                info.locks[bd.release_lock_id].operate_value = bd.release_lock_val
            # Fallback: XAie_DmaSetLock operate values
            if bd.var in lock_operate_map:
                acq_val, rel_val = lock_operate_map[bd.var]
                if bd.acquire_lock_id >= 0:
                    info.locks[bd.acquire_lock_id].operate_value = acq_val
                if bd.release_lock_id >= 0:
                    info.locks[bd.release_lock_id].operate_value = rel_val
        # Populate init values from XAie_LockSetValue
        for lid, lk in info.locks.items():
            if loc in lock_init_map and lid in lock_init_map[loc]:
                lk.init_value = lock_init_map[loc][lid]

    return tiles

# ---------------------------------------------------------------------------
# Text output
# ---------------------------------------------------------------------------

def _lock_dma_str(lock_id: int, lock_val: int) -> str:
    """Format lock ID + DMA request value from __Runtime_dma_bd_config args."""
    if lock_id < 0:
        return "none"
    return f"lock={lock_id} val={lock_val}"

def _lock_init_str(lock_id: int, locks: Dict[int, 'LockInfo']) -> str:
    """Format lock init value (from XAie_LockSetValue)."""
    if lock_id < 0:
        return ""
    lk = locks.get(lock_id)
    if lk and lk.init_value is not None:
        return f"XAie_LockSetValue(init={lk.init_value})"
    return ""


def render_text(tiles: Dict[Tuple[int, int], TileDmaInfo], out) -> None:
    out.write("=" * 70 + "\n")
    out.write("  DMA Buffer Descriptor Configuration Report\n")
    out.write("=" * 70 + "\n\n")

    for loc in sorted(tiles, key=lambda k: (k[1], k[0])):
        info = tiles[loc]
        header = f"Tile ({loc[0]},{loc[1]}) [{info.tile_type}]"
        out.write(f"--- {header} {'─' * (60 - len(header))}---\n")

        if info.ios:
            dirs = ", ".join(
                f"ch{io.channel_id}:{io.direction}" for io in info.ios
            )
            out.write(f"  IO Channels: {dirs}\n")

        if info.bds:
            out.write("  BD Configurations:\n")
            for bd in info.bds:
                chain = (
                    f" -> BD{bd.next_bd}" if bd.next_bd >= 0 else " (no chain)"
                )
                pkt = f"pkt={bd.packet_id}" if bd.enable_packet else "no-pkt"
                acq_dma = _lock_dma_str(bd.acquire_lock_id, bd.acquire_lock_val)
                rel_dma = _lock_dma_str(bd.release_lock_id, bd.release_lock_val)
                out.write(
                    f"    [BD{bd.bd_id}] len={bd.length:>4}  {pkt:<8}"
                    f"  next{chain:<12}  buf={bd.buf_source}\n"
                )
                out.write(
                    f"           DMA lock: acquire({acq_dma})  release({rel_dma})\n"
                )
                acq_init = _lock_init_str(bd.acquire_lock_id, info.locks)
                rel_init = _lock_init_str(bd.release_lock_id, info.locks)
                init_parts = []
                if acq_init:
                    init_parts.append(f"acq lock {bd.acquire_lock_id}: {acq_init}")
                if rel_init:
                    init_parts.append(f"rel lock {bd.release_lock_id}: {rel_init}")
                if init_parts:
                    out.write(
                        f"           Lock init: {', '.join(init_parts)}\n"
                    )

            # Detect ping-pong pairs
            bd_by_id = {bd.bd_id: bd for bd in info.bds}
            visited = set()
            for bd in info.bds:
                if bd.bd_id in visited:
                    continue
                if bd.next_bd >= 0 and bd.next_bd in bd_by_id:
                    other = bd_by_id[bd.next_bd]
                    if other.next_bd == bd.bd_id:
                        out.write(
                            f"    ** Ping-Pong: BD{bd.bd_id} <-> BD{other.bd_id}"
                            f"  pkt={bd.packet_id}\n"
                        )
                        visited.update([bd.bd_id, other.bd_id])

        # Lock summary for tile
        if info.locks:
            out.write("  Lock Summary:\n")
            for lid in sorted(info.locks):
                lk = info.locks[lid]
                dma_s = f"dma_request_val={lk.operate_value}" if lk.operate_value is not None else "dma_request_val=unset"
                init_s = f"XAie_LockSetValue(init={lk.init_value})" if lk.init_value is not None else "no XAie_LockSetValue"
                out.write(f"    Lock {lid}: {dma_s}  |  {init_s}\n")
        out.write("\n")

# ---------------------------------------------------------------------------
# PNG output (matplotlib)
# ---------------------------------------------------------------------------

def render_png(
    tiles: Dict[Tuple[int, int], TileDmaInfo], output_path: str
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("ERROR: matplotlib is required for PNG mode.", file=sys.stderr)
        sys.exit(1)

    if not tiles:
        print("No tiles to render.", file=sys.stderr)
        return

    all_cols = sorted({loc[0] for loc in tiles})
    all_rows = sorted({loc[1] for loc in tiles}, reverse=True)

    col_set = sorted({loc[0] for loc in tiles})
    row_set = sorted({loc[1] for loc in tiles})
    min_col, max_col = min(col_set), max(col_set)
    min_row, max_row = min(row_set), max(row_set)

    cell_w, cell_h = 3.2, 2.8
    margin = 0.6
    fig_w = (max_col - min_col + 1) * cell_w + 2 * margin
    fig_h = (max_row - min_row + 1) * cell_h + 2 * margin + 0.8

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-margin, (max_col - min_col + 1) * cell_w + margin)
    ax.set_ylim(-margin, (max_row - min_row + 1) * cell_h + margin + 0.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("DMA BD Configuration", fontsize=14, fontweight="bold", pad=12)

    type_colors = {"Shim": "#B3E5FC", "Mem": "#C8E6C9", "AIE": "#FFF9C4"}
    dir_colors = {"S2MM": "#E3F2FD", "MM2S": "#FFF3E0"}

    for loc, info in tiles.items():
        ci = loc[0] - min_col
        ri = loc[1] - min_row
        x0 = ci * cell_w + 0.15
        y0 = ri * cell_h + 0.15
        w = cell_w - 0.3
        h = cell_h - 0.3

        bg = type_colors.get(info.tile_type, "#EEEEEE")
        rect = mpatches.FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle="round,pad=0.08",
            facecolor=bg, edgecolor="#333", linewidth=1.2,
        )
        ax.add_patch(rect)

        # Title
        title = f"({loc[0]},{loc[1]}) {info.tile_type}"
        ax.text(
            x0 + w / 2, y0 + h - 0.22, title,
            ha="center", va="top", fontsize=8, fontweight="bold",
        )

        # IO line
        if info.ios:
            io_str = "  ".join(
                f"ch{io.channel_id}:{io.direction}" for io in info.ios
            )
            ax.text(
                x0 + w / 2, y0 + h - 0.5, io_str,
                ha="center", va="top", fontsize=6.5, color="#555",
            )

        # BD details
        line_y = y0 + h - 0.8
        bd_by_id = {bd.bd_id: bd for bd in info.bds}
        visited_pairs = set()
        for bd in info.bds:
            if line_y < y0 + 0.15:
                break
            pkt = f"pkt{bd.packet_id}" if bd.enable_packet else ""
            chain_str = ""

            acq_s = f"acq=lock{bd.acquire_lock_id}(val={bd.acquire_lock_val})" if bd.acquire_lock_id >= 0 else "acq=none"
            rel_s = f"rel=lock{bd.release_lock_id}(val={bd.release_lock_val})" if bd.release_lock_id >= 0 else "rel=none"
            lock_line = f"{acq_s} {rel_s}"
            init_parts = []
            if bd.acquire_lock_id >= 0:
                lk = info.locks.get(bd.acquire_lock_id)
                if lk and lk.init_value is not None:
                    init_parts.append(f"acq.init={lk.init_value}")
            if bd.release_lock_id >= 0:
                lk = info.locks.get(bd.release_lock_id)
                if lk and lk.init_value is not None:
                    init_parts.append(f"rel.init={lk.init_value}")
            if init_parts:
                lock_line += f"  SetValue({','.join(init_parts)})"

            # Check ping-pong
            pair_key = tuple(sorted([bd.bd_id, bd.next_bd]))
            if (
                bd.next_bd >= 0
                and bd.next_bd in bd_by_id
                and bd_by_id[bd.next_bd].next_bd == bd.bd_id
                and pair_key not in visited_pairs
            ):
                visited_pairs.add(pair_key)
                chain_str = f"BD{bd.bd_id}<->BD{bd.next_bd}"
                ax.text(
                    x0 + w / 2, line_y,
                    f"{chain_str}  {pkt}  len={bd.length}",
                    ha="center", va="top", fontsize=6, color="#1565C0",
                    fontfamily="monospace",
                )
                line_y -= 0.22
                ax.text(
                    x0 + w / 2, line_y, lock_line,
                    ha="center", va="top", fontsize=5, color="#7B1FA2",
                    fontfamily="monospace",
                )
                line_y -= 0.22
            elif pair_key not in visited_pairs:
                ax.text(
                    x0 + w / 2, line_y,
                    f"BD{bd.bd_id} {pkt} len={bd.length}",
                    ha="center", va="top", fontsize=6,
                    fontfamily="monospace",
                )
                line_y -= 0.22
                ax.text(
                    x0 + w / 2, line_y, lock_line,
                    ha="center", va="top", fontsize=5, color="#7B1FA2",
                    fontfamily="monospace",
                )
                line_y -= 0.22

        # Buffer source
        if info.bds:
            sources = list(dict.fromkeys(bd.buf_source for bd in info.bds))
            src_text = ", ".join(sources[:2])
            if len(sources) > 2:
                src_text += "..."
            ax.text(
                x0 + w / 2, y0 + 0.15, f"buf: {src_text}",
                ha="center", va="bottom", fontsize=5.5, color="#777",
                style="italic",
            )

    # Legend
    legend_patches = [
        mpatches.Patch(color="#B3E5FC", label="Shim"),
        mpatches.Patch(color="#C8E6C9", label="Mem"),
        mpatches.Patch(color="#FFF9C4", label="AIE/Core"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG saved to {output_path}")

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

_PKT_COLORS = [
    "#4FC3F7", "#81C784", "#FFB74D", "#E57373",
    "#BA68C8", "#4DB6AC", "#FF8A65", "#A1887F",
]

def render_html(
    tiles: Dict[Tuple[int, int], TileDmaInfo], output_path: str
) -> str:
    all_cols = sorted({loc[0] for loc in tiles})
    all_rows = sorted({loc[1] for loc in tiles}, reverse=True)  # top = high row

    def _pkt_color(pid: int) -> str:
        return _PKT_COLORS[pid % len(_PKT_COLORS)]

    def _lock_html(lock_id: int, lock_val: int, locks: Dict[int, LockInfo], label: str) -> str:
        if lock_id < 0:
            return f'<span class="lock unset">{label}: none</span>'
        lk = locks.get(lock_id)
        init_val = str(lk.init_value) if lk and lk.init_value is not None else "none"
        tooltip = f"DMA {label}: lock_id={lock_id}, request_val={lock_val}\nXAie_LockSetValue: init={init_val}"
        init_badge = ""
        if lk and lk.init_value is not None:
            init_badge = f'<span class="lock-init-badge">init={lk.init_value}</span>'
        return (
            f'<span class="lock" title="{tooltip}">'
            f'{label}=lock{lock_id}(val={lock_val}) {init_badge}'
            f'</span>'
        )

    tile_cards = []
    for loc in sorted(tiles, key=lambda k: (-k[1], k[0])):
        info = tiles[loc]
        type_cls = info.tile_type.lower()

        # BD rows
        bd_rows = []
        bd_by_id = {bd.bd_id: bd for bd in info.bds}
        visited = set()
        for bd in info.bds:
            pair_key = tuple(sorted([bd.bd_id, bd.next_bd]))
            is_ping_pong = (
                bd.next_bd >= 0
                and bd.next_bd in bd_by_id
                and bd_by_id[bd.next_bd].next_bd == bd.bd_id
            )
            if is_ping_pong and pair_key in visited:
                continue
            if is_ping_pong:
                visited.add(pair_key)

            chain_html = ""
            pp_class = ""
            if is_ping_pong:
                other = bd_by_id[bd.next_bd]
                chain_html = (
                    f'<span class="chain">'
                    f'BD{bd.bd_id} &#x21c4; BD{other.bd_id}</span>'
                )
                pp_class = " ping-pong"
            else:
                nxt = f"&rarr;BD{bd.next_bd}" if bd.next_bd >= 0 else "none"
                chain_html = (
                    f'<span class="chain">BD{bd.bd_id} next={nxt}</span>'
                )

            pkt_html = ""
            if bd.enable_packet:
                c = _pkt_color(bd.packet_id)
                pkt_html = (
                    f'<span class="pkt" style="background:{c}">'
                    f"pkt {bd.packet_id}</span>"
                )

            acq_html = _lock_html(bd.acquire_lock_id, bd.acquire_lock_val, info.locks, "acq")
            rel_html = _lock_html(bd.release_lock_id, bd.release_lock_val, info.locks, "rel")

            bd_rows.append(
                f'<div class="bd-row{pp_class}">'
                f"  {chain_html} {pkt_html}"
                f'  <span class="len">len={bd.length}</span>'
                f'  <span class="buf">buf={bd.buf_source}</span>'
                f'  <div class="lock-row">{acq_html} {rel_html}</div>'
                f"</div>"
            )

        # IO rows
        io_rows = []
        for io in info.ios:
            dir_cls = io.direction.lower().replace("2", "to")
            io_rows.append(
                f'<div class="io-row {dir_cls}">'
                f"  ch{io.channel_id}: {io.direction}"
                f"</div>"
            )

        # Lock summary rows
        lock_rows = []
        for lid in sorted(info.locks):
            lk = info.locks[lid]
            dma_s = f"dma_request_val={lk.operate_value}" if lk.operate_value is not None else "dma_request_val=unset"
            if lk.init_value is not None:
                init_s = f'<span class="lock-init-badge">XAie_LockSetValue(init={lk.init_value})</span>'
            else:
                init_s = '<span class="lock-no-init">no XAie_LockSetValue</span>'
            lock_rows.append(
                f'<div class="lock-summary-row">'
                f'  Lock {lid}: {dma_s} &nbsp;|&nbsp; {init_s}'
                f'</div>'
            )
        lock_section = ""
        if lock_rows:
            lock_section = f'<div class="lock-section">{"".join(lock_rows)}</div>'

        card = (
            f'<div class="tile-card {type_cls}"'
            f' style="grid-column:{loc[0] - all_cols[0] + 1};'
            f" grid-row:{all_rows[0] - loc[1] + 1}\">"
            f'  <div class="tile-header">'
            f"    ({loc[0]},{loc[1]}) {info.tile_type}"
            f"  </div>"
            f'  <div class="io-section">{"".join(io_rows)}</div>'
            f'  <div class="bd-section">{"".join(bd_rows)}</div>'
            f"  {lock_section}"
            f"</div>"
        )
        tile_cards.append(card)

    ncols = len(all_cols)
    nrows = len(all_rows)

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DMA BD Configuration</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f5f5f5; padding: 24px;
}}
h1 {{ text-align: center; margin-bottom: 18px; font-size: 20px; color: #333; }}
.grid {{
    display: grid;
    grid-template-columns: repeat({ncols}, 280px);
    grid-template-rows: repeat({nrows}, auto);
    gap: 12px;
    justify-content: center;
}}
.tile-card {{
    border: 2px solid #999; border-radius: 8px;
    padding: 10px; min-height: 120px;
    background: #fff; position: relative;
    transition: box-shadow .15s;
}}
.tile-card:hover {{ box-shadow: 0 4px 16px rgba(0,0,0,.18); }}
.tile-card.shim {{ background: #E3F2FD; border-color: #42A5F5; }}
.tile-card.mem  {{ background: #E8F5E9; border-color: #66BB6A; }}
.tile-card.aie  {{ background: #FFFDE7; border-color: #FDD835; }}
.tile-header {{
    font-weight: 700; font-size: 13px; margin-bottom: 6px;
    padding-bottom: 4px; border-bottom: 1px solid #ccc;
}}
.io-section {{ margin-bottom: 6px; }}
.io-row {{
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 11px; font-weight: 600; margin-right: 4px; margin-bottom: 2px;
}}
.io-row.stommm {{ background: #BBDEFB; color: #0D47A1; }}
.io-row.mmtos  {{ background: #FFE0B2; color: #E65100; }}
.bd-section {{ font-size: 11px; }}
.bd-row {{
    padding: 4px 6px; margin-bottom: 3px; border-radius: 4px;
    background: #fafafa; border: 1px solid #e0e0e0;
    display: flex; flex-wrap: wrap; gap: 6px; align-items: center;
}}
.bd-row.ping-pong {{ background: #E8EAF6; border-color: #7986CB; }}
.chain {{ font-weight: 700; color: #1565C0; font-family: monospace; }}
.pkt {{
    padding: 1px 6px; border-radius: 3px; font-size: 10px;
    color: #fff; font-weight: 600;
}}
.len {{ color: #555; }}
.buf {{ color: #888; font-style: italic; font-size: 10px; }}
.lock-row {{ width: 100%; display: flex; gap: 8px; margin-top: 2px; }}
.lock {{
    font-size: 10px; font-family: monospace; color: #4A148C;
    background: #F3E5F5; padding: 1px 5px; border-radius: 3px;
}}
.lock.unset {{ color: #999; background: #f0f0f0; }}
.lock-init-badge {{
    font-size: 9px; font-weight: 700; color: #fff; background: #7B1FA2;
    padding: 0 4px; border-radius: 3px; margin-left: 2px;
}}
.lock-no-init {{ color: #aaa; font-style: italic; }}
.lock-section {{
    margin-top: 6px; padding-top: 4px; border-top: 1px dashed #ccc;
    font-size: 10px; color: #666;
}}
.lock-summary-row {{ padding: 1px 0; font-family: monospace; }}
.legend {{
    display: flex; gap: 16px; justify-content: center;
    margin-bottom: 14px; font-size: 12px;
}}
.legend-item {{
    display: flex; align-items: center; gap: 4px;
}}
.legend-swatch {{
    width: 16px; height: 16px; border-radius: 3px; border: 1px solid #aaa;
}}
</style>
</head>
<body>
<h1>DMA Buffer Descriptor Configuration</h1>
<div class="legend">
    <div class="legend-item"><div class="legend-swatch" style="background:#E3F2FD"></div> Shim</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#E8F5E9"></div> Mem</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#FFFDE7"></div> AIE/Core</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#E8EAF6"></div> Ping-Pong</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#BBDEFB"></div> S2MM</div>
    <div class="legend-item"><div class="legend-swatch" style="background:#FFE0B2"></div> MM2S</div>
</div>
<div class="grid">
{"".join(tile_cards)}
</div>
</body>
</html>
"""
    Path(output_path).write_text(html)
    print(f"HTML saved to {output_path}")
    return output_path

# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

def serve_html(html_path: str, port: int) -> None:
    import socket

    abs_path = Path(html_path).resolve()
    directory = str(abs_path.parent)
    filename = abs_path.name

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)

        def log_message(self, fmt, *args):
            pass

    hostname = socket.gethostname()
    local_url = f"http://localhost:{port}/{filename}"
    remote_url = f"http://{hostname}:{port}/{filename}"

    print(f"Serving DMA BD visualization:")
    print(f"  Local:  {local_url}")
    print(f"  Remote: {remote_url}")
    print(f"  (Ctrl+C to stop)")

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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze DMA BD configurations in generated host.cc"
    )
    parser.add_argument("host_cc", help="Path to generated host.cc")
    parser.add_argument(
        "-m", "--mode", choices=["text", "png", "html"],
        default=None,
        help="Output mode: text, png, or html (default: html + serve)",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output file (default: auto-named based on mode)",
    )
    parser.add_argument(
        "--no-serve", action="store_true",
        help="Generate HTML file only, do not start HTTP server",
    )
    parser.add_argument(
        "--port", type=int, default=8088,
        help="HTTP server port (default: 8088)",
    )
    args = parser.parse_args()

    tiles = parse_host_cc(args.host_cc)
    if not tiles:
        print("No DMA BD configurations found.", file=sys.stderr)
        sys.exit(1)

    mode = args.mode

    # Default: html + serve
    if mode is None:
        mode = "html"

    if mode == "text":
        out_path = args.output
        if out_path:
            with open(out_path, "w") as f:
                render_text(tiles, f)
            print(f"Text report saved to {out_path}")
        else:
            render_text(tiles, sys.stdout)

    elif mode == "png":
        out_path = args.output or "dma_bd_config.png"
        render_png(tiles, out_path)

    elif mode == "html":
        out_path = args.output or "dma_bd_config.html"
        render_html(tiles, out_path)
        if not args.no_serve:
            serve_html(out_path, args.port)


if __name__ == "__main__":
    main()
