#!/usr/bin/env python3
# Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Scan a routing.cc file and draw a per-block connection topology map.

Usage:
    python routing_topology.py <routing.cc> [-o output.png] [--host host.cc]

The script parses XAie stream-switch API calls, groups them by code block
(identified by "round is N" comments or if-block boundaries), traces
cross-tile connections, and renders an annotated grid diagram with
color-coded paths per block.

When --host is provided, DMA channel information (BD configs, channel
assignments, lock/packet metadata) is parsed from the host file and
annotated inside the tile boxes in the diagram.
"""

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

DIRECTION_OPPOSITES = {
    "NORTH": "SOUTH", "SOUTH": "NORTH",
    "EAST": "WEST", "WEST": "EAST",
}

DIRECTION_DELTA = {
    "NORTH": (0, 1), "SOUTH": (0, -1),
    "EAST": (1, 0), "WEST": (-1, 0),
}

TILE_TYPE_BY_ROW = lambda row: "Shim" if row == 0 else ("Mem" if row <= 2 else "AIE")

BLOCK_COLORS = [
    "#2196F3", "#E91E63", "#4CAF50", "#FF9800",
    "#9C27B0", "#00BCD4", "#795548", "#607D8B",
]


@dataclass
class PktSlaveSlot:
    tile: tuple
    port_type: str
    port_num: int
    slot_num: int
    pkt_id: int
    mask: int
    msel: int
    arbiter: int
    line: int


@dataclass
class PktMasterPort:
    tile: tuple
    port_type: str
    port_num: int
    drop_header: bool
    arbiter: int
    msel_en: int
    line: int


@dataclass
class CircuitConn:
    tile: tuple
    slave_type: str
    slave_port: int
    master_type: str
    master_port: int
    line: int


@dataclass
class ShimDmaPort:
    tile: tuple
    port_num: int
    direction: str  # "aie_to_shim" or "shim_to_aie"
    line: int


@dataclass
class DmaBdConfig:
    tile: tuple
    bd_id: int
    addr: int
    length: int
    next_bd: int          # -1 = no chaining
    packet_id: int        # -1 = no packet header
    acq_lock: tuple       # (lock_id, lock_val) or None
    rel_lock: tuple       # (lock_id, lock_val) or None
    line: int


@dataclass
class DmaChannel:
    tile: tuple
    channel_id: int
    direction: str        # "S2MM" or "MM2S"
    start_bd: int
    line: int


@dataclass
class Block:
    name: str
    circuit_conns: list = field(default_factory=list)
    pkt_slaves: list = field(default_factory=list)
    pkt_masters: list = field(default_factory=list)
    shim_dma: list = field(default_factory=list)


def parse_tile_loc(s: str) -> tuple:
    m = re.search(r"XAie_TileLoc\(\s*(\d+)\s*,\s*(\d+)\s*\)", s)
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)))


def parse_routing_file(filepath: str) -> list[Block]:
    with open(filepath) as f:
        lines = f.readlines()

    blocks: list[Block] = []
    current_block: Optional[Block] = None
    block_idx = 0

    for lineno_0, raw in enumerate(lines):
        lineno = lineno_0 + 1
        line = raw.strip()

        round_m = re.search(r"round\s+is\s+(\d+)", line, re.IGNORECASE)
        if round_m:
            block_idx = int(round_m.group(1))
            current_block = Block(name=f"Round {block_idx}")
            blocks.append(current_block)
            continue

        if current_block is None and re.search(r"XAie_Strm|XAie_Enable", line):
            if re.match(r"(int32_t|void|AieRC)\s+XAie_\w+\s*\(", line):
                continue
            current_block = Block(name=f"Block {block_idx}")
            blocks.append(current_block)

        if current_block is None:
            continue

        if re.match(r"(int32_t|void|AieRC)\s+XAie_\w+\s*\(", line):
            continue

        # --- XAie_StrmConnCctEnable ---
        m = re.search(
            r"XAie_StrmConnCctEnable\([^,]+,\s*(XAie_TileLoc\(\d+\s*,\s*\d+\))\s*,"
            r"\s*(\w+)\s*,\s*(\d+)\s*,\s*(\w+)\s*,\s*(\d+)\s*\)", line)
        if m:
            tile = parse_tile_loc(m.group(1))
            current_block.circuit_conns.append(CircuitConn(
                tile=tile,
                slave_type=m.group(2), slave_port=int(m.group(3)),
                master_type=m.group(4), master_port=int(m.group(5)),
                line=lineno,
            ))
            continue

        # --- XAie_StrmPktSwSlaveSlotEnable ---
        m = re.search(
            r"XAie_StrmPktSwSlaveSlotEnable\([^,]+,\s*(XAie_TileLoc\(\d+\s*,\s*\d+\))\s*,"
            r"\s*(\w+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*\{[^}]*PktId\s*=\s*(\d+)[^}]*\}\s*,"
            r"\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", line)
        if m:
            tile = parse_tile_loc(m.group(1))
            current_block.pkt_slaves.append(PktSlaveSlot(
                tile=tile,
                port_type=m.group(2), port_num=int(m.group(3)),
                slot_num=int(m.group(4)),
                pkt_id=int(m.group(5)),
                mask=int(m.group(6)), msel=int(m.group(7)),
                arbiter=int(m.group(8)),
                line=lineno,
            ))
            continue

        # --- XAie_StrmPktSwMstrPortEnable ---
        m = re.search(
            r"XAie_StrmPktSwMstrPortEnable\([^,]+,\s*(XAie_TileLoc\(\d+\s*,\s*\d+\))\s*,"
            r"\s*(\w+)\s*,\s*(\d+)\s*,\s*\w+\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", line)
        if m:
            tile = parse_tile_loc(m.group(1))
            current_block.pkt_masters.append(PktMasterPort(
                tile=tile,
                port_type=m.group(2), port_num=int(m.group(3)),
                drop_header=True,
                arbiter=int(m.group(4)), msel_en=int(m.group(5)),
                line=lineno,
            ))
            continue

        # --- XAie_EnableAieToShimDmaStrmPort ---
        m = re.search(
            r"XAie_EnableAieToShimDmaStrmPort\([^,]+,\s*(XAie_TileLoc\(\d+\s*,\s*\d+\))\s*,"
            r"\s*(\d+)\s*\)", line)
        if m:
            tile = parse_tile_loc(m.group(1))
            current_block.shim_dma.append(ShimDmaPort(
                tile=tile, port_num=int(m.group(2)),
                direction="aie_to_shim", line=lineno,
            ))
            continue

        # --- XAie_EnableShimDmaToAieStrmPort ---
        m = re.search(
            r"XAie_EnableShimDmaToAieStrmPort\([^,]+,\s*(XAie_TileLoc\(\d+\s*,\s*\d+\))\s*,"
            r"\s*(\d+)\s*\)", line)
        if m:
            tile = parse_tile_loc(m.group(1))
            current_block.shim_dma.append(ShimDmaPort(
                tile=tile, port_num=int(m.group(2)),
                direction="shim_to_aie", line=lineno,
            ))
            continue

    return blocks


def parse_host_file(filepath: str) -> tuple[list[DmaBdConfig], list[DmaChannel]]:
    """Parse a host.cc (or aie_control.cpp) for DMA BD configs and channel assignments.

    Supports two patterns:
      A) Runtime wrappers: __Runtime_dma_bd_config / __Runtime_dma_createio_4
         with structured comments (/* DMA BD Config: ... */, /* Create IO: ... */).
      B) Raw XAie calls: XAie_DmaDescInit, XAie_DmaSetAddrLen, XAie_DmaSetLock,
         XAie_DmaSetNextBd, XAie_DmaSetPkt, XAie_DmaWriteBd,
         XAie_DmaChannelSetStartQueue / XAie_DmaChannelSetStartQueueGeneric.
    """
    with open(filepath) as f:
        lines = f.readlines()

    bds: list[DmaBdConfig] = []
    channels: list[DmaChannel] = []

    # Build a variable → tile lookup for resolving indirect tile references
    # e.g. "XAie_LocType v11 = XAie_TileLoc(2, 0);"
    var_tiles: dict[str, tuple] = {}

    # --- State for raw-XAie BD accumulator (Pattern B) ---
    cur_addr = 0
    cur_len = 0
    cur_next_bd = -1
    cur_pkt_id = -1
    cur_acq_lock = None
    cur_rel_lock = None
    cur_init_tile = None  # tile from XAie_DmaDescInit

    def _reset_bd_accum():
        nonlocal cur_addr, cur_len, cur_next_bd, cur_pkt_id
        nonlocal cur_acq_lock, cur_rel_lock, cur_init_tile
        cur_addr = 0
        cur_len = 0
        cur_next_bd = -1
        cur_pkt_id = -1
        cur_acq_lock = None
        cur_rel_lock = None
        cur_init_tile = None

    def _resolve_tile(token: str) -> Optional[tuple]:
        """Resolve a token to a (col, row) tuple."""
        tile = parse_tile_loc(token)
        if tile:
            return tile
        t = token.strip()
        return var_tiles.get(t)

    # Pending comment data for Pattern A
    pending_bd_comment: dict = {}
    pending_io_comment: dict = {}

    for lineno_0, raw in enumerate(lines):
        lineno = lineno_0 + 1
        line = raw.strip()

        # --- Variable → TileLoc mapping ---
        m = re.search(r"(\w+)\s*=\s*(XAie_TileLoc\(\s*\d+\s*,\s*\d+\s*\))", line)
        if m:
            var_tiles[m.group(1)] = parse_tile_loc(m.group(2))

        # ==================================================================
        # Pattern A: Structured comments + runtime wrappers
        # ==================================================================

        # /* DMA BD Config: bd_id=N, offset=N, len=N, enable_packet=..., packet_id=N, next_bd=N */
        m = re.search(
            r"/\*\s*DMA BD Config:\s*bd_id=(\d+),\s*offset=\d+,\s*len=(\d+)"
            r",\s*enable_packet=\w+,\s*packet_id=(\d+),\s*next_bd=(\d+)", line)
        if m:
            pending_bd_comment = {
                "bd_id": int(m.group(1)),
                "length": int(m.group(2)),
                "packet_id": int(m.group(3)),
                "next_bd": int(m.group(4)),
                "line": lineno,
            }
            nb = pending_bd_comment["next_bd"]
            if nb == 4294967295 or nb == 0xFFFFFFFF:
                pending_bd_comment["next_bd"] = -1
            continue

        # /* Create IO: channel_id=N, bd_id=N, tile=(C,R), direction=S2MM */
        m = re.search(
            r"/\*\s*Create IO:\s*channel_id=(\d+),\s*bd_id=(\d+)"
            r",\s*tile=\((\d+),\s*(\d+)\),\s*direction=(\w+)", line)
        if m:
            pending_io_comment = {
                "channel_id": int(m.group(1)),
                "bd_id": int(m.group(2)),
                "tile": (int(m.group(3)), int(m.group(4))),
                "direction": m.group(5),
                "line": lineno,
            }
            continue

        # __Runtime_dma_bd_config(dev, tile, buf, bd_id, offset, len, next_bd, enable_pkt, pkt_id)
        m = re.search(
            r"__Runtime_dma_bd_config\([^,]+,\s*(\w+)\s*,[^,]+,\s*(\d+)\s*,"
            r"\s*\d+\s*,\s*(\d+)\s*,\s*(-?\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", line)
        if m:
            tile = _resolve_tile(m.group(1))
            bd_id = int(m.group(2))
            length = int(m.group(3))
            next_bd_val = int(m.group(4))
            pkt_id = int(m.group(6)) if int(m.group(5)) else -1
            if next_bd_val < 0 or next_bd_val == 4294967295:
                next_bd_val = -1
            if pending_bd_comment and pending_bd_comment.get("bd_id") == bd_id:
                length = pending_bd_comment.get("length", length)
                pkt_id_c = pending_bd_comment.get("packet_id", -1)
                if pkt_id == -1 and pkt_id_c >= 0:
                    pkt_id = pkt_id_c
                pending_bd_comment = {}
            if tile:
                bds.append(DmaBdConfig(
                    tile=tile, bd_id=bd_id, addr=0, length=length,
                    next_bd=next_bd_val, packet_id=pkt_id,
                    acq_lock=None, rel_lock=None, line=lineno))
            continue

        # __Runtime_dma_createio_4(tile, dma_desc, channel_id, bd_id)
        m = re.search(
            r"__Runtime_dma_createio_4\(\s*(\w+)\s*,[^,]+,\s*(\d+)\s*,\s*(\d+)\s*\)", line)
        if m:
            tile = _resolve_tile(m.group(1))
            ch_id = int(m.group(2))
            bd_id = int(m.group(3))
            direction = "S2MM"
            if pending_io_comment and pending_io_comment.get("channel_id") == ch_id:
                tile = pending_io_comment.get("tile", tile)
                direction = pending_io_comment.get("direction", "S2MM")
                pending_io_comment = {}
            if tile:
                channels.append(DmaChannel(
                    tile=tile, channel_id=ch_id, direction=direction,
                    start_bd=bd_id, line=lineno))
            continue

        # ==================================================================
        # Pattern B: Raw XAie DMA calls
        # ==================================================================

        # XAie_DmaDescInit(&Dev, &Desc, XAie_TileLoc(C,R))
        m = re.search(
            r"XAie_DmaDescInit\([^,]+,\s*[^,]+,\s*(XAie_TileLoc\(\s*\d+\s*,\s*\d+\s*\))\s*\)",
            line)
        if m:
            _reset_bd_accum()
            cur_init_tile = parse_tile_loc(m.group(1))
            continue

        # XAie_DmaSetAddrLen(&Desc, addr, len)
        m = re.search(r"XAie_DmaSetAddrLen\([^,]+,\s*(0x[\da-fA-F]+|\d+)\s*,\s*(\d+)\s*\)", line)
        if m:
            cur_addr = int(m.group(1), 0)
            cur_len = int(m.group(2))
            continue

        # XAie_DmaSetLock(&Desc, XAie_LockInit(id, val), XAie_LockInit(id, val))
        m = re.search(
            r"XAie_DmaSetLock\([^,]+,\s*XAie_LockInit\(\s*(\d+)\s*,\s*(-?\d+)\s*\)\s*,"
            r"\s*XAie_LockInit\(\s*(\d+)\s*,\s*(-?\d+)\s*\)\s*\)", line)
        if m:
            cur_acq_lock = (int(m.group(1)), int(m.group(2)))
            cur_rel_lock = (int(m.group(3)), int(m.group(4)))
            continue

        # XAie_DmaSetNextBd(&Desc, next_bd, enable)
        m = re.search(r"XAie_DmaSetNextBd\([^,]+,\s*(\d+)\s*,", line)
        if m:
            cur_next_bd = int(m.group(1))
            continue

        # XAie_DmaSetPkt(&Desc, {.PktId=N, ...})
        m = re.search(r"XAie_DmaSetPkt\([^,]+,\s*\{[^}]*PktId\s*=\s*(\d+)", line)
        if m:
            cur_pkt_id = int(m.group(1))
            continue

        # XAie_DmaWriteBd(&Dev, &Desc, XAie_TileLoc(C,R), bd_num)
        m = re.search(
            r"XAie_DmaWriteBd\([^,]+,\s*[^,]+,\s*(XAie_TileLoc\(\s*\d+\s*,\s*\d+\s*\))\s*,"
            r"\s*(\d+)\s*\)", line)
        if m:
            tile = parse_tile_loc(m.group(1)) or cur_init_tile
            bd_id = int(m.group(2))
            if tile:
                bds.append(DmaBdConfig(
                    tile=tile, bd_id=bd_id, addr=cur_addr, length=cur_len,
                    next_bd=cur_next_bd, packet_id=cur_pkt_id,
                    acq_lock=cur_acq_lock, rel_lock=cur_rel_lock,
                    line=lineno))
            _reset_bd_accum()
            continue

        # XAie_DmaChannelSetStartQueue(&Dev, TileLoc, ch, dir, bd, repeat, ...)
        m = re.search(
            r"XAie_DmaChannelSetStartQueue(?:Generic)?\([^,]+,\s*"
            r"(XAie_TileLoc\(\s*\d+\s*,\s*\d+\s*\))\s*,"
            r"\s*(\d+)\s*,\s*(DMA_\w+)\s*,\s*(\d+)", line)
        if m:
            tile = parse_tile_loc(m.group(1))
            ch_id = int(m.group(2))
            direction = "S2MM" if "S2MM" in m.group(3) else "MM2S"
            start_bd = int(m.group(4))
            if tile:
                channels.append(DmaChannel(
                    tile=tile, channel_id=ch_id, direction=direction,
                    start_bd=start_bd, line=lineno))
            continue

    return bds, channels


def _build_dma_tile_summary(
    bds: list[DmaBdConfig], channels: list[DmaChannel]
) -> dict[tuple, list[str]]:
    """Group DMA BD/channel info by tile and produce per-tile annotation lines."""
    bds_by_tile: dict[tuple, list[DmaBdConfig]] = defaultdict(list)
    for bd in bds:
        bds_by_tile[bd.tile].append(bd)

    ch_by_tile: dict[tuple, list[DmaChannel]] = defaultdict(list)
    for ch in channels:
        ch_by_tile[ch.tile].append(ch)

    all_tiles = set(bds_by_tile) | set(ch_by_tile)
    result: dict[tuple, list[str]] = {}

    for tile in sorted(all_tiles):
        lines = []
        bd_map = {bd.bd_id: bd for bd in bds_by_tile.get(tile, [])}
        tile_channels = ch_by_tile.get(tile, [])

        seen_bds = set()
        for ch in sorted(tile_channels, key=lambda c: (c.direction, c.channel_id)):
            chain = []
            bd_id = ch.start_bd
            while bd_id >= 0 and bd_id not in seen_bds:
                seen_bds.add(bd_id)
                chain.append(bd_id)
                bd_cfg = bd_map.get(bd_id)
                if bd_cfg and bd_cfg.next_bd >= 0:
                    bd_id = bd_cfg.next_bd
                else:
                    break

            bd_str = "->".join(f"BD{b}" for b in chain) if chain else f"BD{ch.start_bd}"
            first_bd = bd_map.get(ch.start_bd)
            extras = []
            if first_bd:
                extras.append(f"{first_bd.length}B")
                if first_bd.packet_id >= 0:
                    extras.append(f"pkt{first_bd.packet_id}")
                if first_bd.acq_lock:
                    extras.append(f"L{first_bd.acq_lock[0]}/{first_bd.rel_lock[0]}"
                                  if first_bd.rel_lock else f"L{first_bd.acq_lock[0]}")
            suffix = f" ({', '.join(extras)})" if extras else ""
            lines.append(f"{ch.direction} ch{ch.channel_id}: {bd_str}{suffix}")

        # BDs without a channel assignment
        for bd in bds_by_tile.get(tile, []):
            if bd.bd_id not in seen_bds:
                extras = [f"{bd.length}B"]
                if bd.packet_id >= 0:
                    extras.append(f"pkt{bd.packet_id}")
                lines.append(f"BD{bd.bd_id} ({', '.join(extras)})")

        if lines:
            result[tile] = lines
    return result


def resolve_pkt_internal_edges(block: Block):
    """Pair packet slave slots with master ports on the same tile via arbiter/msel."""
    edges = []
    masters_by_tile = defaultdict(list)
    for mp in block.pkt_masters:
        masters_by_tile[mp.tile].append(mp)

    for ss in block.pkt_slaves:
        for mp in masters_by_tile.get(ss.tile, []):
            if mp.arbiter == ss.arbiter and (mp.msel_en >> ss.msel) & 1:
                edges.append((ss, mp))
    return edges


def trace_paths(blocks: list[Block]):
    """
    For each block, build list of end-to-end paths by following
    master-direction → neighbor-slave chains.
    Returns list of (block_name, path) where path is a list of
    (tile, slave_type, slave_port, master_type, master_port, mode) hops.
    """
    all_paths = []
    for block in blocks:
        adj = defaultdict(list)

        for cc in block.circuit_conns:
            adj[(cc.tile, cc.slave_type, cc.slave_port)].append(
                (cc.tile, cc.master_type, cc.master_port, "circuit"))

        for ss, mp in resolve_pkt_internal_edges(block):
            adj[(ss.tile, ss.port_type, ss.port_num)].append(
                (mp.tile, mp.port_type, mp.port_num, "packet"))

        sources = set()
        for ss in block.pkt_slaves:
            if ss.port_type == "DMA":
                sources.add((ss.tile, ss.port_type, ss.port_num))
        for cc in block.circuit_conns:
            incoming_key = (cc.tile, cc.slave_type, cc.slave_port)
            is_from_neighbor = cc.slave_type in DIRECTION_OPPOSITES
            if not is_from_neighbor:
                sources.add(incoming_key)

        def follow(tile, mstr_type, mstr_port, visited):
            """Follow a master port across a tile boundary and continue."""
            if mstr_type not in DIRECTION_DELTA:
                return [[(tile, mstr_type, mstr_port, None, None, "endpoint")]]
            if mstr_type == "SOUTH" and tile[1] == 0:
                return [[(tile, "SOUTH", mstr_port, None, None, "shim_exit")]]
            dc, dr = DIRECTION_DELTA[mstr_type]
            nb = (tile[0] + dc, tile[1] + dr)
            opp = DIRECTION_OPPOSITES[mstr_type]
            nb_key = (nb, opp, mstr_port)
            if nb_key in visited:
                return [[(tile, mstr_type, mstr_port, None, None, "loop")]]
            visited.add(nb_key)
            nexts = adj.get(nb_key, [])
            if not nexts:
                return [[(nb, opp, mstr_port, None, None, "dangling")]]
            results = []
            for (nt, mt, mp, mode) in nexts:
                sub = follow(nt, mt, mp, visited)
                for s in sub:
                    results.append([(nb, opp, mstr_port, mt, mp, mode)] + s)
            return results

        for src_key in sources:
            visited = {src_key}
            nexts = adj.get(src_key, [])
            for (nt, mt, mp, mode) in nexts:
                sub_paths = follow(nt, mt, mp, visited.copy())
                for sp in sub_paths:
                    full = [(src_key[0], src_key[1], src_key[2], mt, mp, mode)] + sp
                    all_paths.append((block.name, full))

    return all_paths


def collect_tiles(blocks: list[Block]) -> set:
    tiles = set()
    for b in blocks:
        for cc in b.circuit_conns:
            tiles.add(cc.tile)
        for ss in b.pkt_slaves:
            tiles.add(ss.tile)
        for mp in b.pkt_masters:
            tiles.add(mp.tile)
        for sd in b.shim_dma:
            tiles.add(sd.tile)
    return tiles


def draw_topology(blocks: list[Block], paths, tiles: set, output: str,
                  dma_tile_summary: Optional[dict] = None):
    if not tiles:
        print("No tiles found, nothing to draw.", file=sys.stderr)
        return

    if dma_tile_summary is None:
        dma_tile_summary = {}

    cols = sorted({t[0] for t in tiles})
    rows = sorted({t[1] for t in tiles})
    min_col, max_col = min(cols), max(cols)
    min_row, max_row = min(rows), max(rows)

    all_cols = list(range(min_col, max_col + 1))
    all_rows = list(range(min_row, max_row + 1))

    cell_w = 2.0
    cell_h = 1.8 if dma_tile_summary else 1.4
    pad_x, pad_y = 1.5, 1.0
    fig_w = len(all_cols) * cell_w + 2 * pad_x
    fig_h = len(all_rows) * cell_h + 2 * pad_y + 1.5

    fig, ax = plt.subplots(figsize=(max(fig_w, 8), max(fig_h, 6)))
    ax.set_xlim(-pad_x, len(all_cols) * cell_w + pad_x)
    ax.set_ylim(-pad_y - 1.0, len(all_rows) * cell_h + pad_y)
    ax.set_aspect("equal")
    ax.axis("off")

    tile_centers = {}
    tile_type_colors = {"Shim": "#FFECB3", "Mem": "#C8E6C9", "AIE": "#BBDEFB"}

    for ci, c in enumerate(all_cols):
        for ri, r in enumerate(all_rows):
            cx = ci * cell_w + cell_w / 2
            cy = ri * cell_h + cell_h / 2
            tile_centers[(c, r)] = (cx, cy)
            ttype = TILE_TYPE_BY_ROW(r)
            is_active = (c, r) in tiles
            fc = tile_type_colors.get(ttype, "#EEEEEE") if is_active else "#F5F5F5"
            ec = "#333333" if is_active else "#CCCCCC"
            lw = 1.5 if is_active else 0.5
            rect = mpatches.FancyBboxPatch(
                (cx - cell_w * 0.42, cy - cell_h * 0.38),
                cell_w * 0.84, cell_h * 0.76,
                boxstyle="round,pad=0.05",
                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=1)
            ax.add_patch(rect)
            ax.text(cx, cy + cell_h * 0.18, f"({c},{r})",
                    ha="center", va="center", fontsize=8, fontweight="bold",
                    color="#333" if is_active else "#AAA", zorder=5)
            ax.text(cx, cy - cell_h * 0.05, ttype,
                    ha="center", va="center", fontsize=6.5,
                    color="#666" if is_active else "#CCC", zorder=5)

    # Axis labels
    for ci, c in enumerate(all_cols):
        cx = ci * cell_w + cell_w / 2
        ax.text(cx, len(all_rows) * cell_h + pad_y * 0.3,
                f"Col {c}", ha="center", va="center", fontsize=9, fontweight="bold")
    for ri, r in enumerate(all_rows):
        cy = ri * cell_h + cell_h / 2
        ax.text(-pad_x * 0.55, cy,
                f"Row {r}", ha="center", va="center", fontsize=9,
                fontweight="bold", rotation=0)

    port_offsets = {
        "NORTH": (0, 0.35), "SOUTH": (0, -0.35),
        "EAST": (0.38, 0), "WEST": (-0.38, 0),
    }

    def tile_port_pos(tile, direction, port_num, total_spread=0.2):
        cx, cy = tile_centers[tile]
        dx, dy = port_offsets.get(direction, (0, 0))
        px = cx + dx * cell_w
        py = cy + dy * cell_h
        if direction in ("NORTH", "SOUTH"):
            px += (port_num - 0.5) * total_spread
        elif direction in ("EAST", "WEST"):
            py += (port_num - 0.5) * total_spread
        return px, py

    drawn_internal = set()

    for bi, block in enumerate(blocks):
        color = BLOCK_COLORS[bi % len(BLOCK_COLORS)]

        for cc in block.circuit_conns:
            if cc.tile not in tile_centers:
                continue
            key = (cc.tile, cc.slave_type, cc.slave_port, cc.master_type, cc.master_port, bi)
            if key in drawn_internal:
                continue
            drawn_internal.add(key)

            sx, sy = tile_port_pos(cc.tile, cc.slave_type, cc.slave_port)
            mx, my = tile_port_pos(cc.tile, cc.master_type, cc.master_port)
            cx, cy = tile_centers[cc.tile]
            ax.annotate("", xy=(mx, my), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=1.3, connectionstyle="arc3,rad=0.15"),
                        zorder=3)

            label = f"{cc.slave_type[0]}:{cc.slave_port}→{cc.master_type[0]}:{cc.master_port}"
            mid_x, mid_y = (sx + mx) / 2, (sy + my) / 2
            offset = 0.12 * cell_h * (1 if bi % 2 == 0 else -1)
            ax.text(mid_x, mid_y + offset, label,
                    ha="center", va="center", fontsize=4.5, color=color,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8),
                    zorder=6)

        pkt_edges = resolve_pkt_internal_edges(block)
        for ss, mp in pkt_edges:
            if ss.tile not in tile_centers:
                continue
            key = (ss.tile, ss.port_type, ss.port_num, mp.port_type, mp.port_num, bi)
            if key in drawn_internal:
                continue
            drawn_internal.add(key)

            sx, sy = tile_port_pos(ss.tile, ss.port_type, ss.port_num if ss.port_type in DIRECTION_DELTA else 0)
            mx, my = tile_port_pos(ss.tile, mp.port_type, mp.port_num if mp.port_type in DIRECTION_DELTA else 0)
            cx, cy = tile_centers[ss.tile]

            if ss.port_type not in DIRECTION_DELTA:
                sx = cx - cell_w * 0.25
                sy = cy - cell_h * 0.15
            if mp.port_type not in DIRECTION_DELTA:
                mx = cx + cell_w * 0.25
                my = cy - cell_h * 0.15

            ax.annotate("", xy=(mx, my), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=1.3, linestyle="dashed",
                                        connectionstyle="arc3,rad=0.2"),
                        zorder=3)

            pkt_label = f"{ss.port_type[0:3]}:{ss.port_num}→{mp.port_type[0:3]}:{mp.port_num}"
            mid_x, mid_y = (sx + mx) / 2, (sy + my) / 2
            offset = -0.15 * cell_h * (1 if bi % 2 == 0 else -1)
            ax.text(mid_x, mid_y + offset, f"[pkt] {pkt_label}",
                    ha="center", va="center", fontsize=4, color=color, style="italic",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8),
                    zorder=6)

    drawn_cross = set()
    for bi, block in enumerate(blocks):
        color = BLOCK_COLORS[bi % len(BLOCK_COLORS)]

        for cc in block.circuit_conns:
            mdir = cc.master_type
            if mdir not in DIRECTION_DELTA:
                continue
            dc, dr = DIRECTION_DELTA[mdir]
            nb = (cc.tile[0] + dc, cc.tile[1] + dr)
            if nb not in tile_centers or cc.tile not in tile_centers:
                continue
            key = (cc.tile, mdir, cc.master_port, nb, bi)
            if key in drawn_cross:
                continue
            drawn_cross.add(key)

            sx, sy = tile_port_pos(cc.tile, mdir, cc.master_port)
            opp = DIRECTION_OPPOSITES[mdir]
            ex, ey = tile_port_pos(nb, opp, cc.master_port)
            ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=1.8, connectionstyle="arc3,rad=0"),
                        zorder=4)

        for ss, mp in resolve_pkt_internal_edges(block):
            mdir = mp.port_type
            if mdir not in DIRECTION_DELTA:
                continue
            dc, dr = DIRECTION_DELTA[mdir]
            nb = (mp.tile[0] + dc, mp.tile[1] + dr)
            if nb not in tile_centers or mp.tile not in tile_centers:
                continue
            key = (mp.tile, mdir, mp.port_num, nb, bi)
            if key in drawn_cross:
                continue
            drawn_cross.add(key)

            sx, sy = tile_port_pos(mp.tile, mdir, mp.port_num)
            opp = DIRECTION_OPPOSITES[mdir]
            ex, ey = tile_port_pos(nb, opp, mp.port_num)
            ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=1.8, linestyle="dashed",
                                        connectionstyle="arc3,rad=0"),
                        zorder=4)

    for bi, block in enumerate(blocks):
        color = BLOCK_COLORS[bi % len(BLOCK_COLORS)]
        for sd in block.shim_dma:
            if sd.tile not in tile_centers:
                continue
            cx, cy = tile_centers[sd.tile]
            arrow = "▼" if sd.direction == "aie_to_shim" else "▲"
            ax.text(cx, cy - cell_h * 0.20,
                    f"{arrow} ShimDMA:{sd.port_num}",
                    ha="center", va="center", fontsize=5.5,
                    color=color, fontweight="bold", zorder=6)

    # --- DMA channel annotations inside tiles ---
    for tile, dma_lines in dma_tile_summary.items():
        if tile not in tile_centers:
            continue
        cx, cy = tile_centers[tile]
        y_start = cy - cell_h * 0.15
        for i, dl in enumerate(dma_lines[:3]):
            y_pos = y_start - (i + 1) * cell_h * 0.12
            dma_color = "#1565C0" if "MM2S" in dl else "#C62828"
            ax.text(cx, y_pos, dl,
                    ha="center", va="center", fontsize=4.2, color=dma_color,
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round,pad=0.08", fc="white",
                              ec="#BDBDBD", alpha=0.85, lw=0.4),
                    zorder=7)
        if len(dma_lines) > 3:
            y_pos = y_start - 4 * cell_h * 0.12
            ax.text(cx, y_pos, f"+{len(dma_lines) - 3} more",
                    ha="center", va="center", fontsize=3.5, color="#888",
                    zorder=7)

    legend_handles = []
    for bi, block in enumerate(blocks):
        color = BLOCK_COLORS[bi % len(BLOCK_COLORS)]
        legend_handles.append(mpatches.Patch(color=color, label=block.name))
    legend_handles.append(plt.Line2D([0], [0], color="gray", lw=1.5,
                                     label="Circuit switch (solid)"))
    legend_handles.append(plt.Line2D([0], [0], color="gray", lw=1.5,
                                     linestyle="dashed", label="Packet switch (dashed)"))
    ax.legend(handles=legend_handles, loc="lower center",
              ncol=len(legend_handles), fontsize=7, frameon=True,
              bbox_to_anchor=(0.5, -0.08))

    title = Path(output).stem.replace("_", " ").title() if output else "Routing Topology"
    ax.set_title(f"Connection Topology — {Path(args.input).name}",
                 fontsize=12, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig(output, dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"Topology diagram saved to: {output}")


def print_text_summary(blocks: list[Block], paths,
                       dma_tile_summary: Optional[dict] = None):
    """Print a text summary of all blocks, traced paths, and DMA channels."""
    for block in blocks:
        print(f"\n{'='*60}")
        print(f"  {block.name}")
        print(f"{'='*60}")

        if block.pkt_slaves or block.pkt_masters:
            print("\n  Packet-Switch Connections:")
            for ss, mp in resolve_pkt_internal_edges(block):
                print(f"    tile({ss.tile[0]},{ss.tile[1]}) "
                      f"{ss.port_type}:{ss.port_num} [PktId={ss.pkt_id}] "
                      f"→ {mp.port_type}:{mp.port_num}  (Arb={ss.arbiter})")

        if block.circuit_conns:
            print("\n  Circuit-Switch Connections:")
            for cc in block.circuit_conns:
                print(f"    tile({cc.tile[0]},{cc.tile[1]}) "
                      f"{cc.slave_type}:{cc.slave_port} → {cc.master_type}:{cc.master_port}")

        if block.shim_dma:
            print("\n  Shim DMA Ports:")
            for sd in block.shim_dma:
                d = "AIE→Shim" if sd.direction == "aie_to_shim" else "Shim→AIE"
                print(f"    tile({sd.tile[0]},{sd.tile[1]}) port {sd.port_num} ({d})")

    print(f"\n{'='*60}")
    print(f"  Traced End-to-End Paths")
    print(f"{'='*60}")
    for i, (bname, path) in enumerate(paths):
        hops = []
        for (tile, stype, sport, mtype, mport, mode) in path:
            if mtype:
                hops.append(f"({tile[0]},{tile[1]}) {stype}:{sport}→{mtype}:{mport}")
            elif mode == "shim_exit":
                hops.append(f"({tile[0]},{tile[1]}) → ShimDMA (SOUTH:{sport})")
            else:
                hops.append(f"({tile[0]},{tile[1]}) {stype}:{sport} [{mode}]")
        print(f"\n  [{bname}] Path {i+1}:")
        for j, h in enumerate(hops):
            prefix = "    └─► " if j == len(hops) - 1 else "    ├─► "
            print(f"{prefix}{h}")

    if dma_tile_summary:
        print(f"\n{'='*60}")
        print(f"  DMA Channels (from host file)")
        print(f"{'='*60}")
        for tile in sorted(dma_tile_summary):
            for dl in dma_tile_summary[tile]:
                print(f"    tile({tile[0]},{tile[1]}) {dl}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse routing.cc and draw connection topology diagram")
    parser.add_argument("input", help="Path to routing.cc")
    parser.add_argument("-o", "--output", default=None,
                        help="Output image path (default: <input>_topology.png)")
    parser.add_argument("--host", default=None,
                        help="Path to host.cc (or aie_control.cpp) for DMA channel info")
    parser.add_argument("--text-only", action="store_true",
                        help="Print text summary only, no diagram")
    args = parser.parse_args()

    if args.output is None:
        args.output = str(Path(args.input).with_suffix("")) + "_topology.png"

    blocks = parse_routing_file(args.input)
    if not blocks:
        print("No routing blocks found.", file=sys.stderr)
        sys.exit(1)

    tiles = collect_tiles(blocks)
    paths = trace_paths(blocks)

    dma_tile_summary = {}
    if args.host:
        dma_bds, dma_channels = parse_host_file(args.host)
        dma_tile_summary = _build_dma_tile_summary(dma_bds, dma_channels)
        for tile in dma_tile_summary:
            tiles.add(tile)

    print_text_summary(blocks, paths, dma_tile_summary)

    if not args.text_only:
        draw_topology(blocks, paths, tiles, args.output, dma_tile_summary)
