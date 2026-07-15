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


def _paths_to_json(paths):
    """Convert traced paths into JSON-serializable list."""
    result = []
    for bname, path in paths:
        hops = []
        for (tile, stype, sport, mtype, mport, mode) in path:
            hops.append({
                "tile": list(tile), "slave_type": stype, "slave_port": sport,
                "master_type": mtype, "master_port": mport, "mode": mode,
            })
        result.append({"block": bname, "hops": hops})
    return result


def _blocks_to_json(blocks):
    """Convert block data into JSON-serializable list for the interactive HTML."""
    result = []
    for block in blocks:
        b = {"name": block.name, "circuit": [], "pkt_edges": [], "shim_dma": []}
        for cc in block.circuit_conns:
            b["circuit"].append({
                "tile": list(cc.tile),
                "slave_type": cc.slave_type, "slave_port": cc.slave_port,
                "master_type": cc.master_type, "master_port": cc.master_port,
            })
        for ss, mp in resolve_pkt_internal_edges(block):
            b["pkt_edges"].append({
                "tile": list(ss.tile),
                "slave_type": ss.port_type, "slave_port": ss.port_num,
                "master_type": mp.port_type, "master_port": mp.port_num,
                "pkt_id": ss.pkt_id,
            })
        for sd in block.shim_dma:
            b["shim_dma"].append({
                "tile": list(sd.tile), "port_num": sd.port_num,
                "direction": sd.direction,
            })
        result.append(b)
    return result


def generate_interactive_html(blocks, paths, tiles, dma_tile_summary, output):
    """Generate a self-contained interactive HTML file with SVG topology and simulation."""
    import json

    tiles_list = sorted(tiles)
    cols = sorted({t[0] for t in tiles})
    rows = sorted({t[1] for t in tiles})
    min_col, max_col = min(cols), max(cols)
    min_row, max_row = min(rows), max(rows)
    all_cols = list(range(min_col, max_col + 1))
    all_rows = list(range(min_row, max_row + 1))

    cell_w, cell_h = 160, 130
    pad = 80

    paths_json = json.dumps(_paths_to_json(paths))
    blocks_json = json.dumps(_blocks_to_json(blocks))
    dma_json = json.dumps({f"{k[0]},{k[1]}": v for k, v in dma_tile_summary.items()})
    tiles_json = json.dumps([list(t) for t in tiles_list])

    tile_type_js = "function tileType(r){return r===0?'Shim':r<=2?'Mem':'AIE';}"

    svg_w = len(all_cols) * cell_w + 2 * pad
    svg_h = len(all_rows) * cell_h + 2 * pad + 60

    # Build SVG tile rectangles
    svg_tiles = []
    for ci, c in enumerate(all_cols):
        for ri, r in enumerate(all_rows):
            x = pad + ci * cell_w
            y = pad + (len(all_rows) - 1 - ri) * cell_h  # row 0 at bottom
            ttype = TILE_TYPE_BY_ROW(r)
            is_active = (c, r) in tiles
            fill = {"Shim": "#FFECB3", "Mem": "#C8E6C9", "AIE": "#BBDEFB"}.get(ttype, "#EEE")
            if not is_active:
                fill = "#F5F5F5"
            svg_tiles.append(
                f'<rect id="tile-{c}-{r}" class="tile{"" if is_active else " inactive"}" '
                f'x="{x+8}" y="{y+8}" width="{cell_w-16}" height="{cell_h-16}" rx="8" '
                f'fill="{fill}" stroke="{"#333" if is_active else "#CCC"}" '
                f'stroke-width="{"1.5" if is_active else "0.5"}" '
                f'data-col="{c}" data-row="{r}" />\n'
                f'<text x="{x+cell_w//2}" y="{y+30}" text-anchor="middle" '
                f'font-size="13" font-weight="bold" fill="{"#333" if is_active else "#AAA"}">'
                f'({c},{r})</text>\n'
                f'<text x="{x+cell_w//2}" y="{y+48}" text-anchor="middle" '
                f'font-size="11" fill="{"#666" if is_active else "#CCC"}">{ttype}</text>'
            )

    # Axis labels
    axis_labels = []
    for ci, c in enumerate(all_cols):
        x = pad + ci * cell_w + cell_w // 2
        axis_labels.append(f'<text x="{x}" y="{pad - 15}" text-anchor="middle" '
                           f'font-size="13" font-weight="bold">Col {c}</text>')
    for ri, r in enumerate(all_rows):
        y = pad + (len(all_rows) - 1 - ri) * cell_h + cell_h // 2
        axis_labels.append(f'<text x="{pad - 30}" y="{y+4}" text-anchor="middle" '
                           f'font-size="13" font-weight="bold">Row {r}</text>')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Interactive Routing Topology</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #f0f2f5; color: #333; }}
.header {{ background: #1a237e; color: white; padding: 16px 24px;
           display: flex; justify-content: space-between; align-items: center; }}
.header h1 {{ font-size: 18px; font-weight: 600; }}
.header .controls {{ display: flex; gap: 10px; }}
.header button {{ padding: 6px 16px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.3);
                  background: rgba(255,255,255,0.15); color: white; cursor: pointer;
                  font-size: 13px; transition: background 0.2s; }}
.header button:hover {{ background: rgba(255,255,255,0.3); }}
.main {{ display: flex; height: calc(100vh - 56px); }}
.svg-container {{ flex: 1; overflow: auto; padding: 20px; }}
.panel {{ width: 340px; background: white; border-left: 1px solid #ddd;
          overflow-y: auto; padding: 16px; }}
.panel h2 {{ font-size: 15px; margin-bottom: 12px; color: #1a237e; }}
.panel .log {{ font-family: 'Fira Code', monospace; font-size: 11.5px;
              line-height: 1.6; white-space: pre-wrap; }}
.panel .log .ok {{ color: #2e7d32; }}
.panel .log .err {{ color: #c62828; font-weight: 600; }}
.panel .log .info {{ color: #1565c0; }}
.panel .log .hop {{ color: #555; }}
svg .tile {{ cursor: pointer; transition: fill 0.3s; }}
svg .tile:hover {{ filter: brightness(0.92); }}
svg .tile.inactive {{ cursor: default; }}
.btn-send, .btn-recv {{ cursor: pointer; }}
.btn-send rect {{ fill: #1565c0; rx: 4; }}
.btn-send:hover rect {{ fill: #0d47a1; }}
.btn-recv rect {{ fill: #2e7d32; rx: 4; }}
.btn-recv:hover rect {{ fill: #1b5e20; }}
.btn-send text, .btn-recv text {{ fill: white; font-size: 9px; font-weight: 600;
                                   pointer-events: none; user-select: none; }}
.dma-label {{ font-family: 'Fira Code', monospace; font-size: 9px; fill: #555; }}
.path-line {{ stroke-width: 2.5; fill: none; opacity: 0; transition: opacity 0.3s; }}
.path-line.active {{ opacity: 1; }}
@keyframes flashRed {{
  0%, 100% {{ fill: inherit; }}
  25% {{ fill: #ef5350; }}
  50% {{ fill: inherit; }}
  75% {{ fill: #ef5350; }}
}}
@keyframes flashGreen {{
  0%, 100% {{ fill: inherit; }}
  25% {{ fill: #66bb6a; }}
  50% {{ fill: inherit; }}
  75% {{ fill: #66bb6a; }}
}}
.flash-red {{ animation: flashRed 0.8s ease 3; }}
.flash-green {{ animation: flashGreen 0.8s ease 2; }}
.error-badge {{ cursor: pointer; }}
.error-badge circle {{ fill: #c62828; }}
.error-badge text {{ fill: white; font-size: 10px; font-weight: bold; pointer-events: none; }}
#modal-overlay {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.45);
                  z-index: 100; justify-content: center; align-items: center; }}
#modal-overlay.visible {{ display: flex; }}
#modal {{ background: white; border-radius: 12px; padding: 24px; max-width: 520px;
          width: 90%; box-shadow: 0 8px 32px rgba(0,0,0,0.2); }}
#modal h3 {{ color: #c62828; margin-bottom: 12px; font-size: 16px; }}
#modal pre {{ background: #fafafa; border: 1px solid #eee; border-radius: 6px;
              padding: 12px; font-size: 12px; line-height: 1.6; overflow-x: auto;
              font-family: 'Fira Code', monospace; white-space: pre-wrap; }}
#modal button {{ margin-top: 16px; padding: 8px 20px; background: #1a237e; color: white;
                 border: none; border-radius: 6px; cursor: pointer; font-size: 13px; }}
#modal button:hover {{ background: #283593; }}
</style>
</head>
<body>
<div class="header">
  <h1>Interactive Routing Topology</h1>
  <div class="controls">
    <button onclick="resetAll()">Reset</button>
  </div>
</div>
<div class="main">
  <div class="svg-container">
    <svg id="topo" width="{svg_w}" height="{svg_h}" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
          <polygon points="0 0, 8 3, 0 6" fill="#888" />
        </marker>
        <marker id="arrowGreen" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
          <polygon points="0 0, 8 3, 0 6" fill="#2e7d32" />
        </marker>
        <marker id="arrowRed" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
          <polygon points="0 0, 8 3, 0 6" fill="#c62828" />
        </marker>
      </defs>
      <g id="axis-labels">{"".join(axis_labels)}</g>
      <g id="tiles">{"".join(svg_tiles)}</g>
      <g id="buttons"></g>
      <g id="dma-labels"></g>
      <g id="connections"></g>
      <g id="error-badges"></g>
    </svg>
  </div>
  <div class="panel">
    <h2>Simulation Log</h2>
    <div id="log" class="log"><span class="info">Click a Send or Receive button to simulate DMA packet flow.</span></div>
  </div>
</div>

<div id="modal-overlay" onclick="closeModal()">
  <div id="modal" onclick="event.stopPropagation()">
    <h3 id="modal-title">Error Details</h3>
    <pre id="modal-body"></pre>
    <button onclick="closeModal()">Close</button>
  </div>
</div>

<script>
const PATHS = {paths_json};
const BLOCKS = {blocks_json};
const DMA = {dma_json};
const TILES = {tiles_json};
const PAD = {pad}, CW = {cell_w}, CH = {cell_h};
const ALL_COLS = {json.dumps(all_cols)};
const ALL_ROWS = {json.dumps(all_rows)};
const NUM_ROWS = ALL_ROWS.length;
{tile_type_js}

function tileXY(col, row) {{
  const ci = ALL_COLS.indexOf(col);
  const ri = ALL_ROWS.indexOf(row);
  const x = PAD + ci * CW + CW / 2;
  const y = PAD + (NUM_ROWS - 1 - ri) * CH + CH / 2;
  return [x, y];
}}

function tileId(c, r) {{ return `tile-${{c}}-${{r}}`; }}
function tileKey(c, r) {{ return `${{c}},${{r}}`; }}

const DIR_DELTA = {{NORTH:[0,1], SOUTH:[0,-1], EAST:[1,0], WEST:[-1,0]}};
const DIR_OPP = {{NORTH:'SOUTH', SOUTH:'NORTH', EAST:'WEST', WEST:'EAST'}};

const activeTiles = new Set(TILES.map(t => tileKey(t[0], t[1])));
let errorBadges = [];
let activeAnimations = [];

function portPos(col, row, dir, portNum) {{
  const [cx, cy] = tileXY(col, row);
  const offsets = {{NORTH:[0,-0.38], SOUTH:[0,0.38], EAST:[0.40,0], WEST:[-0.40,0]}};
  const [dx, dy] = offsets[dir] || [0,0];
  let px = cx + dx * CW, py = cy + dy * CH;
  if (dir === 'NORTH' || dir === 'SOUTH') px += (portNum - 0.5) * 14;
  if (dir === 'EAST' || dir === 'WEST') py += (portNum - 0.5) * 14;
  return [px, py];
}}

// Draw buttons
(function drawButtons() {{
  const btnG = document.getElementById('buttons');
  const dmaG = document.getElementById('dma-labels');
  const drawn = new Set();

  // Send buttons on tiles that are DMA sources in routing paths
  const dmaSources = new Set();
  PATHS.forEach(p => {{
    const h0 = p.hops[0];
    if (h0 && (h0.slave_type === 'DMA' || h0.slave_type === 'CORE'))
      dmaSources.add(tileKey(h0.tile[0], h0.tile[1]));
  }});

  dmaSources.forEach(key => {{
    const [c, r] = key.split(',').map(Number);
    const [cx, cy] = tileXY(c, r);
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.classList.add('btn-send');
    g.innerHTML = `<rect x="${{cx-28}}" y="${{cy+20}}" width="56" height="18" rx="4"/>
                   <text x="${{cx}}" y="${{cy+32}}" text-anchor="middle">Send DMA</text>`;
    g.onclick = () => simulateSend(c, r);
    btnG.appendChild(g);
  }});

  // Receive buttons on shim tiles with DMA ports
  const shimPorts = new Set();
  BLOCKS.forEach(b => b.shim_dma.forEach(sd => shimPorts.add(tileKey(sd.tile[0], sd.tile[1]))));

  shimPorts.forEach(key => {{
    const [c, r] = key.split(',').map(Number);
    const [cx, cy] = tileXY(c, r);
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.classList.add('btn-recv');
    g.innerHTML = `<rect x="${{cx-28}}" y="${{cy+20}}" width="56" height="18" rx="4"/>
                   <text x="${{cx}}" y="${{cy+32}}" text-anchor="middle">Receive</text>`;
    g.onclick = () => simulateReceive(c, r);
    btnG.appendChild(g);
  }});

  // DMA channel labels
  for (const [key, lines] of Object.entries(DMA)) {{
    const [c, r] = key.split(',').map(Number);
    if (!activeTiles.has(key)) continue;
    const [cx, cy] = tileXY(c, r);
    lines.slice(0, 2).forEach((line, i) => {{
      const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      t.classList.add('dma-label');
      t.setAttribute('x', cx);
      t.setAttribute('y', cy + 55 + i * 13);
      t.setAttribute('text-anchor', 'middle');
      t.textContent = line;
      const color = line.includes('MM2S') ? '#1565c0' : '#c62828';
      t.setAttribute('fill', color);
      dmaG.appendChild(t);
    }});
    if (lines.length > 2) {{
      const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      t.classList.add('dma-label');
      t.setAttribute('x', cx);
      t.setAttribute('y', cy + 55 + 2 * 13);
      t.setAttribute('text-anchor', 'middle');
      t.setAttribute('fill', '#888');
      t.textContent = `+${{lines.length - 2}} more`;
      dmaG.appendChild(t);
    }}
  }}
}})();

function log(msg, cls) {{
  const el = document.getElementById('log');
  const span = document.createElement('span');
  if (cls) span.className = cls;
  span.textContent = msg + '\\n';
  el.appendChild(span);
  el.scrollTop = el.scrollHeight;
}}

function clearLog() {{
  document.getElementById('log').innerHTML = '';
}}

function flashTile(c, r, cls) {{
  const el = document.getElementById(tileId(c, r));
  if (!el) return;
  el.classList.remove('flash-red', 'flash-green');
  void el.offsetWidth;
  el.classList.add(cls);
  setTimeout(() => el.classList.remove(cls), 2500);
}}

function addErrorBadge(c, r, details) {{
  const [cx, cy] = tileXY(c, r);
  const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  g.classList.add('error-badge');
  g.innerHTML = `<circle cx="${{cx+CW*0.35}}" cy="${{cy-CH*0.35}}" r="10"/>
                 <text x="${{cx+CW*0.35}}" y="${{cy-CH*0.35+4}}" text-anchor="middle">!</text>`;
  g.onclick = (e) => {{ e.stopPropagation(); showModal(`Error at tile (${{c}},${{r}})`, details); }};
  document.getElementById('error-badges').appendChild(g);
  errorBadges.push(g);
}}

function showModal(title, body) {{
  document.getElementById('modal-title').textContent = title;
  document.getElementById('modal-body').textContent = body;
  document.getElementById('modal-overlay').classList.add('visible');
}}

function closeModal() {{
  document.getElementById('modal-overlay').classList.remove('visible');
}}

function drawPathSegment(x1, y1, x2, y2, color, dashed) {{
  const connG = document.getElementById('connections');
  const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
  line.setAttribute('x1', x1); line.setAttribute('y1', y1);
  line.setAttribute('x2', x2); line.setAttribute('y2', y2);
  line.setAttribute('stroke', color);
  line.setAttribute('stroke-width', '3');
  if (dashed) line.setAttribute('stroke-dasharray', '6,3');
  line.setAttribute('marker-end', color === '#c62828' ? 'url(#arrowRed)' : 'url(#arrowGreen)');
  line.style.opacity = '0';
  connG.appendChild(line);
  activeAnimations.push(line);
  return line;
}}

function animatePath(hops, color, onDone) {{
  let i = 0;
  function step() {{
    if (i >= hops.length) {{ if (onDone) onDone(); return; }}
    const hop = hops[i];
    const [c, r] = hop.tile;
    const mtype = hop.master_type;
    const mport = hop.master_port;
    const stype = hop.slave_type;
    const sport = hop.slave_port;

    if (mtype && DIR_DELTA[mtype]) {{
      const [dc, dr] = DIR_DELTA[mtype];
      const nb = [c + dc, r + dr];
      const [sx, sy] = portPos(c, r, mtype, mport);
      const [ex, ey] = portPos(nb[0], nb[1], DIR_OPP[mtype], mport);
      const line = drawPathSegment(sx, sy, ex, ey, color, false);
      requestAnimationFrame(() => line.style.opacity = '1');
    }}

    log(`  (${{c}},${{r}}) ${{stype}}:${{sport}}` +
        (mtype ? ` → ${{mtype}}:${{mport}}` : ` [${{hop.mode}}]`), 'hop');
    i++;
    setTimeout(step, 350);
  }}
  step();
}}

function simulateSend(col, row) {{
  resetAll();
  clearLog();
  log(`=== Send DMA from tile (${{col}},${{row}}) ===`, 'info');

  const matching = PATHS.filter(p => {{
    const h0 = p.hops[0];
    return h0.tile[0] === col && h0.tile[1] === row;
  }});

  if (matching.length === 0) {{
    log('No routing path originates from this tile.', 'err');
    flashTile(col, row, 'flash-red');
    addErrorBadge(col, row,
      `No routing path found originating from tile (${{col}},${{row}}).\\n` +
      `This tile has no DMA source in the stream-switch configuration.\\n\\n` +
      `Check that routing.cc contains a packet-switch slave or circuit-switch\\n` +
      `connection with DMA as slave type on this tile.`);
    return;
  }}

  let pathIdx = 0;
  function runNext() {{
    if (pathIdx >= matching.length) return;
    const p = matching[pathIdx];
    log(`\\n[${{p.block}}] Path ${{pathIdx + 1}}:`, 'info');
    const lastHop = p.hops[p.hops.length - 1];
    const isOk = lastHop.mode === 'shim_exit' || lastHop.mode === 'endpoint';
    const isDangling = lastHop.mode === 'dangling';

    animatePath(p.hops, isOk ? '#2e7d32' : '#c62828', () => {{
      if (isOk) {{
        const dest = lastHop.tile;
        log(`  ✓ Reached destination at (${{dest[0]}},${{dest[1]}})`, 'ok');
        flashTile(dest[0], dest[1], 'flash-green');
      }} else {{
        const brk = lastHop.tile;
        log(`  ✗ Path broken at (${{brk[0]}},${{brk[1]}}) — ${{lastHop.mode}}`, 'err');
        flashTile(brk[0], brk[1], 'flash-red');
        addErrorBadge(brk[0], brk[1],
          `Routing path from tile (${{col}},${{row}}) is broken at tile (${{brk[0]}},${{brk[1]}}).\\n\\n` +
          `Status: ${{lastHop.mode}}\\n` +
          `Last port: ${{lastHop.slave_type}}:${{lastHop.slave_port}}\\n\\n` +
          (isDangling ?
            `The master port at the previous tile sends data to (${{brk[0]}},${{brk[1]}}),\\n` +
            `but no matching slave connection was found on that tile.\\n\\n` +
            `Fix: Add an XAie_StrmConnCctEnable or packet-switch entry on\\n` +
            `tile (${{brk[0]}},${{brk[1]}}) with slave type ${{lastHop.slave_type}} port ${{lastHop.slave_port}}.`
            :
            `The path entered a ${{lastHop.mode}} state.\\nCheck stream-switch config for this tile.`
          ));
      }}
      pathIdx++;
      setTimeout(runNext, 500);
    }});
  }}
  runNext();
}}

function simulateReceive(col, row) {{
  resetAll();
  clearLog();
  log(`=== Receive at shim tile (${{col}},${{row}}) ===`, 'info');

  const arriving = PATHS.filter(p => {{
    const last = p.hops[p.hops.length - 1];
    return last.mode === 'shim_exit' && last.tile[0] === col && last.tile[1] === row;
  }});

  const broken = PATHS.filter(p => {{
    const last = p.hops[p.hops.length - 1];
    if (last.mode === 'shim_exit' || last.mode === 'endpoint') return false;
    for (const h of p.hops) {{
      if (h.master_type === 'SOUTH' && h.tile[1] === 1 &&
          h.tile[0] === col) return true;
      if (h.master_type === 'SOUTH' && h.tile[0] === col) return true;
    }}
    return false;
  }});

  if (arriving.length === 0 && broken.length === 0) {{
    log('No paths target this shim tile.', 'err');
    flashTile(col, row, 'flash-red');
    addErrorBadge(col, row,
      `No routing path reaches shim tile (${{col}},${{row}}).\\n\\n` +
      `No stream-switch path terminates at this tile's SOUTH port.\\n` +
      `Check that routing.cc has a complete path from AIE tiles down to this shim.`);
    return;
  }}

  if (broken.length > 0) {{
    log(`\\n${{broken.length}} path(s) targeting this column are broken:`, 'err');
    broken.forEach((p, i) => {{
      const src = p.hops[0].tile;
      const last = p.hops[p.hops.length - 1];
      log(`  Path from (${{src[0]}},${{src[1]}}): broken at (${{last.tile[0]}},${{last.tile[1]}}) — ${{last.mode}}`, 'err');
      flashTile(last.tile[0], last.tile[1], 'flash-red');
      addErrorBadge(last.tile[0], last.tile[1],
        `Path from (${{src[0]}},${{src[1]}}) heading toward shim (${{col}},${{row}}) ` +
        `is broken at (${{last.tile[0]}},${{last.tile[1]}}).\\n\\n` +
        `Status: ${{last.mode}}\\nPort: ${{last.slave_type}}:${{last.slave_port}}`);
    }});
  }}

  if (arriving.length > 0) {{
    log(`\\n${{arriving.length}} path(s) successfully reach this shim:`, 'ok');
    let idx = 0;
    function showNext() {{
      if (idx >= arriving.length) return;
      const p = arriving[idx];
      const src = p.hops[0].tile;
      log(`\\n[${{p.block}}] From (${{src[0]}},${{src[1]}}):`, 'info');
      animatePath(p.hops, '#2e7d32', () => {{
        log(`  ✓ Received successfully`, 'ok');
        flashTile(col, row, 'flash-green');
        idx++;
        setTimeout(showNext, 400);
      }});
    }}
    showNext();
  }}
}}

function resetAll() {{
  document.querySelectorAll('.flash-red, .flash-green').forEach(el => {{
    el.classList.remove('flash-red', 'flash-green');
  }});
  errorBadges.forEach(g => g.remove());
  errorBadges = [];
  activeAnimations.forEach(el => el.remove());
  activeAnimations = [];
}}
</script>
</body>
</html>"""

    with open(output, "w") as f:
        f.write(html)
    print(f"Interactive HTML saved to: {output}")


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
    parser.add_argument("--html", default=None,
                        help="Generate interactive HTML file (e.g. --html topology.html)")
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

    if args.html:
        generate_interactive_html(blocks, paths, tiles, dma_tile_summary, args.html)

    if not args.text_only and not args.html:
        draw_topology(blocks, paths, tiles, args.output, dma_tile_summary)
