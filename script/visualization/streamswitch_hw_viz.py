#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Stream switch hardware configuration visualizer.

Parses the JSON output produced by AieRt_PrintStreamSwitchConfig / All
(from src/mlir/runtime/aie_runtime_debug.c) and renders an interactive
HTML page showing the tile grid with stream switch state per tile:

  - Master ports: PKT / CIRC / CTRL, drop-header annotation
  - Slave ports + slots: match mode (exact / wildcard / partial), pkt_id
  - Inter-tile arrows derived from master port directions (NORTH/SOUTH/EAST/WEST)
  - Color coding: PKT=purple, CIRC=blue, CTRL=grey, exact-match slot=red warn
  - Tooltip on every arrow and slot showing event 148 risk

Default input:  src/mlir/mlirfront/tilinglinalg/pass/data/streamswitch.log
Default output: streamswitch_hw.html  (then served via HTTP)

Usage:
    python streamswitch_hw_viz.py                          # uses default log
    python streamswitch_hw_viz.py hwlog2                   # positional log file
    python streamswitch_hw_viz.py --input path/to/log      # named argument
    python streamswitch_hw_viz.py hwlog2 -o out.html --no-serve
    python streamswitch_hw_viz.py hwlog2 --port 8090
"""

import argparse
import http.server
import json
import re
import socket
import sys
import webbrowser
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Default log path (relative to this script's directory)
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_LOG = _SCRIPT_DIR / "../../src/mlir/mlirfront/tilinglinalg/pass/data/streamswitch.json"

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

DIRECTIONS = {"NORTH", "SOUTH", "EAST", "WEST"}
DIRECTION_DELTA = {"NORTH": (0, 1), "SOUTH": (0, -1), "EAST": (1, 0), "WEST": (-1, 0)}
DIRECTION_OPPOSITE = {"NORTH": "SOUTH", "SOUTH": "NORTH", "EAST": "WEST", "WEST": "EAST"}


@dataclass
class StreamFlow:
    """One logical stream originating from a shim tile SOUTH slave port."""
    flow_id: int
    shim_col: int
    shim_row: int
    shim_slave_port: str          # e.g. "SOUTH3"
    arrow_indices: List[int] = field(default_factory=list)
    port_keys: List[Tuple[int, int, str]] = field(default_factory=list)  # (col, row, port_name)
    tile_locs: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"({self.shim_col},{self.shim_row}) {self.shim_slave_port}"


def _dir_of(port_name: str) -> Optional[str]:
    """Return canonical direction string if port is a directional port, else None."""
    for d in DIRECTIONS:
        if port_name.startswith(d):
            return d
    return None


# ---------------------------------------------------------------------------
# Parser — the log is a sequence of JSON objects, possibly not wrapped in [ ]
# ---------------------------------------------------------------------------

def parse_log(text: str) -> List[dict]:
    """Parse a stream of JSON objects from the log file.

    AieRt_PrintStreamSwitchConfig emits one JSON object per tile without
    commas between them.  AieRt_PrintStreamSwitchConfigAll wraps them in
    [ ... ].  Both formats are supported.
    """
    text = text.strip()
    if not text:
        return []

    # Try direct array parse first
    if text.startswith("["):
        try:
            items = json.loads(text)
            return [t for t in items if "tile" in t]
        except json.JSONDecodeError:
            pass

    # Split on top-level { ... } objects
    tiles: List[dict] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                fragment = text[start : i + 1]
                try:
                    obj = json.loads(fragment)
                    tiles.append(obj)
                except json.JSONDecodeError:
                    pass
                start = -1
    return tiles


# ---------------------------------------------------------------------------
# Derive inter-tile connections from master port directions
# ---------------------------------------------------------------------------

def _port_num(port_name: str) -> int:
    """Extract trailing port number from port name, e.g. 'EAST1' -> 1."""
    m = re.search(r"(\d+)$", port_name)
    return int(m.group(1)) if m else 0


def build_cross_tile_arrows(tiles: List[dict]) -> List[dict]:
    """For each enabled master port pointing in a cardinal direction, emit an arrow
    to the nearest tile in that direction present in the log.  Scans past gaps
    (e.g. shim row 0 → core row 2 when no mem-tile at row 1 is logged)."""
    arrows = []
    tile_set = {
        (t["tile"]["col"], t["tile"]["row"])
        for t in tiles
    }
    if not tiles:
        return arrows
    all_cols = [t["tile"]["col"] for t in tiles]
    all_rows = [t["tile"]["row"] for t in tiles]
    max_span = max(max(all_cols) - min(all_cols),
                   max(all_rows) - min(all_rows), 1)
    for tile in tiles:
        col = tile["tile"]["col"]
        row = tile["tile"]["row"]
        for master in tile.get("masters", []):
            if not master.get("enabled"):
                continue
            port = master["port"]
            d = _dir_of(port)
            if d is None:
                continue
            dc, dr = DIRECTION_DELTA[d]
            # Scan in the direction to find the nearest tile in the log
            nb = None
            for step in range(1, max_span + 1):
                candidate = (col + dc * step, row + dr * step)
                if candidate in tile_set:
                    nb = candidate
                    break
            opp_dir = DIRECTION_OPPOSITE[d]
            port_num = _port_num(port)
            if nb is None:
                # Dangling: destination tile not in log — emit a stub arrow
                arrow = {
                    "from_tile": [col, row],
                    "to_tile": [col + dc, row + dr],  # virtual 1-step neighbor
                    "from_dir": d,
                    "to_dir": opp_dir,
                    "port": port_num,
                    "port_name": port,
                    "slave_port_name": f"{opp_dir}{port_num}",
                    "switch_type": master.get("switch_type", "CIRC"),
                    "drop_header": master.get("drop_header", False),
                    "drop_header_note": master.get("drop_header_note", ""),
                    "pkt_id": None,
                    "dangling": True,
                }
                arrows.append(arrow)
                continue
            arrow = {
                "from_tile": [col, row],
                "to_tile": list(nb),
                "from_dir": d,
                "to_dir": opp_dir,
                "port": port_num,
                "port_name": port,
                "slave_port_name": f"{opp_dir}{port_num}",
                "switch_type": master.get("switch_type", "CIRC"),
                "drop_header": master.get("drop_header", False),
                "drop_header_note": master.get("drop_header_note", ""),
                "pkt_id": None,
                "dangling": False,
            }
            arrows.append(arrow)
    return arrows


# ---------------------------------------------------------------------------
# BFS flow tracing from shim SOUTH slave ports
# ---------------------------------------------------------------------------

def trace_shim_flows(tiles: List[dict], arrows: List[dict]) -> List[StreamFlow]:
    """Trace stream flows from each shim tile's SOUTH slave port through the array.

    Uses the master port `config` hex field to determine which slave index feeds
    each master.  BFS follows: slave → masters reading that slave → cross-tile
    arrow → next tile's slave → repeat.
    """
    if not tiles or not arrows:
        return []

    # ---- lookup tables ----
    tile_by_loc: Dict[Tuple[int, int], dict] = {}
    for t in tiles:
        tile_by_loc[(t["tile"]["col"], t["tile"]["row"])] = t

    # Per-tile: slave index → slave port dict
    slave_by_idx: Dict[Tuple[int, int], Dict[int, dict]] = defaultdict(dict)
    # Per-tile: slave index → port name
    slave_idx_to_port: Dict[Tuple[int, int], Dict[int, str]] = defaultdict(dict)
    # Per-tile: slave index → list of master port dicts that read from it
    master_by_slave_idx: Dict[Tuple[int, int], Dict[int, list]] = defaultdict(lambda: defaultdict(list))

    for t in tiles:
        loc = (t["tile"]["col"], t["tile"]["row"])
        for s in t.get("slaves", []):
            idx = s.get("index")
            if idx is not None:
                slave_by_idx[loc][idx] = s
                slave_idx_to_port[loc][idx] = s.get("port", f"slave{idx}")
        for m in t.get("masters", []):
            if not m.get("enabled"):
                continue
            cfg = m.get("config")
            if cfg is None:
                continue
            try:
                src_slave_idx = int(cfg, 16)
            except (ValueError, TypeError):
                continue
            master_by_slave_idx[loc][src_slave_idx].append(m)

    # ---- arrow lookup: (from_col, from_row, master_port_name) → arrow index ----
    arrow_idx_by_from: Dict[Tuple[int, int, str], int] = {}
    for i, ar in enumerate(arrows):
        key = (ar["from_tile"][0], ar["from_tile"][1], ar["port_name"])
        arrow_idx_by_from[key] = i

    # ---- BFS from each shim SOUTH slave ----
    flows: List[StreamFlow] = []
    flow_id = 0

    for t in tiles:
        col, row = t["tile"]["col"], t["tile"]["row"]
        if row != 0:
            continue
        for s in t.get("slaves", []):
            if not s.get("enabled"):
                continue
            port = s.get("port", "")
            if not port.startswith("SOUTH"):
                continue
            slave_idx = s.get("index")
            if slave_idx is None:
                continue

            sf = StreamFlow(
                flow_id=flow_id,
                shim_col=col, shim_row=row,
                shim_slave_port=port,
            )
            sf.port_keys.append((col, row, port))
            sf.tile_locs.append((col, row))

            # BFS queue: (tile_col, tile_row, slave_index)
            queue = [(col, row, slave_idx)]
            visited = set()

            while queue:
                tc, tr, si = queue.pop(0)
                if (tc, tr, si) in visited:
                    continue
                visited.add((tc, tr, si))

                # Find all masters in this tile that read from slave index si
                for mast in master_by_slave_idx[(tc, tr)].get(si, []):
                    mport = mast.get("port", "")
                    sf.port_keys.append((tc, tr, mport))

                    # Find cross-tile arrow for this master
                    ai = arrow_idx_by_from.get((tc, tr, mport))
                    if ai is not None:
                        sf.arrow_indices.append(ai)
                        ar = arrows[ai]
                        if not ar.get("dangling"):
                            nb = (ar["to_tile"][0], ar["to_tile"][1])
                            if nb not in [l for l in sf.tile_locs]:
                                sf.tile_locs.append(nb)
                            # Find the slave port in the destination tile
                            slave_port_name = ar["slave_port_name"]
                            sf.port_keys.append((nb[0], nb[1], slave_port_name))
                            # Look up slave index in destination tile
                            dest_slaves = slave_by_idx.get(nb, {})
                            for ds_idx, ds in dest_slaves.items():
                                if ds.get("port") == slave_port_name:
                                    queue.append((nb[0], nb[1], ds_idx))
                                    break

            flows.append(sf)
            flow_id += 1

    return flows


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

_TYPE_COLORS = {
    "PKT":  {"bg": "#EDE7F6", "border": "#7B1FA2", "fg": "#4A148C"},
    "CIRC": {"bg": "#E3F2FD", "border": "#1565C0", "fg": "#0D47A1"},
    "CTRL": {"bg": "#F5F5F5", "border": "#9E9E9E", "fg": "#616161"},
}

_MATCH_COLORS = {
    "exact":    {"bg": "#FFEBEE", "border": "#C62828", "fg": "#B71C1C"},
    "wildcard": {"bg": "#E8F5E9", "border": "#2E7D32", "fg": "#1B5E20"},
    "partial":  {"bg": "#FFF3E0", "border": "#E65100", "fg": "#BF360C"},
}

_ARROW_COLORS = {
    "PKT":  "#7B1FA2",
    "CIRC": "#1565C0",
    "CTRL": "#9E9E9E",
}


def _type_style(sw_type: str) -> str:
    c = _TYPE_COLORS.get(sw_type, _TYPE_COLORS["CTRL"])
    return f"background:{c['bg']};border-color:{c['border']};color:{c['fg']}"


def _match_style(mode: str) -> str:
    c = _MATCH_COLORS.get(mode, _MATCH_COLORS["partial"])
    return f"background:{c['bg']};border:1px solid {c['border']};color:{c['fg']}"


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

def _esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _render_master(m: dict, col: int, row: int, flow_ids: Optional[List[int]] = None) -> str:
    sw = m.get("switch_type", "CIRC")
    enabled = m.get("enabled", False)
    pkt = m.get("packet_switch", False)
    drop = m.get("drop_header", False)
    note = m.get("drop_header_note", "")
    port = m.get("port", "?")
    style = _type_style(sw)
    opacity = "1" if enabled else "0.4"
    elem_id = f"mport-{col}-{row}-{port}"

    drop_badge = ""
    if pkt:
        if drop:
            drop_badge = '<span class="badge badge-drop" title="Drops header before forwarding">DROP_HDR</span>'
        else:
            drop_badge = '<span class="badge badge-keep" title="Preserves header for downstream slave">KEEP_HDR</span>'

    flow_attr = ""
    if flow_ids:
        flow_attr = f' data-flows="{",".join(str(f) for f in flow_ids)}"'

    tooltip = _esc(note) if note else _esc(f"port={port} type={sw}")
    return (
        f'<div id="{elem_id}" class="port-row" style="{style};opacity:{opacity}" title="{tooltip}"{flow_attr}>'
        f'<span class="port-name">{_esc(port)}</span>'
        f'<span class="type-badge">{_esc(sw)}</span>'
        f'{drop_badge}'
        f'</div>'
    )


def _render_slot(slot: dict) -> str:
    mode = slot.get("match_mode", "partial")
    pkt_id = slot.get("pkt_id", "?")
    mask = slot.get("pkt_mask", "?")
    note = slot.get("match_note", "")
    style = _match_style(mode)
    tooltip = _esc(note) if note else _esc(f"mode={mode} pkt_id={pkt_id} mask={mask}")
    warn = ' &#x26A0;' if mode == "exact" else ""
    return (
        f'<span class="slot-badge" style="{style}" title="{tooltip}">'
        f'slot:{slot.get("slot","?")} id={pkt_id} {_esc(mode)}{warn}'
        f'</span>'
    )


def _render_slave(s: dict, col: int, row: int, flow_ids: Optional[List[int]] = None) -> str:
    sw = s.get("switch_type", "CIRC")
    enabled = s.get("enabled", False)
    port = s.get("port", "?")
    style = _type_style(sw)
    opacity = "1" if enabled else "0.4"
    elem_id = f"sport-{col}-{row}-{port}"
    slots_html = ""
    active_slots = [sl for sl in s.get("slots", []) if sl.get("enabled")]
    if active_slots:
        slots_html = "".join(_render_slot(sl) for sl in active_slots)

    flow_attr = ""
    if flow_ids:
        flow_attr = f' data-flows="{",".join(str(f) for f in flow_ids)}"'

    return (
        f'<div id="{elem_id}" class="port-row slave-row" style="{style};opacity:{opacity}"{flow_attr}>'
        f'<span class="port-name">{_esc(port)}</span>'
        f'<span class="type-badge">{_esc(sw)}</span>'
        f'{slots_html}'
        f'</div>'
    )


def _render_tile_card(tile: dict, min_col: int, max_row: int,
                      port_flow_map: Optional[Dict] = None) -> str:
    col = tile["tile"]["col"]
    row = tile["tile"]["row"]
    tile_type = tile["tile"]["type"]

    gc = col - min_col + 1
    gr = max_row - row + 1

    type_cls = tile_type.lower()
    pfm = port_flow_map or {}

    masters = tile.get("masters", [])
    slaves  = tile.get("slaves", [])
    active_masters = [m for m in masters if m.get("enabled")]
    active_slaves  = [s for s in slaves  if s.get("enabled") or any(sl.get("enabled") for sl in s.get("slots", []))]

    is_active = bool(active_masters or active_slaves)

    masters_html = ""
    if active_masters:
        rows = "".join(
            _render_master(m, col, row, pfm.get((col, row, m.get("port", "?"))))
            for m in active_masters
        )
        masters_html = (
            f'<div class="section-hdr" onclick="toggleSection(this)">&#x25BC; Masters ({len(active_masters)})</div>'
            f'<div class="section-body expanded">{rows}</div>'
        )

    slaves_html = ""
    if active_slaves:
        rows = "".join(
            _render_slave(s, col, row, pfm.get((col, row, s.get("port", "?"))))
            for s in active_slaves
        )
        slaves_html = (
            f'<div class="section-hdr slave-hdr" onclick="toggleSection(this)">&#x25BC; Slaves ({len(active_slaves)})</div>'
            f'<div class="section-body expanded">{rows}</div>'
        )

    return (
        f'<div class="tile-card {type_cls}{"" if is_active else " inactive"}" '
        f'id="tile-{col}-{row}" data-col="{col}" data-row="{row}" '
        f'style="grid-column:{gc};grid-row:{gr}">'
        f'<div class="tile-hdr">({col},{row}) {tile_type}</div>'
        f'{slaves_html}'
        f'{masters_html}'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Full HTML page
# ---------------------------------------------------------------------------

def render_html(tiles: List[dict], arrows: List[dict], output_path: str,
                flows: Optional[List[StreamFlow]] = None) -> str:
    if not tiles:
        print("No tiles to render.", file=sys.stderr)
        return output_path

    flows = flows or []

    all_cols = sorted({t["tile"]["col"] for t in tiles})
    all_rows = sorted({t["tile"]["row"] for t in tiles}, reverse=True)
    min_col = all_cols[0]
    max_row = all_rows[0]
    ncols = len(all_cols)
    nrows = len(all_rows)

    # Build port → flow_ids map for HTML data attributes
    port_flow_map: Dict[Tuple[int, int, str], List[int]] = defaultdict(list)
    for sf in flows:
        for pk in sf.port_keys:
            if sf.flow_id not in port_flow_map[pk]:
                port_flow_map[pk].append(sf.flow_id)

    tile_cards = "".join(
        _render_tile_card(t, min_col, max_row, port_flow_map) for t in sorted(
            tiles, key=lambda t: (-t["tile"]["row"], t["tile"]["col"])
        )
    )

    # Build flow filter panel HTML
    flow_panel_html = ""
    if flows:
        # Group flows by shim column
        shim_groups: Dict[int, List[StreamFlow]] = defaultdict(list)
        for sf in flows:
            shim_groups[sf.shim_col].append(sf)

        panel_rows = ""
        for col in sorted(shim_groups.keys()):
            grp = shim_groups[col]
            checkboxes = ""
            for sf in grp:
                n_arrows = len(sf.arrow_indices)
                n_tiles = len(sf.tile_locs)
                checkboxes += (
                    f'<label class="flow-toggle">'
                    f'<input type="checkbox" checked data-flow-id="{sf.flow_id}" onchange="toggleFlow({sf.flow_id}, this.checked)">'
                    f' {_esc(sf.shim_slave_port)}'
                    f'<span class="flow-stat">{n_arrows} arrows, {n_tiles} tiles</span>'
                    f'</label>'
                )
            panel_rows += (
                f'<div class="flow-group">'
                f'<div class="flow-group-hdr">Shim Col {col}</div>'
                f'{checkboxes}'
                f'</div>'
            )

        flow_panel_html = (
            f'<div class="flow-filter-panel" id="flow-panel">'
            f'<div class="flow-panel-hdr" onclick="toggleFlowPanel()">'
            f'&#x25BC; Stream Flow Filter ({len(flows)} flows)</div>'
            f'<div class="flow-panel-body" id="flow-panel-body">'
            f'<div class="flow-panel-controls">'
            f'<button onclick="showAllFlows()">Show All</button>'
            f'<button onclick="hideAllFlows()">Hide All</button>'
            f'</div>'
            f'{panel_rows}'
            f'</div></div>'
        )

    arrows_json = json.dumps(arrows)
    flows_json = json.dumps([
        {"flow_id": sf.flow_id, "arrow_indices": sf.arrow_indices,
         "label": sf.label}
        for sf in flows
    ])

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AIE Stream Switch HW Configuration</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f0f2f5; padding: 20px; color: #333;
}}
h1 {{ text-align:center; margin-bottom:6px; font-size:21px; color:#1a1a2e; }}
.subtitle {{ text-align:center; margin-bottom:14px; font-size:12px; color:#888; }}

/* ---- controls ---- */
.controls {{
    display:flex; justify-content:center; gap:10px;
    margin-bottom:14px; flex-wrap:wrap;
}}
.controls button {{
    padding:5px 14px; border:1px solid #bbb; border-radius:6px;
    background:#fff; cursor:pointer; font-size:12px; font-weight:600;
    transition:all .15s;
}}
.controls button:hover {{ background:#e3f2fd; border-color:#42a5f5; }}
.controls button.active {{ background:#1976d2; color:#fff; border-color:#1976d2; }}

/* ---- legend ---- */
.legend {{
    display:flex; gap:12px; justify-content:center;
    margin-bottom:14px; font-size:11px; flex-wrap:wrap;
    align-items:center;
}}
.legend-item {{ display:flex; align-items:center; gap:4px; }}
.legend-swatch {{
    width:13px; height:13px; border-radius:3px; border:2px solid;
    flex-shrink:0;
}}

/* ---- grid ---- */
.grid-outer {{ display:flex; justify-content:center; }}
.grid-wrapper {{ position:relative; display:inline-block; }}
.grid {{
    display:grid;
    grid-template-columns: repeat({ncols}, 260px);
    grid-template-rows: repeat({nrows}, auto);
    gap:12px;
    position:relative; z-index:1;
}}
svg.overlay {{
    position:absolute; top:0; left:0;
    width:100%; height:100%;
    pointer-events:none; z-index:10;
}}

/* ---- tile cards ---- */
.tile-card {{
    border:2px solid #bbb; border-radius:10px;
    padding:8px; min-height:60px;
    background:#fff; transition:box-shadow .15s, transform .15s;
}}
.tile-card:hover {{ box-shadow:0 4px 16px rgba(0,0,0,.15); transform:translateY(-1px); }}
.tile-card.inactive {{ opacity:0.35; }}
.tile-card.shim {{ background:#c8e6c9; border-color:#2e7d32; }}
.tile-card.core {{ background:#fffde7; border-color:#fdd835; }}
.tile-card.mem  {{ background:#e8f5e9; border-color:#66bb6a; }}
.tile-hdr {{
    font-weight:700; font-size:13px;
    padding-bottom:4px; border-bottom:1px solid #ddd;
    margin-bottom:5px;
}}
.section-hdr {{
    font-size:11px; font-weight:700; color:#555;
    cursor:pointer; padding:2px 0; user-select:none;
    transition:color .1s;
}}
.section-hdr:hover {{ color:#1976d2; }}
.slave-hdr {{ color:#5d4037; }}
.slave-hdr:hover {{ color:#bf360c; }}
.section-body {{ overflow:hidden; transition:max-height .3s ease; }}
.section-body.expanded {{ max-height:600px; }}
.section-body.collapsed {{ max-height:0; }}

/* ---- port rows ---- */
.port-row {{
    display:flex; align-items:center; gap:4px; flex-wrap:wrap;
    padding:3px 6px; margin:2px 0; border-radius:4px;
    border-left:3px solid transparent; font-size:10px;
    font-family:monospace;
}}
.slave-row {{ border-left-width:2px; border-style:dashed; }}
.port-name {{ font-weight:700; min-width:52px; }}
.type-badge {{
    font-size:9px; font-weight:800; padding:1px 5px;
    border-radius:3px; background:rgba(0,0,0,.08);
    flex-shrink:0;
}}

/* ---- badges ---- */
.badge {{
    font-size:9px; font-weight:700; padding:1px 5px; border-radius:3px; flex-shrink:0;
}}
.badge-drop {{ background:#fce4ec; color:#c62828; border:1px solid #ef9a9a; }}
.badge-keep {{ background:#e8f5e9; color:#2e7d32; border:1px solid #a5d6a7; }}

/* ---- slot badges ---- */
.slot-badge {{
    font-size:9px; padding:1px 5px; border-radius:3px; margin-left:2px;
    flex-shrink:0;
}}

/* ---- tooltip ---- */
.arrow-tip {{
    position:absolute; background:rgba(30,30,30,.92); color:#fff;
    padding:4px 10px; border-radius:5px; font-size:11px;
    pointer-events:none; z-index:100; white-space:nowrap;
    display:none; font-family:monospace;
}}

/* ---- flow filter panel ---- */
.flow-filter-panel {{
    max-width:600px; margin:0 auto 14px; border:1px solid #bbb;
    border-radius:8px; background:#fff; font-size:12px;
}}
.flow-panel-hdr {{
    font-weight:700; padding:6px 12px; cursor:pointer;
    user-select:none; border-bottom:1px solid #eee;
}}
.flow-panel-hdr:hover {{ color:#1976d2; }}
.flow-panel-body {{ padding:6px 12px; }}
.flow-panel-body.collapsed {{ display:none; }}
.flow-panel-controls {{
    display:flex; gap:6px; margin-bottom:6px;
}}
.flow-panel-controls button {{
    padding:3px 10px; border:1px solid #bbb; border-radius:4px;
    background:#fff; cursor:pointer; font-size:11px; font-weight:600;
}}
.flow-panel-controls button:hover {{ background:#e3f2fd; }}
.flow-group {{ margin-bottom:6px; }}
.flow-group-hdr {{
    font-weight:700; font-size:11px; color:#555; margin-bottom:2px;
}}
.flow-toggle {{
    display:flex; align-items:center; gap:4px; padding:1px 0;
    cursor:pointer; font-family:monospace; font-size:11px;
}}
.flow-toggle input {{ cursor:pointer; }}
.flow-stat {{
    font-size:9px; color:#888; margin-left:4px;
}}
.port-row.flow-dimmed {{ opacity:0.15 !important; }}
</style>
</head>
<body>
<h1>AIE Stream Switch — HW Configuration</h1>
<div class="subtitle">Parsed from AieRt_PrintStreamSwitchConfig JSON output</div>

<div class="controls">
    <button onclick="expandAll()">Expand All</button>
    <button onclick="collapseAll()">Collapse All</button>
    <button id="btn-arrows" class="active" onclick="toggleArrows(this)">Stream Arrows</button>
</div>

{flow_panel_html}

<div class="legend">
    <div class="legend-item">
        <div class="legend-swatch" style="background:#EDE7F6;border-color:#7B1FA2"></div> PKT (packet-switch)
    </div>
    <div class="legend-item">
        <div class="legend-swatch" style="background:#E3F2FD;border-color:#1565C0"></div> CIRC (circuit-switch)
    </div>
    <div class="legend-item">
        <div class="legend-swatch" style="background:#F5F5F5;border-color:#9E9E9E"></div> CTRL / FIFO / TRACE
    </div>
    <div class="legend-item">
        <div class="legend-swatch" style="background:#FFEBEE;border-color:#C62828"></div> exact slot (event 148 risk)
    </div>
    <div class="legend-item">
        <div class="legend-swatch" style="background:#E8F5E9;border-color:#2E7D32"></div> wildcard slot (merge point)
    </div>
    <div class="legend-item">
        <div class="legend-swatch" style="background:#fce4ec;border-color:#ef9a9a"></div> DROP_HEADER
    </div>
    <div class="legend-item">
        <div class="legend-swatch" style="background:#e8f5e9;border-color:#a5d6a7"></div> KEEP_HEADER
    </div>
    <div class="legend-item">
        <div class="legend-swatch" style="background:#fffde7;border-color:#fdd835"></div> CORE tile
    </div>
    <div class="legend-item">
        <div class="legend-swatch" style="background:#c8e6c9;border-color:#2e7d32"></div> SHIM tile
    </div>
    <div class="legend-item">
        <div class="legend-swatch" style="background:#FFF3E0;border-color:#E65100"></div> dangling (target not in log)
    </div>
</div>

<div class="grid-outer">
<div class="grid-wrapper">
<div class="grid" id="tile-grid">
{tile_cards}
</div>
<svg class="overlay" id="routing-svg">
<defs>
    <marker id="arr-pkt"  markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
        <polygon points="0 0,8 3,0 6" fill="#7B1FA2"/>
    </marker>
    <marker id="arr-circ" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
        <polygon points="0 0,8 3,0 6" fill="#1565C0"/>
    </marker>
    <marker id="arr-ctrl" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
        <polygon points="0 0,8 3,0 6" fill="#9E9E9E"/>
    </marker>
</defs>
</svg>
<div class="arrow-tip" id="arrow-tip"></div>
</div>
</div>

<script>
const ARROWS = {arrows_json};
const FLOWS = {flows_json};
let arrowsVisible = true;

/* ---- build arrow→flow_ids lookup ---- */
const arrowFlowIds = {{}};
FLOWS.forEach(f => {{
    f.arrow_indices.forEach(ai => {{
        if (!arrowFlowIds[ai]) arrowFlowIds[ai] = [];
        if (!arrowFlowIds[ai].includes(f.flow_id)) arrowFlowIds[ai].push(f.flow_id);
    }});
}});

/* ---- flow filter state ---- */
const hiddenFlows = new Set();

function toggleFlow(flowId, checked) {{
    if (checked) {{
        hiddenFlows.delete(flowId);
    }} else {{
        hiddenFlows.add(flowId);
    }}
    applyFlowVisibility();
    drawArrows();
}}

function showAllFlows() {{
    hiddenFlows.clear();
    document.querySelectorAll('#flow-panel input[data-flow-id]').forEach(cb => {{ cb.checked = true; }});
    applyFlowVisibility();
    drawArrows();
}}

function hideAllFlows() {{
    FLOWS.forEach(f => hiddenFlows.add(f.flow_id));
    document.querySelectorAll('#flow-panel input[data-flow-id]').forEach(cb => {{ cb.checked = false; }});
    applyFlowVisibility();
    drawArrows();
}}

function toggleFlowPanel() {{
    const body = document.getElementById('flow-panel-body');
    const hdr = body.previousElementSibling;
    if (body.classList.contains('collapsed')) {{
        body.classList.remove('collapsed');
        hdr.innerHTML = hdr.innerHTML.replace('\\u25B6','\\u25BC');
    }} else {{
        body.classList.add('collapsed');
        hdr.innerHTML = hdr.innerHTML.replace('\\u25BC','\\u25B6');
    }}
}}

function applyFlowVisibility() {{
    /* dim/undim port rows based on hidden flows */
    document.querySelectorAll('.port-row[data-flows]').forEach(el => {{
        const ids = el.dataset.flows.split(',').map(Number);
        const allHidden = ids.length > 0 && ids.every(id => hiddenFlows.has(id));
        el.classList.toggle('flow-dimmed', allHidden);
    }});
}}

/* ---- section toggle ---- */
function toggleSection(hdr) {{
    const body = hdr.nextElementSibling;
    if (body.classList.contains('collapsed')) {{
        body.classList.remove('collapsed'); body.classList.add('expanded');
        hdr.innerHTML = hdr.innerHTML.replace('\\u25B6','\\u25BC');
    }} else {{
        body.classList.remove('expanded'); body.classList.add('collapsed');
        hdr.innerHTML = hdr.innerHTML.replace('\\u25BC','\\u25B6');
    }}
    setTimeout(drawArrows, 320);
}}
function expandAll() {{
    document.querySelectorAll('.section-body').forEach(b => {{
        b.classList.remove('collapsed'); b.classList.add('expanded');
    }});
    document.querySelectorAll('.section-hdr').forEach(h => {{
        h.innerHTML = h.innerHTML.replace('\\u25B6','\\u25BC');
    }});
    setTimeout(drawArrows, 320);
}}
function collapseAll() {{
    document.querySelectorAll('.section-body').forEach(b => {{
        b.classList.remove('expanded'); b.classList.add('collapsed');
    }});
    document.querySelectorAll('.section-hdr').forEach(h => {{
        h.innerHTML = h.innerHTML.replace('\\u25BC','\\u25B6');
    }});
    setTimeout(drawArrows, 320);
}}
function toggleArrows(btn) {{
    arrowsVisible = !arrowsVisible;
    btn.classList.toggle('active', arrowsVisible);
    document.getElementById('routing-svg').style.display = arrowsVisible ? '' : 'none';
}}

/* ---- arrow drawing (curved paths to avoid overlap) ---- */
function drawArrows() {{
    const svg     = document.getElementById('routing-svg');
    const wrapper = document.querySelector('.grid-wrapper');
    const wRect   = wrapper.getBoundingClientRect();

    svg.setAttribute('width',   wrapper.offsetWidth);
    svg.setAttribute('height',  wrapper.offsetHeight);
    svg.setAttribute('viewBox', `0 0 ${{wrapper.offsetWidth}} ${{wrapper.offsetHeight}}`);

    [...svg.querySelectorAll('.data-arrow')].forEach(el => el.remove());

    const tileEls = {{}};
    document.querySelectorAll('.tile-card').forEach(card => {{
        tileEls[card.dataset.col + ',' + card.dataset.row] = card;
    }});

    const tip   = document.getElementById('arrow-tip');
    const dedup = new Set();

    /* --- first pass: collect unique arrows, group by tile pair --- */
    const items      = [];
    const pairGroups = {{}};

    ARROWS.forEach((arrow, origIdx) => {{
        const fromKey = arrow.from_tile[0] + ',' + arrow.from_tile[1];
        const toKey   = arrow.to_tile[0]   + ',' + arrow.to_tile[1];
        const deKey   = `${{fromKey}}-${{toKey}}-${{arrow.port_name}}-${{arrow.switch_type}}`;
        if (dedup.has(deKey)) return;
        dedup.add(deKey);

        /* skip arrows whose every flow is hidden */
        const fids = arrowFlowIds[origIdx] || [];
        if (fids.length > 0 && fids.every(id => hiddenFlows.has(id))) return;

        const pairKey = fromKey < toKey ? `${{fromKey}}|${{toKey}}` : `${{toKey}}|${{fromKey}}`;
        if (!pairGroups[pairKey]) pairGroups[pairKey] = [];
        const idx = pairGroups[pairKey].length;
        pairGroups[pairKey].push(arrow);
        items.push({{ arrow, pairKey, idx, origIdx }});
    }});

    /* --- second pass: draw curved paths --- */
    const STUB_LEN = 40;   /* px length of dangling stub arrows */

    items.forEach(({{ arrow, pairKey, idx, origIdx }}) => {{
        const fromKey = arrow.from_tile[0] + ',' + arrow.from_tile[1];
        const toKey   = arrow.to_tile[0]   + ',' + arrow.to_tile[1];
        const isDangling = !!arrow.dangling;

        const fromPortId = `mport-${{arrow.from_tile[0]}}-${{arrow.from_tile[1]}}-${{arrow.port_name}}`;
        const toPortId   = `sport-${{arrow.to_tile[0]}}-${{arrow.to_tile[1]}}-${{arrow.slave_port_name}}`;
        const fromPortEl = document.getElementById(fromPortId);
        const toPortEl   = document.getElementById(toPortId);

        let x1, y1, x2, y2;
        const portLevelConnect = !isDangling && fromPortEl && toPortEl;

        if (isDangling) {{
            /* Dangling: destination tile not in log — draw a short stub from tile edge */
            const fromEl = tileEls[fromKey];
            if (!fromEl) return;
            const fR = fromEl.getBoundingClientRect();
            switch (arrow.from_dir) {{
                case 'NORTH':
                    x1 = fR.left + fR.width/2 - wRect.left; y1 = fR.top    - wRect.top;
                    x2 = x1;                                  y2 = y1 - STUB_LEN;
                    break;
                case 'SOUTH':
                    x1 = fR.left + fR.width/2 - wRect.left; y1 = fR.bottom - wRect.top;
                    x2 = x1;                                  y2 = y1 + STUB_LEN;
                    break;
                case 'EAST':
                    x1 = fR.right - wRect.left; y1 = fR.top + fR.height/2 - wRect.top;
                    x2 = x1 + STUB_LEN;         y2 = y1;
                    break;
                case 'WEST':
                    x1 = fR.left - wRect.left;  y1 = fR.top + fR.height/2 - wRect.top;
                    x2 = x1 - STUB_LEN;          y2 = y1;
                    break;
            }}
        }} else if (portLevelConnect) {{
            const fPR = fromPortEl.getBoundingClientRect();
            const tPR = toPortEl.getBoundingClientRect();
            const d   = arrow.from_dir;

            if (d === 'EAST') {{
                x1 = fPR.right  - wRect.left;
                y1 = fPR.top + fPR.height / 2 - wRect.top;
                x2 = tPR.left   - wRect.left;
                y2 = tPR.top + tPR.height / 2 - wRect.top;
            }} else if (d === 'WEST') {{
                x1 = fPR.left   - wRect.left;
                y1 = fPR.top + fPR.height / 2 - wRect.top;
                x2 = tPR.right  - wRect.left;
                y2 = tPR.top + tPR.height / 2 - wRect.top;
            }} else if (d === 'NORTH') {{
                x1 = fPR.left + fPR.width / 2 - wRect.left;
                y1 = fPR.top    - wRect.top;
                x2 = tPR.left + tPR.width / 2 - wRect.left;
                y2 = tPR.bottom - wRect.top;
            }} else {{
                x1 = fPR.left + fPR.width / 2 - wRect.left;
                y1 = fPR.bottom - wRect.top;
                x2 = tPR.left + tPR.width / 2 - wRect.left;
                y2 = tPR.top    - wRect.top;
            }}
        }} else {{
            const fromEl = tileEls[fromKey];
            const toEl   = tileEls[toKey];
            if (!fromEl || !toEl) return;

            const fR = fromEl.getBoundingClientRect();
            const tR = toEl.getBoundingClientRect();

            switch (arrow.from_dir) {{
                case 'NORTH':
                    x1 = fR.left + fR.width/2 - wRect.left; y1 = fR.top    - wRect.top;
                    x2 = tR.left + tR.width/2 - wRect.left; y2 = tR.bottom - wRect.top;
                    break;
                case 'SOUTH':
                    x1 = fR.left + fR.width/2 - wRect.left; y1 = fR.bottom - wRect.top;
                    x2 = tR.left + tR.width/2 - wRect.left; y2 = tR.top    - wRect.top;
                    break;
                case 'EAST':
                    x1 = fR.right - wRect.left; y1 = fR.top + fR.height/2 - wRect.top;
                    x2 = tR.left  - wRect.left; y2 = tR.top + tR.height/2 - wRect.top;
                    break;
                case 'WEST':
                    x1 = fR.left  - wRect.left; y1 = fR.top + fR.height/2 - wRect.top;
                    x2 = tR.right - wRect.left; y2 = tR.top + tR.height/2 - wRect.top;
                    break;
            }}
        }}

        /* --- curve offset: perpendicular displacement to separate overlapping arrows --- */
        const total  = pairGroups[pairKey].length;
        const spread = (idx - (total - 1) / 2) * 30;
        const dx  = x2 - x1;
        const dy  = y2 - y1;
        const len = Math.sqrt(dx * dx + dy * dy) || 1;
        const px  = -dy / len;   /* perpendicular unit vector */
        const py  =  dx / len;

        const mx = (x1 + x2) / 2 + px * spread;
        const my = (y1 + y2) / 2 + py * spread;

        const pathD = (total === 1 && spread === 0)
            ? `M ${{x1}} ${{y1}} L ${{x2}} ${{y2}}`
            : `M ${{x1}} ${{y1}} Q ${{mx}} ${{my}} ${{x2}} ${{y2}}`;

        const sw       = arrow.switch_type;
        const color    = isDangling ? '#E65100'
                       : sw === 'PKT' ? '#7B1FA2' : sw === 'CIRC' ? '#1565C0' : '#9E9E9E';
        const markerId = sw === 'PKT' ? 'arr-pkt'  : sw === 'CIRC' ? 'arr-circ' : 'arr-ctrl';
        const dashArr  = isDangling ? '4,4' : (sw === 'PKT' ? '6,3' : 'none');

        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.classList.add('data-arrow');
        path.setAttribute('d', pathD);
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke', color);
        path.setAttribute('stroke-width', isDangling ? '2' : (sw === 'CIRC' ? '2.5' : '2'));
        if (dashArr !== 'none') path.setAttribute('stroke-dasharray', dashArr);
        path.setAttribute('marker-end', `url(#${{markerId}})`);
        const pathFlowIds = arrowFlowIds[origIdx] || [];
        if (pathFlowIds.length) path.setAttribute('data-flows', pathFlowIds.join(','));
        path.style.pointerEvents = 'stroke';
        path.style.cursor = 'pointer';

        const drop    = arrow.drop_header ? 'DROP_HEADER' : (sw === 'PKT' ? 'KEEP_HEADER' : '');
        let tipText;
        if (isDangling) {{
            tipText = `(${{fromKey}}) ${{arrow.port_name}} \u2192 ${{arrow.from_dir}} [DANGLING: target tile not in log] [${{sw}}]`;
        }} else {{
            const portTgt = portLevelConnect
                ? `${{arrow.port_name}} \u2192 ${{arrow.slave_port_name}}`
                : `${{arrow.port_name}} \u2192 (tile edge fallback)`;
            tipText = `(${{fromKey}}) ${{portTgt}} (${{toKey}}) [${{sw}}${{drop ? ' ' + drop : ''}}]${{arrow.drop_header_note ? ': ' + arrow.drop_header_note : ''}}`;
        }}

        path.addEventListener('mouseenter', () => {{
            tip.textContent = tipText;
            tip.style.display = 'block';
        }});
        path.addEventListener('mousemove', e => {{
            tip.style.left = (e.clientX - wRect.left + 10) + 'px';
            tip.style.top  = (e.clientY - wRect.top  - 24) + 'px';
        }});
        path.addEventListener('mouseleave', () => {{ tip.style.display = 'none'; }});

        svg.appendChild(path);
    }});
}}

window.addEventListener('load', () => {{ drawArrows(); }});
window.addEventListener('resize', drawArrows);
</script>
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
    abs_path = Path(html_path).resolve()
    directory = str(abs_path.parent)
    filename = abs_path.name

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=directory, **kw)
        def log_message(self, fmt, *args):
            pass

    hostname = socket.gethostname()
    local_url  = f"http://localhost:{port}/{filename}"
    remote_url = f"http://{hostname}:{port}/{filename}"
    print("Serving stream switch visualization:")
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
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize AIE stream switch HW configuration from "
            "AieRt_PrintStreamSwitchConfig JSON log."
        )
    )
    parser.add_argument(
        "logfile",
        nargs="?",
        default=None,
        help=f"Log file to visualize (default: {_DEFAULT_LOG})",
    )
    parser.add_argument(
        "--input", "-i",
        default=None,
        help=f"Path to streamswitch.log (default: {_DEFAULT_LOG})",
    )
    parser.add_argument(
        "-o", "--output",
        default="streamswitch_hw.html",
        help="Output HTML file (default: streamswitch_hw.html)",
    )
    parser.add_argument(
        "--no-serve", action="store_true",
        help="Write HTML only, do not start HTTP server",
    )
    parser.add_argument(
        "--port", type=int, default=8091,
        help="HTTP server port (default: 8091)",
    )
    args = parser.parse_args()

    # Positional arg takes priority, then --input, then default
    input_path = Path(args.logfile if args.logfile else (args.input if args.input else str(_DEFAULT_LOG)))
    if not input_path.exists():
        print(f"Error: log file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    text = input_path.read_text()
    tiles = parse_log(text)
    if not tiles:
        print("No tile data found in log.", file=sys.stderr)
        sys.exit(1)

    active = [t for t in tiles if t.get("masters") or t.get("slaves")]
    print(f"Parsed {len(tiles)} tiles ({len(active)} with active ports)")

    arrows = build_cross_tile_arrows(tiles)
    print(f"Derived {len(arrows)} cross-tile stream arrows")

    flows = trace_shim_flows(tiles, arrows)
    print(f"Traced {len(flows)} stream flows from shim SOUTH slave ports")

    # Annotate each arrow with its flow_ids for the JSON payload
    arrow_flow_map: Dict[int, List[int]] = defaultdict(list)
    for sf in flows:
        for ai in sf.arrow_indices:
            if sf.flow_id not in arrow_flow_map[ai]:
                arrow_flow_map[ai].append(sf.flow_id)
    for ai, fids in arrow_flow_map.items():
        arrows[ai]["flow_ids"] = fids

    render_html(tiles, arrows, args.output, flows=flows)

    if not args.no_serve:
        serve_html(args.output, args.port)


if __name__ == "__main__":
    main()
