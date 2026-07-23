#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""debug_ui_mcp - MCP server exposing the *static* schedule-view UI data.

This is a UI-only companion to the `aiegdb` MCP server (aiemcp.py). It is spawned
ONLY by schedule_debug_server.py for the browser "LLM tab", so its tools are NOT
visible to general Claude Code / CLI sessions (those use the repo-root .mcp.json
`aiegdb` server, which stays untouched).

Purpose: give the embedded LLM the same per-tile information the human sees in the
schedule view (host_schedule.html) without it having to read/parse the large
schedule_view.json blob by hand. Where `aiegdb`/aiemcp expose LIVE hardware state,
this server exposes the STATIC compiled schedule.

This is intended to grow: add more UI-facing tools here over time (flow lists,
supply/demand rollups, kernel-arg maps, etc.). Keep live-hardware tools in
aiemcp.py and static schedule/UI tools here.

Configuration (env, set by schedule_debug_server's temp .mcp.json):

    DEBUGUI_JSON_DIR    dir holding schedule_view.json (default: auto-detect)
    AIEMCP_JSON_DIR     fallback dir (shared with aiemcp.py)

Run standalone (stdio transport):  python3 src/tool/debug/debug_ui_mcp.py
Protocol smoke test:               mcp dev src/tool/debug/debug_ui_mcp.py
"""

import json
import os
import sys

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("debugui")


# ── schedule_view.json loading ───────────────────────────────────────────────
_VIEW_CACHE = {"path": None, "data": None}


def _view_candidates():
    """Candidate schedule_view.json paths, in priority order."""
    out = []
    for env in ("DEBUGUI_JSON_DIR", "AIEMCP_JSON_DIR"):
        d = os.environ.get(env)
        if d:
            out.append(os.path.join(d, "schedule_view.json"))
    out += [
        "./aout/worklocal/schedule_view.json",
        "./worklocal/schedule_view.json",
    ]
    return out


def _load_view():
    """Load schedule_view.json (cached by resolved path). Returns dict or None."""
    for p in _view_candidates():
        if os.path.isfile(p):
            ap = os.path.abspath(p)
            if _VIEW_CACHE["path"] == ap and _VIEW_CACHE["data"] is not None:
                return _VIEW_CACHE["data"]
            try:
                with open(ap) as f:
                    data = json.load(f)
            except (OSError, ValueError):
                continue
            _VIEW_CACHE["path"] = ap
            _VIEW_CACHE["data"] = data
            return data
    return None


def _find_tile(view, col, row):
    for t in view.get("tiles", []) or []:
        loc = t.get("loc") or []
        if len(loc) == 2 and loc[0] == col and loc[1] == row:
            return t
    return None


# ── formatting helpers (mirror the host_schedule.html panel) ──────────────────
def _fmt_supply_demand(tile):
    """Dedup per-flow supply/demand verdicts across the tile's channels."""
    seen, rows = set(), []
    for c in tile.get("dma_channels", []) or []:
        fb = c.get("flow_balance")
        if not fb or fb.get("flow_index") in seen:
            continue
        seen.add(fb.get("flow_index"))
        s, d = fb.get("supply_per_round"), fb.get("demand_per_round")
        if fb.get("balanced") is False:
            verd = "OVER-SUPPLY" if (s or 0) > (d or 0) else "UNDER-SUPPLY"
        elif fb.get("balanced") is True:
            verd = "balanced"
        else:
            verd = "unchecked"
        delta = ""
        if fb.get("balanced") is False and s is not None and d is not None:
            delta = "  (delta %sB)" % (s - d)
        rows.append(
            "  flow %s (%s): %s  supply=%sB/round demand=%sB/round%s  [%s]"
            % (fb.get("flow_index"), fb.get("pattern"), verd, s, d, delta,
               fb.get("note", "")))
    return rows


def _fmt_kernel_match(tile):
    km = (tile.get("high_level") or {}).get("kernel_match")
    if not km or not km.get("matches"):
        return []
    out = ["channel <-> kernel argument (by BD buffer address):"]
    for m in km["matches"]:
        adr = "/".join(m.get("addrs_hex") or []) or "-"
        bsy = "/".join(m.get("bcf_syms") or []) or "-"
        arg = ("arg%s" % m["arg"]) if m.get("arg") is not None else "-"
        out.append("  %s%s -> window %s %s  [bd %s = %s] via %s"
                   % (m.get("direction"), m.get("channel"), m.get("window"),
                      arg, adr, bsy, m.get("method")))
    return out


def _fmt_lines(records, prefix="L"):
    """Render a list of {line, code} rows (line may be None)."""
    out = []
    for rec in records or []:
        ln = rec.get("line")
        code = rec.get("code", "")
        if ln is None:
            out.append("        %s" % code)
        else:
            out.append("  %s%-5s %s" % (prefix, ln, code))
    return out


def _section_hi(tile):
    hl = tile.get("high_level") or {}
    out = []
    if hl.get("role"):
        out.append("role: %s" % hl["role"])
    if hl.get("kernel"):
        out.append("kernel: %s" % hl["kernel"])
    sd = _fmt_supply_demand(tile)
    if sd:
        out.append("supply / demand:")
        out.extend(sd)
    if hl.get("summary"):
        out.append("transfers:")
        out.extend("  %s" % s for s in hl["summary"])
    if hl.get("contracts"):
        out.append("contracts:")
        out.extend("  %s" % c for c in hl["contracts"])
    km = _fmt_kernel_match(tile)
    if km:
        out.extend(km)
    return out


def _section_mid(tile):
    mid = tile.get("middle_ir")
    if isinstance(mid, str):
        return [mid]
    return _fmt_lines(mid) or ["(no dfschedule IR slice)"]


def _section_lo(tile):
    lo = tile.get("low_level") or {}
    rows = _fmt_lines(lo.get("code_lines"))
    if not rows:
        return ["(no attributed host.cc lines)"]
    hdr = "host.cc lines %s-%s:" % (lo.get("line_start"), lo.get("line_end"))
    return [hdr] + rows


# ── tools ────────────────────────────────────────────────────────────────────
@mcp.tool()
def tile_info(col: int, row: int, section: str = "all") -> str:
    """Return the schedule-view UI information for one AIE tile.

    This is the same per-tile content the human sees in host_schedule.html when
    they click a tile: the High level summary (role, kernel, transfers,
    supply/demand balance flags, channel<->kernel-arg map), the Middle
    (dfschedule IR) slice, and the Low level (attributed host.cc source lines).

    Args:
      col:     tile column (logical, as shown in the grid)
      row:     tile row (0 = shim, >=3 = core)
      section: which part to return - "hi" | "mid" | "lo" | "all" (default all)

    Returns a readable text digest. Use section="hi" for a quick summary or
    "all" for everything. Coordinates are the logical grid coords shown in the UI
    (not phys_col); this reads the static compiled schedule, not live hardware
    (use the aiegdb tools for live DMA/core/register state).
    """
    view = _load_view()
    if view is None:
        return ("error: schedule_view.json not found (looked in: %s)"
                % ", ".join(_view_candidates()))
    tile = _find_tile(view, col, row)
    if tile is None:
        locs = ", ".join("(%s,%s)" % (t["loc"][0], t["loc"][1])
                         for t in view.get("tiles", []) if t.get("loc"))
        return ("error: tile (%s,%s) not in schedule_view.json. Available: %s"
                % (col, row, locs))

    sec = (section or "all").strip().lower()
    valid = {"hi", "mid", "lo", "all"}
    if sec not in valid:
        return "error: section must be one of hi|mid|lo|all (got %r)" % section

    lines = ["=== Tile (%s,%s)  type=%s ==="
             % (col, row, tile.get("type"))]
    if sec in ("hi", "all"):
        lines.append("")
        lines.append("--- High level ---")
        lines.extend(_section_hi(tile))
    if sec in ("mid", "all"):
        lines.append("")
        lines.append("--- Middle (dfschedule IR) ---")
        lines.extend(_section_mid(tile))
    if sec in ("lo", "all"):
        lines.append("")
        lines.append("--- Low level (host.cc) ---")
        lines.extend(_section_lo(tile))
    return "\n".join(lines)


@mcp.tool()
def get_backend_status() -> dict:
    """Return the currently active debug backend and its connection state.

    Call this first in any debug session to understand what backend is active
    and whether live hardware reads are possible.

    Returns a dict with:
      backend      "simulator" | "hardware" | "unknown"
      ipc_ready    (simulator only) True when the IPC debug socket is open
      dbg_socket   (simulator only) filesystem path to the *.sock.dbg socket
      target       (hardware only) aiedbg target string, e.g. xsdb://host:3121
      device       aiedbg device string, e.g. "pal" or "npi"
      startcol     physical column offset for this partition
      aie_version  register layout version string, e.g. "2ps" or "5"
      note         human-readable status summary

    When backend is "simulator" and ipc_ready is True, the aiegdb MCP server
    (aie_exec / aie_scope tools) is configured to issue READ32 requests directly
    over the IPC debug socket — no JTAG or xsdb needed. When ipc_ready is False,
    the simulator is not yet running; start it from the UI Run button first.

    When backend is "hardware", use aie_exec / aie_scope to interact with the
    physical board via aiedbg. The target string identifies the xsdb endpoint.
    """
    backend = os.environ.get("AIEMCP_BACKEND", "unknown").strip().lower()
    dbg_socket = os.environ.get("AEG_PS_IPC_DBG_SOCKET", "").strip() or None
    ipc_ready = backend == "simulator" and bool(dbg_socket)
    target = os.environ.get("AIEDBG_TARGET", "").strip() or None
    device = os.environ.get("AIEMCP_DEVICE", "").strip() or None
    startcol = os.environ.get("AIEMCP_STARTCOL", "").strip() or None
    aie_version = os.environ.get("AIEMCP_AIE_VERSION", "").strip() or None

    if backend == "simulator":
        if ipc_ready:
            note = ("Simulator is running. Live register reads are active via "
                    "the IPC debug socket. Use aie_exec/aie_scope to read DMA, "
                    "core, and event registers.")
        else:
            note = ("Simulator backend selected but IPC debug socket is not yet "
                    "ready. Start the simulator from the Run button in the UI, "
                    "then retry.")
    elif backend == "hardware":
        note = ("Hardware board backend. Live reads go through aiedbg/xsdb. "
                "Use aie_exec/aie_scope to read DMA, core, and event registers.")
    else:
        note = "Backend unknown — no debug_ui_config.json or server not started."

    return {
        "backend": backend,
        "ipc_ready": ipc_ready,
        "dbg_socket": dbg_socket,
        "target": target,
        "device": device,
        "startcol": startcol,
        "aie_version": aie_version,
        "note": note,
    }


@mcp.tool()
def tile_list() -> str:
    """List every tile in the current schedule view with its type and role.

    Use this first to discover which tiles exist before calling tile_info.
    Returns one line per tile: (col,row) type - role.
    """
    view = _load_view()
    if view is None:
        return ("error: schedule_view.json not found (looked in: %s)"
                % ", ".join(_view_candidates()))
    out = []
    for t in view.get("tiles", []) or []:
        loc = t.get("loc") or [None, None]
        role = (t.get("high_level") or {}).get("role", "")
        out.append("(%s,%s) %-5s %s" % (loc[0], loc[1], t.get("type"), role))
    return "\n".join(out) if out else "(no tiles in schedule_view.json)"


# ── search index (mirrors buildSearchIndex() in host_schedule.html) ──────────

def _build_search_index(view):
    """Return list of hit dicts: {kind, col, row, fi, label, description}."""
    hits = []

    def add(kind, col, row, fi, label, description):
        hits.append(dict(kind=kind, col=col, row=row, fi=fi,
                         label=label, label_lc=label.lower(), description=description))

    for t in view.get("tiles", []) or []:
        if not t or not t.get("loc"):
            continue
        tc, tr = t["loc"]

        # Kernel name from high_level
        hl = t.get("high_level") or {}
        if hl.get("kernel"):
            add("kernel", tc, tr, None, hl["kernel"],
                "kernel on (%s,%s)" % (tc, tr))

        # DMA channels — contracts, window names, BD lengths, bcf_syms
        for ch in t.get("dma_channels", []) or []:
            fi = ch.get("flow_index")
            if ch.get("contract"):
                add("contract", tc, tr, fi, ch["contract"],
                    "%s ch%s on (%s,%s)" % (ch.get("direction", "?"), ch.get("channel"), tc, tr))
            for bd in ch.get("bd_chain", []) or []:
                if bd.get("len") is not None:
                    add("bd_len", tc, tr, fi, str(bd["len"]),
                        "BD%s len=%s %s ch%s (%s,%s)" % (bd.get("bd_id"), bd["len"],
                                                          ch.get("direction"), ch.get("channel"), tc, tr))
                for sym in bd.get("bcf_syms", []) or []:
                    if sym:
                        add("buffer", tc, tr, fi, sym,
                            "BD%s buffer on (%s,%s) %s ch%s" % (bd.get("bd_id"), tc, tr,
                                                                  ch.get("direction"), ch.get("channel")))

    # Kernel window names from top-level "kernel" block
    kern = view.get("kernel") or {}
    if kern.get("function"):
        for w in kern.get("windows", []) or []:
            if w.get("name"):
                add("window", None, None, None, w["name"],
                    "kernel window (%s)" % kern["function"])

    # comm_paths — net IDs, flow indices, GMIO names, hop port names
    for p in view.get("comm_paths", []) or []:
        if not p:
            continue
        fi = p.get("flow_index")
        dma_tiles = p.get("dma_tiles") or []
        rep = dma_tiles[0] if dma_tiles else None
        rc, rr = (rep[0], rep[1]) if rep else (p.get("prod_col"), p.get("prod_row"))

        net_id = p.get("id") or p.get("net_id")
        if net_id:
            add("net", rc, rr, fi, net_id, "net (%s) f%s" % (net_id, fi))
        if fi is not None:
            add("flow", rc, rr, fi, "f%s" % fi, "flow index %s" % fi)
        if p.get("config_ref"):
            add("gmio", rc, rr, fi, p["config_ref"], "config ref for f%s" % fi)
        for h in p.get("hops", []) or []:
            if h.get("port_name"):
                add("port", rc, rr, fi, h["port_name"], "hop port in f%s" % fi)

    # flow_summary — kernel port strings
    for fs in view.get("flow_summary", []) or []:
        if not fs:
            continue
        fi = fs.get("flow_index")
        for entry in fs.get("entries", []) or []:
            tc2, tr2 = entry.get("tile_col"), entry.get("tile_row")
            if entry.get("kernel_port"):
                add("port", tc2, tr2, fi, entry["kernel_port"],
                    "supply/demand port f%s" % fi)

    return hits


_SEARCH_CACHE = {"path": None, "index": None}


def _get_search_index():
    view = _load_view()
    if view is None:
        return None, None
    ap = _VIEW_CACHE["path"]
    if _SEARCH_CACHE["path"] == ap and _SEARCH_CACHE["index"] is not None:
        return view, _SEARCH_CACHE["index"]
    idx = _build_search_index(view)
    _SEARCH_CACHE["path"] = ap
    _SEARCH_CACHE["index"] = idx
    return view, idx


@mcp.tool()
def symbol_search(query: str, kinds: str = "") -> str:
    """Search the compiled AIE design for any symbol matching a substring.

    This mirrors the Search bar in host_schedule.html (the "kernel, window, net,
    GMIO, port, len…" input at the bottom of the device-map panel), returning the
    same hits the browser would highlight — but as structured text instead of
    yellow SVG halos.

    Searchable fields (kinds):
      kernel   — kernel function names (e.g. "matmul", "dskernel_receiver")
      window   — kernel window / buffer argument names (e.g. "in_0", "out")
      buffer   — BCF buffer symbols attributed to BD descriptors
      contract — DMA channel contract strings (e.g. "S2MM ch0: ping-pong receive, 1024B")
      bd_len   — BD transfer lengths in bytes (e.g. "1024", "256")
      net      — net / comm-path IDs (e.g. "push_0", "net7")
      flow     — flow index strings (e.g. "f0", "f7")
      gmio     — GMIO config-ref names
      port     — kernel-port / graph-port strings from flow_summary

    Args:
      query: case-insensitive substring to search (e.g. "receiver", "1024", "push_0")
      kinds: comma-separated list of kinds to restrict results (default: all kinds).
             Example: "kernel,contract"  or  "bd_len,buffer"

    Returns one line per match: kind  (col,row)  f<fi>  label  —  description
    Tile coords may be "(-,-)" for design-wide entries (kernel windows, etc.).
    Use tile_info(col, row) to drill into any returned tile.
    """
    view, idx = _get_search_index()
    if idx is None:
        return ("error: schedule_view.json not found (looked in: %s)"
                % ", ".join(_view_candidates()))

    q = (query or "").strip().lower()
    if not q:
        return "error: query must not be empty"

    kind_filter = set()
    if kinds:
        kind_filter = {k.strip().lower() for k in kinds.split(",") if k.strip()}

    hits = [h for h in idx
            if q in h["label_lc"]
            and (not kind_filter or h["kind"] in kind_filter)]

    if not hits:
        return "no matches for %r%s" % (query, (" (kinds: %s)" % kinds) if kinds else "")

    lines = ["symbol_search %r  →  %d match%s" % (query, len(hits), "es" if len(hits) != 1 else "")]
    seen = set()
    for h in hits:
        dedup_key = (h["kind"], h.get("col"), h.get("row"), h.get("fi"), h["label"])
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        col_s = str(h["col"]) if h["col"] is not None else "-"
        row_s = str(h["row"]) if h["row"] is not None else "-"
        fi_s = ("f%s" % h["fi"]) if h["fi"] is not None else "—"
        lines.append("  %-10s (%s,%s)  %-5s  %s  —  %s"
                     % (h["kind"], col_s, row_s, fi_s, h["label"], h["description"]))
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()
