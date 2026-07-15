#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""aiemcp - an MCP (Model Context Protocol) server exposing aiegdb live AIE debug.

This is a *third* front-end onto the same stateful scoped console that powers the
browser/daemon tier (see doc/design/aiegdb_live_debug_framework.md). Instead of
framing commands over a pipe (`aiegdb.py --server`), it imports `aiegdb.AieGdb`
**in-process** and keeps one long-lived instance whose scope
(partition -> tile -> channel) persists across MCP tool calls.

Model of the console
--------------------
The console has three nested scopes:

    partition(startcol=N)                 <- always present (board/array config)
        -> tile(col,row)                  <- `target tile <col> <row>`
            -> mm2s0 / s2mm1              <- `target channel <dir_ch>`

At tile scope, col/row are auto-injected into aiedbg calls (phys_col =
col + startcol). At channel scope, direction/channel are auto-injected too, so
`dma status`, `bd`, `event` need no coordinates.

Tools
-----
- aie_exec(cmd)   run any aiegdb command line (primary tool)
- aie_scope()     show the current scope + config (`where`)
- aie_commands()  list the commands valid at the current scope (`?`)
- aie_help()      full command reference (`help`)

Configuration comes from environment variables (MCP servers launch via
`.mcp.json`, not argv):

    AIEDBG_TARGET       aiedbg target, e.g. xsdb://xxx.xxx.xxx.213:3121
                        (falls back to ~/.aiedbg_env like aiegdb)
    AIEMCP_DEVICE       aiedbg -d device            (default: pal)
    AIEMCP_STARTCOL     physical column offset       (default: from provenance JSON)
    AIEMCP_AIE_VERSION  register offsets: 5 | 2ps    (default: from provenance JSON)
    AIEMCP_JSON_DIR     provenance/schedule_view JSON dir (default: auto-detect)
    AIEMCP_DRY_RUN      if set (1/true/yes), print aiedbg commands instead of
                        running them (no board needed)

Run standalone (stdio transport):  python3 src/tool/debug/aiemcp.py
Protocol smoke test:               mcp dev src/tool/debug/aiemcp.py
"""

import os
import re
import sys
import threading

# Import aiediag + aiegdb as libraries. Both live in the same dir
# (src/tool/debug/) and guard main() behind __main__, so importing them has no
# side effects. Same shim as aiegdb.py:32-35 / schedule_debug_server.py.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
import aiediag  # noqa: E402
import aiegdb  # noqa: E402

from mcp.server.fastmcp import FastMCP  # noqa: E402

mcp = FastMCP("aiegdb")

# ── strip ANSI color (aiediag disables color off-tty, but be defensive) ───────
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _env_bool(name):
    val = os.environ.get(name, "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _build_gdb():
    """Construct the single long-lived AieGdb, resolving config exactly like
    aiegdb.main() (src/tool/debug/aiegdb.py:788-807) but from env vars."""
    target = (os.environ.get("AIEDBG_TARGET")
              or aiegdb._target_from_aiedbg_env())
    device = os.environ.get("AIEMCP_DEVICE", "pal")
    json_dir = os.environ.get("AIEMCP_JSON_DIR") or None
    dry_run = _env_bool("AIEMCP_DRY_RUN")

    startcol_env = os.environ.get("AIEMCP_STARTCOL")
    aiever_env = os.environ.get("AIEMCP_AIE_VERSION")

    gdb = aiegdb.AieGdb(
        target=target, device=device,
        startcol=startcol_env if startcol_env not in (None, "") else 0,
        aie_version=aiever_env if aiever_env not in (None, "") else "5",
        json_dir=json_dir, dry_run=dry_run)

    # Resolve startcol: explicit env wins, else read from provenance JSON.
    if startcol_env in (None, ""):
        gdb._load_jsons()
        gdb.startcol = aiediag.startcol_from_jsons(gdb._dfsche, gdb._dmaphop) or 0
    else:
        try:
            gdb.startcol = int(startcol_env)
        except ValueError:
            gdb.startcol = 0

    # Resolve aie_version: JSON wins; explicit env normalized through the mapper;
    # else warn + fallback. Mirrors aiegdb/daemon (JSON-first) so a manual
    # AIEMCP_AIE_VERSION=5 (compiler number) is not misread as debug AIEML.
    gdb._load_jsons()
    ver = aiediag.aie_version_from_jsons(gdb._dfsche, gdb._dmaphop)
    if ver is None and aiever_env not in (None, ""):
        ver = aiediag.debug_aie_version_from_gen(aiever_env)
    if ver is None:
        ver = "2ps"
        print("warning: could not resolve aie_version; falling back to 2ps",
              file=sys.stderr)
    gdb.aie_version = ver

    return gdb


_gdb = _build_gdb()
_lock = threading.Lock()


def _run(line):
    """Serialized, OS-fd-level capture around gdb.run_line(line).

    aiegdb prints results to stdout AND its _passthrough runs subprocess.run(cmd)
    without capture, so the child inherits fd 1. FastMCP stdio uses fd 1 for
    JSON-RPC, so we must dup2 fd 1/2 to a pipe around each command (Python prints
    *and* child-process output) and restore before returning. A lock serializes
    the fd swap because it must not interleave with other tool calls.
    """
    with _lock:
        # Save the real stdio fds and the Python-level stream.
        saved_out_fd = os.dup(1)
        saved_err_fd = os.dup(2)
        saved_sys_out = sys.stdout
        saved_sys_err = sys.stderr
        r_fd, w_fd = os.pipe()
        captured = b""
        try:
            os.dup2(w_fd, 1)
            os.dup2(w_fd, 2)
            os.close(w_fd)
            # Point Python's buffered streams at the new fd 1 as well.
            sys.stdout = os.fdopen(1, "w", closefd=False)
            sys.stderr = os.fdopen(2, "w", closefd=False)
            try:
                _gdb.run_line(line)
            except SystemExit as e:
                # aiediag.run_aiedbg_reg_read calls sys.exit(1) when aiedbg is
                # missing; never let that kill the server.
                print(f"error: command exited (SystemExit: {e})")
            except Exception as e:  # noqa: BLE001 - keep server alive
                print(f"error: {e}")
            finally:
                try:
                    sys.stdout.flush()
                    sys.stderr.flush()
                except Exception:  # noqa: BLE001
                    pass
        finally:
            # Restore real fds first so the pipe write end is fully closed.
            os.dup2(saved_out_fd, 1)
            os.dup2(saved_err_fd, 2)
            os.close(saved_out_fd)
            os.close(saved_err_fd)
            sys.stdout = saved_sys_out
            sys.stderr = saved_sys_err
            # Drain the read end (writers are closed now, so read to EOF).
            chunks = []
            while True:
                buf = os.read(r_fd, 65536)
                if not buf:
                    break
                chunks.append(buf)
            os.close(r_fd)
            captured = b"".join(chunks)

    text = captured.decode("utf-8", errors="replace")
    text = _ANSI_RE.sub("", text)
    return {"output": text.rstrip("\n"), "scope": _gdb.prompt().rstrip()}


@mcp.tool()
def aie_exec(cmd: str) -> dict:
    """Run one aiegdb command line against the live AIE board and return its
    output plus the resulting scope.

    The console is stateful and scoped: partition -> tile -> channel. Scope
    persists across calls, so descend with `target ...` first, then issue
    scope-aware commands.

    Navigation (any scope):
      target tile <col> <row>        descend to a tile   (e.g. "target tile 0 3")
      target channel <dir_ch>        descend to a channel (e.g. "target channel mm2s0")
      up                             pop one level        top    go to partition
      set startcol N | set target .. | set device .. | set aie 5|2ps

    Tile scope (col/row auto-injected as phys_col = col + startcol):
      dma <dir_ch>                   decoded DMA status
      pc                             core PC -> source line (linemap)
      event                          event-status regs (decoded)
      channels                       list this tile's DMA channels
      reg read OFF | reg write OFF VAL | mem read ADDR LEN
      scan dma|cores | tile list | show ...   (array-wide aiedbg passthrough)

    Channel scope (direction/channel also auto-injected):
      dma status | status            decoded DMA status register
      bd                             BD chain (JSON) + live HW lengths
      event                          per-channel DMA start/finish/error
      dma counter                    AIE perf counters (read-only)
      dma counter setup [finished|started]   (INTRUSIVE write)

    Example session:
      aie_exec("target tile 0 3")
      aie_exec("target channel mm2s0")
      aie_exec("dma status")

    Returns {"output": <captured text>, "scope": <current prompt>}.
    """
    return _run(cmd)


@mcp.tool()
def aie_scope() -> dict:
    """Show the current scope and partition config (breadcrumb + phys_col/row +
    target/device), via the console `where` command.

    Returns {"output": <where text>, "scope": <current prompt>}.
    """
    return _run("where")


@mcp.tool()
def aie_commands() -> dict:
    """List only the commands valid at the *current* scope level (console `?`).

    Use this to discover what you can do at partition vs tile vs channel scope
    before issuing an aie_exec command.

    Returns {"output": <command list>, "scope": <current prompt>}.
    """
    return _run("?")


@mcp.tool()
def aie_help() -> str:
    """Return the full aiegdb command reference (console `help`)."""
    return _run("help")["output"]


if __name__ == "__main__":
    mcp.run()
