<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
-->

# `src/tool/debug` — AIE Live Debug & Schedule Visualization Tools

This directory centralizes the **HTML AIE-schedule / live-debug** solution: a set
of Python tools that turn the compiler's provenance JSON + generated `host.cc`
into a readable schedule view, and connect a browser (or Claude Code) to live AIE
hardware over JTAG for DMA/core/event inspection.

All tools are pure-Python (stdlib only, except `aiemcp.py` which needs the `mcp`
package). None of them modify `host.cc` or any generated source.

## Files

| File | Role | Depends on |
|------|------|------------|
| `aiediag.py` | DMA diagnostic **library + CLI** — register offsets, `aiedbg` reg reads, status decoders, provenance-map cross-reference | none (leaf) |
| `aiegdb.py` | Scoped **GDB-like CLI** over `aiedbg` (partition → tile → channel) | `aiediag` |
| `aiemcp.py` | **MCP server** exposing `aiegdb` to Claude Code (in-process, persistent scope) | `aiediag`, `aiegdb`, `mcp` |
| `schedule_view.py` | Generates `schedule_view.json` + `host_schedule.html` from provenance JSON + `host.cc` | none (stdlib) |
| `schedule_debug_server.py` | **Live debug daemon** (`http.server`) behind `host_schedule.html`; deploys ELF, tails `applog`, overlays live DMA/core/event status, drives an LLM tab | `aiediag`, `aiegdb`; spawns `aiemcp.py` |

### Dependency graph

```
aiediag  <-- aiegdb  <-- aiemcp
   ^          ^            ^  (spawned as subprocess via --mcp-config)
   |          |            |
   +----------+---- schedule_debug_server  --> host_schedule.html (served)
                    schedule_view --> host_schedule.html + schedule_view.json
```

---

## Prerequisites

- **Provenance JSONs** in a workdir (e.g. `aout/worklocal/`):
  `dfscheduleprovenancemap.json`, `dmaphopprovenacemap.json`, and `host.cc`.
  These are produced by the tilinglinalg pipeline (see the repo `CLAUDE.md`).
- **Live HW debug** requires an `aiedbg` JTAG target, set via either:
  - `export AIEDBG_TARGET=xsdb://<host>:3121`, or
  - `~/.aiedbg_env` (written by `aiedbg-setup`), or
  - a `--target xsdb://<host>:3121` flag.
- `aiemcp.py` additionally needs: `pip install "mcp[cli]"`.

All live tools accept `--dry-run` (or `AIEMCP_DRY_RUN=1`) to print the `aiedbg`
commands **without a board** — useful for testing offline.

---

## 1. `schedule_view.py` — generate the readable schedule view

Consumes `<workdir>/dfscheduleprovenancemap.json` + `<workdir>/host.cc` and writes
`<workdir>/schedule_view.json` and `<workdir>/host_schedule.html` (a self-contained
static page — all HTML/CSS/JS is generated inline).

```bash
python3 src/tool/debug/schedule_view.py [workdir]      # workdir default: aout/worklocal
# → wrote <workdir>/schedule_view.json
# → wrote <workdir>/host_schedule.html
```

Open `host_schedule.html` in a browser for an offline view. This step is run
automatically by `script/aiehlc.sh` after a build.

---

## 2. `aiediag.py` — flow-aware DMA diagnostics

Reads DMA status registers from live hardware, decodes them, and cross-references
provenance JSON to name the connected tiles / routing paths.

```bash
# Dig into a tile's DMA channel (logical col, physical row; startcol optional)
python3 src/tool/debug/aiediag.py dig <col> <row> [startcol N] [--dry-run]

# Read a core's PC and map it to kernel source file:line
python3 src/tool/debug/aiediag.py pc <col> <row> [--dry-run]
```

Common flags: `--json-dir DIR`, `--aie-version {5,2ps}`, `--target xsdb://...`,
`--device pal`, `--dry-run`. It is also imported as a **library** by the other
tools (offsets, `run_aiedbg_reg_read`, decoders, `startcol_from_jsons`).

---

## 3. `aiegdb.py` — scoped GDB-like CLI

A stateful REPL that keeps a *current scope* (partition → tile → channel). Scope
persists across commands; tile scope auto-injects `phys_col = col + startcol`,
channel scope also auto-injects direction/channel.

```bash
# Interactive REPL
python3 src/tool/debug/aiegdb.py --target xsdb://<host>:3121

# Non-interactive: run ';'-separated commands, then exit
python3 src/tool/debug/aiegdb.py -c "target tile 0 3; target channel mm2s0; dma status"

# Run a command script
python3 src/tool/debug/aiegdb.py --script cmds.txt

# No board — print aiedbg commands only
python3 src/tool/debug/aiegdb.py --dry-run -c "target tile 0 3; dma status"
```

Inside the REPL: `target tile <col> <row>`, `target channel <dir_ch>` (e.g.
`mm2s0`), `up`, `top`, `where`, `?` (per-scope command list), plus channel
helpers `dma status`, `bd`, `event`, `pc`. Flags mirror `aiediag`
(`--startcol`, `--aie-version`, `--json-dir`, `--device`). `--server` runs a
framed stdin/stdout REPL used by the debug daemon.

---

## 4. `aiemcp.py` — MCP server for Claude Code

Exposes `aiegdb` to Claude Code as MCP tools, keeping **one long-lived** `AieGdb`
instance so scope persists across tool calls. Registered as `aiegdb` in the
repo-root `.mcp.json`.

Tools: `aie_exec(cmd)` (primary — e.g. `target tile 0 3`, `dma status`, `bd`,
`pc`), `aie_scope()` (`where`), `aie_commands()` (`?`), `aie_help()` (`help`).

Config via environment (set in `.mcp.json`):

| Env var | Meaning | Default |
|---------|---------|---------|
| `AIEDBG_TARGET` | JTAG target `xsdb://host:port` | `~/.aiedbg_env` |
| `AIEMCP_DEVICE` | `aiedbg -d` device | `pal` |
| `AIEMCP_STARTCOL` | physical column offset | from provenance JSON |
| `AIEMCP_AIE_VERSION` | register offsets `5` \| `2ps` | from provenance JSON |
| `AIEMCP_JSON_DIR` | provenance JSON dir | auto-detect |
| `AIEMCP_DRY_RUN` | print `aiedbg` cmds, no board | off |

```bash
# Run standalone (stdio transport)
python3 src/tool/debug/aiemcp.py

# Protocol smoke test (needs mcp[cli]); no board
AIEMCP_DRY_RUN=1 mcp dev src/tool/debug/aiemcp.py
```

---

## 5. `schedule_debug_server.py` — live debug daemon

An `http.server` daemon that backs the `host_schedule.html` page. It deploys the
ELF via `apppaltest.py -y -nonreboot`, tails the repo-root `applog`, exposes
polling JSON endpoints (`/run`, `/applog`, `/grid`, `/cmd`) so the browser shows
live DMA/core/event overlays, and drives an in-browser LLM tab wired to the
`aiegdb` MCP server. The daemon issues **read-only** hardware ops only.

```bash
python3 src/tool/debug/schedule_debug_server.py <workdir> \
    --elf <path/to/main.elf> --aie-version 5 --open
```

Key flags: `--port 8091`, `--host 0.0.0.0`, `--target xsdb://...`, `--device pal`,
`--startcol N`, `--apppaltest PATH` (default `script/test/apppaltest.py`),
`--applog PATH` (default repo-root `applog`), `--no-llm`, `--password` /
`--no-password`, `--no-mcp-probe`. Launched automatically by
`script/aiehlc.sh --prettydebug`.

### Path anchoring

This file resolves the repo root as three levels up (`src/tool/debug` → repo
root) and locates sibling tools (`aiegdb.py`, `aiemcp.py`) in its own directory.
`apppaltest.py` stays under `script/test/` and `applog` is written at the repo
root.

---

## Quick reference

```bash
# Offline: generate the static schedule view
python3 src/tool/debug/schedule_view.py aout/worklocal

# Offline: dry-run the live tools (no board)
python3 src/tool/debug/aiegdb.py --dry-run -c "target tile 0 3; dma status"
AIEMCP_DRY_RUN=1 mcp dev src/tool/debug/aiemcp.py

# Live: scoped CLI
export AIEDBG_TARGET=xsdb://xxx.xxx.xxx.213:3121
python3 src/tool/debug/aiegdb.py

# Live: full browser debug server
python3 src/tool/debug/schedule_debug_server.py aout/worklocal --elf aout/worklocal/build/host --open
```

## See also

- `doc/design/live_debug_framework.md` — daemon design
- `doc/design/aiegdb_live_debug_framework.md` — scoped CLI / MCP design
- `.cursor/skills/aiehwdmadebug/SKILL.md` — live DMA debug workflow
- repo-root `.mcp.json` — registers `aiemcp.py` as the `aiegdb` MCP server
