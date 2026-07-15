# Design: Guarantee the HTML LLM tab's Claude auto-connects to the aiegdb MCP server

Date: 2026-07-06

## Problem

`script/visualization/schedule_debug_server.py` runs an "LLM tab": it spawns a
headless Claude Code subprocess (`_llm_spawn`, ~line 399) that the browser drives
over `/llm`, `/llm/poll`, `/llm/reset`. Today that Claude connects to the
`aiegdb` MCP server (`script/aiemcp.py`) **only implicitly** — it relies on:

1. `claude_cwd` happening to equal the repo root, and
2. the gitignored repo-root `.mcp.json` being present there,

so Claude Code's cwd-based `.mcp.json` auto-discovery picks it up. There is:

- **no `--mcp-config` flag** (breaks if launched from a different cwd or if the
  repo `.mcp.json` is absent),
- **no verification** that the aiegdb tools actually loaded,
- **no guarantee** the MCP server uses the *same* hardware config
  (target/device/startcol/aie_version) the daemon already resolved.

## Goal

When the daemon spawns Claude:

1. **#1 — the aiegdb MCP server starts.** (Claude Code itself launches
   `python3 script/aiemcp.py` as a stdio child; we make that deterministic.)
2. **#2 — the HTML's Claude auto-connects to it**, cwd-independently, using the
   same HW config as the daemon, with a startup probe that warns if it didn't.

## Approach (chosen)

All changes live in `script/visualization/schedule_debug_server.py`. No changes
to `schedule_view.py` / HTML (server-log warning only). No changes to
`script/aiemcp.py` (it already reads all config from env vars).

### 1. Auto-generate a temp MCP config at spawn

New `DebugState._write_mcp_config()`:

- Build the same shape as `.mcp.json`:

  ```json
  {"mcpServers": {"aiegdb": {
     "command": "python3",
     "args": ["<abs>/script/aiemcp.py"],
     "env": {
        "AIEDBG_TARGET":      "<resolved target>",
        "AIEMCP_DEVICE":      "<resolved device>",
        "AIEMCP_STARTCOL":    "<resolved startcol>",
        "AIEMCP_AIE_VERSION": "<resolved aie_version>",
        "AIEMCP_JSON_DIR":    "<workdir>"
     }}}}
  ```

- Write to `tempfile.NamedTemporaryFile(delete=False, suffix=".mcp.json")`;
  store path on `self._mcp_config_path`.
- Reuses target/device/startcol/aie_version resolved in `main()`
  (lines ~1158-1178) so both tiers share ONE hardware config.
- Never touches the repo-root `.mcp.json`.
- On write failure: log and fall back to cwd auto-discovery (do not crash).

Constructor (`DebugState.__init__`) gains `aie_version` / `startcol` / `device` /
`target` / `workdir` awareness for the config (these values already flow into
`DebugState`; wire the missing ones through as needed) and calls
`_write_mcp_config()` (or does it lazily on first `_llm_spawn`).

### 2. Wire config + tools into the spawn command (`_llm_spawn`)

```
claude -p --input-format stream-json --output-format stream-json \
  --include-partial-messages --verbose \
  --permission-mode bypassPermissions \
  --mcp-config <self._mcp_config_path> \
  --strict-mcp-config \
  --allowedTools mcp__aiegdb__aie_exec mcp__aiegdb__aie_scope \
                 mcp__aiegdb__aie_commands mcp__aiegdb__aie_help
```

- `--mcp-config` → deterministic, cwd-independent (#2).
- `--strict-mcp-config` → ignore any other `.mcp.json`; only our server loads.
- `--allowedTools` → explicit grant of the four aiegdb tools (belt-and-suspenders
  with `bypassPermissions`).
- If `_mcp_config_path` is None (write failed), omit the `--mcp-config`/
  `--strict-mcp-config` flags and keep cwd fallback.

### 3. Startup verification probe (ON by default; `--no-mcp-probe` to skip)

After `DebugState` is built and before `serve_forever()`, run a one-shot probe:

```
claude -p --output-format json \
  --mcp-config <path> --strict-mcp-config \
  --permission-mode bypassPermissions \
  --allowedTools mcp__aiegdb__aie_scope \
  "call the aiegdb aie_scope tool and reply with only its scope line"
```

- Parse the JSON result. Success = ran without an MCP-load error (tool callable).
- On failure/timeout: print `LLM MCP: aiegdb NOT connected (<reason>)` to server
  logs near the existing `LLM:` line (~1218); **keep serving** (verify + warn).
- On success: print `LLM MCP: aiegdb connected`.
- Skip the probe when `claude` is NOT FOUND or `--no-llm`.
- `--no-mcp-probe` CLI flag skips it (probe costs one model call).

### 4. Cleanup

On `KeyboardInterrupt` shutdown (~line 1238), `os.unlink(self._mcp_config_path)`
if set (guard `FileNotFoundError`/`OSError`).

## Error handling

| Failure | Behavior |
|---------|----------|
| Temp-file write fails | log; fall back to cwd `.mcp.json` discovery; keep serving |
| `claude` binary missing | existing `NOT FOUND` path; skip probe |
| Probe fails / times out | warn in logs; keep serving |
| Probe reports MCP not loaded | warn in logs; keep serving |

## Testing

- **Unit-ish:** assert generated JSON parses; `--mcp-config` path exists on disk;
  env dict contains all five keys.
- **Offline smoke:** inject `AIEMCP_DRY_RUN=1` into the config env → probe should
  still show `aie_scope` callable without a board.
- **Live:** start server → log shows `LLM MCP: aiegdb connected` → in the LLM tab
  ask Claude to run `aie_scope`; verify it returns the partition scope
  (`partition(startcol=N)...`).

## Non-goals

- No browser/HTML banner (server-log warning only).
- No changes to `script/aiemcp.py` behavior.
- No new hardware write paths (MCP stays read-only except the existing explicit
  `dma counter setup`).

## Files changed

- `script/visualization/schedule_debug_server.py` — `_write_mcp_config`,
  `_llm_spawn` flags, startup probe, `--no-mcp-probe` flag, cleanup.
- `docs/plans/2026-07-06-html-claude-aiegdb-mcp-autoconnect-design.md` (this doc).
