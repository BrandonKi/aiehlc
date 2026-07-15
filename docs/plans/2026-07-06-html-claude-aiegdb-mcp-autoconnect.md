# HTML LLM tab → aiegdb MCP auto-connect Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Guarantee that the Claude Code subprocess spawned by `schedule_debug_server.py` (the HTML "LLM tab") always loads the `aiegdb` MCP server (`script/aiemcp.py`) with the daemon's resolved hardware config, cwd-independently, and warn at startup if the connection fails.

**Architecture:** In `DebugState`, generate a temp `.mcp.json` (embedding `AIEDBG_TARGET`, `AIEMCP_DEVICE`, `AIEMCP_STARTCOL`, `AIEMCP_AIE_VERSION`, `AIEMCP_JSON_DIR` from values the daemon already resolved). Pass it to `claude` via `--mcp-config <temp> --strict-mcp-config` and explicitly allow the four `mcp__aiegdb__*` tools. Add a one-shot startup probe (on by default, `--no-mcp-probe` to skip) that verifies the tool is callable and warns (keeps serving) on failure. Clean up the temp file on shutdown.

**Tech Stack:** Python 3 stdlib (`tempfile`, `subprocess`, `json`, `os`), Claude Code CLI (`--mcp-config`, `--strict-mcp-config`, `--allowedTools` — all verified present in the installed CLI).

**Design doc:** `docs/plans/2026-07-06-html-claude-aiegdb-mcp-autoconnect-design.md`

**Single file touched:** `script/visualization/schedule_debug_server.py`

**Note on testing:** This module has no existing unit-test harness and its real behavior needs the `claude` CLI + a board. Tests here are lightweight, board-free assertions runnable with `python3 -c` against imported helpers plus manual verification steps. Follow TDD where a pure helper exists (`_write_mcp_config`); use manual verification for the subprocess/probe wiring.

---

### Task 1: Add `_write_mcp_config()` helper to `DebugState`

**Files:**
- Modify: `script/visualization/schedule_debug_server.py` (add import `tempfile`; add method + attribute init in `DebugState.__init__` around line 131)

**Step 1: Write the failing test (board-free helper test)**

Create a throwaway check (run from repo root). This imports the module and asserts the generated config is well-formed:

```bash
python3 - <<'PY'
import importlib.util, os, json, tempfile, sys
sys.path.insert(0, "script")   # for aiediag/aiegdb imports the module may need
spec = importlib.util.spec_from_file_location(
    "sds", "script/visualization/schedule_debug_server.py")
sds = importlib.util.module_from_spec(spec); spec.loader.exec_module(sds)
st = sds.DebugState(workdir=".", elf=None, aie_version="5", device="pal",
                    target="xsdb://1.2.3.4:3121", apppaltest="x", startcol=3,
                    applog="/tmp/claude/applog")
path = st._write_mcp_config()
assert path and os.path.isfile(path), "config file not written"
cfg = json.load(open(path))
srv = cfg["mcpServers"]["aiegdb"]
assert srv["args"][0].endswith("aiemcp.py"), srv["args"]
env = srv["env"]
for k in ("AIEDBG_TARGET","AIEMCP_DEVICE","AIEMCP_STARTCOL",
          "AIEMCP_AIE_VERSION","AIEMCP_JSON_DIR"):
    assert k in env, f"missing {k}"
assert env["AIEMCP_STARTCOL"] == "3"
assert env["AIEDBG_TARGET"] == "xsdb://1.2.3.4:3121"
print("OK", path)
PY
```

Expected: FAIL with `AttributeError: 'DebugState' object has no attribute '_write_mcp_config'`.

**Step 2: Add `import tempfile`**

At the top of the file with the other stdlib imports, add `import tempfile` (place it alphabetically near `import subprocess`).

**Step 3: Init the attribute in `DebugState.__init__`**

After line 131 (`self._llm_lock = threading.Lock()`), add:

```python
        # Path to the auto-generated MCP config handed to the claude subprocess
        # via --mcp-config (written lazily by _write_mcp_config; cleaned up on
        # shutdown). None => fall back to cwd .mcp.json auto-discovery.
        self._mcp_config_path = None
```

**Step 4: Add the method**

Add this method to `DebugState` (place it just before `_llm_spawn`, i.e. before line ~396 `def _llm_spawn`):

```python
    def _write_mcp_config(self):
        """Write a temp .mcp.json registering the aiegdb MCP server with the
        SAME hardware config the daemon already resolved, and return its path.

        Handed to `claude` via --mcp-config so the LLM tab's connection to the
        aiegdb server (script/aiemcp.py) does not depend on cwd / the repo-root
        .mcp.json. Returns None on failure (caller falls back to cwd discovery).
        """
        if self._mcp_config_path and os.path.isfile(self._mcp_config_path):
            return self._mcp_config_path
        aiemcp = os.path.join(_SCRIPT_DIR, "aiemcp.py")
        cfg = {
            "mcpServers": {
                "aiegdb": {
                    "command": sys.executable,
                    "args": [aiemcp],
                    "env": {
                        "AIEDBG_TARGET": self.target or "",
                        "AIEMCP_DEVICE": self.device or "pal",
                        "AIEMCP_STARTCOL": str(self.startcol),
                        "AIEMCP_AIE_VERSION": str(self.aie_version),
                        "AIEMCP_JSON_DIR": self.workdir,
                    },
                }
            }
        }
        try:
            fd, path = tempfile.mkstemp(suffix=".mcp.json", prefix="aiegdb_")
            with os.fdopen(fd, "w") as f:
                json.dump(cfg, f, indent=2)
            self._mcp_config_path = path
            return path
        except OSError as e:
            print(f"warning: could not write MCP config ({e}); "
                  f"falling back to cwd .mcp.json discovery", file=sys.stderr)
            self._mcp_config_path = None
            return None
```

Note: uses `sys.executable` (the running interpreter) for `command` so the MCP
server runs under the same Python. This is stricter than the repo `.mcp.json`'s
literal `"python3"` and avoids PATH ambiguity.

**Step 5: Run the test to verify it passes**

Run the Step 1 heredoc again.
Expected: `OK /tmp/.../aiegdb_XXXX.mcp.json`.

**Step 6: Commit**

```bash
git add script/visualization/schedule_debug_server.py
git commit -m "feat(debug-server): generate temp MCP config for LLM tab claude"
```

---

### Task 2: Wire `--mcp-config` + `--strict-mcp-config` + `--allowedTools` into `_llm_spawn`

**Files:**
- Modify: `script/visualization/schedule_debug_server.py:399-403` (`_llm_spawn` cmd list)

**Step 1: Read current cmd**

Current (lines 399-403):

```python
        cmd = [self.claude_bin, "-p",
               "--input-format", "stream-json",
               "--output-format", "stream-json",
               "--include-partial-messages", "--verbose",
               "--permission-mode", "bypassPermissions"]
        if self.claude_model:
            cmd += ["--model", self.claude_model]
```

**Step 2: Replace with MCP-aware cmd**

```python
        cmd = [self.claude_bin, "-p",
               "--input-format", "stream-json",
               "--output-format", "stream-json",
               "--include-partial-messages", "--verbose",
               "--permission-mode", "bypassPermissions"]
        mcp_cfg = self._write_mcp_config()
        if mcp_cfg:
            cmd += ["--mcp-config", mcp_cfg, "--strict-mcp-config",
                    "--allowedTools",
                    "mcp__aiegdb__aie_exec", "mcp__aiegdb__aie_scope",
                    "mcp__aiegdb__aie_commands", "mcp__aiegdb__aie_help"]
        if self.claude_model:
            cmd += ["--model", self.claude_model]
```

Rationale: if `_write_mcp_config()` fails (returns None), we omit the flags and
Claude falls back to cwd `.mcp.json` discovery (existing behavior). When present,
`--strict-mcp-config` guarantees ONLY our server loads.

**Step 3: Verify the cmd builds (board-free)**

```bash
python3 - <<'PY'
import importlib.util, sys
sys.path.insert(0, "script")
spec = importlib.util.spec_from_file_location(
    "sds", "script/visualization/schedule_debug_server.py")
sds = importlib.util.module_from_spec(spec); spec.loader.exec_module(sds)
st = sds.DebugState(".", None, "5", "pal", "xsdb://1.2.3.4:3121", "x", 3,
                    "/tmp/claude/applog")
# Simulate what _llm_spawn does to build the flag list (no Popen).
cfg = st._write_mcp_config()
assert cfg
print("mcp_config:", cfg)
print("OK")
PY
```

Expected: prints a temp path and `OK`.

**Step 4: Commit**

```bash
git add script/visualization/schedule_debug_server.py
git commit -m "feat(debug-server): pass --mcp-config/--strict/--allowedTools to LLM claude"
```

---

### Task 3: Add the startup MCP probe (verify + warn) and `--no-mcp-probe` flag

**Files:**
- Modify: `script/visualization/schedule_debug_server.py` — add `probe_mcp()` method to `DebugState`; add `--no-mcp-probe` argparse flag; call probe in `main()` after the `LLM:` print block (~1220) and before `serve_forever()` (~1236).

**Step 1: Add the probe method**

Add to `DebugState` (near `_write_mcp_config`):

```python
    def probe_mcp(self, timeout=90):
        """One-shot check that the LLM tab's claude can load + call the aiegdb
        MCP server. Returns (ok: bool, detail: str). Never raises."""
        claude = shutil.which(self.claude_bin) or (
            self.claude_bin if os.path.isfile(self.claude_bin) else None)
        if not claude:
            return False, "claude binary not found"
        cfg = self._write_mcp_config()
        if not cfg:
            return False, "MCP config not written"
        cmd = [self.claude_bin, "-p", "--output-format", "json",
               "--permission-mode", "bypassPermissions",
               "--mcp-config", cfg, "--strict-mcp-config",
               "--allowedTools", "mcp__aiegdb__aie_scope",
               "Call the aiegdb aie_scope tool and reply with only its "
               "scope line."]
        if self.claude_model:
            cmd += ["--model", self.claude_model]
        try:
            proc = subprocess.run(cmd, cwd=self.claude_cwd,
                                  capture_output=True, text=True,
                                  timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, f"probe timed out after {timeout}s"
        except OSError as e:
            return False, f"probe failed to run: {e}"
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            return False, f"claude exited {proc.returncode}: " \
                          f"{tail[-1] if tail else '(no output)'}"
        # Success heuristic: claude ran and the scope word appears in output.
        out = (proc.stdout or "")
        ok = "partition" in out or "scope" in out.lower()
        return ok, ("scope tool responded" if ok else
                    "claude ran but aie_scope output not detected")
```

**Step 2: Add the CLI flag**

Near the other LLM argparse flags (after `--no-password`, ~line 1144), add:

```python
    ap.add_argument("--no-mcp-probe", action="store_true",
                    help="skip the startup probe that verifies the LLM tab's "
                         "claude can reach the aiegdb MCP server")
```

**Step 3: Call the probe in `main()`**

In the `else` branch that prints the `LLM:` line (after ~line 1222
`print(f"  LLM auth: ...")`), add:

```python
        if not args.no_mcp_probe and _claude:
            ok, detail = Handler.state.probe_mcp()
            status = "connected" if ok else "NOT connected"
            print(f"  LLM MCP:    aiegdb {status} ({detail})")
            if not ok:
                print("  WARNING: LLM tab claude could not reach the aiegdb "
                      "MCP server; the chat still works but AIE debug tools "
                      "may be unavailable.", file=sys.stderr)
```

(Reuses the `_claude` variable computed at ~1216. Place this inside the same
`else:` block so it is skipped when `--no-llm`.)

**Step 4: Verify flag parses (board-free)**

```bash
python3 script/visualization/schedule_debug_server.py --help 2>&1 | grep -- --no-mcp-probe
```

Expected: the `--no-mcp-probe` help line prints.

**Step 5: Commit**

```bash
git add script/visualization/schedule_debug_server.py
git commit -m "feat(debug-server): startup probe verifying aiegdb MCP connection"
```

---

### Task 4: Clean up the temp MCP config on shutdown

**Files:**
- Modify: `script/visualization/schedule_debug_server.py:1237-1251` (the `KeyboardInterrupt` handler)

**Step 1: Add unlink in the shutdown handler**

After the `llm_proc` termination block (~line 1250, before `server.shutdown()`),
add:

```python
        mcp_cfg = Handler.state._mcp_config_path
        if mcp_cfg:
            try:
                os.unlink(mcp_cfg)
            except OSError:
                pass
```

**Step 2: Verify it parses / imports cleanly**

```bash
python3 -c "import ast; ast.parse(open('script/visualization/schedule_debug_server.py').read()); print('syntax OK')"
```

Expected: `syntax OK`.

**Step 3: Commit**

```bash
git add script/visualization/schedule_debug_server.py
git commit -m "chore(debug-server): remove temp MCP config on shutdown"
```

---

### Task 5: End-to-end manual verification

**Files:** none (verification only).

**Step 1: Offline (no board) probe smoke test**

Temporarily force dry-run in the injected env is not wired via CLI; instead verify
the probe path runs against the CLI without a board by confirming the config +
flags. Start the server with the LLM tab and a dummy target:

```bash
python3 script/visualization/schedule_debug_server.py \
  --workdir <path-with-host_schedule.html> \
  --no-password --target xsdb://127.0.0.1:3121
```

Expected in stdout: a line `  LLM MCP:    aiegdb connected (scope tool responded)`
OR `NOT connected (...)` with a clear reason. Either way the server keeps serving
and prints `Ctrl-C to stop.`

**Step 2: Confirm the temp config exists while running**

In another shell:

```bash
ls -la /tmp/aiegdb_*.mcp.json && cat /tmp/aiegdb_*.mcp.json
```

Expected: one file containing `"aiegdb"` with the five env keys and the resolved
`AIEMCP_STARTCOL` / `AIEDBG_TARGET`.

**Step 3: Live LLM-tab check**

Open the browser URL, go to the LLM tab, send: `run aie_scope`.
Expected: the transcript shows `[tool: mcp__aiegdb__aie_scope ...]` and returns a
`partition(startcol=N)...` scope line.

**Step 4: Shutdown cleanup check**

Ctrl-C the server, then:

```bash
ls /tmp/aiegdb_*.mcp.json 2>/dev/null || echo "cleaned up"
```

Expected: `cleaned up` (no leftover temp config).

**Step 5: Final commit (if any doc updates)**

Update `CLAUDE.md`'s `script/aiemcp.py` / `schedule_debug_server.py` bullets to
note the auto-generated `--mcp-config` + startup probe, then:

```bash
git add CLAUDE.md
git commit -m "docs: note LLM-tab auto-generated MCP config + startup probe"
```

---

## Rollback

All changes are in one file plus one design doc. `git revert` the feature commits
to restore cwd-based `.mcp.json` auto-discovery (the prior behavior).
