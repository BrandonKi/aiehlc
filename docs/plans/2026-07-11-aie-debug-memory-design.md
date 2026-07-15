# AIE Schedule Debug Long-Term Memory — Design

Date: 2026-07-11
Status: Approved (design phase)

## Problem

Live AIE debug sessions via `aiegdb` (CLI, HTML console, and MCP) produce a
stream of decoded results (DMA status, BD chains, events, PC, perf counters)
that are lost when the REPL exits. There is no durable, structured memory of an
investigation: what was checked, what was found, and how state evolved across
board runs. We want a long-term memory that consolidates debug results into
structured, re-loadable context and records debug-state history over time.

## Requirements (from request)

1. A **local memory server** that receives debug information (results),
   consolidates it into structured context, and maintains a state structure
   recording current debug state, history, etc.
2. A **`save` command** in both `aiegdb` (CLI) and the HTML aiegdb command line
   that saves the process-so-far into the server.
3. The `save` command records a **timestamp watermark** so the next save does
   not resend duplicate information.

Persistent memory files live under `debugcache/memory/`. All new solution code
lives under `src/tool/debug/agent/`.

## Decisions (from brainstorming)

- **Storage format:** JSON state + Markdown log (machine-readable state +
  human/LLM-readable narrative).
- **Consolidation:** rule-based parsing (deterministic, no LLM/network
  dependency), reusing aiediag's already-decoded output text.
- **Save source:** the aiegdb command+output history (REPL transcript) since the
  last save watermark.
- **Transport:** HTTP to a standalone stdlib `http.server` daemon (uniform for
  CLI, HTML, MCP; mirrors `schedule_debug_server.py`).
- **State model:** sessions + typed findings (dedup-keyed) + chronological
  timeline; `state.json` holds current state + per-session watermark.
- **Session id:** auto-derived from provenance (design/workdir), overridable via
  `save <session_name>`.

## Architecture

```
   aiegdb CLI  ─┐   memory daemon (agent/memory_server.py)
   HTML console ─┼─▶ stdlib http.server on 127.0.0.1:PORT
   MCP (future) ─┘    POST /save  POST /session
                      GET /state  /session /timeline /context /health
                             │ imports
                      agent/memory_store.py  (importable library)
                        - rule-based parse: aiegdb output -> findings
                        - dedup by content hash
                        - session / state / timeline model
                        - atomic read/write (tmp + os.replace, lock)
                             │ writes
                      debugcache/memory/
                        state.json           (current state + watermarks)
                        sessions/<id>.json   (findings + timeline)
                        sessions/<id>.md      (append-only readable log)
```

### Components (`src/tool/debug/agent/`)

1. **`memory_store.py`** — core importable library (no HTTP). Owns the data
   model, rule-based parser, dedup, watermark logic, atomic file I/O.
2. **`memory_server.py`** — thin stdlib `http.server` daemon wrapping
   `memory_store`. Follows `schedule_debug_server.py` conventions
   (`BaseHTTPRequestHandler`, `_send_json`, string-dispatch do_GET/do_POST).
3. **`memory_client.py`** — small `urllib`-based helper so `aiegdb.py` posts a
   save without HTTP boilerplate; handles daemon-offline gracefully.

### Integration edits (outside agent/)

4. **`aiegdb.py`** — record a per-process transcript (`{ts, scope, command,
   output}`) in `AieGdb`; add a universal `save [session] [-m note]` verb that
   flushes transcript-since-watermark via `memory_client`, advancing the local
   watermark only on success.
5. **`schedule_view.py`** — HTML `save` affordance in the aiegdb console that
   sends the `save` command through the existing `/aiegdb` path (reuses the same
   subprocess, so it inherits the transcript + watermark).

## Data Model & File Schema

### `debugcache/memory/state.json`
```json
{
  "version": 1,
  "active_session": "simplematmul_2x2",
  "sessions": {
    "simplematmul_2x2": {
      "created": "2026-07-11T00:20:03Z",
      "last_save": "2026-07-11T00:41:12Z",
      "watermark_ts": "2026-07-11T00:41:12Z",
      "design": "simplematmul",
      "workdir": "aout/worklocal",
      "startcol": 3, "aie_version": "2ps",
      "finding_count": 7, "save_count": 3
    }
  }
}
```

### `debugcache/memory/sessions/<id>.json`
```json
{
  "id": "simplematmul_2x2",
  "findings": [
    {
      "hash": "a1b2c3d4",
      "kind": "dma_status",
      "scope": "partition(startcol=3)/tile(0,3)/mm2s0",
      "tile": [0,3], "channel": "mm2s0",
      "command": "dma status",
      "summary": "MM2S ch0 tile(0,3): started, not finished (stuck)",
      "detail": "<parsed fields>",
      "first_seen": "2026-07-11T00:22:10Z",
      "last_seen": "2026-07-11T00:41:12Z",
      "seen_count": 2
    }
  ],
  "timeline": [
    {"ts":"2026-07-11T00:22:10Z","save_id":1,"commands":12,
     "new_findings":5,"note":"initial scan of stuck mm2s chain"}
  ]
}
```

### `debugcache/memory/sessions/<id>.md`
```markdown
## Save 3 — 2026-07-11T00:41:12Z
Scope: tile(0,3)/mm2s0
New findings (2):
- [dma_status] MM2S ch0 tile(0,3): started, not finished (stuck)
- [bd] BD0 len=1024 mismatch vs schedule 2048
Commands run: dma status; bd; event
```

### Dedup
- **Layer 1 (client watermark):** only transcript entries newer than
  `self._save_watermark` are sent; advanced to `now` on successful save.
- **Layer 2 (server content hash):** `hash(kind + scope + command +
  normalized_summary)`. Repeated identical findings bump `seen_count` /
  `last_seen` instead of appending. Catches re-sends after a client restart.

## Save Protocol

`save [session_name] [-m "note"]` — universal verb in `AieGdb._dispatch`:
1. Collect transcript entries with `ts > self._save_watermark`.
2. POST `/save` `{session, entries, note, meta:{startcol,aie_version,workdir,
   design}}`.
3. On 200: `self._save_watermark = now`.
4. On daemon-offline: warn, do NOT advance watermark (retry-safe).

Server `memory_store.consolidate`:
- Classify each entry by command verb + scope → `kind` (`dma_status`, `bd`,
  `event`, `pc`, `counter`, `scan`, `raw`).
- Extract one-line `summary` from the already-decoded aiediag output.
- Compute dedup hash; merge into session findings; append timeline + `.md`
  section; update `state.json`.

## HTTP API (127.0.0.1, default port 8790, `--port` override)

| Method | Path | Body / Query | Returns |
|--------|------|--------------|---------|
| GET  | `/health` | — | `{ok, version, active_session}` |
| POST | `/save` | `{session?, entries[], note?, meta{}}` | `{ok, session, new_findings, dup_findings, save_id}` |
| POST | `/session` | `{session, activate?}` | `{ok, session}` |
| GET  | `/state` | — | `state.json` |
| GET  | `/session` | `id` | session `.json` |
| GET  | `/timeline` | `id` | timeline array |
| GET  | `/context` | `id, kind?, tile?` | consolidated markdown/text for LLM re-feed |

## Error Handling

- Daemon offline → client prints `[memory: daemon offline; save skipped]`,
  watermark unchanged, REPL unaffected.
- Atomic writes: `*.tmp` + `os.replace` under a per-store `threading.Lock`.
- Corrupt/missing `state.json` → rebuild from `sessions/*.json`.
- Malformed POST → `{error}` 400 (mirrors existing `_send_json`).
- Session auto-id from provenance `design`/workdir; `save <name>` overrides.

## Testing (`agent/unitest/`)

- `test_memory_store.py` — parser classification, dedup hash stability,
  watermark filtering, atomic write + corrupt recovery, session auto-id. Pure
  Python, no HW/network.
- `test_memory_server.py` — daemon on ephemeral port; POST `/save` twice with
  overlapping entries → second reports all dups; GET `/state` / `/context`.
- aiegdb `save`: extend dry-run/batch path — `aiegdb -c "save testsess"` against
  a stub daemon.

## Files Created / Changed

- New: `src/tool/debug/agent/memory_store.py`,
  `src/tool/debug/agent/memory_server.py`,
  `src/tool/debug/agent/memory_client.py`,
  `src/tool/debug/agent/unitest/test_memory_store.py`,
  `src/tool/debug/agent/unitest/test_memory_server.py`,
  `src/tool/debug/agent/README.md`
- Changed: `src/tool/debug/aiegdb.py` (transcript + `save` verb + help),
  `src/tool/debug/schedule_view.py` (HTML save affordance)
- New skill (per repo Learn rule): `.cursor/skills/aiedebugmemory/SKILL.md`
- Memory data (runtime, gitignored): `debugcache/memory/`
