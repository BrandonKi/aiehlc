# AIE Schedule Debug Long-Term Memory Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a local debug-memory subsystem that consolidates `aiegdb` REPL
transcripts into durable, structured sessions (findings + timeline) and lets a
watermark-deduped `save` command persist "the process so far" from both the
aiegdb CLI and the HTML console.

**Architecture:** A stdlib-only importable store (`memory_store.py`) owns the
data model, rule-based parser, dedup, watermark, and atomic JSON+Markdown I/O
under `debugcache/memory/`. A thin stdlib `http.server` daemon
(`memory_server.py`) wraps it. `aiegdb.py` records a per-process transcript and
gains a `save` verb that POSTs new-since-watermark entries via a small
`memory_client.py`. The HTML console reuses the existing `/aiegdb` path.

**Tech Stack:** Python 3 stdlib only (`http.server`, `urllib`, `json`,
`hashlib`, `threading`, `unittest`). No third-party deps. Mirrors the
conventions in `src/tool/debug/schedule_debug_server.py`.

**Design reference:** `docs/plans/2026-07-11-aie-debug-memory-design.md`

---

## Conventions (read before starting)

- All new code under `src/tool/debug/agent/`. Stdlib only.
- Follow `schedule_debug_server.py` house style: `_THIS_DIR`/`_REPO_ROOT`
  module globals; `BaseHTTPRequestHandler` with a `_send_json(obj, code=200)`
  helper; string-dispatch in `do_GET`/`do_POST`; module docstrings; keep any
  function under 200 lines (repo rule in CLAUDE.md).
- Timestamps: UTC ISO-8601 with `Z`, e.g. `datetime.now(timezone.utc)
  .strftime("%Y-%m-%dT%H:%M:%SZ")`. Provide a single `_now_iso()` helper.
- Atomic writes: write `path + ".tmp"` then `os.replace(tmp, path)`, all under
  a `threading.Lock` held by the store.
- Tests use `unittest` (stdlib), runnable with
  `python3 -m unittest -v src.tool.debug.agent.unitest.test_memory_store`
  from repo root, OR directly `python3 <path>/test_memory_store.py`. Each test
  file must be runnable standalone via `unittest.main()`.
- Memory data dir default: `<repo>/debugcache/memory`. Resolve once via a
  `default_memory_dir()` helper (based on `_REPO_ROOT`).

---

## Task 1: Scaffold agent package + memory dir helper

**Files:**
- Create: `src/tool/debug/agent/__init__.py`
- Create: `src/tool/debug/agent/unitest/__init__.py`
- Create: `src/tool/debug/agent/memory_store.py` (stub only)
- Create: `src/tool/debug/agent/unitest/test_memory_store.py`

**Step 1: Write the failing test**

In `src/tool/debug/agent/unitest/test_memory_store.py`:

```python
#!/usr/bin/env python3
import os
import sys
import unittest

_AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)
import memory_store  # noqa: E402


class TestMemoryDir(unittest.TestCase):
    def test_default_memory_dir_ends_with_debugcache_memory(self):
        d = memory_store.default_memory_dir()
        self.assertTrue(d.replace("\\", "/").endswith("debugcache/memory"))


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `python3 src/tool/debug/agent/unitest/test_memory_store.py`
Expected: FAIL — `AttributeError: module 'memory_store' has no attribute
'default_memory_dir'`.

**Step 3: Write minimal implementation**

In `src/tool/debug/agent/memory_store.py`:

```python
#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""memory_store - importable long-term debug-memory store for aiegdb.

Consolidates aiegdb REPL transcripts into durable sessions (typed findings +
timeline), written under debugcache/memory/ as JSON state + a Markdown log.
No HTTP, no third-party deps; memory_server.py wraps this over http.server.

Design reference: docs/plans/2026-07-11-aie-debug-memory-design.md
"""

import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# agent/ -> debug/ -> tool/ -> src/ -> repo root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(_THIS_DIR))))


def default_memory_dir():
    """Absolute path to the memory data dir (<repo>/debugcache/memory)."""
    return os.path.join(_REPO_ROOT, "debugcache", "memory")
```

Create empty `src/tool/debug/agent/__init__.py` and
`src/tool/debug/agent/unitest/__init__.py`.

**Step 4: Run test to verify it passes**

Run: `python3 src/tool/debug/agent/unitest/test_memory_store.py`
Expected: PASS (1 test).

**Step 5: Commit**

```bash
git add src/tool/debug/agent/__init__.py \
  src/tool/debug/agent/unitest/__init__.py \
  src/tool/debug/agent/memory_store.py \
  src/tool/debug/agent/unitest/test_memory_store.py
git commit -m "feat(debug-memory): scaffold agent package + memory dir helper"
```

---

## Task 2: `_now_iso()` + finding hash (dedup key)

**Files:**
- Modify: `src/tool/debug/agent/memory_store.py`
- Test: `src/tool/debug/agent/unitest/test_memory_store.py`

**Step 1: Write the failing test** (append a class)

```python
class TestHashing(unittest.TestCase):
    def test_finding_hash_is_stable_and_ignores_whitespace(self):
        h1 = memory_store.finding_hash(
            "dma_status", "tile(0,3)/mm2s0", "dma status",
            "MM2S ch0 stuck")
        h2 = memory_store.finding_hash(
            "dma_status", "tile(0,3)/mm2s0", "dma status",
            "  MM2S   ch0 stuck  ")
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 8)

    def test_finding_hash_differs_on_scope(self):
        h1 = memory_store.finding_hash("dma_status", "tile(0,3)/mm2s0",
                                       "dma status", "x")
        h2 = memory_store.finding_hash("dma_status", "tile(1,3)/mm2s0",
                                       "dma status", "x")
        self.assertNotEqual(h1, h2)

    def test_now_iso_format(self):
        s = memory_store._now_iso()
        self.assertRegex(s, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
```

**Step 2: Run test to verify it fails**

Run: `python3 src/tool/debug/agent/unitest/test_memory_store.py`
Expected: FAIL — no `finding_hash` / `_now_iso`.

**Step 3: Write minimal implementation** (add imports + functions)

```python
import hashlib
import re
from datetime import datetime, timezone

_WS_RE = re.compile(r"\s+")


def _now_iso():
    """UTC ISO-8601 timestamp with trailing Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _norm(s):
    """Collapse whitespace + strip, for hash stability."""
    return _WS_RE.sub(" ", (s or "").strip())


def finding_hash(kind, scope, command, summary):
    """Stable 8-hex dedup key over normalized (kind, scope, command,
    summary)."""
    key = "|".join(_norm(x) for x in (kind, scope, command, summary))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
```

**Step 4: Run test to verify it passes**

Run: `python3 src/tool/debug/agent/unitest/test_memory_store.py`
Expected: PASS (4 tests).

**Step 5: Commit**

```bash
git add src/tool/debug/agent/memory_store.py \
  src/tool/debug/agent/unitest/test_memory_store.py
git commit -m "feat(debug-memory): add _now_iso and stable finding hash"
```

---

## Task 3: Rule-based transcript-entry classifier

Classifies one transcript entry `{scope, command, output}` into a `finding`
dict `{kind, scope, tile, channel, command, summary, detail}` (no timestamps
yet — the store adds those). Reuse aiediag's decoded text: the `summary` is the
most salient decoded line (e.g. lines containing `stuck`, `completed`,
`ERROR`, `mismatch`, `PC=`), falling back to the first non-empty output line.

**Files:**
- Modify: `src/tool/debug/agent/memory_store.py`
- Test: `src/tool/debug/agent/unitest/test_memory_store.py`

**Step 1: Write the failing test**

```python
class TestClassify(unittest.TestCase):
    def test_dma_status_kind_and_tile_channel(self):
        entry = {"scope": "partition(startcol=3)/tile(0,3)/mm2s0",
                 "command": "dma status",
                 "output": "MM2S ch0 tile(3,3): started but never finished "
                           "(stuck)"}
        f = memory_store.classify(entry)
        self.assertEqual(f["kind"], "dma_status")
        self.assertEqual(f["tile"], [0, 3])
        self.assertEqual(f["channel"], "mm2s0")
        self.assertIn("stuck", f["summary"])

    def test_bd_command_kind(self):
        entry = {"scope": "partition/tile(0,3)/mm2s0", "command": "bd",
                 "output": "BD0 len=1024 ..."}
        self.assertEqual(memory_store.classify(entry)["kind"], "bd")

    def test_pc_command_summary_prefers_pc_line(self):
        entry = {"scope": "partition/tile(0,3)", "command": "pc",
                 "output": "  PC=0x00123 -> foo.cc:42"}
        f = memory_store.classify(entry)
        self.assertEqual(f["kind"], "pc")
        self.assertIn("PC=0x00123", f["summary"])

    def test_unknown_command_is_raw(self):
        entry = {"scope": "partition", "command": "scan dma",
                 "output": "..."}
        self.assertEqual(memory_store.classify(entry)["kind"], "scan")
        entry2 = {"scope": "partition", "command": "wat", "output": "x"}
        self.assertEqual(memory_store.classify(entry2)["kind"], "raw")
```

**Step 2: Run test to verify it fails**

Run: `python3 src/tool/debug/agent/unitest/test_memory_store.py`
Expected: FAIL — no `classify`.

**Step 3: Write minimal implementation**

```python
# scope parsing: pull tile(c,r) and trailing dir_ch from an aiegdb prompt/scope
_TILE_RE = re.compile(r"tile\((\d+),(\d+)\)")
_CHAN_RE = re.compile(r"/((?:mm2s|s2mm)\d+)\b", re.IGNORECASE)

# command verb -> finding kind
_KIND_BY_VERB = {
    "dma": "dma_status",   # "dma status" / "dma <dir_ch>"
    "status": "dma_status",
    "bd": "bd",
    "event": "event",
    "pc": "pc",
    "counter": "counter",
    "scan": "scan",
}
# lines that are "interesting" for a one-line summary, most-salient first
_SALIENT = ("ERROR", "stuck", "mismatch", "never started", "never finished",
            "completed", "PC=", "started")


def _scope_tile_channel(scope):
    tile = None
    m = _TILE_RE.search(scope or "")
    if m:
        tile = [int(m.group(1)), int(m.group(2))]
    ch = None
    mc = _CHAN_RE.search(scope or "")
    if mc:
        ch = mc.group(1).lower()
    return tile, ch


def _summary_from_output(output):
    lines = [ln.strip() for ln in (output or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    for needle in _SALIENT:
        for ln in lines:
            if needle in ln:
                return ln
    return lines[0]


def classify(entry):
    """Turn a transcript entry into a finding dict (no timestamps)."""
    scope = entry.get("scope", "")
    command = entry.get("command", "")
    output = entry.get("output", "")
    verb = (command.strip().split() or [""])[0].lower()
    # "dma counter ..." is a counter finding, not dma_status
    if verb == "dma" and "counter" in command.lower():
        kind = "counter"
    else:
        kind = _KIND_BY_VERB.get(verb, "raw")
    tile, ch = _scope_tile_channel(scope)
    summary = _summary_from_output(output)
    return {"kind": kind, "scope": scope, "tile": tile, "channel": ch,
            "command": command, "summary": summary, "detail": output}
```

**Step 4: Run test to verify it passes**

Run: `python3 src/tool/debug/agent/unitest/test_memory_store.py`
Expected: PASS (all classify tests).

**Step 5: Commit**

```bash
git add src/tool/debug/agent/memory_store.py \
  src/tool/debug/agent/unitest/test_memory_store.py
git commit -m "feat(debug-memory): rule-based transcript-entry classifier"
```

---

## Task 4: `MemoryStore` class — load/save state, atomic writes, session auto-id

**Files:**
- Modify: `src/tool/debug/agent/memory_store.py`
- Test: `src/tool/debug/agent/unitest/test_memory_store.py`

**Step 1: Write the failing test**

```python
import tempfile


class TestStoreBasics(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = memory_store.MemoryStore(self.tmp)

    def test_fresh_state_has_no_sessions(self):
        st = self.store.get_state()
        self.assertEqual(st["sessions"], {})
        self.assertIsNone(st["active_session"])

    def test_ensure_session_creates_and_activates(self):
        sid = self.store.ensure_session(None, meta={"design": "simplematmul",
                                                    "workdir": "aout/worklocal"})
        self.assertTrue(sid)
        st = self.store.get_state()
        self.assertIn(sid, st["sessions"])
        self.assertEqual(st["active_session"], sid)

    def test_explicit_session_name_wins(self):
        sid = self.store.ensure_session("mysess", meta={})
        self.assertEqual(sid, "mysess")

    def test_state_persists_across_instances(self):
        self.store.ensure_session("s1", meta={})
        store2 = memory_store.MemoryStore(self.tmp)
        self.assertIn("s1", store2.get_state()["sessions"])
```

**Step 2: Run test to verify it fails**

Run: `python3 src/tool/debug/agent/unitest/test_memory_store.py`
Expected: FAIL — no `MemoryStore`.

**Step 3: Write minimal implementation**

```python
import json
import threading

STATE_VERSION = 1


def _safe_id(s):
    """Filesystem-safe session id fragment."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", (s or "").strip()) or "session"


class MemoryStore:
    """Durable debug-memory: state.json + sessions/<id>.{json,md}.

    Thread-safe via a single lock; all writes are atomic (tmp + os.replace).
    """

    def __init__(self, memory_dir=None):
        self.dir = os.path.abspath(memory_dir or default_memory_dir())
        self.sessions_dir = os.path.join(self.dir, "sessions")
        os.makedirs(self.sessions_dir, exist_ok=True)
        self._lock = threading.Lock()

    # ---- paths -----------------------------------------------------------
    def _state_path(self):
        return os.path.join(self.dir, "state.json")

    def _session_json(self, sid):
        return os.path.join(self.sessions_dir, _safe_id(sid) + ".json")

    def _session_md(self, sid):
        return os.path.join(self.sessions_dir, _safe_id(sid) + ".md")

    # ---- atomic io -------------------------------------------------------
    def _read_json(self, path, default):
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return default

    def _write_json(self, path, obj):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)

    def _append_text(self, path, text):
        with open(path, "a") as f:
            f.write(text)

    # ---- state -----------------------------------------------------------
    def _load_state(self):
        return self._read_json(self._state_path(),
                               {"version": STATE_VERSION,
                                "active_session": None, "sessions": {}})

    def get_state(self):
        with self._lock:
            return self._load_state()

    def ensure_session(self, session, meta=None):
        """Create (if new) + activate a session; return its id. Explicit name
        wins, else auto-derive from meta.design/workdir."""
        meta = meta or {}
        with self._lock:
            state = self._load_state()
            sid = _safe_id(session) if session else self._auto_id(meta, state)
            if sid not in state["sessions"]:
                state["sessions"][sid] = {
                    "created": _now_iso(), "last_save": None,
                    "watermark_ts": None,
                    "design": meta.get("design"),
                    "workdir": meta.get("workdir"),
                    "startcol": meta.get("startcol"),
                    "aie_version": meta.get("aie_version"),
                    "finding_count": 0, "save_count": 0}
                self._write_json(self._session_json(sid),
                                 {"id": sid, "findings": [], "timeline": []})
            state["active_session"] = sid
            self._write_json(self._state_path(), state)
            return sid

    def _auto_id(self, meta, state):
        design = meta.get("design")
        if design:
            return _safe_id(design)
        wd = meta.get("workdir")
        if wd:
            return _safe_id(os.path.basename(os.path.normpath(wd)))
        return state.get("active_session") or "session"
```

**Step 4: Run test to verify it passes**

Run: `python3 src/tool/debug/agent/unitest/test_memory_store.py`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/tool/debug/agent/memory_store.py \
  src/tool/debug/agent/unitest/test_memory_store.py
git commit -m "feat(debug-memory): MemoryStore state + atomic io + session id"
```

---

## Task 5: `consolidate()` — merge entries into findings/timeline/md + watermark

**Files:**
- Modify: `src/tool/debug/agent/memory_store.py`
- Test: `src/tool/debug/agent/unitest/test_memory_store.py`

**Step 1: Write the failing test**

```python
class TestConsolidate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = memory_store.MemoryStore(self.tmp)

    def _entry(self, out, cmd="dma status",
               scope="partition/tile(0,3)/mm2s0", ts="2026-01-01T00:00:01Z"):
        return {"ts": ts, "scope": scope, "command": cmd, "output": out}

    def test_first_save_adds_findings(self):
        res = self.store.consolidate(
            None, [self._entry("MM2S ch0 stuck")], note="init", meta={})
        self.assertEqual(res["new_findings"], 1)
        self.assertEqual(res["dup_findings"], 0)
        sess = self.store.get_session(res["session"])
        self.assertEqual(len(sess["findings"]), 1)
        self.assertEqual(len(sess["timeline"]), 1)

    def test_duplicate_entry_is_deduped(self):
        r1 = self.store.consolidate(None, [self._entry("MM2S ch0 stuck")],
                                    meta={})
        r2 = self.store.consolidate(r1["session"],
                                    [self._entry("MM2S ch0 stuck")], meta={})
        self.assertEqual(r2["new_findings"], 0)
        self.assertEqual(r2["dup_findings"], 1)
        sess = self.store.get_session(r1["session"])
        self.assertEqual(len(sess["findings"]), 1)
        self.assertEqual(sess["findings"][0]["seen_count"], 2)

    def test_watermark_advances_to_latest_entry_ts(self):
        self.store.consolidate(
            "s", [self._entry("x", ts="2026-01-01T00:00:05Z")], meta={})
        st = self.store.get_state()
        self.assertEqual(st["sessions"]["s"]["watermark_ts"],
                         "2026-01-01T00:00:05Z")

    def test_md_log_appended(self):
        r = self.store.consolidate("s", [self._entry("MM2S ch0 stuck")],
                                   note="hello", meta={})
        with open(self.store._session_md(r["session"])) as f:
            md = f.read()
        self.assertIn("MM2S ch0 stuck", md)
        self.assertIn("hello", md)
```

**Step 2: Run test to verify it fails**

Run: `python3 src/tool/debug/agent/unitest/test_memory_store.py`
Expected: FAIL — no `consolidate` / `get_session`.

**Step 3: Write minimal implementation**

```python
    def get_session(self, sid):
        with self._lock:
            return self._read_json(self._session_json(sid),
                                   {"id": _safe_id(sid), "findings": [],
                                    "timeline": []})

    def consolidate(self, session, entries, note="", meta=None):
        """Merge transcript entries into the session; dedup by finding hash.

        Returns {ok, session, new_findings, dup_findings, save_id}.
        """
        sid = self.ensure_session(session, meta=meta)  # takes+releases lock
        with self._lock:
            sess = self._read_json(self._session_json(sid),
                                   {"id": sid, "findings": [], "timeline": []})
            by_hash = {f["hash"]: f for f in sess["findings"]}
            new_cnt = dup_cnt = 0
            md_new = []
            latest_ts = None
            for e in entries:
                ts = e.get("ts") or _now_iso()
                if latest_ts is None or ts > latest_ts:
                    latest_ts = ts
                f = classify(e)
                h = finding_hash(f["kind"], f["scope"], f["command"],
                                 f["summary"])
                if h in by_hash:
                    ex = by_hash[h]
                    ex["seen_count"] += 1
                    ex["last_seen"] = ts
                    dup_cnt += 1
                else:
                    f.update({"hash": h, "first_seen": ts, "last_seen": ts,
                              "seen_count": 1})
                    sess["findings"].append(f)
                    by_hash[h] = f
                    new_cnt += 1
                    md_new.append(f)

            state = self._load_state()
            meta_s = state["sessions"].setdefault(sid, {})
            save_id = int(meta_s.get("save_count", 0)) + 1
            sess["timeline"].append(
                {"ts": _now_iso(), "save_id": save_id,
                 "commands": len(entries), "new_findings": new_cnt,
                 "note": note or ""})
            self._write_json(self._session_json(sid), sess)

            meta_s["save_count"] = save_id
            meta_s["finding_count"] = len(sess["findings"])
            meta_s["last_save"] = _now_iso()
            if latest_ts:
                meta_s["watermark_ts"] = latest_ts
            state["active_session"] = sid
            self._write_json(self._state_path(), state)

            self._append_text(self._session_md(sid),
                              self._render_md(save_id, note, md_new, entries))
            return {"ok": True, "session": sid, "new_findings": new_cnt,
                    "dup_findings": dup_cnt, "save_id": save_id}

    def _render_md(self, save_id, note, new_findings, entries):
        cmds = "; ".join(e.get("command", "") for e in entries)
        lines = [f"\n## Save {save_id} — {_now_iso()}"]
        if note:
            lines.append(f"Note: {note}")
        lines.append(f"New findings ({len(new_findings)}):")
        for f in new_findings:
            lines.append(f"- [{f['kind']}] {f['summary']}")
        lines.append(f"Commands run: {cmds}\n")
        return "\n".join(lines)
```

Note: `ensure_session` acquires the lock and releases it before the `with
self._lock` block in `consolidate`, so there is no re-entrant deadlock (the
lock is a plain `threading.Lock`). Do NOT call `ensure_session` while holding
the lock.

**Step 4: Run test to verify it passes**

Run: `python3 src/tool/debug/agent/unitest/test_memory_store.py`
Expected: PASS (all consolidate tests).

**Step 5: Commit**

```bash
git add src/tool/debug/agent/memory_store.py \
  src/tool/debug/agent/unitest/test_memory_store.py
git commit -m "feat(debug-memory): consolidate entries with dedup + watermark"
```

---

## Task 6: `context()` read-back (structured context for LLM/human)

**Files:**
- Modify: `src/tool/debug/agent/memory_store.py`
- Test: `src/tool/debug/agent/unitest/test_memory_store.py`

**Step 1: Write the failing test**

```python
class TestContext(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = memory_store.MemoryStore(self.tmp)
        self.store.consolidate(
            "s", [{"ts": "2026-01-01T00:00:01Z",
                   "scope": "partition/tile(0,3)/mm2s0",
                   "command": "dma status", "output": "MM2S ch0 stuck"}],
            meta={})

    def test_context_returns_markdown_with_finding(self):
        ctx = self.store.context("s")
        self.assertIn("MM2S ch0 stuck", ctx)

    def test_context_kind_filter(self):
        self.assertIn("stuck", self.store.context("s", kind="dma_status"))
        self.assertNotIn("stuck", self.store.context("s", kind="bd"))
```

**Step 2: Run test to verify it fails**

Run: `python3 src/tool/debug/agent/unitest/test_memory_store.py`
Expected: FAIL — no `context`.

**Step 3: Write minimal implementation**

```python
    def context(self, sid, kind=None, tile=None):
        """Consolidated text context for a session, optionally filtered by
        finding kind and/or tile [c,r]. Returns a markdown-ish string."""
        sess = self.get_session(sid)
        rows = []
        for f in sess.get("findings", []):
            if kind and f.get("kind") != kind:
                continue
            if tile and f.get("tile") != list(tile):
                continue
            rows.append(f"- [{f['kind']}] {f.get('scope','')}: "
                        f"{f.get('summary','')} (seen {f.get('seen_count',1)})")
        header = f"# Debug memory: session {sid}"
        if not rows:
            return header + "\n(no matching findings)\n"
        return header + "\n" + "\n".join(rows) + "\n"
```

**Step 4: Run test to verify it passes**

Run: `python3 src/tool/debug/agent/unitest/test_memory_store.py`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/tool/debug/agent/memory_store.py \
  src/tool/debug/agent/unitest/test_memory_store.py
git commit -m "feat(debug-memory): context read-back with kind/tile filters"
```

---

## Task 7: Corrupt-state recovery

**Files:**
- Modify: `src/tool/debug/agent/memory_store.py`
- Test: `src/tool/debug/agent/unitest/test_memory_store.py`

**Step 1: Write the failing test**

```python
class TestRecovery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = memory_store.MemoryStore(self.tmp)

    def test_corrupt_state_json_rebuilds_from_sessions(self):
        self.store.consolidate("s1", [{"ts": "2026-01-01T00:00:01Z",
            "scope": "partition/tile(0,3)/mm2s0", "command": "dma status",
            "output": "x"}], meta={})
        with open(self.store._state_path(), "w") as f:
            f.write("{ this is not json")
        store2 = memory_store.MemoryStore(self.tmp)
        st = store2.get_state()          # must not raise
        # rebuild_state() reconstructs the session entry from sessions/*.json
        store2.rebuild_state()
        self.assertIn("s1", store2.get_state()["sessions"])
```

**Step 2: Run test to verify it fails**

Run: `python3 src/tool/debug/agent/unitest/test_memory_store.py`
Expected: FAIL — no `rebuild_state` (and/or `get_state` already tolerant).

**Step 3: Write minimal implementation**

```python
    def rebuild_state(self):
        """Reconstruct state.json from sessions/*.json (recovery path)."""
        with self._lock:
            state = {"version": STATE_VERSION, "active_session": None,
                     "sessions": {}}
            for name in sorted(os.listdir(self.sessions_dir)):
                if not name.endswith(".json"):
                    continue
                sid = name[:-len(".json")]
                sess = self._read_json(
                    os.path.join(self.sessions_dir, name),
                    {"findings": [], "timeline": []})
                tl = sess.get("timeline", [])
                state["sessions"][sid] = {
                    "created": (tl[0]["ts"] if tl else _now_iso()),
                    "last_save": (tl[-1]["ts"] if tl else None),
                    "watermark_ts": None,
                    "finding_count": len(sess.get("findings", [])),
                    "save_count": len(tl)}
                state["active_session"] = sid
            self._write_json(self._state_path(), state)
            return state
```

`get_state`/`_load_state` already return the default dict on `ValueError`
(corrupt JSON), so `get_state()` does not raise — the test's first assertion
passes without change; `rebuild_state()` repopulates sessions.

**Step 4: Run test to verify it passes**

Run: `python3 src/tool/debug/agent/unitest/test_memory_store.py`
Expected: PASS (all store tests).

**Step 5: Commit**

```bash
git add src/tool/debug/agent/memory_store.py \
  src/tool/debug/agent/unitest/test_memory_store.py
git commit -m "feat(debug-memory): rebuild state.json from sessions on corrupt"
```

---

## Task 8: `memory_server.py` HTTP daemon

**Files:**
- Create: `src/tool/debug/agent/memory_server.py`
- Test: `src/tool/debug/agent/unitest/test_memory_server.py`

**Step 1: Write the failing test**

`src/tool/debug/agent/unitest/test_memory_server.py`:

```python
#!/usr/bin/env python3
import json
import os
import sys
import tempfile
import unittest
import urllib.request

_AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)
import memory_server  # noqa: E402


def _post(url, obj):
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read().decode("utf-8"))


def _get(url):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read().decode("utf-8"))


class TestServer(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.srv = memory_server.make_server("127.0.0.1", 0, self.tmp)
        self.port = self.srv.server_address[1]
        import threading
        self.t = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.t.start()
        self.base = f"http://127.0.0.1:{self.port}"

    def tearDown(self):
        self.srv.shutdown()

    def test_health(self):
        r = _get(self.base + "/health")
        self.assertTrue(r["ok"])

    def test_save_then_dup(self):
        entry = {"ts": "2026-01-01T00:00:01Z",
                 "scope": "partition/tile(0,3)/mm2s0",
                 "command": "dma status", "output": "MM2S ch0 stuck"}
        r1 = _post(self.base + "/save",
                   {"session": "s", "entries": [entry], "meta": {}})
        self.assertEqual(r1["new_findings"], 1)
        r2 = _post(self.base + "/save",
                   {"session": "s", "entries": [entry], "meta": {}})
        self.assertEqual(r2["dup_findings"], 1)

    def test_state_and_context(self):
        _post(self.base + "/save",
              {"session": "s", "entries": [
                  {"ts": "2026-01-01T00:00:01Z",
                   "scope": "partition/tile(0,3)/mm2s0",
                   "command": "dma status", "output": "MM2S ch0 stuck"}],
               "meta": {}})
        st = _get(self.base + "/state")
        self.assertIn("s", st["sessions"])
        ctx = _get(self.base + "/context?id=s")
        self.assertIn("stuck", ctx["context"])


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `python3 src/tool/debug/agent/unitest/test_memory_server.py`
Expected: FAIL — no `memory_server`.

**Step 3: Write minimal implementation**

`src/tool/debug/agent/memory_server.py`:

```python
#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""memory_server - stdlib http.server daemon wrapping memory_store.

Local-only (127.0.0.1) HTTP front-end so aiegdb (CLI + HTML console + MCP) can
POST debug transcripts to the long-term debug memory. Mirrors the conventions
of src/tool/debug/schedule_debug_server.py.

Endpoints:
  GET  /health                      {ok, version, active_session}
  POST /save    {session?,entries[],note?,meta{}}
  POST /session {session, activate?}
  GET  /state
  GET  /session?id=
  GET  /timeline?id=
  GET  /context?id=&kind=&tile=c,r

Run: python3 memory_server.py [--host 127.0.0.1] [--port 8790] [--dir PATH]
"""

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
import memory_store  # noqa: E402

DEFAULT_PORT = 8790


class _Handler(BaseHTTPRequestHandler):
    store = None  # set by make_server

    def log_message(self, *a):  # quiet
        pass

    def _send_json(self, obj, code=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b""
        return json.loads(raw.decode("utf-8")) if raw else {}

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        st = self.store
        if u.path == "/health":
            state = st.get_state()
            self._send_json({"ok": True, "version": memory_store.STATE_VERSION,
                             "active_session": state.get("active_session")})
        elif u.path == "/state":
            self._send_json(st.get_state())
        elif u.path == "/session":
            self._send_json(st.get_session(q.get("id", [""])[0]))
        elif u.path == "/timeline":
            self._send_json(
                {"timeline": st.get_session(q.get("id", [""])[0])
                 .get("timeline", [])})
        elif u.path == "/context":
            tile = None
            if q.get("tile"):
                try:
                    tile = [int(x) for x in q["tile"][0].split(",")]
                except ValueError:
                    tile = None
            ctx = st.context(q.get("id", [""])[0],
                             kind=q.get("kind", [None])[0], tile=tile)
            self._send_json({"context": ctx})
        else:
            self._send_json({"error": f"unknown path: {u.path}"}, code=404)

    def do_POST(self):
        u = urlparse(self.path)
        try:
            body = self._body()
        except (ValueError, json.JSONDecodeError):
            self._send_json({"error": "invalid JSON body"}, code=400)
            return
        st = self.store
        try:
            if u.path == "/save":
                self._send_json(st.consolidate(
                    body.get("session"), body.get("entries", []),
                    note=body.get("note", ""), meta=body.get("meta", {})))
            elif u.path == "/session":
                sid = st.ensure_session(body.get("session"),
                                        meta=body.get("meta", {}))
                self._send_json({"ok": True, "session": sid})
            else:
                self._send_json({"error": f"unknown path: {u.path}"}, code=404)
        except Exception as e:  # never crash the daemon
            self._send_json({"error": str(e)}, code=500)


def make_server(host, port, memory_dir):
    """Build (but do not start) a ThreadingHTTPServer bound to host:port."""
    handler = type("_H", (_Handler,),
                   {"store": memory_store.MemoryStore(memory_dir)})
    return ThreadingHTTPServer((host, port), handler)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="memory_server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--dir", default=None,
                    help="memory data dir (default debugcache/memory)")
    args = ap.parse_args(argv)
    srv = make_server(args.host, args.port, args.dir)
    print(f"memory_server on http://{args.host}:{srv.server_address[1]}  "
          f"dir={args.dir or memory_store.default_memory_dir()}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `python3 src/tool/debug/agent/unitest/test_memory_server.py`
Expected: PASS (3 tests).

**Step 5: Commit**

```bash
git add src/tool/debug/agent/memory_server.py \
  src/tool/debug/agent/unitest/test_memory_server.py
git commit -m "feat(debug-memory): stdlib http.server memory daemon"
```

---

## Task 9: `memory_client.py` — thin urllib client for aiegdb

**Files:**
- Create: `src/tool/debug/agent/memory_client.py`
- Test: `src/tool/debug/agent/unitest/test_memory_server.py` (add a class that
  reuses the live server from Task 8)

**Step 1: Write the failing test** (append to test_memory_server.py)

```python
class TestClient(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.srv = memory_server.make_server("127.0.0.1", 0, self.tmp)
        self.port = self.srv.server_address[1]
        import threading
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        import memory_client
        self.client = memory_client.MemoryClient(
            f"http://127.0.0.1:{self.port}")

    def tearDown(self):
        self.srv.shutdown()

    def test_save_returns_result(self):
        r = self.client.save("s", [{"ts": "2026-01-01T00:00:01Z",
            "scope": "partition/tile(0,3)/mm2s0", "command": "dma status",
            "output": "MM2S ch0 stuck"}], note="n", meta={})
        self.assertTrue(r["ok"])
        self.assertEqual(r["new_findings"], 1)

    def test_offline_returns_none_not_raise(self):
        import memory_client
        c = memory_client.MemoryClient("http://127.0.0.1:1")  # nothing here
        self.assertIsNone(c.save("s", [], meta={}))
```

**Step 2: Run test to verify it fails**

Run: `python3 src/tool/debug/agent/unitest/test_memory_server.py`
Expected: FAIL — no `memory_client`.

**Step 3: Write minimal implementation**

`src/tool/debug/agent/memory_client.py`:

```python
#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""memory_client - tiny urllib client for the debug memory daemon.

Used by aiegdb's `save` command. All calls fail soft: on any connection/HTTP
error they return None so a missing daemon never breaks the REPL.
"""

import json
import os
import urllib.error
import urllib.request

DEFAULT_URL = os.environ.get("AIE_MEMORY_URL", "http://127.0.0.1:8790")


class MemoryClient:
    def __init__(self, base_url=None, timeout=3):
        self.base = (base_url or DEFAULT_URL).rstrip("/")
        self.timeout = timeout

    def _post(self, path, obj):
        data = json.dumps(obj).encode("utf-8")
        req = urllib.request.Request(
            self.base + path, data=data,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError):
            return None

    def save(self, session, entries, note="", meta=None):
        return self._post("/save", {"session": session, "entries": entries,
                                    "note": note, "meta": meta or {}})

    def health(self):
        try:
            with urllib.request.urlopen(self.base + "/health",
                                        timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError):
            return None
```

**Step 4: Run test to verify it passes**

Run: `python3 src/tool/debug/agent/unitest/test_memory_server.py`
Expected: PASS (all server + client tests).

**Step 5: Commit**

```bash
git add src/tool/debug/agent/memory_client.py \
  src/tool/debug/agent/unitest/test_memory_server.py
git commit -m "feat(debug-memory): fail-soft urllib memory client"
```

---

## Task 10: aiegdb transcript recording

Record each command+output in `AieGdb` so `save` can flush it. The cleanest
hook is `run_line`, which already brackets each command. Capture the printed
output by redirecting stdout during `_dispatch` while still echoing to the real
stdout (tee), so interactive output is unchanged AND the transcript is stored.

**Files:**
- Modify: `src/tool/debug/aiegdb.py` (`AieGdb.__init__`, `run_line`)
- Test: `src/tool/debug/agent/unitest/test_aiegdb_save.py` (new)

**Step 1: Write the failing test**

`src/tool/debug/agent/unitest/test_aiegdb_save.py`:

```python
#!/usr/bin/env python3
import os
import sys
import unittest

_DEBUG_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
if _DEBUG_DIR not in sys.path:
    sys.path.insert(0, _DEBUG_DIR)
import aiegdb  # noqa: E402


class TestTranscript(unittest.TestCase):
    def test_run_line_records_transcript(self):
        gdb = aiegdb.AieGdb(dry_run=True)
        gdb.run_line("where")
        self.assertTrue(gdb._transcript)
        last = gdb._transcript[-1]
        self.assertEqual(last["command"], "where")
        self.assertIn("scope", last)
        self.assertIn("output", last)
        self.assertIn("ts", last)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `python3 src/tool/debug/agent/unitest/test_aiegdb_save.py`
Expected: FAIL — `AieGdb` has no `_transcript`.

**Step 3: Write minimal implementation**

In `aiegdb.py` `__init__`, after `self._reg_trace = []`:

```python
        # per-process debug transcript for the `save` command; each entry is
        # {ts, scope, command, output}. self._save_watermark is the ISO ts of
        # the last successful save (entries with ts <= it are not re-sent).
        self._transcript = []
        self._save_watermark = ""
```

Add a small `_now_iso` at module scope in aiegdb.py (or import from agent):

```python
from datetime import datetime, timezone


def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
```

Rewrite `run_line` to tee-capture output into the transcript:

```python
    def run_line(self, line):
        """Run one command; tee its printed output into the transcript (for
        `save`) and still flush the reg-read block as before."""
        import io
        self._reg_trace = []
        scope = self.prompt().rstrip("> ").strip()
        cmd = line.strip()
        buf = io.StringIO()

        class _Tee:
            def __init__(self, real, cap):
                self.real, self.cap = real, cap

            def write(self, s):
                self.real.write(s)
                self.cap.write(s)

            def flush(self):
                self.real.flush()

        old = sys.stdout
        sys.stdout = _Tee(old, buf)
        try:
            return self._dispatch(line)
        finally:
            block = aiediag.format_reg_read_block(self._reg_trace)
            if block:
                print(block)
            sys.stdout = old
            if cmd and not cmd.startswith("#"):
                self._transcript.append(
                    {"ts": _now_iso(), "scope": scope, "command": cmd,
                     "output": _strip_ansi_local(buf.getvalue())})
```

Add a local ansi stripper near the top of aiegdb.py (schedule_debug_server has
one; keep aiegdb self-contained):

```python
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi_local(s):
    return _ANSI_RE.sub("", s or "")
```

Note: `save` itself is dispatched through `run_line`, so guard against
recording the `save` command's own output as a finding — Task 11 makes the
`save` verb return before producing finding-like output, and its transcript
entry is harmless (classified `raw`). Acceptable for v1.

**Step 4: Run test to verify it passes**

Run: `python3 src/tool/debug/agent/unitest/test_aiegdb_save.py`
Expected: PASS.

Also re-run existing dry-run smoke to ensure no regression:
Run: `python3 src/tool/debug/aiegdb.py --dry-run -c "where; help"`
Expected: prints scope + help, no traceback.

**Step 5: Commit**

```bash
git add src/tool/debug/aiegdb.py \
  src/tool/debug/agent/unitest/test_aiegdb_save.py
git commit -m "feat(debug-memory): record aiegdb command transcript"
```

---

## Task 11: aiegdb `save` verb (client watermark + dedup)

**Files:**
- Modify: `src/tool/debug/aiegdb.py` (`_dispatch` universal verbs, `_help`,
  `_commands`, new `_save` method)
- Test: `src/tool/debug/agent/unitest/test_aiegdb_save.py` (add class)

**Step 1: Write the failing test**

```python
class TestSaveVerb(unittest.TestCase):
    def test_save_filters_by_watermark_and_calls_client(self):
        gdb = aiegdb.AieGdb(dry_run=True)
        sent = {}

        class FakeClient:
            def save(self, session, entries, note="", meta=None):
                sent["entries"] = entries
                sent["session"] = session
                return {"ok": True, "session": session or "auto",
                        "new_findings": len(entries), "dup_findings": 0,
                        "save_id": 1}
        gdb._memory_client = FakeClient()

        gdb.run_line("where")          # entry 1
        gdb.run_line("save mysess")    # flush -> sends entry 1 (+ maybe self)
        first_count = len(sent["entries"])
        self.assertGreaterEqual(first_count, 1)
        self.assertEqual(sent["session"], "mysess")
        self.assertTrue(gdb._save_watermark)

        # second save with no new commands sends nothing new
        sent.clear()
        gdb.run_line("save mysess")
        self.assertEqual(sent.get("entries", []), [])
```

**Step 2: Run test to verify it fails**

Run: `python3 src/tool/debug/agent/unitest/test_aiegdb_save.py`
Expected: FAIL — `save` unknown / no `_memory_client`.

**Step 3: Write minimal implementation**

In `aiegdb.py` top imports, add the agent dir to path + import client lazily:

```python
_AGENT_DIR = os.path.join(_THIS_DIR, "agent")
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)
```

In `__init__`, add:

```python
        self._memory_client = None  # lazy MemoryClient (see _save)
```

In `_dispatch`, add a universal verb (before scope-specific handling, alongside
`where`/`up`):

```python
        if verb == "save":
            self._save(args)
            return True
```

Add the method:

```python
    def _save(self, args):
        """Flush transcript-since-watermark to the memory daemon.

        Usage: save [session_name] [-m "note"]
        On daemon-offline the watermark is NOT advanced (retry-safe).
        """
        session = None
        note = ""
        i = 0
        while i < len(args):
            a = args[i]
            if a in ("-m", "--note"):
                note = " ".join(args[i + 1:])
                break
            if session is None and not a.startswith("-"):
                session = a
            i += 1

        entries = [e for e in self._transcript
                   if e["ts"] > self._save_watermark
                   and e["command"].split()[0].lower() != "save"]
        if self._memory_client is None:
            import memory_client
            self._memory_client = memory_client.MemoryClient()
        meta = {"startcol": self.startcol, "aie_version": self.aie_version,
                "workdir": self.json_dir,
                "design": self._design_name()}
        res = self._memory_client.save(session, entries, note=note, meta=meta)
        if res is None:
            print(aiediag.yellow(
                "  [memory: daemon offline; save skipped, watermark kept]"))
            return
        # advance watermark only on success
        if entries:
            self._save_watermark = entries[-1]["ts"]
        print(aiediag.green(
            f"  [memory] session={res.get('session')} "
            f"new={res.get('new_findings')} dup={res.get('dup_findings')} "
            f"save_id={res.get('save_id')}"))

    def _design_name(self):
        """Best-effort design name from provenance JSON (falls back to
        workdir basename)."""
        try:
            dfsche, _ = self._load_jsons()
            if isinstance(dfsche, dict):
                nm = dfsche.get("design") or dfsche.get("name")
                if nm:
                    return nm
        except Exception:
            pass
        if self.json_dir:
            return os.path.basename(os.path.normpath(self.json_dir))
        return None
```

Add `save` to `_help` (universal block) and `_commands` (universal line):

```python
        print("  save [session] [-m note]  persist transcript to debug memory")
```

**Step 4: Run test to verify it passes**

Run: `python3 src/tool/debug/agent/unitest/test_aiegdb_save.py`
Expected: PASS.

Regression: `python3 src/tool/debug/aiegdb.py --dry-run -c "where; save t"`
Expected: prints `[memory: daemon offline; save skipped ...]` (no daemon),
no traceback.

**Step 5: Commit**

```bash
git add src/tool/debug/aiegdb.py \
  src/tool/debug/agent/unitest/test_aiegdb_save.py
git commit -m "feat(debug-memory): aiegdb save verb with watermark dedup"
```

---

## Task 12: HTML console `save` affordance

Add a "Save to memory" button next to the aiegdb console "Reload" button that
sends the `save` command through the existing `/aiegdb` path (so it uses the
same persistent subprocess, inheriting transcript + watermark). Optionally a
free-text session/note box.

**Files:**
- Modify: `src/tool/debug/schedule_view.py` (the aiegdb console HTML + the
  `conreload` JS block near lines 1776–1780)

**Step 1: Locate the console controls** — find the `conreload` button markup
(search `conreload`) and its click handler (`schedule_view.py:1776`).

**Step 2: Add the button markup** next to the Reload button (same container):

```html
<button id="consave" title="Save this debug session to long-term memory">
  Save to memory</button>
```

**Step 3: Add the click handler** after the `conreload.onclick` block:

```javascript
document.getElementById('consave').onclick = () => {
  const sess = (prompt('Session name (blank = auto):') || '').trim();
  const note = (prompt('Note (optional):') || '').trim();
  let cmd = 'save';
  if (sess) cmd += ' ' + sess;
  if (note) cmd += ' -m ' + note;
  conSend(cmd);   // reuses the persistent aiegdb subprocess + transcript
};
```

**Step 4: Manual verification** — regenerate the HTML view (whatever step
produces `host_schedule.html`) and confirm the button renders and that clicking
it appends `save ...` to the console. If no daemon, the console shows the
offline message from Task 11. (No unit test — this is browser glue; behavior is
covered by the aiegdb `_save` tests.)

Run: `python3 -c "import ast; ast.parse(open('src/tool/debug/schedule_view.py').read())"`
Expected: no `SyntaxError` (the file is valid Python after the string edits).

**Step 5: Commit**

```bash
git add src/tool/debug/schedule_view.py
git commit -m "feat(debug-memory): HTML console Save-to-memory button"
```

---

## Task 13: docs, README, skill, gitignore

**Files:**
- Create: `src/tool/debug/agent/README.md`
- Modify: `src/tool/debug/README.md` (add a "Debug Memory" subsection)
- Create: `.cursor/skills/aiedebugmemory/SKILL.md`
- Modify: `CLAUDE.md` (add the new tool + skill to the doc list, per Document
  rule)
- Modify: `.gitignore` (ignore `debugcache/memory/` runtime data)

**Step 1: Write `agent/README.md`** — how to start the daemon
(`python3 src/tool/debug/agent/memory_server.py`), the `save` command usage in
aiegdb CLI + HTML, the file layout under `debugcache/memory/`, and the HTTP API
table (copy from the design doc).

**Step 2: Add `.gitignore` entry:**

```
debugcache/memory/
```

**Step 3: Write the skill** `.cursor/skills/aiedebugmemory/SKILL.md` describing
when to use debug memory (persisting a live investigation, reloading prior
findings via `/context`), the `save` watermark semantics, and the daemon
lifecycle. Follow the format of existing skills under `.cursor/skills/`.

**Step 4: Update `CLAUDE.md`** — add bullets for `memory_server.py`,
`memory_store.py`, `memory_client.py`, and the new skill under "Additional
Documentation", matching the existing bullet style.

**Step 5: Commit**

```bash
git add src/tool/debug/agent/README.md src/tool/debug/README.md \
  .cursor/skills/aiedebugmemory/SKILL.md CLAUDE.md .gitignore
git commit -m "docs(debug-memory): README, skill, CLAUDE.md, gitignore"
```

---

## Task 14: Full test sweep + optional daemon integration smoke

**Step 1: Run all agent unit tests**

Run:
```bash
python3 src/tool/debug/agent/unitest/test_memory_store.py
python3 src/tool/debug/agent/unitest/test_memory_server.py
python3 src/tool/debug/agent/unitest/test_aiegdb_save.py
```
Expected: all PASS, 0 failures.

**Step 2: End-to-end smoke (real daemon + real aiegdb, dry-run reads)**

```bash
# terminal A
python3 src/tool/debug/agent/memory_server.py --dir /tmp/claude/aiemem &
# terminal B
AIE_MEMORY_URL=http://127.0.0.1:8790 \
  python3 src/tool/debug/aiegdb.py --dry-run \
  -c "where; save smoke -m hello; where; save smoke"
# then inspect
cat /tmp/claude/aiemem/state.json
cat /tmp/claude/aiemem/sessions/smoke.md
```
Expected: first `save` reports `new>=1`; second `save` reports `new=0`
(watermark dedup); `state.json` has session `smoke` with `save_count=2`;
`smoke.md` contains the note `hello`.

**Step 3: Commit any fixes discovered during the sweep** (if none, skip).

---

## Notes for the Implementer

- **DRY:** `_now_iso`, `finding_hash`, `classify` live only in `memory_store`;
  aiegdb has its own tiny `_now_iso`/ansi-strip to stay import-light — that
  minor duplication is intentional (aiegdb must not hard-depend on the daemon
  package at import time; the client is imported lazily inside `_save`).
- **YAGNI:** No auth on the memory daemon (local-only, like the grid endpoints
  are gated but memory is read/write local). No LLM summarization. No SQLite.
- **Keep functions < 200 lines** (CLAUDE.md rule) — `consolidate` is the
  longest; if it grows, split `_render_md` further.
- **Lock discipline:** never call `ensure_session` while holding
  `self._lock` (it acquires the lock itself). See the note in Task 5.
- **Process transparency (CLAUDE.md):** after each task, list changed/created
  files in the commit body or task summary.
