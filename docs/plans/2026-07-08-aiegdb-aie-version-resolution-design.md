# Design: Correct `aie_version` for aiegdb launched from the HTML schedule-debug server

Date: 2026-07-08

## Problem

When the live schedule-debug server (`schedule_debug_server.py`) is launched from
`aiehlc.sh --prettydebug`, the aiegdb console (HTML tab) and the MCP server (LLM
tab) read AIE DMA/register status at the **wrong offsets** on AIE2PS hardware,
returning `0x00000000` (false "Idle") instead of the real value.

### Root cause (verified)

Two different version-numbering schemes collide at the launch boundary:

| Layer | `5` means | `2` / `1` means | `2ps` means |
|-------|-----------|-----------------|-------------|
| **Compiler** (`aiehlc.sh --aie-version`) | AIE2PS (Cortex-A78) | AIEML (Cortex-A72) | n/a |
| **Debug tools** (`aiediag`/`aiegdb`/daemon offset maps) | AIEML (`shim_5` = 0x1D2xx) | n/a | AIE2PS (`shim_2ps` = 0x9320/0x9328) |

`script/aiehlc.sh:376` passes the **compiler** number straight into the daemon:

```bash
python3 .../schedule_debug_server.py "${WORKLOCAL_DIR}" \
    --elf ... --aie-version "${aie_version}" --open
```

So `aiehlc.sh --aie-version 5` (AIE2PS) becomes `schedule_debug_server.py
--aie-version 5`, which the debug layer interprets as **AIEML**. That explicit
flag then *overrides* the correct JSON auto-resolution and selects the wrong
offset map. `compute_reg_offset("shim","mm2s",1,"5") = 0x1D22C`, which is
unmapped on AIE2PS and reads `0`; the correct offset is `0x932C` = `0x02080012`.

### What is already correct

- Provenance JSONs carry the authoritative gen: both
  `dfscheduleprovenancemap.json` and `dmaphopprovenacemap.json` contain
  `aie_gen: "Gen5"` (verified).
- `aiediag.aie_version_from_jsons(...)` → `"2ps"` for this board (verified).
- The mapping already exists in ONE place: `aiediag.debug_aie_version_from_gen`
  (`"5"`/`"Gen5"` → `"2ps"`; `"1"`/`"2"` → `"5"`).
- Both spawn paths forward `self.aie_version` faithfully:
  - `_gdb_spawn` (schedule_debug_server.py:315): `--aie-version str(self.aie_version)`
  - `_write_mcp_config` (schedule_debug_server.py:424): `AIEMCP_AIE_VERSION=str(self.aie_version)`

The bug is solely that `self.aie_version` is set to the wrong string because the
explicit compiler-numbered flag beats the correct JSON value.

## Design

Make the provenance JSON `aie_gen` the **source of truth**, and normalize any
explicit `--aie-version` flag through the existing `debug_aie_version_from_gen`
mapper so compiler numbers (`5`) become debug strings (`2ps`).

### Resolution order (both `schedule_debug_server.py` and `aiegdb.py`)

1. If provenance JSON has `aie_gen` → `debug_aie_version_from_gen(aie_gen)`
   (authoritative; verified correct for this board).
2. Else if an explicit `--aie-version` flag was given →
   `debug_aie_version_from_gen(flag)` (so `5`→`2ps`, `2`→`5`, `2ps`→`2ps`).
3. Else warn loudly and fall back to a default.

This makes the HTML→aiegdb path correct **regardless of what number
`aiehlc.sh` passes**, because the JSON wins.

### `debug_aie_version_from_gen` extension

Currently accepts `"5"`/`"Gen5"`/`"1"`/`"2"`. Add idempotent passthrough for the
already-normalized debug strings so it is safe to call on any input:
- `"2ps"` → `"2ps"`
- `"5"` (bare) → `"2ps"` (compiler semantics; this is the ONLY behavior change to
  bare `"5"`, and it is required so the aiehlc.sh flag is correct)

Note: this makes bare `"5"` mean AIE2PS everywhere (compiler semantics). The old
debug meaning of bare `"5"` = AIEML is dropped in favor of a single consistent
scheme. AIEML is selected by `"1"`/`"2"`.

### CLI arg loosening

- `aiegdb.py:772` currently restricts `--aie-version` to `choices=["5","2ps"]`.
  Remove `choices` (or widen to `["1","2","5","2ps"]`) so compiler numbers pass
  through the normalizer instead of being rejected by argparse.
- `schedule_debug_server.py:1208` has no `choices` (already permissive).

## Components touched

| File | Change |
|------|--------|
| `src/tool/debug/aiediag.py` | Extend `debug_aie_version_from_gen` for idempotent `"2ps"` passthrough; ensure bare-number handling is centralized |
| `src/tool/debug/schedule_debug_server.py` | Resolution block (~1272-1275): JSON-first, normalize explicit flag through mapper; warn on unresolved |
| `src/tool/debug/aiegdb.py` | Mirror JSON-first + normalize logic (~803-807); loosen `--aie-version` choices (772) |
| `_gdb_spawn` / `_write_mcp_config` | No change (already forward `self.aie_version`) |
| `script/aiehlc.sh` | No change required (daemon now normalizes) |

## Error handling

- Unresolved version (no JSON `aie_gen`, no flag): print a clear warning to
  stderr naming the workdir and the fallback used, instead of silently
  defaulting to a number that reads `0` everywhere.

## Verification plan

1. **Unit** (`debug_aie_version_from_gen`):
   - `"5"`→`"2ps"`, `"Gen5"`→`"2ps"`, `"2ps"`→`"2ps"`, `"2"`→`"5"`, `"1"`→`"5"`.
2. **Integration** (daemon resolution against real workdir):
   - `aie_version_from_jsons(load_jsons("aout/worklocal"))` → `"2ps"`.
   - Simulate explicit flag `5` → normalized → `"2ps"`.
3. **End-to-end** (on board, HW at 10.23.224.213):
   - `aiegdb --aie-version 5 --json-dir aout/worklocal` reading shim(3,0)
     mm2s1 (`target tile 0 0; target channel mm2s1; dma status`) → must return
     `raw=0x02080012` (not `0x00000000`).
   - Confirm the daemon's `_gdb_spawn` and `_write_mcp_config` both carry the
     normalized `2ps`.
