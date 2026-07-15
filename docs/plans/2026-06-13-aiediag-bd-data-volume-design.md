# aiediag `[2] BD Chain` — intended-vs-real data volume & HW BD length

## Goal
Enrich section `[2] BD Chain (from JSON)` of `script/aiediag.py` to report, for the
**queried tile / channel**:

- the **data intended to send/receive** (the `contract` string + a computed total volume),
- the **BDs intended to be used** (bd_ids from the JSON chain),
- the **ping-pong buffer logic** (existing cycle check, retained),
- the **intended size each BD handles** (per-BD `len` from JSON), and
- the **real BD length read from hardware** (per-BD `Buffer_Length` register), compared
  against the intended size.

## Scope (confirmed with user)
- Queried tile only (section `[2]`); connected tiles (`[4]/[5]`) unchanged.
- Only the BDs that appear in the channel's JSON `bd_chain` are read from HW.
- Enrich the existing `[2]` section in place (no new section).

## Register facts (AIE2PS, verified from aie2pstile.json / aie2psshim.json)
- Core memory module: `DMA_BD0_0` base `0x1D000`, BD stride `0x20`.
  `Buffer_Length` = word0 bits[13:0], unit = 32-bit words. bytes = words * 4.
- Shim NoC module: `DMA_BD0_0` base `0x9000`, BD stride `0x30`.
  `Buffer_Length` = word0 bits[31:0], unit = 32-bit words. bytes = words * 4.
- Word 0 of each BD holds Buffer_Length for both tile types.

## Intended data (from `dfscheduleprovenancemap.json`)
- `channel.contract` — human-readable intent string.
- `channel.bd_chain[].{bd_id,len,next_bd}` — per-BD intended bytes + chaining.
- `channel.start_io[].repeat_count` — how many times the chain runs.
- Total intended volume = `sum(bd.len) * max(repeat_count, 1)`.

## Implementation (`script/aiediag.py`, single file)

### Constants (near existing DMA_STATUS_OFFSETS)
```python
BD_BASE_STRIDE = {            # tile_type -> (bd0_base, bd_stride)
    "core":     (0x1D000, 0x20),
    "shim_5":   (0x9000,  0x30),
    "shim_2ps": (0x9000,  0x30),
}
BD_LEN_MASK = {"core": 0x3FFF, "shim": 0xFFFFFFFF}
```

### Helpers
- `bd_length_offset(tile_type, bd_id)` -> `base + bd_id * stride` (word 0).
- `read_bd_hw_lengths(phys_col, row, tile_type, bd_ids, target, device, dry_run)`
  -> `{bd_id: bytes}`; reads each BD word0 via existing `run_aiedbg_reg_read`,
  masks the length field, multiplies words*4. Returns `None` in dry-run
  (commands still printed by `run_aiedbg_reg_read`).

### Formatter
- Extend `format_bd_chain(channel_entry, hw_lengths=None)`:
  - keep existing per-BD line + ping-pong + contract + start_io,
  - add a "Total intended" line (sum(len) * repeat),
  - add a per-BD intended-vs-real block when `hw_lengths` is provided:
    `BD<id>: intended=<n>B  hw=<m>B (<w> words)  OK|MISMATCH|hw not configured`.

### Wiring in `main()` section `[2]`
After resolving `ch_entry`, collect `bd_ids` from `bd_chain`, call
`read_bd_hw_lengths(...)` (runs in dry-run to emit commands), pass result to
`format_bd_chain`.

## Verification
1. Dry-run core: `aiediag.py dig 4 3 -s2mm0 --dry-run` emits BD word0 reads at
   `0x1D000 + bd_id*0x20` for each chain BD.
2. Dry-run shim: `aiediag.py dig 0 0 -mm2s0 --dry-run` emits reads at
   `0x9000 + bd_id*0x30`.
3. No-JSON tile: existing "not found" path still works, no crash.
4. HW run: healthy channel shows `OK`; misconfigured/stuck BD shows `MISMATCH`
   or `hw not configured` (hw=0).

## Out of scope
- Connected tiles (`[4]/[5]`).
- All-16-BD dump (only JSON-chain BDs).
- New section (enrich `[2]` in place).
