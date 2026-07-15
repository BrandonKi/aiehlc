# In-Pass Tiling Reconstruction for `fullconnect_auto = 0`

## Problem

When a kernel opts into `aie::GlobalPolicy { .fullconnect_auto = 0 }` (e.g.
`conv2d_spatial` at `example/tileprogram/ccode/simpleconv2d.cc:327`), the front-end
(`src/llvm/aiehlc.cc`) stops emitting the IR module attributes that describe the
M×N cartesian repeat / spatial sub-tiling — `routing.tile_m`, `routing.tile_rows`,
`routing.m_rounds`, `routing.tile_n`, `routing.tile_cols`, `routing.n_rounds`.

For a GEMM with `fullconnect_auto = 0` this is fine: the shim BD fires once and
K-accumulation (`routing.k_rounds`) still drives the only iteration, so every
downstream consumer falls through to the correct pure-K-round path.

But `conv2d_spatial` **needs** the M-round tiling values to survive:
`classifyTiling` (`passblueprinttoschedule.cpp:529`) computes
`mRounds = tile_rows / tile_m` and `buildOutputTileDescriptor`
(`passblueprinttoschedule.cpp:604`) uses `tileM`/`tileRows` for the host `scf.for`
round loop and the output DDR strides. With the attrs dropped, `classifyTiling`
returns `mMode = Match, mRounds = 1` and `passState.tileM/tileRows = 0`
(`0/0` division in `buildOutputTileDescriptor`) — the per-tile-row round loop
disappears and conv2d breaks.

Note: for the concrete `conv2d_spatial` example `kRounds == 1` (full-K), so the
K-round attr block is skipped anyway and `effective_k/full_k` are never emitted; the
values that actually matter for conv2d are `tile_m/tile_rows/m_rounds`, previously
supplied by the single-kernel *halo M-round fallback* in `aiehlc.cc`.

## Decision

**Attr = override, pass computes fallback.** Keep the front-end gating (drop the
M×N attrs when `fullconnect_auto = 0`) and make the *pass* reconstruct only what
conv2d needs — `tile_m / tile_rows` — from the IR when the attrs are absent. When
the attrs are present (`fullconnect_auto = 1`, GEMM/matmul) the fallback never fires,
so those paths stay byte-identical.

## Why the `#routing.tiling` attr alone is insufficient

The `#routing.tiling` LevelAttr is populated only on the **halo INPUT** tensor's
`partitiontensor` (`routingmanager.cpp:922-950`); the output `win_c`
`partitiontensor` gets an empty `TilingAttr{}`. Its row-dim `outer` level encodes
`slice = haloSlice` (INPUT rows), `rounds = splitnum` (the *mesh* split, not the
on-core M-round count) and a nested `slice_tiling.rounds = l2Rounds`. Crucially the
**width split `w_rounds` is not in the tiling attr at all** — it lives only in the
`tensor_N.halo` dict. Since `classifyTiling` needs the **output** `tile_m/tile_rows`
and `m_rounds = spatialMRounds = w_rounds × l2_rounds`, the tiling attr cannot
supply them.

## Reconstruction source (hybrid)

Guaranteed-present sources at the dmaphop→blueprint stage:

- `tensor_N.halo` dict (`tilinglinalg_pipeline.cpp:322`) — emitted
  **unconditionally** for halo tensors; carries `w_rounds`, `l2_rounds`, `slice`,
  `ow_t`, `row_pitch`. →  `m_rounds = max(1, w_rounds) × max(1, l2_rounds)`.
- output `partitiontensor` / `tensor.extract_slice` result shape → `tile_rows`
  (per-tile output row count). → `tile_m = tile_rows / m_rounds`.

## Architecture — pass-entry backfill

`passblueprinttoschedule` reads `routing.tile_m/tile_rows` from ~10 sites, all
funnelling through `passState` (`passblueprinttoschedule.cpp:3002-3003`) plus direct
`getAttrOfType` reads. Rather than edit every read site, the pass **backfills** the
values once and re-`setAttr`s them on the module, so all existing (already
null-guarded) consumers work unchanged.

Backfill runs at the **start of `passdmaphoptodfscheblueprint`** (shared pipeline
stage 4). This is before the module is cloned into the host and kernel paths, so both
clones inherit the values, and the K-tiling block later in the *same* pass
(`passdmaphoptodfscheblueprint.cpp:988-1066`) also sees them.

### Steps (only when `routing.tile_m` is absent — fallback guard)

1. Scan module attrs for a `tensor_N.halo` dict; compute
   `m_rounds = max(1, w_rounds) × max(1, l2_rounds)`.
2. If `m_rounds > 1`: locate the **output** `partitiontensor` (the one whose flow is
   the output/merge direction) and read its per-tile slice row count → `tile_rows`;
   `tile_m = tile_rows / m_rounds`.
3. `moduleOp->setAttr("routing.tile_m", …)` and `("routing.tile_rows", …)` with a
   `[TilingLinalg] Backfilled tile_m/tile_rows …` log. `m_rounds` is implied
   downstream as `tile_rows / tile_m` (classifyTiling).

### 988-1066 K-tiling block ("for completeness")

Add a local fallback so `effective_k/full_k/tile_m` are taken from the same
reconstruction when the module attrs are absent. This is a no-op for `conv2d_spatial`
(`kRounds == 1`, block skipped) but satisfies the named example.

## Files changed

- `src/llvm/aiehlc.cc` — gate the M×N tiling attribute emission on
  `fullconnect_auto` (multi-kernel K-round block, single-kernel K-round block with a
  hoisted `singleFullConnectAuto` flag, single-kernel halo M-round fallback guard).
  *`effective_k/k_rounds/full_k` and `spatial_halo_buf_size` remain ungated.*
- `src/mlir/mlirfront/tilinglinalg/pass/passdmaphoptodfscheblueprint/passdmaphoptodfscheblueprint.cpp`
  — pass-entry backfill of `routing.tile_m/tile_rows` from the halo dict + output
  partition/extract shape (fallback, only when absent); local reconstruction fallback
  in the 988-1066 K-tiling block.
- No edits to `passblueprinttoschedule` read sites (all null-guarded; fed by the
  backfilled attrs).

## Verification

1. Build: `cd build && make -j$(nproc)` → RC 0.
2. `simpleconv2d.cc` (`conv_policy = {.fullconnect_auto = 0}`): regenerate; confirm
   host `scf.for` round loop present, `routing.tile_m/tile_rows` backfilled,
   `routing.fullconnect_auto = 0`, `routing.spatial_halo_buf_size` intact, RC 0, host
   builds.
3. `simplematmul2.cc` (`fullconnect_auto = 1`): regenerate; byte-compare
   `6_BlueprintToSchedulePass.mlir` host block to the pre-change baseline (must be
   identical — fallback never fires).
4. HW run (if board): `python3 ./script/test/apppaltest.py -y -nonreboot` →
   `device_teardown done` for simpleconv2d.

## Residual risk

Reliably locating the *output* `partitiontensor` in the pass to read `tile_rows`.
Identify it via the output/merge flow direction; verify against generated conv2d IR
during implementation.
