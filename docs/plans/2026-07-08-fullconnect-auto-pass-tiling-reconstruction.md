# In-Pass Tiling Reconstruction for `fullconnect_auto = 0` — Implementation Plan

> **SUPERSEDED (2026-07-08).** This plan is abandoned and was never implemented.
> Decision reversed: for `fullconnect_auto = 0` we do **NOT** reconstruct
> `routing.tile_m` / `routing.tile_rows`. Instead the M/N round auto-generation
> (`m_rounds = tile_rows / tile_m`, `n_rounds = tile_cols / tile_n`) is **explicitly
> gated off** for fca=0 (`isFullConnectAuto` guard in `classifyTiling` and the
> `buildOutputTileDescriptor` call site, `passblueprinttoschedule.cpp`). conv2d
> (fca=0, halo) derives its output tiling **purely from halo attrs** via the 2D
> `shimDimStrides` fallback — never from `tile_m`/`tile_rows`. See plan
> `~/.claude/plans/sharded-roaming-hennessy.md`. The rest of this document is kept
> for historical context only.

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** When a kernel sets `aie::GlobalPolicy { .fullconnect_auto = 0 }`, the
front-end drops the M×N tiling module attrs; make `passdmaphoptodfscheblueprint`
reconstruct `routing.tile_m` / `routing.tile_rows` from the IR (halo dict + output
`pull`-flow extract shape) so `conv2d_spatial` keeps its host round loop, while the
`fullconnect_auto = 1` path stays byte-identical (fallback never fires).

**Architecture:** Attr = override, pass computes fallback. Front-end edits (already
applied to `aiehlc.cc`) gate the M×N attr emission on `fullconnect_auto`. A new
pass-entry backfill in `passdmaphoptodfscheblueprint::runOnOperation` — running
*before* `applyPartialConversion` (which erases `pull`/`partitiontensor`) — sets
`routing.tile_m` / `routing.tile_rows` on the module when absent, so all ~10
already-null-guarded consumers in `passblueprinttoschedule` work unchanged. A local
fallback in the K-tiling block (988-1066) uses the same reconstruction.

**Tech Stack:** C++/MLIR (LLVM), aiehlc TilingLinalg pipeline, EmitC → host.cc /
kernel.cc, XAie runtime on AIE2PS board.

---

## Background / already-done context

- **Design doc:** `doc/design/fullconnect_auto_pass_tiling_reconstruction.md`
  (committed, 621e748).
- **aiehlc.cc front-end edits are ALREADY APPLIED** (uncommitted working tree, 71+/37-):
  gated M×N attr emission (multi-kernel K-round block, single-kernel K-round block
  with hoisted `singleFullConnectAuto`, single-kernel halo M-round fallback guard).
  These are done — this plan does NOT re-do them; Task 0 only verifies + commits them.
- `simplematmul2.cc` is at `fullconnect_auto = 1` (must NOT be modified).
- `simpleconv2d.cc` is at `fullconnect_auto = 0`, kernel `conv2d_spatial`
  (`simpleconv2d.cc:327`). This is the target that must keep working.

## Reconstruction formula (verified against IR + front-end aiehlc.cc:2201-2330)

- `m_rounds = max(1, w_rounds) * max(1, l2_rounds)` — both read from the
  `tensor_N.halo` DictionaryAttr (emitted unconditionally for halo tensors at
  `tilinglinalg_pipeline.cpp:322`; `w_rounds`/`l2_rounds` present only when >1, so
  default to 1 when absent).
- `tile_rows` = per-tile OUTPUT row count = shape of the OUTPUT `pull` flow's
  extract tensor along the split dim. In the IR, `dmaphop.pull %V from ...` where
  `%V = routing.routingextract_data %P` and `%P = routing.partitiontensor`. So:
  `pull.getData()` → RankedTensorType → `shape[splitDim]`; `splitDim` from the
  source `partitiontensor.getPartition().getSplitdim()`.
- `tile_m = tile_rows / m_rounds`.

Verified numeric example (GEMM `3_DmapToDmaphopPass.mlir`): output
`routingextract_data ... -> tensor<64x256xi8>`, splitDim=0 → tile_rows=64 =
`macroDimM/effectiveMeshRows = 256/4`. Matches front-end `aiehlc.cc:2013`.

## Insertion point (verified)

`passdmaphoptodfscheblueprint.cpp:1567` `runOnOperation()`. Insert the backfill
**after `auto module = getOperation();` (1572) and before `applyPartialConversion`
(1607)** — the conversion erases `dmaphop::pull`, `routing::partitiontensor`,
`routing::extract_data` (`target.addIllegalOp` at 1601-1605), so reconstruction MUST
read them first.

## Consumers (no edits — all null-guarded, fed by backfill)

- `passblueprinttoschedule.cpp` `classifyTiling` (518, reads tile_m 529 / tile_rows
  530, `mRounds = rows/m` 540), `buildOutputTileDescriptor` (604), `passState->tileM
  /tileRows` (3002-3003), + direct reads. All return safe defaults when attr absent;
  after backfill they see real values.

---

## Task 0: Verify + commit the front-end aiehlc.cc gating (already applied)

**Files:**
- Verify: `src/llvm/aiehlc.cc` (uncommitted edits present)
- Verify: `example/tileprogram/ccode/simplematmul2.cc:41` (must be `.fullconnect_auto = 1`)
- Verify: `example/tileprogram/ccode/simpleconv2d.cc:327` (must be `.fullconnect_auto = 0`)

**Step 1: Inspect the applied diff**

Run: `git diff src/llvm/aiehlc.cc`
Expected: 3 edits — (a) multi-kernel K-round block splits K-accum (always) vs M×N
(only `mkd.fullConnectAuto`); (b) hoisted `bool singleFullConnectAuto` before
single-kernel K-round block + split emission; (c) `singleFullConnectAuto &&` added to
the single-kernel halo M-round fallback guard. `spatial_halo_buf_size` +
`fullconnect_auto` emission unchanged.

**Step 2: Confirm the two example files are correct**

Run: `git diff --stat example/tileprogram/ccode/simplematmul2.cc example/tileprogram/ccode/simpleconv2d.cc`
Expected: empty (no changes to either — simplematmul2 stays fca=1, simpleconv2d
already fca=0 on main).

**Step 3: Build to confirm the applied edits compile**

Run: `cd build && make -j$(nproc) aiehlc 2>&1 | tail -5`
Expected: `[100%] Built target aiehlc`, RC 0.

**Step 4: Commit the front-end gating**

```bash
git add src/llvm/aiehlc.cc
git commit -m "$(cat <<'EOF'
feat(aiehlc): gate M×N tiling attr emission on fullconnect_auto

Drop routing.tile_m/tile_rows/tile_n/tile_cols/m_rounds/n_rounds when a
kernel sets fullconnect_auto=0; keep effective_k/k_rounds/full_k and
spatial_halo_buf_size. The pass backfills the conv2d-needed values from IR.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 1: Add the reconstruction helper (pure, testable)

**Files:**
- Modify: `src/mlir/mlirfront/tilinglinalg/pass/passdmaphoptodfscheblueprint/passdmaphoptodfscheblueprint.cpp`
  (add a file-local static helper near the other file-local helpers, e.g. after
  `resolveFuncArgIndexForRootView` at line 66-~120)

**Step 1: Write the helper**

Add a static helper that scans the module and returns reconstructed
`{tileRows, mRounds, tileM, valid}`. Read halo dict for `m_rounds`; walk `dmaphop::pull`
ops for `tile_rows`.

```cpp
// Reconstruct conv2d's M-round tiling (tile_m/tile_rows) from IR when the
// front-end dropped the module attrs (fullconnect_auto = 0). Returns valid=false
// when no halo dict is present or m_rounds<=1 (nothing to reconstruct).
struct ReconTiling {
    int64_t tileRows = 0;
    int64_t mRounds = 1;
    int64_t tileM = 0;
    bool valid = false;
};

static ReconTiling reconstructMTilingFromIR(ModuleOp module) {
    ReconTiling r;

    // 1. m_rounds from any tensor_N.halo dict: max(1,w_rounds)*max(1,l2_rounds).
    int64_t wRounds = 1, l2Rounds = 1;
    bool foundHalo = false;
    for (NamedAttribute na : module->getAttrs()) {
        StringRef name = na.getName().strref();
        if (!name.endswith(".halo")) continue;
        auto dict = dyn_cast<DictionaryAttr>(na.getValue());
        if (!dict) continue;
        foundHalo = true;
        if (auto a = dict.getAs<IntegerAttr>("w_rounds"))
            wRounds = std::max<int64_t>(1, a.getInt());
        if (auto a = dict.getAs<IntegerAttr>("l2_rounds"))
            l2Rounds = std::max<int64_t>(1, a.getInt());
        break; // one halo (the IFM) is enough for the M-round product
    }
    if (!foundHalo) return r; // no conv halo → nothing to reconstruct
    r.mRounds = std::max<int64_t>(1, wRounds * l2Rounds);
    if (r.mRounds <= 1) return r; // no on-core M-round loop needed

    // 2. tile_rows from the OUTPUT pull-flow extract shape (per-tile row count).
    //    pull.getData() == routingextract_data result; splitDim from its source
    //    partitiontensor.
    module->walk([&](dmaphop::pull pullOp) {
        if (r.tileRows > 0) return WalkResult::interrupt();
        Value data = pullOp.getData();
        auto dataTy = dyn_cast<RankedTensorType>(data.getType());
        if (!dataTy || dataTy.getRank() != 2) return WalkResult::advance();
        int64_t splitDim = 0;
        if (auto ext = dyn_cast_or_null<routing::extract_data>(data.getDefiningOp())) {
            if (auto part = dyn_cast_or_null<routing::partitiontensor>(
                    ext.getTensor().getDefiningOp()))
                splitDim = part.getPartition().getSplitdim();
        }
        r.tileRows = dataTy.getShape()[splitDim];
        return WalkResult::interrupt();
    });
    if (r.tileRows <= 0) return r;

    // 3. tile_m = tile_rows / m_rounds (must divide evenly).
    if (r.tileRows % r.mRounds != 0) {
        llvm::errs() << "[TilingLinalg] Backfill WARN: tile_rows=" << r.tileRows
                     << " not divisible by m_rounds=" << r.mRounds
                     << "; skipping reconstruction\n";
        return r;
    }
    r.tileM = r.tileRows / r.mRounds;
    r.valid = (r.tileM > 0);
    return r;
}
```

Note: confirm the accessor names (`pullOp.getData()`, `extract_data.getTensor()`,
`partitiontensor.getPartition().getSplitdim()`) against the current op defs before
building; they were verified against IR + pass usage (532, 557-558, 1229) but the
exact getter name for `extract_data`'s input may be `.getTensor()` (matches line 557's
`op.getTensor()` for the same op class). Adjust if the build errors.

**Step 2: Build (compile-check the helper)**

Run: `cd build && make -j$(nproc) aiehlc 2>&1 | tail -8`
Expected: RC 0, `[100%] Built target aiehlc`. If accessor names are wrong, fix per
compiler error and rebuild.

**Step 3: Commit**

```bash
git add src/mlir/mlirfront/tilinglinalg/pass/passdmaphoptodfscheblueprint/passdmaphoptodfscheblueprint.cpp
git commit -m "$(cat <<'EOF'
feat(passdmaphop): add reconstructMTilingFromIR helper

Reconstructs conv2d tile_m/tile_rows from the halo dict (m_rounds) and the
output pull-flow extract shape (tile_rows) for the fullconnect_auto=0 path.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Wire the backfill into `runOnOperation`

**Files:**
- Modify: `passdmaphoptodfscheblueprint.cpp:1572` (insert after `auto module =
  getOperation();`, before `applyPartialConversion` at 1607)

**Step 1: Insert the backfill (guarded on attr absence)**

```cpp
    auto module = getOperation();
    OpBuilder builder(module->getContext());

    // Backfill conv2d M-round tiling (fullconnect_auto=0 drops these attrs).
    // Only when routing.tile_m is absent (fallback guard) — when present
    // (fullconnect_auto=1) this is a no-op and the fca=1 path stays identical.
    if (!module->getAttrOfType<IntegerAttr>("routing.tile_m")) {
        ReconTiling rt = reconstructMTilingFromIR(module);
        if (rt.valid) {
            Builder b(module->getContext());
            module->setAttr("routing.tile_m",
                            b.getI64IntegerAttr(rt.tileM));
            module->setAttr("routing.tile_rows",
                            b.getI64IntegerAttr(rt.tileRows));
            llvm::errs() << "[TilingLinalg] Backfilled tile_m=" << rt.tileM
                         << " tile_rows=" << rt.tileRows
                         << " (m_rounds=" << rt.mRounds << ") from IR\n";
        }
    }
```

**Step 2: Build**

Run: `cd build && make -j$(nproc) aiehlc 2>&1 | tail -5`
Expected: RC 0.

**Step 3: Verify no-op on GEMM (attr present OR no halo → no backfill)**

Run: `source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simplematmul2.cc 2>&1 | grep -E "Backfilled tile_m" || echo "NO BACKFILL (correct for GEMM)"`
Expected: `NO BACKFILL (correct for GEMM)` — simplematmul2 has no halo dict, so
`reconstructMTilingFromIR` returns `valid=false`.

**Step 4: Commit**

```bash
git add src/mlir/mlirfront/tilinglinalg/pass/passdmaphoptodfscheblueprint/passdmaphoptodfscheblueprint.cpp
git commit -m "$(cat <<'EOF'
feat(passdmaphop): backfill tile_m/tile_rows at pass entry when absent

Runs reconstructMTilingFromIR before applyPartialConversion (which erases
pull/partitiontensor); sets routing.tile_m/tile_rows so downstream null-guarded
consumers work unchanged. No-op when the attr is present (fullconnect_auto=1).

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Local fallback in the 988-1066 K-tiling block ("for completeness")

**Files:**
- Modify: `passdmaphoptodfscheblueprint.cpp:990-997` (K-tiling attr reads)

**Step 1: Add local fallback for `tile_m` in the K-block**

The block already reads `routing.tile_m` at 992 (`tileMAttr`). After the module
backfill (Task 2), `routing.tile_m` is already set, so `tileMAttr` is non-null for
conv2d. This task adds a *defensive* local reconstruction so the block is correct even
if the backfill were skipped. For `conv2d_spatial` `kRounds==1` so this block is
skipped entirely (guard `effectiveK < fullK` at 999 is false) — this is a no-op for
the named example but satisfies the design's "for completeness" requirement.

Change line 997 from:
```cpp
                        int64_t tileM = tileMAttr ? tileMAttr.getInt() : 0;
```
to:
```cpp
                        int64_t tileM = tileMAttr ? tileMAttr.getInt() : 0;
                        if (tileM == 0) {
                            // Local fallback: reconstruct if module attr absent.
                            ReconTiling rt = reconstructMTilingFromIR(moduleOp);
                            if (rt.valid) tileM = rt.tileM;
                        }
```

(Uses `moduleOp` — confirm the in-scope module handle name at this point; the block
reads `moduleOp->getAttrOfType` at 990-992, so `moduleOp` is in scope.)

**Step 2: Build**

Run: `cd build && make -j$(nproc) aiehlc 2>&1 | tail -5`
Expected: RC 0.

**Step 3: Commit**

```bash
git add src/mlir/mlirfront/tilinglinalg/pass/passdmaphoptodfscheblueprint/passdmaphoptodfscheblueprint.cpp
git commit -m "$(cat <<'EOF'
feat(passdmaphop): local tile_m fallback in K-tiling block

Defensive reconstruction if routing.tile_m is absent when the K-tiling
multi-dim addressing block runs. No-op for conv2d_spatial (kRounds==1, block
skipped) and for backfilled modules.

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: End-to-end verification — conv2d (fullconnect_auto=0)

**Files:** (no edits — verification only)

**Step 1: Regenerate simpleconv2d**

Run: `source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simpleconv2d.cc 2>&1 | tee /tmp/claude/conv_gen.log | tail -20`
Expected: RC 0, host + kernel emitted, no `ERROR`.

**Step 2: Confirm the backfill fired**

Run: `grep -E "Backfilled tile_m=.* tile_rows=.* \(m_rounds=" /tmp/claude/conv_gen.log`
Expected: one line, e.g. `[TilingLinalg] Backfilled tile_m=196 tile_rows=3136
(m_rounds=16) from IR` (exact numbers per conv geometry).

**Step 3: Confirm backfilled attrs land on the module**

Run: `grep -E "routing.tile_m|routing.tile_rows|routing.fullconnect_auto|routing.spatial_halo_buf_size" ir/dfschedule/6_BlueprintToSchedulePass.mlir | head -1`
Expected: module attrs include `routing.tile_m`, `routing.tile_rows`,
`routing.fullconnect_auto = 0`, `routing.spatial_halo_buf_size = <N>`.

**Step 4: Confirm the host round loop survived (the actual conv2d fix)**

Run: `grep -E "scf.for" worklocal/host.cc 2>/dev/null | head -5 || grep -c "scf.for" ir/dfschedule/6_BlueprintToSchedulePass.mlir`
Expected: host `scf.for` round loop present (non-zero). Absence = regression
(0/0 division in `buildOutputTileDescriptor` — the exact bug this fixes).

**Step 5: Confirm host compiles**

The `aiehlc.sh` run compiles host+kernel; confirm no compile error in the log.
Run: `grep -iE "error:" /tmp/claude/conv_gen.log | head` → expected: empty.

---

## Task 5: Regression — GEMM baseline byte-identical (fullconnect_auto=1)

**Files:** (no edits — verification only)

**Step 1: Capture pre-change baseline** (do this BEFORE Task 0's commit if possible;
otherwise regenerate on the committed state — the fca=1 path is unaffected either way)

Run: `git stash list; source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simplematmul2.cc 2>&1 | tail -5`
Expected: RC 0.

**Step 2: Confirm all 9 K/tiling attrs present (fca=1 emits them)**

Run: `grep -oE "routing.(tile_m|tile_rows|tile_n|tile_cols|m_rounds|n_rounds|effective_k|full_k|k_rounds)" ir/dfschedule/6_BlueprintToSchedulePass.mlir | sort -u`
Expected: for simplematmul2 (currently fca=1) the M×N + K attrs present; and
`routing.fullconnect_auto = 1`. NOTE: simplematmul2 on main is fca=1 — do NOT change
it. (If it is fca=0 on main, adjust expectation to K-only + no M×N; still no
`Backfilled` since GEMM has no halo dict.)

**Step 3: Confirm no backfill fired for GEMM**

Run: `source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simplematmul2.cc 2>&1 | grep "Backfilled tile_m" || echo "NO BACKFILL (correct)"`
Expected: `NO BACKFILL (correct)`.

---

## Task 6: HW run (if board available)

**Files:** (no edits — verification only)

**Step 1: Run conv2d on board**

Run: `source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simpleconv2d.cc && python3 ./script/test/apppaltest.py -y -nonreboot > ./applog 2>&1; grep -E "device_teardown done|AIE ERROR" ./applog | tail`
Expected: `device_teardown done` (pass). `AIE ERROR` = fail → debug via
`data-mismatch-debug` / `aiehwdmadebug`.

**Step 2: Run GEMM on board (regression)**

Run: `source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simplematmul2.cc && python3 ./script/test/apppaltest.py -y -nonreboot > ./applog 2>&1; grep -E "device_teardown done|AIE ERROR" ./applog | tail`
Expected: `device_teardown done`.

---

## Files changed (summary)

- `src/llvm/aiehlc.cc` — front-end gating (Task 0, already applied; committed in Task 0).
- `src/mlir/mlirfront/tilinglinalg/pass/passdmaphoptodfscheblueprint/passdmaphoptodfscheblueprint.cpp`
  — `reconstructMTilingFromIR` helper (Task 1) + pass-entry backfill (Task 2) + local
  K-block fallback (Task 3).
- No edits to `passblueprinttoschedule` (null-guarded consumers), `simplematmul2.cc`,
  or `simpleconv2d.cc`.

## Residual risks + mitigations

| Risk | Mitigation |
|------|-----------|
| Accessor names (`getData`/`getTensor`/`getSplitdim`) differ from op defs | Verified against pass usage (532, 557-558, 1229); Task 1 Step 2 build catches any mismatch |
| Multiple `pull` flows → wrong tile_rows | Walk interrupts on first 2D pull; all output sub-flows share the same root tensor (same splitDim/slice), so first is representative |
| `tile_rows % m_rounds != 0` | Helper warns + returns invalid → backfill skipped (no wrong attr written) |
| conv geometry numbers unknown pre-run | Task 4 Step 2 prints actual backfilled values; cross-checked by host scf.for presence (Step 4) |
