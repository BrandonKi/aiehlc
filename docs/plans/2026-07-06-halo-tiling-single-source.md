# Halo Tiling Single-Source Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** For halo (spatial-halo conv) mode, make `#routing.tiling` the single in-IR source of both the M-round count and the IFM slab buffer size, deleting the redundant `routing.m_rounds`/`tile_m`/`tile_rows` **and** `routing.spatial_halo_buf_size` module-scalar paths, and carrying the derived values to the blueprint/kernel stages via the durable `tensor_N.halo` dict (`m_rounds`, `buf_size`).

**Architecture:** `#routing.tiling` on the `partitiontensor` op is extended with a nested W-level so its row (split) dim fully encodes `L1(splitnum) → L2(l2_rounds) → W(w_rounds)`. The pipeline reads back that attribute after routing IR is built and derives `tensor_N.halo.m_rounds = product of the row-dim temporal levels (L2 × W)` and `tensor_N.halo.buf_size = slabRows × raw_wc` (slabRows = L2 level slice if present else the L1 row slice; raw_wc = col-dim `outer.base`). Consumers switch from the module scalars to the `tensor_N.halo` dict: the host `scf.for` round loop + `classifyTiling` read `m_rounds`; the kernel/host window allocators + conv-C reconstruction read `buf_size`. `routing.spatial_halo_buf_size` is not durable-fragile like `#routing.tiling` (it survives), but it duplicates op-origin info — so it moves into the derived `tensor_N.halo` carrier. Matmul (`k_rounds`) paths and `routing.pp_depth_map` are untouched and must stay byte-identical.

**Why `tensor_N.halo`, not `#routing.tiling` directly:** `#routing.tiling` rides to late stages only as a *discardable* `{tiling=...}` attribute on `memref.subview`/`dfschedule.memref_mapping` ops — any canonicalizer may drop it, and consumers would need to trace a flow config back to the owning op. The `tensor_N.halo` module dict is a *durable* carrier no pass touches, read identically by the cloned host + kernel modules. So the single source is `#routing.tiling` (op-origin), and `tensor_N.halo` is its durable derived view.

**Tech Stack:** C++17, MLIR (custom `routing` dialect: `TilingAttr`/`DimAttr`/`LevelAttr`), the tilinglinalg unitest harness (`pass/unitest/test.cpp`), golden IR under `ir/dfschedule/*.mlir` and `ir/simplerouting/*.mlir`, E2E via `script/test/apppaltest.py`.

---

## Ground-truth reference (verified against current golden IR)

Current halo case (conv 230x230x4, R=4, l2_rounds=4, w_rounds=4, k_rounds=4):

- `ir/dfschedule/0_initial.mlir` module scalars: `routing.tile_m=196`, `routing.tile_rows=3136`, `routing.m_rounds=16` (16 = w_rounds×l2_rounds = 4×4).
- `tensor_0.halo` dict keys: `slice=61, step=56, split_dim=0, w_slice=61, w_step=56, w_rounds=4, row_pitch=920, ow_t=28, l2_slice=19, l2_step=14, l2_rounds=4, k_slice=244, k_step=224, k_rounds=4`. **No `m_rounds` key today.**
- `ir/simplerouting/0_initial.mlir` `#routing.tiling` d0 (row):
  `outer=<base=230,total=244,slice=61,step=56,rounds=4, slice_tiling=<base=61,total=76,slice=19,step=14,rounds=4>>`.
  - L1 `rounds=4` = **splitnum** (spatial, NOT a temporal round).
  - nested `slice_tiling rounds=4` = **l2_rounds**.
  - **w_rounds (4) is absent** — this plan adds it as a 3rd nesting level under the L2 `slice_tiling`.
- Therefore `m_rounds` derived from tiling = **product of the row-dim TEMPORAL levels only (L2 × W)** = 4×4 = 16. The L1 splitnum level is excluded.

**Critical reconciliation fact:** In the current code the host `scf.for` outer-round count is derived from `classification.mRounds` (= `tile_rows/tile_m` = 16). When `k_rounds>1` (`kAccumHaloSlab=true`), the L2/K multipliers at `passblueprinttoschedule.cpp:2269,2282,2290` are gated OFF, so the loop runs exactly `mRounds` times and the K/L2 bands are handled by a multi-dim BD. After this change, halo `mRounds` must come from `tensor_N.halo.m_rounds` and equal **16** for this golden case — no more, no less. Every task that touches the round count MUST be diffed against golden IR + applog before proceeding.

## Task order

1. Add W-split fields to `TensorSplitDesc` + mirror them from `shimDma` (pure plumbing, no IR change yet).
2. Build the nested W-level in `#routing.tiling` (routingmanager) — unit test asserts 3-level tiling.
3. Derive `tensor_N.halo.m_rounds` from `#routing.tiling` in the pipeline — unit test asserts value.
4. Add `haloMRounds` helper + switch halo consumers to it in `passblueprinttoschedule.cpp`.
5. Delete the halo `m_rounds/tile_m/tile_rows` module-scalar block in `aiehlc.cc:4450-4460`.
6. Derive `tensor_N.halo.buf_size` from `#routing.tiling` in the pipeline — unit test asserts value.
7. Switch the 5 `spatial_halo_buf_size` readers to `tensor_N.halo.buf_size` + delete both `aiehlc.cc` emission sites.
8. Golden-IR regen + E2E verification (covers both `m_rounds` and `buf_size` removals).

---

### Task 1: Add W-split fields to `TensorSplitDesc` and mirror from shimDma

**Files:**
- Modify: `src/mlir/mlirfront/tilinglinalg/pass/tilinglinalg_pipeline.h:83-97` (add fields to struct)
- Modify: `src/mlir/mlirfront/tilinglinalg/pass/tilinglinalg_pipeline.cpp:207-217` (mirror in `effectiveSplit`)

This task is pure C++ plumbing. `TensorSplitDesc` currently has `haloL2*` and `kAccum*` but no W-split fields, so `routingmanager` cannot build the W-level. Add `haloWSlice/haloWStep/haloWRounds` and mirror them from `shimDma.wSlice/wStep/wRounds`.

**Step 1: Add fields to the struct**

In `tilinglinalg_pipeline.h`, after the `haloL2*` fields (line 90, right before the K-contraction comment block at line 91), insert:

```cpp
    // Nested W (on-core WIDTH temporal) split, innermost under the L2 level on the
    // SAME split axis. Mirrored from DmaAddressing.wSlice/wStep/wRounds. When
    // haloWRounds > 1 each L2 row-slab is further chunked into haloWRounds width
    // rounds of haloWSlice cols advancing by haloWStep. Nested as the innermost
    // #routing.level under the L2 slice_tiling. Zero / 1 = no W split.
    int haloWSlice = 0;  // cols per on-core width round (e.g. 61)
    int haloWStep = 0;   // col stride between W rounds (e.g. 56)
    int haloWRounds = 0; // number of W rounds (e.g. 4); 0/1 = no W split
```

**Step 2: Mirror from shimDma in the pipeline**

In `tilinglinalg_pipeline.cpp`, inside the `if (sd.mode == 1 && sd.haloSlice > 0)` block, after the `haloL2*` mirroring (line 212, before the K-contraction comment at 213), insert:

```cpp
            // Nested W (on-core width temporal) split: propagate so
            // createroutingfuncBySplitModel nests the innermost W #routing.level
            // under the L2 slice_tiling in #routing.tiling.
            effectiveSplit.tensorSplits[i].haloWSlice = sd.wSlice;
            effectiveSplit.tensorSplits[i].haloWStep = sd.wStep;
            effectiveSplit.tensorSplits[i].haloWRounds = sd.wRounds;
```

**Step 3: Build to verify it compiles**

Run: build the tilinglinalg unitest (see "Build commands" at the bottom).
Expected: compiles clean. No IR change yet (`routingmanager` does not read the new fields).

**Step 4: Commit**

```bash
git add src/mlir/mlirfront/tilinglinalg/pass/tilinglinalg_pipeline.h \
        src/mlir/mlirfront/tilinglinalg/pass/tilinglinalg_pipeline.cpp
git commit -m "feat(halo): plumb W-split fields into TensorSplitDesc"
```

---

### Task 2: Build the nested W-level in `#routing.tiling`

**Files:**
- Modify: `src/mlir/mlirfront/tilinglinalg/routing/routingmanager.cpp:922-933` (nest W under L2)
- Test: `src/mlir/mlirfront/tilinglinalg/pass/unitest/test.cpp:1450-1618` (Conv2d Spatial-Halo Integration Test)

**Step 1: Write the failing test additions**

In `test.cpp`, the Conv2d Spatial-Halo test does not currently set W-split. Add W params next to the L2 params (after line 1489, `int l2_rounds = 4;`):

```cpp
    // Nested W (on-core width temporal) split: each L2 row-slab is chunked into
    // w_rounds width rounds. Coverage mirrors the height halo geometry.
    int w_slice = 61; // cols per on-core width round
    int w_step = 56;  // col stride between W rounds (overlap = 61-56 = 5)
    int w_rounds = 4; // number of width rounds
```

Set them on the `DmaAddressing halo` (after line 1513, `halo.rowPitch = raw_wc;`):

```cpp
    halo.wSlice = w_slice;
    halo.wStep = w_step;
    halo.wRounds = w_rounds;
    halo.owT = OW / R; // 28 (per-chunk output cols; required when wRounds>1)
```

Set them on the SplitModel (after line 1538, `sm.tensorSplits[0].kAccumRounds = k_rounds;`):

```cpp
        sm.tensorSplits[0].haloWSlice = w_slice;
        sm.tensorSplits[0].haloWStep = w_step;
        sm.tensorSplits[0].haloWRounds = w_rounds;
```

Then add a `#routing.tiling` assertion after the existing `tensor_0.halo` checks (after line 1591, the closing brace of the `if (haloAttr)` block, before the shim_dma check at 1593). This walks the row-dim `outer → slice_tiling(L2) → slice_tiling(W)` chain:

```cpp
    // Verify the partitiontensor #routing.tiling row dim has the 3-level chain
    // L1(splitnum) -> L2(l2_rounds) -> W(w_rounds).
    {
        routing::TilingAttr tilingA;
        module->walk([&](routing::partitiontensor pt) {
            if (!tilingA)
                if (auto t = pt.getTilingAttr())
                    tilingA = t;
        });
        if (!tilingA) {
            std::cerr << "FAIL: no #routing.tiling found on any partitiontensor" << std::endl;
        } else {
            auto dims = tilingA.getDims();
            // split_dim == 0 -> row dim is d0.
            auto rowOuter = dims[0].getOuter();
            auto l2Level = rowOuter.getSliceTiling(); // L2
            if (!l2Level) {
                std::cerr << "FAIL: #routing.tiling row dim missing L2 slice_tiling" << std::endl;
            } else if (l2Level.getRounds() != l2_rounds) {
                std::cerr << "FAIL: L2 rounds " << l2Level.getRounds() << " != " << l2_rounds << std::endl;
            } else {
                auto wLevel = l2Level.getSliceTiling(); // W (innermost)
                if (!wLevel) {
                    std::cerr << "FAIL: #routing.tiling missing nested W slice_tiling" << std::endl;
                } else if (wLevel.getRounds() != w_rounds) {
                    std::cerr << "FAIL: W rounds " << wLevel.getRounds() << " != " << w_rounds << std::endl;
                } else if (wLevel.getSlice() != w_slice || wLevel.getStep() != w_step) {
                    std::cerr << "FAIL: W slice/step mismatch (got slice=" << wLevel.getSlice()
                              << " step=" << wLevel.getStep() << ")" << std::endl;
                } else {
                    std::cout << "PASS: #routing.tiling row dim is 3-level L1->L2->W (w_rounds="
                              << wLevel.getRounds() << ")" << std::endl;
                }
            }
        }
    }
```

**Step 2: Run test to verify it fails**

Run the unitest (see "Build commands"), filter for the Conv2d Spatial-Halo section.
Expected: FAIL — "missing nested W slice_tiling" (routingmanager does not build the W-level yet).

**Step 3: Build the W-level in routingmanager**

In `routingmanager.cpp`, replace the L2-level construction (lines 922-933) so the W-level is nested inside the L2 `slice_tiling`. Current code:

```cpp
                    // Row (HW-split) dim: outer L1 level + optional nested L2 slice_tiling.
                    routing::LevelAttr rowSliceTiling;
                    if (l2Rounds > 1) {
                        rowSliceTiling = routing::LevelAttr::get(
                            ctx2, /*base=*/split.haloSlice, /*total=*/(int64_t)l2Slice * l2Rounds,
                            /*slice=*/l2Slice, /*step=*/l2Step, /*rounds=*/l2Rounds,
                            /*slice_tiling=*/routing::LevelAttr{});
                    }
```

Replace with (adds the W-level read + nests it under L2):

```cpp
                    // Innermost W (on-core width temporal) split: nested under L2.
                    // Opt-in: only emitted when haloWRounds > 1.
                    int32_t wSliceLv = 0, wStepLv = 0, wRoundsLv = 1;
                    if (split.haloWRounds > 1 && split.haloWSlice > 0) {
                        wSliceLv = split.haloWSlice;
                        wStepLv = split.haloWStep;
                        wRoundsLv = split.haloWRounds;
                    }
                    routing::LevelAttr wSliceTiling;
                    if (wRoundsLv > 1) {
                        wSliceTiling = routing::LevelAttr::get(
                            ctx2, /*base=*/wSliceLv, /*total=*/(int64_t)wSliceLv * wRoundsLv,
                            /*slice=*/wSliceLv, /*step=*/wStepLv, /*rounds=*/wRoundsLv,
                            /*slice_tiling=*/routing::LevelAttr{});
                    }
                    // Row (HW-split) dim: outer L1 level + optional nested L2 slice_tiling,
                    // which itself nests the optional innermost W slice_tiling.
                    routing::LevelAttr rowSliceTiling;
                    if (l2Rounds > 1) {
                        rowSliceTiling = routing::LevelAttr::get(
                            ctx2, /*base=*/split.haloSlice, /*total=*/(int64_t)l2Slice * l2Rounds,
                            /*slice=*/l2Slice, /*step=*/l2Step, /*rounds=*/l2Rounds,
                            /*slice_tiling=*/wSliceTiling);
                    }
```

Note: the W-level is only reachable when `l2Rounds > 1` (it nests under L2). This matches the golden geometry (L2 always present when W is). If a W-only-without-L2 case is ever needed, that is out of scope here.

**Step 4: Run test to verify it passes**

Run the unitest, Conv2d Spatial-Halo section.
Expected: PASS — "#routing.tiling row dim is 3-level L1->L2->W (w_rounds=4)". Existing L2/K-accum PASS lines still present.

**Step 5: Commit**

```bash
git add src/mlir/mlirfront/tilinglinalg/routing/routingmanager.cpp \
        src/mlir/mlirfront/tilinglinalg/pass/unitest/test.cpp
git commit -m "feat(halo): nest W-level in #routing.tiling row dim"
```

---

### Task 3: Derive `tensor_N.halo.m_rounds` from `#routing.tiling`

**Files:**
- Modify: `src/mlir/mlirfront/tilinglinalg/pass/tilinglinalg_pipeline.cpp:322` (add `m_rounds` to the halo dict)
- Test: `src/mlir/mlirfront/tilinglinalg/pass/unitest/test.cpp` (assert `tensor_0.halo.m_rounds`)

The halo dict is currently built purely from `sd` (shimDma) at lines 281-322. We add `m_rounds` derived from the row-dim temporal levels. Two equivalent sources are available: (a) read back the partitiontensor `#routing.tiling` and multiply the row-dim temporal levels' `rounds`, or (b) compute `max(1,sd.l2Rounds) * max(1,sd.wRounds)` directly from `sd`. Per the design, `#routing.tiling` is the canonical source, so use (a) — read it back and derive from it. This guarantees the halo dict is a *derived* view of the tiling, not a parallel copy.

**Step 1: Write the failing test assertion**

In `test.cpp`, inside the `if (haloAttr)` block, after the K-accum checks (after line 1587, before the closing `} else {` at 1588), add:

```cpp
        // m_rounds derived from #routing.tiling row-dim temporal levels (L2 x W).
        auto mRoundsA = haloAttr.getAs<IntegerAttr>("m_rounds");
        if (mRoundsA) {
            int64_t expected = (int64_t)l2_rounds * w_rounds; // 4 * 4 = 16
            std::cout << "  m_rounds=" << mRoundsA.getInt() << std::endl;
            if (mRoundsA.getInt() == expected)
                std::cout << "PASS: tensor_0.halo.m_rounds == l2_rounds*w_rounds (" << expected << ")" << std::endl;
            else
                std::cerr << "FAIL: tensor_0.halo.m_rounds " << mRoundsA.getInt() << " != " << expected << std::endl;
        } else {
            std::cerr << "FAIL: tensor_0.halo missing m_rounds attr" << std::endl;
        }
```

**Step 2: Run test to verify it fails**

Run the unitest, Conv2d Spatial-Halo section.
Expected: FAIL — "tensor_0.halo missing m_rounds attr".

**Step 3: Implement the derivation**

In `tilinglinalg_pipeline.cpp`, inside the halo-dict `for` loop, right before `m->setAttr("tensor_" + std::to_string(i) + ".halo", ...)` at line 322, add a block that reads back the partitiontensor's `#routing.tiling` for this tensor and multiplies the row-dim temporal levels' rounds. Add near the top of the file (with the other includes) if not already present:

```cpp
#include "routing/routingattr.h" // routing::TilingAttr / DimAttr / LevelAttr
```

(Verify the exact include path used elsewhere in this file for `routing::partitiontensor` / `routing::TilingAttr`; match it.)

Insert before the `setAttr` at line 322:

```cpp
            // Derive m_rounds (host round-loop trip count) from the CANONICAL
            // #routing.tiling on this tensor's partitiontensor op: the product of
            // the row (split) dim's TEMPORAL levels' rounds (L2 x W). The L1 outer
            // level carries splitnum (spatial) and is excluded. This makes the halo
            // dict a derived view of #routing.tiling, not a parallel copy.
            {
                int splitDimI = sd.splitDim; // 0 or 1
                int64_t mRounds = 1;
                m->walk([&](routing::partitiontensor pt) {
                    // Match by tensor operand value == tensorValues[i].
                    // (op operand is named $tensor -> accessor getTensor(); verified
                    //  against routingop.td.)
                    if (pt.getTensor() != tensorValues[i])
                        return;
                    auto til = pt.getTilingAttr();
                    if (!til)
                        return;
                    auto dims = til.getDims();
                    if (splitDimI < 0 || splitDimI >= (int)dims.size())
                        return;
                    // Walk the nested slice_tiling chain BELOW the L1 outer level
                    // (outer = splitnum spatial; children = temporal rounds).
                    auto lvl = dims[splitDimI].getOuter().getSliceTiling();
                    while (lvl) {
                        int64_t r = lvl.getRounds();
                        if (r > 1)
                            mRounds *= r;
                        lvl = lvl.getSliceTiling();
                    }
                });
                if (mRounds > 1)
                    entries.append("m_rounds", builder.getI32IntegerAttr((int32_t)mRounds));
            }
```

Note: `pt.getTensor()` is the verified accessor for the `partitiontensor` tensor operand (operand `$tensor` in `routingop.td`). Match tensor `i` to the same value passed into `tensorValues` used by `createroutingfuncBySplitModel`. If matching by value is unreliable, an acceptable fallback is to derive directly: `mRounds = std::max(1, sd.l2Rounds) * std::max(1, sd.wRounds);` — but prefer the tiling read-back per the design.

**Step 4: Run test to verify it passes**

Run the unitest, Conv2d Spatial-Halo section.
Expected: PASS — "tensor_0.halo.m_rounds == l2_rounds*w_rounds (16)".

**Step 5: Commit**

```bash
git add src/mlir/mlirfront/tilinglinalg/pass/tilinglinalg_pipeline.cpp \
        src/mlir/mlirfront/tilinglinalg/pass/unitest/test.cpp
git commit -m "feat(halo): derive tensor_N.halo.m_rounds from #routing.tiling"
```

---

### Task 4: Add `haloMRounds` helper + switch halo consumers to it

**Files:**
- Modify: `src/mlir/mlirfront/tilinglinalg/pass/passblueprinttoschedule/passblueprinttoschedule.cpp`
  - Add helper near `classifyTiling` (`:518`).
  - Update `classifyTiling` (`:529-544`) and/or the round-loop (`:2229-2291`).

This is the highest-risk task. **Do not change numeric behavior** — the golden host `scf.for` for the reference case must still run 16 rounds. Because `aiehlc.cc` still emits `tile_m/tile_rows` until Task 5, this task adds the *fallback* path (`tensor_N.halo.m_rounds`) without removing the scalar reads yet, so Task 4 and Task 5 are individually golden-safe.

**Step 1: Add the helper**

After `classifyTiling` (after line 565), add:

```cpp
// Returns tensor_N.halo.m_rounds for the given func-arg index, or 0 if absent.
// This is the canonical halo M-round count derived from #routing.tiling
// (product of the split dim's temporal levels: L2 x W).
static int64_t haloMRounds(ModuleOp moduleOp, int argIdx) {
    if (!moduleOp || argIdx < 0)
        return 0;
    std::string name = "tensor_" + std::to_string(argIdx) + ".halo";
    if (auto dict = moduleOp->getAttrOfType<DictionaryAttr>(name))
        if (auto a = dict.getAs<IntegerAttr>("m_rounds"))
            return a.getInt();
    return 0;
}
```

**Step 2: Use the helper as the halo mRounds source in the round loop**

In the round-loop block, `outerRounds` is set at line 2277 from `classification.mRounds`. For the halo sender flow, prefer `haloMRounds`. Locate the `int haloArgIdxL2 = traceFlowConfigToFuncArgIndex(shimFlowConfig);` at line 2254 (already computed inside the `if (moduleOp && shimIsSender)` block). Extend that block to also fetch the m_rounds fallback, then override `outerRounds` when present.

After line 2277 (`int64_t outerRounds = nOuterPolicy ? classification.nRounds : classification.mRounds;`), insert:

```cpp
                // Halo single-source: when the shim tensor carries tensor_N.halo.m_rounds
                // (derived from #routing.tiling), use it as the authoritative M-round
                // count. This replaces the routing.tile_m/tile_rows-derived value for the
                // halo path. When absent (matmul / non-halo), keep classification.mRounds.
                if (!nOuterPolicy && shimIsSender) {
                    int haloArgIdxM = traceFlowConfigToFuncArgIndex(shimFlowConfig);
                    int64_t hm = haloMRounds(moduleOp, haloArgIdxM);
                    if (hm > 1)
                        outerRounds = hm;
                }
```

**Important reconciliation:** with `outerRounds` now equal to `m_rounds` (= L2×W = 16 for golden), the subsequent multiplies at lines 2282 (`*= haloL2RoundsForLoop`) and 2290 (`*= haloKRoundsForLoop`) would DOUBLE-COUNT. In the golden `k_rounds>1` case those are already gated OFF (`kAccumHaloSlab`), but for a pure-L2 (no-k) case `haloL2RoundsForLoop>1` would multiply again. Guard them: when the m_rounds override fired, `m_rounds` ALREADY includes L2×W, so skip both multiplies. Change the conditions at 2282 and 2290 to also require that the override did NOT fire. Introduce a flag:

Replace the inserted block's tail and the two multiply sites so that, when `hm > 1`, a local `bool haloMRoundsOverride = true;` disables the L2/W/K folding that `m_rounds` already accounts for. Concretely:
- set `bool haloMRoundsOverride = (hm > 1);` in the inserted block (declare it before line 2282's scope),
- change `if (haloL2RoundsForLoop > 1) {` (line 2282) to `if (haloL2RoundsForLoop > 1 && !haloMRoundsOverride) {`,
- leave the K multiply (2290) as-is ONLY if K is NOT part of m_rounds. **Per golden, m_rounds = L2×W excludes k_rounds**, and the K bands are handled by the multi-dim BD (kAccumHaloSlab), so the K multiply must also NOT apply to the loop trip count. It is already gated by `kAccumHaloSlab` upstream (line 2269) for the force-loop, but the multiply at 2290 is not. Guard it too: `if (haloKRoundsForLoop > 1 && !haloMRoundsOverride)`.

**Step 3: Verify behavior is unchanged (golden diff)**

Regenerate the halo IR and diff the host round-loop bounds. Run the unitest to produce `worklocal/host.cc`, then compare the scf.for trip count and the emitted `outerRounds=` debug line against a pre-change capture.

Run: unitest (produces `applog` + `worklocal/host.cc`).
Expected: `[BlueprintToSchedule] ... outerRounds=16` for the halo input flow — identical to pre-change. `git diff` on regenerated `ir/dfschedule/*.mlir` shows the added W-level + `m_rounds` key, and NO change to BD lengths, iter dims, lock values, or round counts.

**Step 4: Commit**

```bash
git add src/mlir/mlirfront/tilinglinalg/pass/passblueprinttoschedule/passblueprinttoschedule.cpp
git commit -m "feat(halo): consume tensor_N.halo.m_rounds in round loop"
```

---

### Task 5: Delete the halo module-scalar block in `aiehlc.cc`

**Files:**
- Modify: `src/llvm/aiehlc.cc:4450-4460` (remove the halo `tile_m/tile_rows/m_rounds` emission)

Only the inner `if (derivedTilingParams.spatialMRounds > 1 ...)` block that sets `routing.tile_m`/`routing.tile_rows`/`routing.m_rounds` is deleted. Keep the enclosing `spatial_halo_buf_size` emission (lines 4438-4443, 4461). Do NOT touch the K-round (`kRounds>1`, matmul) emission blocks elsewhere in the file (`:3846`, `:4419`).

**Step 1: Delete the block**

Remove lines 4450-4460 (the entire inner `if` including its `llvm::outs()`), leaving the `spatial_halo_buf_size` set and its closing brace intact.

**Step 2: Verify `classifyTiling` degrades gracefully for halo**

With `routing.tile_m`/`tile_rows` no longer set on the halo path, `classifyTiling` returns `mMode=Match, mRounds=1` for halo (since `m==0`). The round loop's `needsOuterLoop` then depends on `classification.mMode == Multiple` (now false) OR the l2/k force at line 2269. For the golden `k_rounds>1` case, line 2269 is gated OFF (`kAccumHaloSlab`), so `needsOuterLoop` would become **false** — regressing the 16-round loop.

Therefore Task 5 REQUIRES that `needsOuterLoop` also be forced when `haloMRounds > 1`. Add to the force condition. After the block at line 2264 (`}` closing the `if (moduleOp && shimIsSender)`), and before line 2269, fetch the halo m_rounds and include it in the force:

```cpp
            int64_t haloMRoundsForLoop = 0;
            if (moduleOp && shimIsSender) {
                int haloArgIdxM = traceFlowConfigToFuncArgIndex(shimFlowConfig);
                haloMRoundsForLoop = haloMRounds(moduleOp, haloArgIdxM);
            }
```

Change line 2269 from:

```cpp
            if ((haloL2RoundsForLoop > 1 || haloKRoundsForLoop > 1) && !kAccumHaloSlab)
                needsOuterLoop = true;
```

to:

```cpp
            if (((haloL2RoundsForLoop > 1 || haloKRoundsForLoop > 1) && !kAccumHaloSlab) ||
                haloMRoundsForLoop > 1)
                needsOuterLoop = true;
```

(This makes the halo loop self-sufficient from `tensor_N.halo.m_rounds`, independent of the deleted scalars.)

**Step 3: Build + regenerate + golden diff**

Run: unitest (regenerates halo IR + host.cc + applog).
Expected:
- `routing.tile_m`/`routing.tile_rows`/`routing.m_rounds` GONE from regenerated `ir/dfschedule/0_initial.mlir` (halo module).
- `[BlueprintToSchedule] ... outerRounds=16` unchanged.
- host `scf.for` bounds, BD lengths, iter dims, lock values byte-identical to pre-refactor golden (except the removed 3 module scalars and the added `#routing.tiling` W-level + `m_rounds` key).

**Step 4: Confirm matmul is byte-identical**

Regenerate a matmul case (e.g. `simplematmul2.cc`) and `git diff` its IR: it must be byte-identical (matmul never had `tensor_N.halo`, and its `k_rounds` emission at `aiehlc.cc:3846/4419` was untouched).

**Step 5: Commit**

```bash
git add src/llvm/aiehlc.cc \
        src/mlir/mlirfront/tilinglinalg/pass/passblueprinttoschedule/passblueprinttoschedule.cpp
git commit -m "refactor(halo): remove redundant routing.m_rounds/tile_m/tile_rows scalars"
```

---

### Task 6: Derive `tensor_N.halo.buf_size` from `#routing.tiling`

**Files:**
- Modify: `src/mlir/mlirfront/tilinglinalg/pass/tilinglinalg_pipeline.cpp:322` (add `buf_size` to the halo dict, next to `m_rounds` from Task 3)
- Test: `src/mlir/mlirfront/tilinglinalg/pass/unitest/test.cpp` (assert `tensor_0.halo.buf_size`)

`routing.spatial_halo_buf_size` today equals the IFM slab element count
`convHaloBufSize = slabRows × raw_wc` (`aiehlc.cc:2340-2373`), where
`slabRows = (l2Rounds>1 && l2Slice>0) ? l2Slice : halo_slice` and `raw_wc` is the
full input row width. Both are already in `#routing.tiling`:
- `slabRows` = the L2 level slice (one nesting level under the row-dim `outer`),
  falling back to the row `outer.slice` when there is no L2 level.
- `raw_wc` = the **col-dim** `outer.base` (the full, un-split row width — d1 outer base).

**Critical:** take exactly ONE level under the row `outer` for `slabRows`. The new
W-level (Task 2) nests *below* L2, so a naive recurse would pick up W. `slabRows`
is L2 only. This produces the SAME integer as `convHaloBufSize` (element count, not
bytes) so every downstream consumer that divides by `elementSizeBytes` or by
`haloSlice` keeps working unchanged.

Golden reference: `slabRows=19` (l2_slice), `raw_wc=920` → `buf_size = 19*920 = 17480`
(element count). Confirm this equals the current `routing.spatial_halo_buf_size` value
in `ir/dfschedule/0_initial.mlir` **before** writing the derivation (read the golden
scalar first; if it differs, the derivation formula is wrong — STOP).

**Step 1: Confirm the golden `spatial_halo_buf_size` value**

Read the current halo `routing.spatial_halo_buf_size` from `ir/dfschedule/0_initial.mlir`
(regenerate if the tree was wiped). Record the exact integer `B`. The derivation MUST
reproduce `B` exactly.

**Step 2: Write the failing test assertion**

In `test.cpp`, inside the `if (haloAttr)` block, after the `m_rounds` checks from Task 3, add:

```cpp
        // buf_size derived from #routing.tiling: slabRows (L2 slice) x raw_wc (col outer.base).
        auto bufA = haloAttr.getAs<IntegerAttr>("buf_size");
        if (bufA) {
            int64_t slabRows = (l2_rounds > 1) ? l2_slice : halo_slice; // 19
            int64_t expected = slabRows * raw_wc; // 19 * 920 = 17480
            std::cout << "  buf_size=" << bufA.getInt() << std::endl;
            if (bufA.getInt() == expected)
                std::cout << "PASS: tensor_0.halo.buf_size == slabRows*raw_wc (" << expected << ")" << std::endl;
            else
                std::cerr << "FAIL: tensor_0.halo.buf_size " << bufA.getInt() << " != " << expected << std::endl;
        } else {
            std::cerr << "FAIL: tensor_0.halo missing buf_size attr" << std::endl;
        }
```

**Step 3: Run test to verify it fails**

Run the unitest, Conv2d Spatial-Halo section.
Expected: FAIL — "tensor_0.halo missing buf_size attr".

**Step 4: Implement the derivation**

In `tilinglinalg_pipeline.cpp`, inside the same halo-dict `for` loop (reuse the
`partitiontensor` match from Task 3 — ideally in the SAME `m->walk` so the tiling is
read once), before the `setAttr` at line 322, compute `buf_size`:

```cpp
            // Derive buf_size (IFM slab element count) from the CANONICAL #routing.tiling:
            //   slabRows = L2 level slice (one level under the row-dim outer), else row outer.slice
            //   raw_wc   = col-dim (d1) outer.base (full un-split row width)
            // buf_size = slabRows * raw_wc, element count identical to the old
            // convHaloBufSize -> routing.spatial_halo_buf_size.
            {
                int splitDimI = sd.splitDim;      // row (split) dim: 0 or 1
                int colDimI = (splitDimI == 0) ? 1 : 0;
                int64_t slabRows = 0, rawWc = 0;
                m->walk([&](routing::partitiontensor pt) {
                    if (pt.getTensor() != tensorValues[i]) return;
                    auto til = pt.getTilingAttr();
                    if (!til) return;
                    auto dims = til.getDims();
                    if (splitDimI >= (int)dims.size() || colDimI >= (int)dims.size()) return;
                    auto rowOuter = dims[splitDimI].getOuter();
                    auto l2 = rowOuter.getSliceTiling();   // exactly ONE level under outer = L2
                    slabRows = l2 ? l2.getSlice() : rowOuter.getSlice();
                    rawWc = dims[colDimI].getOuter().getBase();
                });
                if (slabRows > 0 && rawWc > 0)
                    entries.append("buf_size", builder.getI32IntegerAttr((int32_t)(slabRows * rawWc)));
            }
```

**Step 5: Run test to verify it passes**

Run the unitest, Conv2d Spatial-Halo section.
Expected: PASS — "tensor_0.halo.buf_size == slabRows*raw_wc (17480)" (or whatever `B` is).
The printed `buf_size` MUST equal the golden `routing.spatial_halo_buf_size` `B` from Step 1.

**Step 6: Commit**

```bash
git add src/mlir/mlirfront/tilinglinalg/pass/tilinglinalg_pipeline.cpp \
        src/mlir/mlirfront/tilinglinalg/pass/unitest/test.cpp
git commit -m "feat(halo): derive tensor_N.halo.buf_size from #routing.tiling"
```

---

### Task 7: Switch `spatial_halo_buf_size` consumers + delete `aiehlc.cc` emission

**Files:**
- Modify: `src/mlir/mlirfront/tilinglinalg/pass/passblueprinttoschedule/passblueprinttoschedule.cpp` (3 uses: `:664` gate, `:1706` value, `:2409` value)
- Modify: `src/mlir/mlirfront/tilinglinalg/pass/passblueprinttoschedulekernel/passblueprinttoschedulekernel.cpp:892` (1 value use)
- Modify: `src/mlir/mlirfront/tilinglinalg/pass/passdmaphoptodfscheblueprint/passdmaphoptodfscheblueprint.cpp:1085` (1 value use — 2D shim-BD gather for `w_rounds>1`)
- Modify: `src/llvm/aiehlc.cc` (delete emission at `:3869` mesh-loop and `:4440` non-mesh halo)

`routing.spatial_halo_buf_size` has **5** reader sites (verified by grep). All read a
single integer (element count). Replace each with `tensor_N.halo.buf_size` via a small
helper mirroring `haloMRounds`. Because Task 6 guarantees the carried integer is
byte-identical to the old scalar, this is a pure source-substitution with no numeric
change.

Note: line numbers below are approximate (source drifts ±2 between edits); grep
`routing.spatial_halo_buf_size` to relocate each `getAttrOfType<IntegerAttr>` reader.
The `passdmaphoptodfscheblueprint.cpp:1085` reader runs in the SHARED stage (before the
host/kernel module clone). Task 6 adds `buf_size` in `tilinglinalg_pipeline.cpp` (very
first stage), so the dict value is already present there — reachability is fine.

**Step 1: Add a `haloBufSize` helper**

In `passblueprinttoschedule.cpp`, next to `haloMRounds` (Task 4), add:

```cpp
// Returns tensor_N.halo.buf_size for the given func-arg index, or 0 if absent.
static int64_t haloBufSize(ModuleOp moduleOp, int argIdx) {
    if (!moduleOp || argIdx < 0)
        return 0;
    std::string name = "tensor_" + std::to_string(argIdx) + ".halo";
    if (auto dict = moduleOp->getAttrOfType<DictionaryAttr>(name))
        if (auto a = dict.getAs<IntegerAttr>("buf_size"))
            return a.getInt();
    return 0;
}
```

Add the same helper (or a shared one in a common header) to
`passblueprinttoschedulekernel.cpp` for its single use.

**Step 2: Switch the gate use (`passblueprinttoschedule.cpp:664`)**

The gate currently checks `module has routing.spatial_halo_buf_size && value>0`, then
reads `w_rounds`/`ow_t` from the `tensor_N.halo` dict. Replace the module-scalar
presence/value check with `haloBufSize(moduleOp, argIdx) > 0`, tracing `argIdx` from the
same flow config already available at that site (mirror how `:664` obtains the tensor
index for the `.halo` dict lookup it already does).

**Step 3: Switch the two value uses (`:1706`, `:2409`)**

- `:1706` `haloBufAttr = ...getAttrOfType<IntegerAttr>("routing.spatial_halo_buf_size")` →
  `pingPongBufferSize = haloBufAttr.getInt() / elementSizeBytes;` → replace
  `haloBufAttr.getInt()` with `haloBufSize(moduleOp, argIdx)`. The `/elementSizeBytes`
  stays (Task 6 carries the element count, same integer as before).
- `:2409` `bufSize = ...getInt(); chunkWidthElems = bufSize/haloSlice; convC = chunkWidthElems/haloWSlice;`
  → replace the `getInt()` source with `haloBufSize(moduleOp, argIdx)`; arithmetic unchanged.

**Step 4: Switch the kernel value use (`passblueprinttoschedulekernel.cpp:892`)**

`pingPongBufSize = haloBufAttr.getInt();` → `pingPongBufSize = haloBufSize(moduleOp, argIdx);`
Keep `isSpatialHaloPort=true; paramInfo.singleBuffer=true;` as-is. Add the `haloBufSize`
helper (or share it) in this file.

**Step 5: Switch the shared-stage value use (`passdmaphoptodfscheblueprint.cpp:1085`)**

This site (inside the `w_rounds>1` 2D shim-BD gather) already has `tensorIdx` and the
`tensor_N.halo` `haloAttr` in scope (`:1079-1081`). The simplest switch reads `buf_size`
directly from that dict rather than the module scalar:

```cpp
// was: auto bufSizeAttr = moduleOp->getAttrOfType<IntegerAttr>("routing.spatial_halo_buf_size");
auto bufSizeAttr = haloAttr.getAs<IntegerAttr>("buf_size");
```

`chunkWidthElems = bufSizeAttr.getInt() / haloSlice` and everything below is unchanged
(same integer). Since this pass runs before the host/kernel clone, no helper indirection
is needed — the dict is right there.

**Step 6: Delete both `aiehlc.cc` emission sites**

- Non-mesh halo: delete the `routing.spatial_halo_buf_size` set at `:4440` (plus its
  `llvm::outs()` at `:4442`). After Task 5 removed the inner `tile_m/tile_rows/m_rounds`
  block, this leaves only the `spatial_halo_buf_size` set to remove here.
- Mesh-loop: delete the `routing.spatial_halo_buf_size` set at `:3869` (plus its
  `llvm::outs()` at `:3871`).
- Leave `convHaloBufSize` computation (`:2360-2373`, `derivedTilingParams.convHaloBufSize`
  at `:2369`) if it is still read elsewhere; grep `convHaloBufSize` first. If it is now
  dead (only fed the deleted setAttr), delete the computation too. Do NOT touch matmul
  `k_rounds` blocks.

**Step 7: Build + regenerate + golden diff**

Run: unitest (regenerates halo IR + host.cc + kernel.cc + applog).
Expected:
- `routing.spatial_halo_buf_size` GONE from regenerated `ir/dfschedule/0_initial.mlir`.
- `tensor_0.halo.buf_size` present with the golden value `B`.
- host `scf.for`, BD lengths, ping-pong buffer sizes, conv-C reconstruction, kernel
  window sizes byte-identical to pre-refactor (except the moved attribute).

**Uncertainty to resolve during execution:** the kernel comment at
`passblueprinttoschedulekernel.cpp:915-921` says the K-split halo path has NO
`spatial_halo_buf_size` (it uses `l2_rounds*k_rounds` windows). But the golden case has
`k_rounds=4`. Before deleting the emission, regenerate and confirm whether the golden
`0_initial.mlir` actually contains `routing.spatial_halo_buf_size` at all for this
`k_rounds>1` case. If it is ABSENT in golden, Task 6/7 for buf_size only matters for a
pure-L2 (no-K) halo case — still do the substitution, but the golden diff for THIS case
will show no buf_size change. Verify against a no-K halo case if one exists.

**Step 8: Confirm matmul byte-identical**

Regenerate a matmul case; `git diff` its IR must be byte-identical.

**Step 9: Commit**

```bash
git add src/llvm/aiehlc.cc \
        src/mlir/mlirfront/tilinglinalg/pass/passblueprinttoschedule/passblueprinttoschedule.cpp \
        src/mlir/mlirfront/tilinglinalg/pass/passblueprinttoschedulekernel/passblueprinttoschedulekernel.cpp \
        src/mlir/mlirfront/tilinglinalg/pass/passdmaphoptodfscheblueprint/passdmaphoptodfscheblueprint.cpp
git commit -m "refactor(halo): remove redundant routing.spatial_halo_buf_size scalar"
```

---

### Task 8: Golden-IR regen + E2E verification

**Files:**
- Regenerate: `ir/dfschedule/*.mlir`, `ir/simplerouting/*.mlir`
- Run: `example/tileprogram/ccode/simpleconv2d.cc` via `script/test/apppaltest.py`

**Step 1: Regenerate all golden IR and review the diff**

Regenerate the halo golden set. Review `git diff ir/`:
- Expected additions: W-level in every `#routing.tiling` d0 (`slice_tiling=<... l2 ...>` now nests `slice_tiling=<... w_rounds=4 ...>`); `m_rounds=16` and `buf_size=B` keys in `tensor_0.halo`.
- Expected removals: `routing.tile_m`, `routing.tile_rows`, `routing.m_rounds`, and `routing.spatial_halo_buf_size` from the halo module.
- Everything else (BD lens, iter_step/iter_wrap, lock ids/values, channel repeats, kernel rounds, ping-pong buffer sizes) MUST be unchanged.

If any BD/round/lock/buffer value changed, STOP — the reconciliation in Task 4 or the buf_size derivation in Task 6 is wrong. Use the systematic-debugging skill.

**Step 2: E2E on hardware**

Compile + run the spatial-halo conv end-to-end:

```bash
source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simpleconv2d.cc
python3 ./script/test/apppaltest.py -y -nonreboot > ./applog 2>&1
```

Expected in `applog`: `device_teardown done`, correct conv output, host round-loop bounds identical to pre-refactor. No `AIE ERROR`.

**Step 3: Commit regenerated golden IR**

```bash
git add ir/dfschedule ir/simplerouting example/tileprogram/ccode/simpleconv2d.cc
git commit -m "test(halo): regenerate golden IR for single-source tiling"
```

**Step 4: Update architecture docs (per project Document rule)**

Update `doc/tilinglinalg.md` (and any `#routing.tiling` level description) to document the 3-level halo row-dim nesting (L1→L2→W) and that both `tensor_N.halo.m_rounds` and `tensor_N.halo.buf_size` are derived from it (replacing the deleted `routing.m_rounds`/`tile_m`/`tile_rows`/`spatial_halo_buf_size` module scalars). List all changed files per the Process-transparent rule.

```bash
git add doc/tilinglinalg.md
git commit -m "docs(halo): document 3-level #routing.tiling + m_rounds/buf_size derivation"
```

---

## Build commands

Tilinglinalg unitest (standalone) — used in Tasks 1-7 for compile + unit assertions:

```bash
# From repo root; build the unitest per CLAUDE.md "TilingLinalg unitest" flow.
# (Follow the project's existing unitest build steps under pass/unitest/build.)
```

Full E2E (Task 8):

```bash
source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simpleconv2d.cc
python3 ./script/test/apppaltest.py -y -nonreboot > ./applog 2>&1
```

## Risks & guardrails

- **Double-counting L2/W** — highest risk. `m_rounds` already = L2×W. Every place that
  previously multiplied `haloL2RoundsForLoop`/`haloKRoundsForLoop` into the trip count
  must be guarded by `!haloMRoundsOverride`. Verify `outerRounds=16` in applog at each step.
- **Accessor names** — verified against `routingattr.td` / `routingop.td`: `LevelAttr` →
  `getBase/getTotal/getSlice/getStep/getRounds/getSliceTiling`; `DimAttr` → `getOuter`;
  `TilingAttr` → `getDims`; `partitiontensor` tensor operand → `getTensor()`.
- **W nests only under L2** — the implementation nests W inside the L2 `slice_tiling`; a
  W-without-L2 case is out of scope and would need a separate nesting site.
- **Matmul byte-identical** — all halo branches are additive and guarded on `tensor_N.halo`
  presence; confirm with a matmul IR diff in Tasks 5 and 7.
- **`#routing.tiling` reachability** — as a discardable attr it can be dropped by later
  passes/canonicalizers, so both the round count and slab buffer size are carried forward
  via the durable `tensor_N.halo` dict (`m_rounds` in Task 3, `buf_size` in Task 6), NOT
  the attr itself.
- **buf_size: L2 vs W ambiguity** — `slabRows` must read exactly ONE nesting level under
  the row `outer` (L2), never recurse into the new W-level. Wrong level → wrong buffer
  size → HW hang or wrong output. The Task 6 unit test asserts `buf_size == slabRows*raw_wc`
  against the golden `B` to catch this.
- **buf_size: element vs byte identity** — the carried integer is an ELEMENT count (same as
  the old scalar). Consumers apply their own `/elementSizeBytes` or `/haloSlice`; the
  substitution must not change the units, or every downstream size silently shifts.
- **buf_size: k_rounds golden presence** — unverified whether the `k_rounds>1` golden case
  actually emits `routing.spatial_halo_buf_size` (kernel comment claims K-split has none).
  Task 7 Step 6 resolves this against regenerated golden before deleting the emission.
