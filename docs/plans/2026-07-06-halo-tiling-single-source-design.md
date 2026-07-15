# Halo mode: `#routing.tiling` as single source of M-round truth

Date: 2026-07-06
Scope: **halo mode only** (spatial-halo conv). Matmul / GEMM (`k_rounds`) paths unchanged.

## Problem

For halo mode there are three parallel copies of the same tiling data:

1. **`#routing.tiling`** on `partitiontensor` — built from `shimDma` in
   `routingmanager.cpp:895-950`. The intended operation-level descriptor.
2. **`tensor_N.halo`** module dict — built from the *same* `shimDma` in
   `tilinglinalg_pipeline.cpp:281-322`.
3. **`routing.m_rounds` / `routing.tile_m` / `routing.tile_rows`** module scalars
   — derived *independently* from `derivedParams.spatialMRounds` in
   `aiehlc.cc:4438-4461` (and the mesh-loop equivalent).

Copy #3 is the redundant "m_round logic" to remove for halo. Downstream,
`classifyTiling` (`passblueprinttoschedule.cpp:518`) and the host `scf.for` round
loop (`:2277`) read those scalars to size the loop.

### Reachability constraint

`#routing.tiling` is a routing-dialect attribute consumed and dropped by
`DmaphopTodfscheblueprintPass` (`passdmaphoptodfscheblueprint.cpp:567`). It does
**not** survive to the blueprint-to-schedule stage where `classifyTiling` / the
round loop run. At that stage halo info arrives via the `tensor_N.halo` dict.

Therefore "single source in `#routing.tiling`" is realized as:

> `#routing.tiling` is the **canonical in-IR descriptor**; the halo round count is
> carried to the blueprint stage via the operation-derived `tensor_N.halo` dict
> (populated *from* the tiling levels); the `derivedParams`→module-scalar halo
> path (copy #3) is deleted entirely.

## Round-count mapping

The halo M-round count (`aiehlc.cc:2242,2251`):

```
spatialMRounds = max(1, w_rounds) * max(1, l2_rounds)
tile_m         = tile_rows / spatialMRounds
```

The host `scf.for` needs **`w_rounds × l2_rounds`** (× `k_rounds` folds in via the
existing K-accum multiplier). Iteration decomposition
(`passdfscheduletoapi.cpp:3430,3442`): `hc = iv / w_rounds`, `wc = iv % w_rounds`
→ height/L2 is the **outer** digit, width the **inner**.

Where each round component lives today:

| Component | In `#routing.tiling`? | In `tensor_N.halo`? |
|-----------|----------------------|---------------------|
| `l2_rounds` (on-core row temporal) | yes — row dim nested `slice_tiling.rounds` (`routingmanager.cpp:924`) | yes — `l2_rounds` |
| `k_rounds` (K-accum) | yes — col dim `outer.rounds` (`:941`) | yes — `k_rounds` |
| `w_rounds` (on-core width temporal) | **no** | yes — `w_rounds` |

**Decision (full fidelity):** extend `#routing.tiling` to encode the width-split
level so it fully represents `l2 × w × k`, then derive `tensor_N.halo.m_rounds`
from the complete tiling.

## Design

### 1. W-level in `#routing.tiling` (`routingmanager.cpp:923-933`)

Extend the row (split) dim from 2 levels to 3, nesting width as the innermost
level (matches `iv = hc*w_rounds + wc`, hc outer, wc inner):

```
rowOuter (L1)              rounds=splitnum,  slice=haloSlice, step=haloStep
  slice_tiling (L2 height) rounds=l2_rounds, slice=l2_slice,  step=l2_step
    slice_tiling (W width)  rounds=w_rounds,  slice=w_slice,   step=w_step
```

Emitted only when `w_rounds > 1` (IR byte-identical otherwise). K-accum level
stays on the col dim. Requires plumbing `wSlice/wStep/wRounds` from
`split`/`shimDma` into `routingmanager` (they already reach the halo dict).

### 2. Derived carrier (`tilinglinalg_pipeline.cpp`)

After `createroutingfuncBySplitModel`, read back the `partitiontensor`'s
`#routing.tiling` and compute
`m_rounds = product of the row-dim temporal levels' rounds` (L2 × W). Add it as
`tensor_N.halo.m_rounds`. This makes `#routing.tiling` the source and the halo
dict the derived blueprint carrier. (Only for halo tensors, `m_rounds > 1`.)

### 3. Deletions (`aiehlc.cc`)

Remove the halo-only scalar emission at `:4438-4461`
(`routing.tile_m/tile_rows/m_rounds`) and any halo equivalent in the mesh loop
(`~:3846`, currently gated on `kRounds>1` so halo usually skips it — verify).
Matmul (`kRounds>1`) emission blocks are untouched.

### 4. Consumer edits (`passblueprinttoschedule.cpp`)

Add a helper `haloMRounds(moduleOp, argIdx)` returning `tensor_N.halo.m_rounds`.

- In `classifyTiling` and the round loop (`:2277`): when the flow's tensor is a
  halo tensor, use `haloMRounds` as `mRounds` instead of `tile_rows/tile_m`.
- Preserve exact current numeric behavior: reconcile the existing
  `haloL2RoundsForLoop` / `haloKRoundsForLoop` multipliers so the total equals
  `w_rounds × l2_rounds × k_rounds` with no double-count (today `classification.mRounds`
  already equals `w_rounds × l2_rounds`, then `:2282` multiplies l2 again — this
  reconciliation must be verified against golden IR).
- `passState.tileM` / `tileRows` for halo derive from `m_rounds` + tensor shape.

## Data flow (after change, halo path)

```
shimDma (conv geom) --> #routing.tiling on partitiontensor   [CANONICAL, l2 x w x k]
                              | (read back)
                              v
                     tensor_N.halo { slice, step, l2_*, w_*, k_*, m_rounds }  [derived carrier]
                              |
      DmaphopTodfscheblueprint / BlueprintToSchedule read halo dict
                              v
        classifyTiling / scf.for round loop  (mRounds = halo m_rounds)
```

`routing.m_rounds/tile_m/tile_rows` are no longer set on the halo path.

## Testing / verification

- **Unit** (`pass/unitest/test.cpp`): extend the conv2d spatial-halo test to
  assert the 3-level `#routing.tiling` (L1 -> L2 -> W) and
  `tensor_0.halo.m_rounds == w_rounds * l2_rounds`.
- **Golden IR**: regenerate `ir/dfschedule/*.mlir`; diff to confirm halo cases
  retain identical BD/round counts and matmul IR is byte-identical.
- **E2E**: run `example/tileprogram/ccode/simpleconv2d.cc` through
  `apppaltest.py`; verify `device_teardown done`, correct output, and unchanged
  host round-loop bounds vs. pre-refactor.

## Risks

- W-split modeled as a nested level on the **row/split dim** (the temporal-round
  carrier), not the geometric width axis. Consistent with how `mRounds` folds
  both splits, but reviewers should confirm the modeling choice.
- The consumer reconciliation (avoiding l2/w double-count) is the highest-risk
  edit; gate every change behind golden-IR diffs.
- Matmul must stay byte-identical — all halo branches are additive/guarded on the
  presence of `tensor_N.halo`.
