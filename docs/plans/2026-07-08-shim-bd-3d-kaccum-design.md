# Prefer 3D D0/D1/D2 shim BD for conv2d halo K-accum

Date: 2026-07-08

## Scope

- Producer-side representation change only: make the conv2d spatial-halo shim BD
  use all of D0/D1/D2 (a real 3D BD) instead of falling back to iteration+repeat.
- Applies to the conv2d spatial-halo **K-accum path only** (`tensor_N.halo.k_rounds > 1`).
- Does NOT change the runtime channel repeat (repeat = l2_rounds stays) and does
  not attempt to resolve the mm2s1 stall — that is explicitly out of scope.

## Root cause

In `src/mlir/mlirfront/tilinglinalg/pass/passdmaphoptodfscheblueprint/passdmaphoptodfscheblueprint.cpp`
the shim BD dimension builders are a chain of branches, each gated on
`!inputShimDimStrides` (first match wins):

| order | branch                | line | gate         | output                       |
|-------|-----------------------|------|--------------|------------------------------|
| 1     | 2D width-split halo   | 1078 | `w_rounds>1` | 2D `[4,920]/[61,19]`         |
| 2     | 3D K-accum halo       | 1128 | `k_rounds>1` | 3D `[4,920,224]/[61,19,4]`   |

For the failing conv both `w_rounds=4` and `k_rounds=4`, so the 2D width-split
branch fires first and blocks the 3D K-accum branch. The BD then only carries
D0/D1 and the round count is pushed into iteration + channel repeat downstream.

The downstream pass `passblueprinttoschedule.cpp` already assumes the 3D K-accum
representation: `kAccumHaloSlab` is derived purely from
`tensor_N.halo.k_rounds > 1` (line ~944), and the repeat logic (line ~2822-2828)
already expects "one multi-dim BD (D2 = k_rounds) + channel repeat = l2_rounds".
So the downstream is consistent with 3D; only the producer emits the wrong (2D)
shape.

## Fix

Give the 3D K-accum branch priority over the 2D width-split branch when
`k_rounds > 1`. Add a guard to the width-split branch so it does not fire when
the K-accum condition also holds:

```cpp
// width-split branch (~line 1082)
auto kRoundsGuard = haloAttr ? haloAttr.getAs<IntegerAttr>("k_rounds") : nullptr;
bool kAccumWins = kRoundsGuard && kRoundsGuard.getInt() > 1;
if (wRoundsAttr && wRoundsAttr.getInt() > 1 && !kAccumWins) {
    ... existing 2D width-split body ...
}
```

The 3D K-accum branch then fires and sets `strides=[4,920,224]`,
`wraps=[61,19,4]`.

## Why minimal / safe

- Purely producer-side (scope: "3D BD default only").
- Only affects `k_rounds>1` halo tensors (scope: "halo K-accum only"); pure
  width-split (`w_rounds>1, k_rounds<=1`) is untouched — no regression there.
- Downstream needs no change; it already expects the 3D BD for `kAccumHaloSlab`.

## Verification

1. Build the pass incrementally.
2. Re-run generation; `ir/dfschedule/5_DmaphopTodfscheblueprintPass.mlir`
   `flow_src_4` should show `shim_dim_strides=[4,920,224]`,
   `shim_dim_wraps=[61,19,4]`.
3. Log `[ShimMultiDim] Push input (3D halo K-accum)` appears and
   `(2D halo width-split)` does NOT for tensor_0.
4. A `w_rounds>1, k_rounds<=1` case still hits the 2D width-split branch.
