# Design: Structured `tile_dim` (size/stride/groups) in `SpatialPolicy`

## Status
Implemented. Replaces the flat scalar `tile_m / tile_n / tile_k` hints in
`aie::SpatialPolicy` with a structured per-dimension descriptor `tile_dim`, and
unifies the conv halo/overlap split into the same descriptor.

> **Superseded layout:** the `tile_dim m/n/k` fields shown below as members of
> `SpatialPolicy` have since been moved out into per-op iteration spaces
> (`GemmSpace`, `Conv2dSpace`) that *compose* a now-lean `SpatialPolicy`. The
> `tile_dim` descriptor itself is unchanged. See
> [spatial_space_composition.md](spatial_space_composition.md).

## Motivation
`SpatialPolicy` described tiling as bare partition counts (`tile_m`, `tile_n`,
`tile_k`) with stride implicitly equal to size. Conv overlap/stride lived in a
*separate* `DmaTransform` struct (`halo_slice` / `halo_step`, `mode == 1`). This
split the description of *how a dimension is divided* across two unrelated types.

`tile_dim` makes each dimension self-describing and lets the same field express
both exact GEMM partitioning and overlapping conv halos.

## New types

```cpp
namespace aie {
enum class TileMode { Partition = 0, Overlap = 1 };

// Full description of how ONE dimension is split.
// Current user-facing layout:
//   {tile_size, stride, fullsize, pad_hi, pad_lo, win, win_stride}.
// `groups` is NOT a user field; it is derived internally from `fullsize`
// (groups = ceil((fullsize - tile_size) / stride) + 1) so downstream code
// keeps working unchanged.
struct tile_dim {
    int tile_size  = 0;  // per-tile covered length (incl. halo), e.g. 61
    int stride     = 0;  // start spacing between adjacent tiles, e.g. 56; overlap = tile_size - stride
    int fullsize   = 0;  // total dim length (replaces user-facing groups)
    int pad_hi     = 0;  // optional padding (default 0)
    int pad_lo     = 0;  // optional padding (default 0)
    int win        = 0;  // NEW (field 5): conv kernel window length (height) — drives conv-halo derivation
    int win_stride = 0;  // NEW (field 6): conv stride — drives conv-halo derivation
};

struct SpatialPolicy {
    Pattern  pattern      = Pattern::Broadcast;
    Layout   distribution = Layout::Row;
    Flow     merge_order  = Flow::Default;
    int      pp_depth     = 2;
    int      max_buffer_bytes = 4096;
    tile_dim m;                                   // replaces tile_m
    tile_dim n;                                   // replaces tile_n
    tile_dim k;                                   // replaces tile_k
    LayoutTransform layout_transform = LayoutTransform::None;
    TileMode mode = TileMode::Partition;          // intent annotation + validation switch
    bool     require_full_coverage = true;        // validation: union of tiles must cover the dim
};
}
```

## Semantics
- **Partition** (`stride == size`, `overlap == 0`): exact tiling, existing GEMM
  behavior. `m = {size=16, stride=16}` ≡ old `tile_m=16`.
- **Overlap** (`stride < size`, `overlap = size - stride`): replaces conv halo.
  `m = {size=61, stride=56, groups=4}` ≡ old `halo_slice=61, halo_step=56`,
  `HW_ROWS=4`.
- **`groups` derivation** when 0:
  `groups = (total - size + stride - 1) / stride + 1` (ceil).
- **`require_full_coverage`** validates `(groups-1)*stride + size >= total`;
  for Partition it additionally requires exact coverage `== total`.

### What "total" means per mode
- **Partition `m`/`n`**: temporal sub-tiling *inside a core*.
  `total = tileRows = M/HW_ROWS` (resp. `tileCols`). `groups = tileRows/size`.
- **Overlap `m`**: spatial halo split *across mesh rows* over the raw input.
  `total = raw_h`, `groups = HW_ROWS`.
- **`k`**: always temporal, `total = K`, `groups = K/size`.

## Implementation map

All changes are confined to the frontend + the pipeline's struct re-emit; the
MLIR dialect passes are untouched because `DerivedTilingParams`
(`tilinglinalg_pipeline.h`) keeps `tileM/tileN/effectiveK/convHaloSlice` — only
the *source* of those values changed.

| Site | File:area | Change |
|------|-----------|--------|
| ParsedTensorInfo | `src/llvm/aiehlc.cc` (~53) | `int tileM/N/K` → `TileDim tdM/tdN/tdK` + `tileMode` + `requireFullCoverage` |
| AST extraction | `src/llvm/aiehlc.cc` (~1068) | nested `tile_dim` read via `readTileDim` lambda; field order `0..10` (m/n/k at 5/6/7, mode at 9, require_full_coverage at 10) |
| Derive + validation | `src/llvm/aiehlc.cc` (~1320) | explicit hints from `td*.size`; `validateDim()` coverage checks; rounds prefer explicit `groups` |
| Overlap → halo | `src/llvm/aiehlc.cc` (~1285) | `mode==Overlap` drives `shimDma.{mode,halo_slice,halo_step}` from `tdM` |
| Struct emit (EmitC stub) | `src/llvm/aiehlc.cc` (~2376) | emit `TileMode`, `tile_dim`, new `SpatialPolicy` |
| Struct emit (host append) | `src/llvm/aiehlc.cc` (~2999) | identical |
| Struct emit (pipeline host) | `src/mlir/.../tilinglinalg_pipeline.cpp` (~660) | identical |
| Example | `example/tileprogram/ccode/simplematmul2.cc` | policies rewritten to `m/n/k` + `mode` |

> Note: the struct is emitted as a **string in three places** (Clang-phase stub,
> host append in `aiehlc.cc`, and the pipeline host re-emit). All three must stay
> in sync or the generated `host.cc` won't compile the user `constexpr`
> definitions.

## Validation example
```cpp
.m = {.size = 16, .stride = 16, .groups = 3}   // on a 64-row tile
```
emits:
```
[TilingLinalg] ERROR: dim m Partition coverage 48 != total 64 (size=16 stride=16 groups=3).
```

## Verification performed
1. `make -j` — clean build.
2. Frontend on `simplematmul2.cc`: `Policy resolved` prints
   `m/n/k {size,stride,groups}` and `mode`; `Two-level tiling: tileM=16 tileN=16
   effectiveK=64 kRounds=4 spatialMRounds=4 spatialNRounds=4`.
3. Generated `host.cc` (both struct copies) compiles → "Host built successfully".
4. Coverage validation confirmed firing on `groups=3` under-coverage.

## Migration notes
- `simplematmul.cc` / `simpleconv2d.cc` use only default `m/n/k` (auto-derive),
  so they are source-compatible with the new struct (designated initializers in
  declaration order).
- Future: `simpleconv2d.cc` may express its halo declaratively via
  `.m = {.size=halo_slice, .stride=halo_step, .groups=HW_ROWS}, .mode =
  aie::TileMode::Overlap` instead of `ConvTiling::spatial`.

## `win` / `win_stride`: conv-halo geometry derived from the descriptor

`tile_dim` gained two fields (positional indices 5 and 6) so the **GemmSpace
spatial-halo path derives the conv tiling itself** instead of relying on a
separate `Conv2dSpace_Spatial` type + out-of-band `SP_*` macros:

| field | meaning |
|-------|---------|
| `win`        | conv kernel window height (e.g. `SP_KH = 7`) |
| `win_stride` | conv stride (e.g. `SP_S = 2`) |

When `d1.win > 0`, the spatial-halo path (`aiehlc.cc`) computes the per-tile
output-row count produced by a `tile_size`-row slab (the first/top tile sees
`pad_lo`):

```
oh_per_row = (tile_size + pad_lo - win) / win_stride + 1
```

and lifts `kernel_h = win`, `stride = win_stride`, `oh_per_row` into `shimDma`
(with `haloExplicit = true`), emitting:

```
[TilingLinalg] GemmSpace conv-halo: kernel_h=7 stride=2 oh_per_row=8
```

This makes the halo split (`tile_size` input rows) and the kernel's per-tile
output-row count **self-consistent**: with `tile_size=18, pad_lo=3, win=7,
win_stride=2` → `oh_per_row = 8`, matching `OH_T`. Previously the kernel used a
hard-coded `SP_OHR=28` (= `OUTPUT_H/HW_ROWS`) while a slab of 18 input rows can
only yield 8 output rows, causing A-input starvation and the board hang
(`DmaGetPendingBdCount Invalid start queue size`). `SP_OHR` is now defined as
`(TILE_H + PAD_H_LO - SP_KH)/SP_S + 1` so kernel demand == slab supply.

> **Field-order contract:** the `tile_dim` prelude is emitted as a C++ string in
> THREE places (`aiehlc.cc` ×2, `tilinglinalg_pipeline.cpp` ×1) and read
> positionally. `win`/`win_stride` were appended AFTER `pad_lo`, so existing
> indices 0–4 are unchanged. The 3 copies MUST stay byte-identical.
