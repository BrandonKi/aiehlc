# Design: Composition-based Spatial Op Spaces (GemmSpace + Conv2dSpace)

## Status
Implemented. Refactors the "fat" `aie::SpatialPolicy` (which baked the GEMM
iteration space `tile_dim m/n/k` directly into the generic policy) into a
**composition** design: a lean generic `SpatialPolicy` plus per-op iteration
spaces (`GemmSpace`, `Conv2dSpace`) that *compose* (not inherit) the policy.

See also: [tile_dim_structured_design.md](tile_dim_structured_design.md) for the
`tile_dim {tile_size, stride, fullsize, pad_hi, pad_lo}` descriptor reused here
(`groups` is derived internally from `fullsize`).

## Motivation
The previous `SpatialPolicy` mixed two concerns:
- **Generic spatial behavior** (pattern, distribution, merge_order, pp_depth,
  buffer budget, layout transform, mode, coverage validation).
- **GEMM-specific iteration space** (`tile_dim m, n, k`).

This made the policy non-reusable: a conv2d port had no clean place to express
its `oh/ow/oc/ic/kh/kw` iteration space, and conv geometry leaked into the
separate `DmaTransform` struct. Composition separates the reusable policy from
the per-op iteration space so each op carries exactly the dims it needs.

## New types

```cpp
namespace aie {
// Reusable per-dimension descriptor (see tile_dim_structured_design).
// groups is derived internally from fullsize, not a user field.
struct tile_dim { int tile_size = 0; int stride = 0; int fullsize = 0; int pad_hi = 0; int pad_lo = 0; };

// 3-PART ORTHOGONAL generic policy — split into (1) array mapping,
// (2) materialization, (3) resource/pipeline. `mode` is GONE (Overlap is now
// DERIVED from the iteration-space split dim: stride < tile_size ⇒ Overlap).
enum class PadMaterialize { DDR = 0, Memtile = 1 };
enum class Im2col         { None = 0, Dma = 1 };
struct Bytes { int value = 0; };

struct SpatialMap {                          // (1) array mapping
    Pattern act         = Pattern::Broadcast;   // pattern for 1st input port (act)
    Pattern wgt         = Pattern::Broadcast;   // pattern for 2nd+ input port (wgt)
    Layout  layout      = Layout::Row;          // -> pti.distribution
    Flow    merge_order = Flow::Default;         // -> pti.mergeOrder (output Gather)
};
struct Materialize {                         // (2) materialization
    PadMaterialize pad    = PadMaterialize::DDR;     // parsed/metadata
    Im2col         im2col = Im2col::None;            // Dma -> pti.layoutTransform=1
};
struct Schedule {                            // (3) resource / pipeline
    int   pp_depth  = 2;                          // -> pti.pingPong
    Bytes l1_budget = Bytes{32*1024};            // .value -> pti.maxBufferBytes
};
struct SpatialPolicy {            // field order CONTRACTUAL for AST extraction
    SpatialMap  map;     // field 0
    Materialize mat;     // field 1
    Schedule    sched;   // field 2
};
// Pattern resolution is ROLE-AWARE (by input ordinal), not a single flat field:
//   output port      -> Pattern::Gather + map.merge_order
//   1st input  port  -> map.act
//   2nd+ input port  -> map.wgt
// The struct NAME stays `SpatialPolicy` so GemmSpace/Conv2dSpace field indices
// (`SpatialPolicy policy;` at field 0) and the `.policy = {...}` initializer are
// unchanged. require_full_coverage is no longer a user field (always true).

// GEMM iteration space — composes the policy.
struct GemmSpace {
    SpatialPolicy policy;
    tile_dim m;   // M sub-tiling   (legacy/global GEMM dim, also conv-halo)
    tile_dim n;   // N sub-tiling
    tile_dim k;   // K chunking
    tile_dim d1;  // per-port dim 1 (e.g. spatial-halo row split / H)
    tile_dim d2;  // per-port dim 2 (e.g. width / W)
    tile_dim d3;  // per-port dim 3 (e.g. channels / C) — bookkeeping only
};
// Per-port 2D/3D form: when d1/d2(/d3) are set the port describes its OWN
// operand shape (role-aware) instead of the global m/n/k path. For a
// spatial-halo input the three axes are folded into the existing flat slab:
//   raw_wc = d2.fullsize * d3.fullsize   (W * C)
//   raw_h  = d1.fullsize                 (full input H)
// d3 is *bookkeeping only* — no new 3D shim BD descriptor is generated, and
// an absent d3 is treated as the multiplicative identity (1) so the old 2D
// form (d2.fullsize = W*C, no d3) is unchanged. pad_lo/pad_hi on d1 are now
// FUNCTIONAL for the conv-halo derivation (boundary slab top/bottom pad).
//
// GemmSpace drives conv tiling: when d1 ALSO carries win/win_stride (the conv
// kernel window height + stride, tile_dim fields 5/6) the spatial-halo path
// derives the conv geometry itself — kernel_h = d1.win, stride = d1.win_stride,
// and oh_per_row = (d1.tile_size + d1.pad_lo - d1.win) / d1.win_stride + 1 —
// instead of a separate `Conv2dSpace_Spatial` type + `SP_*` macros. This makes
// the halo split (d1.tile_size input rows) and the kernel's per-tile output-row
// count self-consistent. See tile_dim_structured_design.md (`win`/`win_stride`).

// Conv2d iteration space — composes the policy + scalar stride/pad.
// Carries EXACT input spatial dims (ih/iw); output OH/OW are DERIVED forward.
struct Conv2dSpace {
    SpatialPolicy policy;
    tile_dim ih;  // input height  (exact)
    tile_dim iw;  // input width   (exact)
    tile_dim ic;  // input channels
    tile_dim oc;  // output channels (num filters)
    tile_dim kh;  // kernel height
    tile_dim kw;  // kernel width
    int stride = 1;
    int pad    = 0;
    tile_dim m;   // OPTIONAL explicit spatial-halo split (Overlap mode only)
};
}
```

### Optional explicit spatial-halo split (`Conv2dSpace.m`)
For Overlap (spatial-halo) convs the compiler normally derives the per-tile
input slab from `OH / HW_ROWS`:
`halo_slice = (OH/HW_ROWS - 1)*S + KH`, `halo_step = (OH/HW_ROWS)*S`. The
optional **field 9** `tile_dim m` lets the user *pin* this split directly,
bypassing the `OH/HW_ROWS` auto-derivation:

```cpp
.m = {.size = 61, .stride = 56, .groups = HW_ROWS}  // IFM 61 rows/tile, step 56, 5-row overlap
```

- `m.size`   = `halo_slice` — input rows per tile-row
- `m.stride` = `halo_step`  — input-row stride between tile-rows (`overlap = size - stride`)
- `m.groups` = number of tile-rows → `oh_per_row = OH / groups`

Semantics match the existing `tile_dim` overlap convention
(`tile_dim_structured_design.md`). When `m` is omitted (`m.size == 0`) the
auto-derivation is used unchanged (full backward compatibility). The explicit
split is stored via a new `ShimDma.haloExplicit` flag (and
`halo_slice`/`halo_step`/`oh_per_row`), **not** through `tdM` — routing it
through `tdM` would corrupt the conv's im2col-GEMM tiling math (`explicitTileM`,
working-set check, `spatialMRounds`, `validateDim`). The Overlap auto-block
skips its three computations when `haloExplicit` is set, and emits a coverage
warning if `(groups-1)*halo_step + halo_slice < IH`.

### Why input dims (ih/iw), not output dims (oh/ow)
An earlier draft encoded **output** dims and reconstructed the input via the
inverse formula `IW = (OW-1)*S + KW - 2*pad`. That reconstruction is **lossy for
strided convs**: integer division in the forward formula is non-invertible. For
the `simpleconv2d` example (`IW=224, S=2, KW=7, pad=3`) the forward gives
`OW = (224 + 6 - 7)/2 + 1 = 112`, but the inverse of `OW=112` yields `IW=223`,
corrupting the im2col row stride (`IW*C = 669` instead of the correct `672`).
Carrying the **exact** input dims and deriving the output forward is
value-preserving and matches what the `DmaTransform::im2col`/`spatial` factories
consume (they take exact input H/W directly).

Forward derivation used by the extractor:
```
OW = (IW + 2*pad - KW) / stride + 1
OH = (IH + 2*pad - KH) / stride + 1
```

### 3-part SpatialPolicy field order
The policy is now three nested sub-structs (m/n/k were removed earlier; `mode`
and `require_full_coverage` are no longer user fields). The authoritative field
order, used by the AST extractor (`readPolicy`), is:

```
field 0 map   { 0 act, 1 wgt, 2 layout, 3 merge_order }
field 1 mat   { 0 pad, 1 im2col }
field 2 sched { 0 pp_depth, 1 l1_budget{ 0 value } }
```

Mapping onto the internal `pti` fields:

| Source                                            | Internal `pti` field   |
|---------------------------------------------------|------------------------|
| `map.act` / `map.wgt` (by input ordinal) / Gather (output) | `pti.pattern`  |
| `map.layout`                                       | `pti.distribution`     |
| `map.merge_order`                                  | `pti.mergeOrder`       |
| `mat.im2col == Dma` ? 1 : 0                        | `pti.layoutTransform`  |
| `sched.pp_depth`                                   | `pti.pingPong`         |
| `sched.l1_budget.value`                            | `pti.maxBufferBytes`   |
| derived: split dim `stride < tile_size` ⇒ Overlap | `pti.tileMode`         |
| (constant)                                        | `pti.requireFullCoverage = true` |

## Key design decisions

### 1. `port` accepts any Space via `auto` NTTP
The port template changed from a typed `SpatialPolicy` NTTP to a generic `auto`:

```cpp
// before
template<typename T, SpatialPolicy P, DmaTransform D = DmaTransform::flat()>
struct port { using type = T; };
// after
template<typename T, auto Space, DmaTransform D = DmaTransform::flat()>
struct port { using type = T; };
```

Because `SpatialPolicy`, `GemmSpace`, and `Conv2dSpace` are all structural /
literal C++20 types, `auto` accepts a bare `SpatialPolicy` (legacy), a
`GemmSpace`, or a `Conv2dSpace` as the 2nd template argument. Attachment is
**per-port**: each `aie::port` carries its own Space.

### 2. AST extraction (bare Policy vs composed Space)
In `aiehlc.cc`, the Space shape is detected at parse time. **Important:** the old
`getStructField(0).isStruct()` heuristic no longer works — the 3-part bare
`SpatialPolicy` *also* has a struct at field 0 (`map`). Detection now uses the
**record type name** (`spaceTypeName`):

```cpp
// GemmSpace / Conv2dSpace -> composed; SpatialPolicy -> bare;
// fallback: nf >= 4 -> composed (bare Policy has exactly 3 fields).
```

- **composed** (`GemmSpace` / `Conv2dSpace`) → field 0 is a nested
  `SpatialPolicy` read via the `readPolicy` lambda, then disambiguate the op:
  - `GemmSpace`  → fields 1,2,3 = `m`, `n`, `k`; fields 4,5,6 = `d1`, `d2`,
    `d3` per-port 2D/3D dims (via `readTileDim`).
  - `Conv2dSpace` → fields 1..6 = `ih, iw, ic, oc, kh, kw`; fields 7,8 =
    scalar `stride`, `pad`. `ih.size`/`iw.size` are the **exact** input H/W;
    OH/OW are derived forward (see above).
  - Disambiguation uses the **record type name** (`spaceTypeName ==
    "Conv2dSpace"` / `"GemmSpace"`) with a field-count fallback (`>= 8` ⇒ conv;
    `GemmSpace` now has 7 fields, `Conv2dSpace` has 10).
- **bare** → the 2nd template arg is a `SpatialPolicy` directly (exactly 3
  fields: map/mat/sched). `readPolicy` runs on the whole APValue.

The `readPolicy` lambda walks the three nested sub-structs (map/mat/sched),
resolves `pti.pattern` role-aware (output ⇒ Gather; input ordinal 0 ⇒ `map.act`,
ordinal 1+ ⇒ `map.wgt`), and is shared by the composed (nested field-0 struct)
and bare paths. `pti.tileMode` (Overlap) is derived separately from the
iteration-space split dim (`stride < tile_size`), replacing the removed `mode`.

### 3. Conv2d dims as semantic source; DmaTransform precedence preserved
`Conv2dSpace` carries **exact input** dims (`ih`/`iw`) plus `ic`/`oc`, kernel
`kh`/`kw`, and scalar `stride`/`pad`. The output geometry is derived **forward**:

```
OW = (IW + 2*pad - KW) / stride + 1
OH = (IH + 2*pad - KH) / stride + 1
```

(Input dims are taken verbatim — no lossy reconstruction; see "Why input dims".)

When a `Conv2dSpace` is present **and** the explicit 3rd template-arg
`DmaTransform` is `flat()` (default), the shim DMA is derived from the
`Conv2dSpace` dims + `policy.mode`:

- **Partition mode** ⇒ im2col multi-dim BD (matches `DmaTransform::im2col`):
  `dims[0]={1, KW*C}`, `dims[1]={IW*C, KH}`, `dims[2]={S*C, OW}`,
  `num_dims=3`, `iter_step=IW*C*S`, `iter_wrap=OH`.
- **Overlap mode** ⇒ spatial-halo (matches `ConvTiling::spatial`):
  `oh_per_row = OH / meshRows`, `halo_slice=(oh_per_row-1)*S+KH`,
  `halo_step=oh_per_row*S`, `raw_h=IH`, `raw_wc=IW*C`.

If an explicit `DmaTransform` is given, **it wins** (single, unambiguous source
of truth). The derivation is guarded by `num_dims == 0 && mode == 0`, which is
true only when the explicit DmaTransform is flat — so existing
`DmaTransform::im2col(...)` / `ConvTiling::spatial(...)` examples keep working.

#### Flat-DmaTransform guard (defaulted-NTTP gotcha)
The `port` template's 3rd arg defaults to `DmaTransform::flat()`. Because of how
Clang records template args, the specialization **always** has a `targs[2]`
(the flat default) even when the user omits it. The naive DmaTransform extractor
read every `DmaTransform` field unconditionally, so for a Conv2dSpace port with
the defaulted flat transform it would **clobber** the geometry the Conv2dSpace
branch had just set (`kernel_h/input_c/stride → 0`), corrupting the later
derivation (observed: `halo_slice=27 raw_wc=0`).

Fix: detect a flat transform *before* applying it and skip the overwrite:
```cpp
int dmaModePeek = (numFields >= 10)
    ? (int)dmaApval->getStructField(4).getInt().getExtValue() : 0;
bool dmaIsFlat = (numDims == 0 && dmaModePeek == 0);
if (!dmaIsFlat && numFields >= 10) { /* fields 4-9 halo/raw   */ }
if (!dmaIsFlat && numFields >= 16) { /* fields 10-15 conv geom */ }
```
A flat transform now leaves the Conv2dSpace-derived geometry intact; a
**non-flat** explicit transform still overrides it (precedence preserved).

## Implementation map

The struct definitions are emitted **as a string in THREE places** that must
stay byte-for-byte in sync (same warning as the tile_dim design):

| Location | Role |
|----------|------|
| `src/llvm/aiehlc.cc` (Clang stub emit, ~L2368) | injected into user TU for parsing |
| `src/llvm/aiehlc.cc` (host append emit, ~L3012) | appended to generated `host.cc` |
| `src/mlir/.../tilinglinalg_pipeline.cpp` (host re-emit, ~L666) | pipeline host path |

Other touch points (all in `aiehlc.cc`):
- **ShimDma struct** (~L78): added `fromConvSpace`, `conv_ih`, `conv_iw`,
  `conv_oh`, `conv_oc`, `pad` for Conv2dSpace-derived geometry.
- **AST extraction** (~L1155): composed/legacy branch, `readPolicy`/`readTileDim`
  lambdas, `spaceTypeName` disambiguation. Conv2dSpace reads `ih,iw,ic,oc,kh,kw`
  (exact input) and derives OH/OW forward.
- **Flat-DmaTransform guard** (~L1314): `dmaIsFlat = (num_dims==0 && mode==0)`
  skips the DmaTransform field overwrite so a defaulted `flat()` 3rd arg doesn't
  clobber Conv2dSpace geometry; a non-flat explicit transform still wins.
- **Conv2dSpace derivation** (~L1406): im2col / spatial-halo population guarded by
  `fromConvSpace && num_dims==0 && mode==0`.

## What does NOT change
- MLIR dialect passes: `DerivedTilingParams` (tileM/tileN/effectiveK/
  convHaloSlice) remains the downstream contract; only the *source* of those
  values changes.
- Kernel-side builtins (`get_kernel_h`, `get_tile_rows`, `get_effective_k`, …)
  still read from `shimDma` / derived params.

## 3-part policy refactor (map / mat / sched)
The flat `SpatialPolicy` (`pattern/distribution/merge_order/pp_depth/
max_buffer_bytes/layout_transform/mode/require_full_coverage`) was replaced by
the three orthogonal sub-structs documented in **New types** above. Notes:
- `mode` was **removed**: Overlap is now derived from the iteration-space split
  dim (`stride < tile_size`) → `pti.tileMode`.
- `require_full_coverage` is no longer a user field (always `true`).
- Pattern is **role-aware**: `map.act` (1st input), `map.wgt` (2nd+ input),
  `Gather` + `map.merge_order` (output).
- `simplematmul2.cc` migrated to the new `.policy = {.map=..., .mat=..., .sched=...}`
  form. `l1_budget` kept at `Bytes{4096}` to preserve the prior
  `max_buffer_bytes`. Effective per-port behavior unchanged: win_a Broadcast(Row),
  win_b Broadcast(Col), win_c Gather(LeftToRight).
- **Follow-up (not yet migrated):** `simplematmul.cc` and `simpleconv2d.cc` still
  use the OLD flat policy and will not compile with the new struct. They only
  build when selected as `--runtime-source-file`, so they don't block the
  `simplematmul2.cc` flow.

### Validation (3-part refactor)
1. `make -C build aiehlc` clean build. ✔ EXIT=0
2. End-to-end generate + HW run on `simplematmul2.cc`. Policy resolved identical
   to pre-refactor: win_a `pattern=0 distribution=0 mergeOrder=0 ppDepth=2
   maxBufferBytes=4096 mode=0`; win_b `pattern=0 distribution=1 ...`; win_c
   `pattern=3 distribution=0 mergeOrder=1 ...`. ✔
3. IR (`ir/dfschedule/0_initial.mlir`): `pp_depth_map = {2,2,2}`, k_rounds=4,
   m/n_rounds=16, tile_m/n=16, effective_k=64. ✔
4. HW run: `PASS: all 65536 elements match.`, AIE ERROR count=0. ✔

## Migration (composition refactor — historical)
- `simplematmul2.cc` → migrated to `GemmSpace` (`.policy = {...}`, `.m/.n/.k`).
- `simpleconv2d.cc` → migrated to `Conv2dSpace`: two constants `RowBC_im2col`
  (`.policy.mode = Partition`) and `RowBC_spatial` (`.policy.mode = Overlap`),
  each carrying `.ih/.iw/.ic/.oc/.kh/.kw` + `.stride/.pad`. The `conv2d_im2col`
  and `conv2d_spatial` `win_a` ports drop their explicit `DmaTransform::im2col`/
  `ConvTiling::spatial` 3rd arg (now derived). `ColBC`/`LtoR_Merge` (filter,
  output) stay bare `SpatialPolicy` — only the conv input carries a Conv2dSpace.
- `simplematmul.cc` → **kept** on bare `SpatialPolicy` without m/n/k (proves the
  lean-bare legacy path).

### Update: `RowBC_spatial` is now a lean `GemmSpace` (supersedes `Conv2dSpace_Spatial`)
`simpleconv2d.cc`'s spatial-halo input `RowBC_spatial` was converted from
`Conv2dSpace_Spatial` to a generic `aie::GemmSpace`. Its `d1` carries the
HEIGHT-halo split (`tile_size=18, stride=13, fullsize=224, pad_lo/pad_hi=3`) AND
the conv window (`win=SP_KH=7, win_stride=SP_S=2`); `d2` carries `W*C=896`. The
GemmSpace spatial-halo path derives `kernel_h/stride/oh_per_row` from these
fields, so no `Conv2dSpace_Spatial` is needed for the spatial path. The kernel's
`SP_OHR` is now `(TILE_H + PAD_H_LO - SP_KH)/SP_S + 1 = 8` (was a hard-coded
`28 = OUTPUT_H/HW_ROWS`), fixing the slab-supply vs kernel-demand mismatch that
caused the board hang. `OH_T`/`OW_T` were changed from `constexpr int` to
`#define` so the kernel TU (which carries only `#define`s) sees them via
`TILE_H`/`SP_OHR`. Verified: `GemmSpace spatial-halo: halo_slice=18 halo_step=13
raw_h=224 raw_wc=896` + `GemmSpace conv-halo: kernel_h=7 stride=2 oh_per_row=8`;
IR `memref<224x896xi8>` with `partitiontensor {overlap=5, slice_size=18,
step=13}`.

## Validation (performed)
1. `make -j` clean build of `aiehlc`. ✔ EXIT=0
2. Frontend resolve on migrated `simplematmul2.cc`:
   `Policy resolved: ... m{size=16,stride=16} n{size=16,stride=16}
   k{size=64,stride=64} mode=0`, `tileM=16 tileN=16 effectiveK=64 kRounds=4`,
   "Host built successfully", 0 errors. ✔
3. Frontend resolve on migrated `simpleconv2d.cc` (`Conv2dSpace`, Overlap mode):
   `conv2d{KH=7,KW=7,C=3,S=2,P=3,OH=112,OW=112,IH=224,IW=224} mode=1`,
   `Conv2dSpace spatial-halo derived: halo_slice=61 halo_step=56 raw_h=224
   raw_wc=672 oh_per_row=28` (exact match to the previous explicit-DmaTransform
   output), "Host built successfully". ✔
4. Legacy `simplematmul.cc` (bare lean policy, no m/n/k) resolves with defaults
   `tileM=64 tileN=64 effectiveK=256`, "Host built successfully". ✔
5. No-regression re-check after the flat-DmaTransform guard: `simplematmul2.cc`
   and `simplematmul.cc` both still resolve identically and build the host. ✔
