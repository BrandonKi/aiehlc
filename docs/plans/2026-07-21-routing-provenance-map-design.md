# Routing Provenance Map — Design

Date: 2026-07-21
Status: Implemented

> Implementation note: the design was refined once the real IR was inspected.
> The `routinghw.*` ops only exist *before* `RoutingHWLowerPass` (that pass
> lowers them into `emitc.call`), and they are grouped inside
> `routing.RoutingCreate` regions. The pass therefore runs **after
> `RoutingHWVerifyPass` and before `RoutingHWLowerPass`**, and the JSON is
> organized as `routing_groups[]` (one per `RoutingCreate`, tagged by
> `memo`+`scf_idx`) each holding `tiles[]` and an ordered `connections[]`.

## Problem

The TilingLinalg pipeline already emits provenance JSON for two stages:

- `DmaphopProvenanceMapPass` → `dmaphopprovenacemap.json` — **logical** communication
  paths (producers/consumers, hop chains, partition info).
- `DfscheduleProvenanceMapPass` → `dfscheduleprovenancemap.json` — **low-level**
  schedule (DMA BDs, locks) for the host path.

There is no provenance for the **physical routing** — the stream-switch port
connections and packet-flow configuration that Phase 5 lowers into `routing.cc`.
When a hop is stuck on hardware, the debug tools (`aiediag`/`aiegdb`) can describe
the logical hop but cannot map it back to the concrete stream-switch master/slave
ports and packet IDs that were programmed.

## Goal

Add a new MLIR pass that walks the Phase 5 `routinghw` IR (the module that becomes
`routing.cc`) and emits `routingprovenancemap.json`: a physical, port-level record of
the routing decisions, keyed by resolved `(col,row)` tiles.

**In scope:** the new pass + pipeline wiring + build wiring.
**Out of scope:** any `aiediag.py`/`aiegdb.py` consumption logic (follow-up).

## Source IR

Phase 5 in `tilinglinalg_pipeline.cpp` runs on `routingDmaphopModule`:

```
DmaphopToRoutinghwPass → RoutingHWVerifyPass → RoutingHWLowerPass
→ RoutingDeadArgPass → RoutingConstantFoldPass → Canonicalizer → EmitC → routing.cc
```

The pass reads the module **right after `RoutingHWLowerPass`**, when all `routinghw`
ops are materialized but before the routing-op cleanup and EmitC translation.

Relevant `routinghw` ops (from `routinghw/td/routinghwop.td`):

| Op | Key attributes | Result |
|----|----------------|--------|
| `tilecreate` | `row`, `col`, `comments` | `i32` tile handle |
| `ioshimtilecreate` | `row`, `col`, `IOID`, `dmadirection`, `channelused` | `i32` handle |
| `createshimstreamswitchport` | `shimile` (operand), `shimmasterport`, `shimmasterportidx`, `shimmasterporttype` | `i32` |
| `connectstreamswitchport` | `srctile`/`dsttile` (operands), src/dst master/slave port dir+idx | `i32` |
| `connectsinglestreamswitchport` | `curtile` (operand), slave dir+idx, master dir+idx | `i32` |
| `connectpktstreamswitchport` | `curtile` (operand), recv-slave (dir/idx/pktid/pkttype), local-dma (dir/idx/pktid/pkttype), forward-master (dir/idx), `preserveheader` | `i32` |
| `enableexttoaieshimport` / `enableaietoextshimport` | `curtile` (operand), `portdirection`, `portidx` | `i32` |

**Core mechanic:** tile identity is carried as an `i32` SSA handle. Each `connect*`/
`enable*`/`createshim*` op operand (`$curtile`/`$srctile`/`$dsttile`/`$shimile`) is
traced to its defining `tilecreate`/`ioshimtilecreate` op to resolve `(col,row)`.

## JSON shape (as implemented)

```json
{
  "version": 1,
  "startcol": <int>,        // absolute physical start column (phys_col = col + startcol); omitted if -1
  "aie_gen": "<raw --aie-version>",
  "module_attrs": { "tile_m": .., "tile_n": .., "tile_rows": .., "tile_cols": ..,
                    "effective_k": .., "full_k": .., "k_rounds": .., "m_rounds": .., "n_rounds": .. },
  "routing_groups": [
    {
      "id": "group_0",
      "memo": "col",            // "col" | "row": which mesh axis this split serves
      "scf_idx": 0,             // the split index
      "tiles": [
        { "col": 0, "row": 3, "type": "core", "comments": "core_tile" },
        { "col": 0, "row": 0, "type": "shim", "comments": "shim_dma_13",
          "ioid": 13, "dma_direction": 0, "channel_used": 0 }
      ],
      "connections": [
        { "kind": "shim_ext_to_aie", "tile": {"col":6,"row":0,"type":"shim"},
          "port": {"dir":"SOUTH","idx":7} },
        { "kind": "circuit_connect", "tile": {"col":6,"row":0,"type":"shim"},
          "slave": {"dir":"SOUTH","idx":7}, "master": {"dir":"NORTH","idx":1} },
        { "kind": "packet_connect", "tile": {"col":0,"row":4,"type":"core"},
          "recv_slave": {"dir":"NONE","idx":0,"pktid":0,"pkttype":0},
          "local_dma":  {"dir":"DMA","idx":0,"pktid":5,"pkttype":0},
          "forward_master": {"dir":"EAST","idx":0}, "preserve_header": true },
        { "kind": "shim_aie_to_ext", "tile": {"col":3,"row":0,"type":"shim"},
          "port": {"dir":"NORTH","idx":3} }
      ]
    }
  ]
}
```

- `routing_groups[]`: one entry per `routing.RoutingCreate` (a logical data-movement
  group), tagged by `memo` (mesh axis) and `scf_idx` (split index).
- `tiles[]`: every physical tile declared in the group, deduped by `(col,row,type)`.
  `type` is `"shim"` for `ioshimtilecreate` (adds `ioid`/`dma_direction`/`channel_used`),
  `"core"` for `tilecreate`.
- `connections[]`: the **ordered** stream-switch program for the group. `kind` is one of:
  - `circuit_connect` — `connectsinglestreamswitchport`: `slave`+`master` {dir,idx}.
  - `packet_connect` — `connectpktstreamswitchport`: `recv_slave`/`local_dma`
    {dir,idx,pktid,pkttype}, `forward_master` {dir,idx}, `preserve_header`.
  - `shim_ext_to_aie` / `shim_aie_to_ext` — the shim external-port enables: `port` {dir,idx}.
  - `circuit_connect_pair` — `connectstreamswitchport` (multi-tile): `src_tile`/`dst_tile`
    with their slave/master ports.
  - `shim_stream_switch_port` — `createshimstreamswitchport`: `shim_master` {port,idx,type}.

Every connection's `tile` is resolved from the op's tile-handle operand back to the
defining `tilecreate`/`ioshimtilecreate`. The ordered per-group connection list plus
the tile list is sufficient to fully reconstruct the physical routing. The
`module_attrs` block mirrors the two existing passes for cross-file consistency.

## Components

- **`pass/passroutingprovenancemap/passroutingprovenancemap.h`**
  `class RoutingProvenanceMapPass : public PassWrapper<RoutingProvenanceMapPass, OperationPass<>>`
  with ctors `()`, `(outputDir)`, `(outputDir, startCol)`, `(outputDir, startCol, aieGen)`;
  `getArgument()="routing-provenance-map"`; `getDependentDialects` inserts
  `routinghw`, `routing`, `func`, `arith`.
- **`pass/passroutingprovenancemap/passroutingprovenancemap.cpp`**
  Local `JsonWriter` (same pattern as the other two passes — they each keep their own
  copy, no shared header). Helpers: `resolveTile(i32Value) → TileInfo` (walks the
  defining op), per-op processors, `runOnOperation()` that walks the module and writes
  JSON to `<outputDir>/routingprovenancemap.json` (fallback to cwd on dir failure,
  matching the existing passes).

## Data flow

```
routingDmaphopModule (post RoutingHWLowerPass)
   └─ walk tilecreate / ioshimtilecreate  → physical_tiles[]
   └─ walk connect*/enable*/createshim*    → resolveTile(operand) → flows[]
   └─ JsonWriter → routingprovenancemap.json
```

## Pipeline wiring

In `tilinglinalg_pipeline.cpp`, Phase 5, immediately after `RoutingHWVerifyPass`
and **before** `RoutingHWLowerPass` (which lowers `routinghw.*` into `emitc.call`):

```cpp
{
    auto routingProvenancePass =
        std::make_unique<RoutingProvenanceMapPass>(outputDir, partStartCol, aieGen);
    runPipelineSinglePass(ctx, routingDmaphopModule, std::move(routingProvenancePass),
                          routingIrDir, rstage, "RoutingProvenanceMapPass");
}
```

Include `passroutingprovenancemap.h` alongside the other provenance includes (top of file).

## Build wiring

Add the new source + include dir to the three CMake locations that already list the
two existing provenance passes:

- `src/mlir/mlirfront/CMakeLists.txt` — `list(APPEND SOURCE_LIB_FILES ...)` + `include_directories(...)`.
- `src/mlir/mlirfront/tilinglinalg/pass/unitest/CMakeLists.txt` — source + include.
- `src/mlir/mlirfront/frontend/aietriton/CMakeLists.txt` — include dir.

## Error handling

- If the top op is not a `ModuleOp`: emit error + `signalPassFailure()` (as in the
  existing passes).
- If `outputDir` create fails: warn to `errs()` and fall back to a cwd-relative path
  (non-fatal), matching `DmaphopProvenanceMapPass`.
- If a tile handle cannot be resolved (no defining `tilecreate`): record `col=-1,row=-1`
  so the entry is still emitted and visibly flagged, never crash.
- The pass is non-fatal in the pipeline (like the dmaphop/dfschedule provenance calls,
  its return value is not checked to gate the pipeline).

## Testing

- Build the tilinglinalg unitest target.
- Run the standard generation flow
  (`source script/aiehlc.sh --aie-version 5 --runtime-source-file ./example/tileprogram/ccode/simplematmul.cc`)
  and confirm `routingprovenancemap.json` is produced next to `routing.cc`, is valid
  JSON, and that resolved tiles/ports match the `connect*` calls emitted in `routing.cc`.

## Documentation & follow-up

- Update the architecture docs per the repo Document rule (note the new provenance file
  in the debug-tools / provenance discussion).
- Follow-up (separate task): teach `aiediag.py` to locate and use
  `routingprovenancemap.json` for a physical-port debug step.
