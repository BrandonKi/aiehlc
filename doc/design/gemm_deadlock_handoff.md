# GEMM 4×4 DMA Deadlock — Debug Handoff Document

**Branch:** `GEMM`  
**Target file:** `aout/worklocal/host.cc`  
**Board:** portobello13 (vek385-27)  
**Test command:** `python3 script/test/appvek385.py -y -nonreboot aout/worklocal/build/host`  
**Build command:** `bash script/hostcompile.sh`

---

## 1. Problem Statement

The `exp61` GEMM benchmark (`example/tileprogram/ccode/simplematmul2_oob_compare.cc`) targets a **4×4 AIE2PS compute mesh** (cols 0–3, rows 3–6) with shim tiles at row 0. Every board run ends with `wait_io TIMEOUT` on all DMA channels — the design deadlocks and never produces output.

The symptom is always `pending=1` (3 of 4 BD runs complete, 1 stuck) across all B-shim MM2S channels and A-shim MM2S channels.

---

## 2. Architecture

### Tile layout
```
Row 0:  Shim tiles  col 0–3   (DMA to/from DDR)
Row 1:  MemTiles    col 0–3   (inter-tile cache)
Row 2:  MemTiles    col 0–3
Row 3–6: Compute tiles  col 0–3  (16 tiles total)
```

### Data flow per GEMM outer loop iteration
- **A shim (ch=1 MM2S):** col 0 → row 3, col 1 → row 4, col 2 → row 5, col 3 → row 6  
  Routing: **packet-switched** (`XAie_StrmPktSw*`) — independent per column  
  Each startio: `repeat=4`, `len=16384`, 3-dim BD with `d2_wrap=4`

- **B shim (ch=0 MM2S):** col 0 broadcasts to rows 3, 4, 5, 6; same for cols 1–3  
  Routing: **circuit-switched** (`XAie_StrmConnCctEnable`) — fan-out chain  
  Each startio: `repeat=4`, `iter_wrap=4`, `iter_step_size=16384`  
  Fan-out path per col: shim `SOUTH,3→NORTH,0` → at each row: `SOUTH,0→{NORTH,0, DMA,1}`

- **C shim (ch=0/1 S2MM):** col 2–3 collect output from compute tiles  
  Uses out-of-order (OOO) BDs with packet routing

### Kernel demand per tile per outer loop iteration
```
matmul.cc: for(mr=0..15) { for(kr=0..3) { acq(win_a); acq(win_b); } write(win_c); }
→ 64 win_a + 64 win_b + 16 win_c per tile per outer loop iter
```

### win_b buffer layout per compute tile (local memory)
```
BD0: 0x8000 (32768)   ping  4096B → next_bd=1
BD1: 0x9000 (36864)   pong  4096B → next_bd=6   [changed from 0]
BD6: 0xC800 (51200)   new   4096B → next_bd=7   [added]
BD7: 0xD800 (55296)   new   4096B → next_bd=0   [added]
lock2 init=4 (producer credit), lock3 init=0 (consumer credit)
```

### win_a buffer layout per compute tile
```
BD2: 0xA000 (40960)   ping  4096B → next_bd=3
BD3: 0xB000 (45056)   pong  4096B → next_bd=2
lock0 init=2 (producer credit), lock1 init=0 (consumer credit)
```

### win_c buffer layout per compute tile
```
BD4: 0xC000 (49152)   ping  1024B → next_bd=5
BD5: 0xC400 (50176)   pong  1024B → next_bd=4
lock4 init=0 (producer/consumer for output)
```

---

## 3. Root Cause Analysis

### The circuit-switched multicast deadlock

B shim routing is **circuit-switched fan-out**:
```
SOUTH,3(shim) → NORTH,0(shim) → NORTH,0(row3) →
  ├─ DMA,1 (win_b S2MM at row3)
  └─ NORTH,0(row4) →
       ├─ DMA,1 (win_b S2MM at row4)
       └─ NORTH,0(row5) → ... → NORTH,0(row6) → DMA,1
```

**The SOUTH,0 port at each compute row fans out to BOTH DMA,1 AND NORTH,0 simultaneously.** This is a shared circuit-switched connection — if either destination applies backpressure, the entire column stalls.

**Deadlock condition:** When all 4 rows' win_b ping-pong is full (lock2=0), B shim cannot send any more data. But kernels can only start consuming win_b once they have win_a too. If win_a arrives after win_b fills up, kernels stall waiting for win_a while B shim stalls waiting for lock2 credits — and the SOUTH,0→NORTH,0 path blocks A shim data at the memtile hops (since the stream switch is shared for all NS traffic).

### Timeline of deadlock with ping-pong (lock2=2):

```
t=0: B shim starts broadcasting. A shim starts (packet-switched, through memtile, slower).
t=1: B shim delivers win_b[0] to all rows. lock2: 2→1
t=2: B shim delivers win_b[1] to all rows. lock2: 1→0
t=3: B shim tries win_b[2]. lock2=0 at all rows → S2MM stalls.
     SOUTH,0 at row3 is blocked (DMA,1 stalled).
     NORTH,0 at row3 also blocked (shared circuit connection).
     A shim data can't pass through row3 NORTH chain.
     Kernels at all rows waiting for win_a → can't drain win_b.
     DEADLOCK.
```

### Why quad-buffering (lock2=4) is still deadlocking

With lock2=4, the B shim can deliver 4 windows before stalling. But `iter_wrap=4` means each shim BD run delivers exactly 4 windows — then the shim issues the next BD run for the next outer loop iteration immediately (no blocking on the shim side). Meanwhile the kernel still hasn't gotten win_a. So at the 5th window (2nd BD run), lock2=0 again and the same deadlock occurs.

**The real fix needed:** The B shim must be **rate-limited to match kernel consumption** — i.e., the shim should not be able to get ahead by more than the kernel can drain. This requires either:

1. **Lock-gated B shim** — add a shim-side lock that only allows the shim to send after the kernel has released win_b. This requires changes to the shim BD `acquire_lock_id` field (currently 0, meaning no acquire).

2. **Reduce B shim iter_wrap to 1** — send 1 window per startio, and sequence: send 1 win_b, send 1 win_a, repeat × 64. This serializes A/B and prevents race. Requires restructuring the outer loop to repeat 64× with each iteration sending 1 win_a + 1 win_b.

3. **Single-window repeat loop** — set `iter_wrap=1` at the shim, `repeat=1` at the shim per outer loop iteration, and change the outer loop from `v374 < 4` to `v374 < 64` (64 iterations of 1 window each, serializing A and B).

4. **Fix the routing to packet-switched for B** — remove circuit-switched fan-out, use packet routing so each row's B path is independent and only that row is backpressured. This requires regenerating `routing.cc` from the pass pipeline.

---

## 4. History of Changes Made

### Session 1 (previous context)
- Identified initial deadlock: all channels pending=3–4
- Changed B shim `repeat` from 4→1 (wrong fix — reduces supply to 25%)
- Re-inserted A shim col3 that was missing

### Session 2 (this context)
- Restored B shim `repeat=4`
- Restructured outer loop: all A shims before all B shims
- Built and tested → `pending=1` (improvement from 3–4, but still deadlocked)
- Deep analysis: proved circuit-switched fan-out requires lock2 ≥ iter_wrap
- Implemented quad-buffering: lock2 init=2→4, BD chain BD0→BD1→BD6→BD7→BD0
- Built and tested → still `pending=1` (quad-buffer insufficient, see root cause above)

---

## 5. Current State of host.cc

Key sections:

### Outer loop structure (lines ~760–980)
```
for (v374 = 0; v374 < 4; v374++) {
  // A shims (col0→row3, col1→row4, col2→row5, col3→row6) — packet-switched
  startio(A shim col0, repeat=4)   → v398
  startio(A shim col1, repeat=4)   → v429
  startio(A shim col2, repeat=4)   → v460
  startio(A shim col3, repeat=4)   → v491

  // B shims — circuit-switched multicast
  startio(B shim col0, repeat=4)   → v378  [OOO BD, iter_wrap=4]
  startio(B shim col1, repeat=4)   → v382  [OOO BD, iter_wrap=4]
  startio(B shim col2, repeat=4)   → v386  [OOO BD, iter_wrap=4]
  startio(B shim col3, repeat=4)   → v390  [OOO BD, iter_wrap=4]

  // C shims (OOO, packet-switched output collection)
  startio(C col3 ch0, repeat=16)   → v421
  startio(C col3 ch1, repeat=16)   → v452
  startio(C col2 ch0, repeat=16)   → v483
  startio(C col2 ch1, repeat=16)   → v514

  // Wait all
  wait(v378, v382, v386, v390)
  wait(v398, v421, v429, v452)
  wait(v460, v483, v491, v514)
}
```

### win_b S2MM init (lines 151–420)
- 16 tiles: BD0 (ping, 0x8000, next_bd=1), BD1 (pong, 0x9000, next_bd=6), BD6 (0xC800, next_bd=7), BD7 (0xD800, next_bd=0)
- lock2 init=4, lock3 init=0
- startio outside loop with repeat=1 (S2MM always running, consuming on kernel demand)

### win_a S2MM init
- 16 tiles: BD2 (ping, 0xA000, next_bd=3), BD3 (pong, 0xB000, next_bd=2)
- lock0 init=2, lock1 init=0

---

## 6. Recommended Next Steps

### Option A — Single-window serialized loop (least invasive)

Change the outer loop iteration count from 4 to 64, and change each shim startio `repeat` from 4 to 1, and change `iter_wrap` from 4 to 1. This way each outer loop iteration sends exactly 1 win_a window + 1 win_b window, and the 4-entry win_b quad-buffer provides enough slack (can hold 4 windows in-flight while kernel drains).

Specific changes:
1. `v6 = 4` → `v6 = 64` (loop bound, defined near top of main)
2. `v8` (A shim stride per iteration) may need to stay the same if it accounts for 1 window
3. B shim `iter_wrap=4` → `iter_wrap=1` (last parameter of `dma_bd_config_multidim_ooo`)
4. B shim `iter_step_size=16384` → stays (still 1 window = 16384B)
5. A shim `d2_wrap=4` → `d2_wrap=1` if it controls the number of windows per startio

**Risk:** Need to verify A shim 3D BD counting matches 1 window per startio, and C shim repeat=16 may need adjustment.

### Option B — Lock-gate B shim at shim tile

Add a lock at the shim tile (e.g. shim lock 0) that:
- Is initialized to 4 (allowing 4 windows in-flight)
- B shim BD acquires this lock before each window send (`acquire_lock_id=0, acquire_lock_val=-1`)
- When compute tile finishes consuming win_b, it releases the shim lock (`+1`)

This requires inter-tile lock signaling (compute tile → shim tile), which is not currently in the architecture. Likely requires routing.cc changes.

### Option C — Change B routing to packet-switched (proper fix)

Modify `routing.cc` to use `XAie_StrmPktSw*` instead of `XAie_StrmConnCctEnable` for the B broadcast path, with one packet type per row. This eliminates the fan-out backpressure coupling. Requires regenerating from the `RoutingHW` pass or manually patching routing.cc.

**This is the correct long-term fix.** The root cause is that circuit-switched multicast inherently creates fan-out deadlock when any receiver stalls. Packet-switched routing gives each receiver an independent path.

### Option D — Reduce quad-buffer to 1-deep but add shim-side back-pressure

Revert lock2=4 to lock2=1, and set `iter_wrap=1` at the shim with `repeat=64` outer iterations. Only 1 win_b in flight at a time. Kernel drains it, releases lock, shim sends next.

---

## 7. Key Files

| File | Role |
|------|------|
| `aout/worklocal/host.cc` | Main patch target — all DMA/lock/startio host-side config |
| `aout/worklocal/routing.cc` | Stream switch config — B shim circuit-switched here |
| `aout/worklocal/matmul.cc` | Kernel — defines win_a/win_b/win_c consumption pattern |
| `debugcache/code/tile_c0_r3_s2mm1.cc` | Reference: win_b S2MM BD config for tile(0,3) |
| `debugcache/code/tile_c0_r3_s2mm0.cc` | Reference: win_a S2MM BD config for tile(0,3) |
| `debugcache/code/tile_c3_r0_mm2s1.cc` | Reference: A shim MM2S for tile(3,0) |
| `example/tileprogram/ccode/simplematmul2_oob_compare.cc` | Source — OOB benchmark driving the pipeline |
| `script/test/appvek385.py` | Board test runner |
| `script/hostcompile.sh` | Host binary build script |

---

## 8. Critical Notes for Next Engineer

1. **Never run `appvek385.py` in Claude sandbox** — DNS to portobello13 fails in sandbox. Use `dangerouslyDisableSandbox: true`.

2. **`pending=N` interpretation:** `XAie_DmaGetPendingBdCount` returns the number of BDs still queued (not yet dispatched to execute), not how many have run. `pending=1` with `repeat=4` means 3 BD runs submitted, 1 waiting, current BD run stalled mid-execution.

3. **B shim routing is circuit-switched** — confirmed in `routing.cc` via `XAie_StrmConnCctEnable`. Any fix that relies on independent per-row backpressure for B requires changing routing to packet-switched.

4. **A shim routing is packet-switched** — uses `XAie_StrmPktSw*`. A shim is NOT the deadlock source.

5. **The outer loop runs 4 times** (v374 = 0..3). Each iteration supplies 1/4 of total GEMM data. Total per tile: 64 win_a + 64 win_b + 16 win_c.

6. **B shim OOO BD:** `iter_wrap=4` means the shim sends 4 × 16384B = 65536B per `startio` call. With `repeat=4`, each outer loop calls startio once per B column, each sending 4 windows.

7. **Quad-buffer vars:** new BD6/BD7 variables are `v700`–`v763` (4 vars per tile × 16 tiles). These must not conflict with existing variables when adding more changes.

8. **Memory map is tight:** 0xD800 + 4096 = 0xE800. Tile LM is 64KB (0x10000). Only ~6KB free after BD7 (0xE800–0x10000). No room for more win_b BDs.

