<!-- Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->
---
name: aiesim-debug-socket
description: How the aiehlc aiesim (aie2pssimmsm) simulator serves live debug register reads/writes over a Unix socket, and the SystemC thread-affinity constraint behind it. Use when changing src/sim/aiehlc_dbg_server.*, the PS wrapper's debug threads, the runsim.sh debug env, or when live sim reads hang, refuse, or crash the simulator.
---

# aiesim debug register socket

The aiehlc simulator runs the host program *in-process* inside `aie2pssimmsm`
on a single SystemC thread (`PSIP_aiehlc::main_action` -> `aiehlc_ps_main`).
Register access is a direct call `ess_Read32` -> `PSRead32` ->
`PSIP_aiehlc::read32` -> XTLM `b_transport`. There is no external entry point,
so historically the debug UI refused live reads for `sim_kind == "aiesim"`.

`aiehlc_dbg_server` (linked into `aiehlc_ps.so`) adds a read-only,
write-gated debug socket, wire-compatible with the AEG IPC debug socket so
aiedbg's `sim_ipc_read32` drains either backend unchanged.

## Files

- `src/sim/aiehlc_dbg_protocol.h` — wire format, byte-identical to
  naiebaremetal `src/ipc/ps_ipc_protocol.h`: packed `{u8 cmd; u8[3]; u64 arg1;
  u32 arg2}` request, `{u8 status; u8[7]; u64 value}` response; `PING 0x01`,
  `WRITE32 0x10`, `READ32 0x11`, `NPI_WRITE32 0x12`, `NPI_READ32 0x13`. Matches
  aiedbg's `_IPC_REQ_FMT="<BxxxQI"` / `_IPC_RESP_FMT="<BxxxxxxxQ"`.
- `src/sim/aiehlc_dbg_server.{h,cpp}` — AF_UNIX listener + per-connection
  threads, a mutex/condvar work queue, and `aiehlc_dbg_drain()` for the SystemC
  side. No SystemC includes.
- `src/sim/aiehlc_ps_wrapper.cpp` — starts the server, provides the register
  callbacks, runs the `debug_action` drain thread and the end-of-app hold.
- `script/sim/dbg_probe.py` — standalone verifier (PING/READ32/WRITE32).

## Reuse the connection: one per scan, not one per register

`client_thread` loops on `recv_all`, so **a single connection carries any number
of requests**. Clients must hold one open for the duration of a scan. aiedbg's
one-shot `sim_ipc_read32` opens and closes a socket per 32-bit register; a
Switch scan is ~228 registers per tile over the whole grid rectangle, so one
press of Scan cost the simulator thousands of accept/thread-spawn/close cycles
and the run died partway through (symptom: the tail of the array reads
`unreachable`, then `ECONNREFUSED` on a `*.sock.dbg` that is still on disk).
`SimIpcConn` in `schedule_debug_server.py` is the reusing client — 2000 reads,
1 accept. Reads through it are serialized by a lock, because responses carry no
request id and two threads sharing a connection would take each other's values.

Two server-side rules that follow from it:

- `accept_thread` must not exit on a transient `accept()` error. It used to
  `break` on any failure, which retired the listener for the rest of the run
  while the socket file stayed on disk — every later connect got
  `ECONNREFUSED` from something that still looked alive. Retry on `EINTR`,
  `ECONNABORTED`, `EMFILE`, `ENFILE`, `ENOBUFS`, `ENOMEM`.
- `client_thread` must wait on the work item with a **timed** wait re-checked
  against `g_stop`. An unbounded `cv.wait` parks the thread and its fd forever
  once the SystemC side stops draining (i.e. after the hold ends and
  `sc_start()` unwinds), which is how the process reaches fd exhaustion with no
  diagnostic.

## The one hard constraint: SystemC thread affinity

SystemC is **not thread-safe**. A foreign POSIX (socket) thread must never call
`b_transport` or touch SystemC state. So:

1. The socket thread only enqueues a work item and blocks on its condvar.
2. `aiehlc_dbg_drain()` runs the register callbacks and MUST be called only from
   a SystemC thread (`debug_action`, or the hold loop).
3. The AXI initiator socket utils are shared between the app thread and the
   drain thread, so `read32/write32/read128/write128` take `m_axi_lock`
   (`sc_mutex`) around `b_transport`.
4. **The wake is part of "SystemC state".** `sc_event::notify()` is NOT
   thread-safe — see below.

### Why pure time-polling does not work, and how to wake safely

After the graph finishes, `aie2pssimmsm` quiesces the AIE array clock, so a
`wait(N, SC_NS)` timed wait stops resuming and a poll-only drain never runs —
reads hang. The socket thread must therefore wake the SystemC side.

It must NOT do that by calling `m_dbg_wake.notify()` directly. The kernel's
event queues take no lock, so a notify from a foreign POSIX thread races
`sc_simcontext::simulate()` mutating the same queue. It survives light traffic
and corrupts the queue under a scan (thousands of wakes in seconds). Signature:

```
[AIESIMULATOR ERROR] Segmentation fault detected.
#2 libsystemc.so(_ZN7sc_core8sc_event7triggerEv+0x3c)
#3 libsystemc.so(_ZN7sc_core13sc_simcontext8simulateERKNS_7sc_timeE+0x911)
```

i.e. the kernel tripping over an entry the socket thread edited. The run's own
output looks healthy right up to the crash — all iterations PASS, the hold line
prints, then it dies mid-scan and the UI reports "simulator stopped answering
during the scan (after tile (C,R))". **A partway-through-a-scan death is this
bug, not fd exhaustion** — that one gives `ECONNREFUSED` on a live socket file.

`async_request_update()` is the only entry point into the kernel that is legal
from a foreign thread. `DbgWakeChannel` (an `sc_prim_channel` in
`aiehlc_ps_wrapper.cpp`, constructed during elaboration) implements it:
`wake_async()` sets the pending flag from the socket thread, the kernel calls
`update()` on its own thread at the next delta, and only there does it
`m_dbg_wake.notify(SC_ZERO_TIME)`. `debug_action` still waits on the event with
a timed backstop: `wait(poll_ns, SC_NS, m_dbg_wake)`, and the hold loop's
`wait(1, SC_MS)` keeps timed activity pending so an async update is always
drained promptly. `g_dbg_wake_chan` is nulled after `aiehlc_dbg_stop()` so a
late socket thread cannot reach a kernel that is unwinding.

Do not "simplify" this back to a direct notify because another IPC server
appears to get away with it.

### Keeping the simulator alive past app completion (the hold)

`main_action` must NOT restructure its own waits into a long event wait after
the app returns — doing so kills `aie2pssimmsm` at hold entry (observed:
process dies right after the "holding array readable" line, before any client
connects). The working shape:

- `debug_action` stays the single drainer for both the run and the hold, woken
  by `m_dbg_wake`.
- `main_action` keeps the kernel alive with the same short `wait(1, SC_MS)`
  cadence the app used, and tracks idleness via `aiehlc_dbg_service_count()`.
- The hold is an **idle timeout**: it exits only after `AIEHLC_DBG_HOLD_SEC`
  seconds (default 600) with no request serviced, so an active scan keeps it
  alive. After the hold, set `g_dbg_thread_stop` and notify so `debug_action`
  exits and `sc_start()` unwinds.

## dbg_info.json (address contract)

aiehlc has no `Work/ps/c_rts/aie_control_config.json`, so the server writes
`<dir>/dbg_info.json` after `listen()` with `base_address`, `column_shift`,
`row_shift`, `aie_gen`, `writes_enabled` — taken from the `XAIE_BASE_ADDR` /
`XAIE_COL_SHIFT` / `XAIE_ROW_SHIFT` macros the `.so` is compiled with (so they
cannot drift). aiedbg's `_load_dbg_info_addr_params` reads it into
`_sim_addr_params`, and `sim_ipc_reg_read` computes
`base + (phys_col << col_shift) + (row << row_shift) + offset`.

## Runtime env (set by script/runsim.sh)

| var | meaning | default |
|---|---|---|
| `AIEHLC_DBG_DIR` | dir to bind the socket + `dbg_info.json` (next to `sim_config.sh`, i.e. `<app>/dbg`) | required to enable |
| `AIEHLC_DBG_HOLD_SEC` | idle timeout (s) before the held-open sim exits; `0` disables the hold | 600 |
| `AIEHLC_DBG_ALLOW_WRITE` | `1` accepts `WRITE32`/`NPI_WRITE32`; otherwise they return `ERR_PROTO` | 0 |
| `AIEHLC_DBG_POLL_NS` | drain-thread timed backstop (ns) | 1000 |
| `AIEHLC_DBG_VERBOSE` | server stderr diagnostics | off |

## aiedbg side (sibling checkout)

`schedule_debug_server.py`: `_sim_watch_dbg_socket` picks `<example>/dbg` for
`sim_kind=="aiesim"` (vs `<example>/ipc` for IPC) and loads `dbg_info.json`; the
watcher is started for both kinds; the three live gates (`/grid`, `/cmd`,
`/aiegdb`) block aiesim only when `not _sim_ipc_ready`; `_devices_for_ui` sets
`live_reads` for aiesim. Contract also noted in `adapters/aiehlc.py` and
`docs/debug-ui/bundle-contract.md`.

**Three front-ends must follow the socket, not just one.** Each caches the
backend differently, so each needs its own hook:

| front-end | how it picks the backend | what makes it follow |
|---|---|---|
| DMA/Cores/Events/Switch scans | `sim_ipc_reg_read` on the daemon itself | nothing — always current |
| LLM tab / `aie_exec` (`aiemcp.py`) | re-reads `backend_status.json` per call | `_ensure_backend_current` |
| Tools → aiegdb console | env of a long-lived `aiegdb.py --server` subprocess | `_gdb_spawn` sets `AEG_PS_IPC_DBG_SOCKET` + `AEG_SIM_*` and withholds `--target`; `_backend_changed` → `_gdb_drop` kills it on every flip |

The transport itself (`patch_gdb_for_simulator`, `sim_ipc_read32`) lives in
`aiegdb.py` so the console and the MCP server share one copy. It was originally
private to `aiemcp.py`, which is why the console kept dialling JTAG and printing
`ConnectionRefusedError` while the sim held the array — if you add a fourth
front-end, wire it to the same helper.

## Verify

```bash
# build (kernel objects must be linked — go through runsim, not a bare make)
source script/aiehlc.sh --platform sim --aie-version 5 --sim-tiles 4:2 \
  --runtime-source-file ./tutorial/example.cpp
bash script/runsim.sh aout/            # launch (holds open after the app)
python3 script/sim/dbg_probe.py aout/dbg   # PING + READ32 pass; WRITE32 = ERR_PROTO
AIEHLC_DBG_ALLOW_WRITE=1 bash script/runsim.sh aout/   # then WRITE32 = OK
```

## Pitfalls

- **PS.so load segfault after a bare `make build/aiehlc_ps.so`.** Relinking
  without `KERNEL_NAMES`/`KERNEL_OBJS` drops the embedded kernel; the
  `kernel_elf_init.cc` constructor then references an undefined
  `_binary_kernel_<name>_start` and `dlopen` crashes. Build through
  `runsim.sh --no-launch` (which sources `sim_config.sh`). See `aiesimloaddebug`.
- **Reads that time out only during heavy compute** are expected: the drain
  runs on simulated time, which advances slowly in real time under load. Reads
  are responsive in the idle hold window, which is when the UI scans.
- **`pkill -f Work_gen5` / `-f runsim` kills your own shell** (its command line
  matches). Kill by pid or match `aie2pssimmsm` via `comm`.
