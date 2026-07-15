#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""
schedule_view.py - Readable spatial schema + HTML viewer for the machine-generated
`host_canonicalized` function in <workdir>/host.cc.

It consumes:
  - <workdir>/dfscheduleprovenancemap.json  (high-level, already grouped by tile/data)
  - <workdir>/host.cc                        (low-level, final emitted C++)

and produces:
  - <workdir>/schedule_view.json             (readable schema: grid + per-tile hi/lo)
  - <workdir>/host_schedule.html             (self-contained interactive tile grid)

host.cc is never modified.

Usage:
    python3 src/tool/debug/schedule_view.py [workdir]
    (workdir defaults to aout/worklocal)
"""

import glob
import json
import os
import re
import sys


# ----------------------------------------------------------------------------
# host.cc parsing -> attribute each line to a spatial owner: (col,row) or "__global__"
# ----------------------------------------------------------------------------

RE_TILELOC_DECL = re.compile(
    r'XAie_LocType\s+(v\d+)\s*=\s*XAie_TileLoc\(\s*(\d+)\s*,\s*(\d+)\s*\)')
RE_BD_CONFIG = re.compile(
    r'__Runtime_dma_bd_config\w*\(\s*v\d+\s*,\s*(v\d+)\s*,')
RE_CREATEIO = re.compile(
    r'__Runtime_dma_createio_\d+\(\s*(v\d+)\s*,')
RE_ENABLE_OOO = re.compile(
    r'__Runtime_dma_channel_enable_ooo\(\s*v\d+\s*,\s*(v\d+)\s*,')
RE_LOCKSET = re.compile(
    r'XAie_LockSetValue\(\s*\w+\s*,\s*XAie_TileLoc\(\s*(\d+)\s*,\s*(\d+)\s*\)')
RE_COMMENT_TILE = re.compile(
    r'tile[ =]\(\s*(\d+)\s*,\s*(\d+)\s*\)')
RE_LAUNCH = re.compile(
    r'(v\d+)\s*=\s*__Runtime_launch_kernel_group')

# for-loop context: header detection + provenance-comment boundary for preamble.
RE_FOR = re.compile(r'\bfor\s*\(')
RE_PROV_COMMENT = re.compile(
    r'/\*\s*(Allocated BD|DMA BD Config|Create IO|Enable out-of-order)')


def find_loops(lines, fstart, fend):
    """Flat/nested for-loops in [fstart,fend]. Returns list of dicts:
    {header, end, preamble} (0-based line indices). preamble = the loop's
    index/offset math: statement lines after the `for(` header up to the first
    provenance comment (/* Allocated BD | DMA BD Config | ... */)."""
    loops, stack, depth = [], [], 0
    for i in range(fstart, fend + 1):
        ln = lines[i]
        if RE_FOR.search(ln):
            stack.append([i, depth])          # body { opens on this line
        depth += ln.count('{') - ln.count('}')
        while stack and depth <= stack[-1][1]:
            hdr, _ = stack.pop()
            loops.append({'header': hdr, 'end': i})
    for lp in loops:                          # compute preamble (index math)
        pre = []
        for j in range(lp['header'] + 1, lp['end']):
            s = lines[j]
            if RE_PROV_COMMENT.search(s):
                break
            if _is_comment_or_blank(s):
                continue
            pre.append(j)
        lp['preamble'] = pre
    return loops


# Simple SSA constant assignments: `<type> vN = <literal>;` (RHS is a bare int).
RE_CONST_ASSIGN = re.compile(r'\b(v\d+)\s*=\s*(-?\d+)\s*;')
RE_VAR = re.compile(r'\bv\d+\b')
# LHS assignment to vN (`<type> vN = ...` or `vN = ...`). Negative lookahead on
# `=` excludes the `==` comparison operator; `<=`/`>=`/`!=` never match because
# the leading operator char sits between vN and `=`.
RE_VAR_DEF = re.compile(r'\b(v\d+)\s*=(?!=)')


def build_const_map(lines, fstart, fend):
    """Map SSA var -> literal string for `vN = <int>;` assignments (SSA: once)."""
    cmap = {}
    for i in range(fstart, fend + 1):
        m = RE_CONST_ASSIGN.search(lines[i])
        if m:
            cmap.setdefault(m.group(1), m.group(2))
    return cmap


def build_var_def_map(lines, fstart, fend):
    """Map SSA var -> 1-based line where it is first assigned/declared.

    Function parameters (the `vN` on the signature line at `fstart`) have no `=`
    assignment, so they are seeded first, all pointing at the signature line;
    this lets the full-block trace-back surface the whole function signature
    (every parameter) when a param like `v1`/`v2` is referenced. SSA guarantees a
    single def per var, so the first match wins (setdefault)."""
    var_def = {}
    for m in RE_VAR.finditer(lines[fstart]):        # function parameters
        var_def.setdefault(m.group(0), fstart + 1)
    for i in range(fstart, fend + 1):
        for m in RE_VAR_DEF.finditer(lines[i]):
            var_def.setdefault(m.group(1), i + 1)
    return var_def


def channel_full_block(host_lines, entries, cmap, var_def, var2loc):
    """Contiguous host.cc block for ONE channel, with arg-def trace-back.

    `entries` are the channel-scoped mapping records (already include the
    enclosing loop header/index/close context). We union those lines with the
    defining line of every SSA `vN` referenced on them, following the def chain
    TRANSITIVELY (a pulled-in def line like `v174 = buffer_arg(v44)` drags in
    v44's def too, and so on) so the whole argument provenance is visible. SSA
    (one def per var) guarantees the closure terminates. Returns the same shape
    as the tile full block (`ranges/line_start/line_end/code_lines`) plus a
    `params` list annotating each referenced var's resolved value.
    """
    base = {e['line'] for e in entries}          # 1-based, channel scoped

    def resolve(v):
        val = cmap.get(v)
        if val is None and v in var2loc:
            c, r = var2loc[v]
            val = 'TileLoc(%d,%d)' % (c, r)
        return val

    # Transitive closure over the def chain. Seed the worklist with every var
    # referenced on a base line, then for each var pull its def line into the
    # block and enqueue the vars referenced on THAT def line. `varset` collects
    # all vars seen (for param annotation); SSA bounds the loop to finite vars.
    union = set(base)
    varset = set()
    work = []
    for n in base:
        work.extend(m.group(0) for m in RE_VAR.finditer(host_lines[n - 1]))
    while work:
        v = work.pop()
        if v in varset:
            continue
        varset.add(v)
        dl = var_def.get(v)
        if dl is None or dl in union:
            continue
        union.add(dl)                            # pull arg-def line into block
        work.extend(m.group(0)                   # trace vars on the def line
                    for m in RE_VAR.finditer(host_lines[dl - 1]))

    ordered = sorted(union)
    ranges = contiguous_ranges(ordered)
    code_lines = [{'line': n, 'code': host_lines[n - 1]} for n in ordered]

    params = []
    for v in sorted(varset):
        val = resolve(v)
        dl = var_def.get(v)
        if val is None and dl is None:
            continue                             # nothing meaningful to annotate
        params.append({'var': v, 'value': val, 'def_line': dl})

    return {
        'ranges': ranges,
        'line_start': ranges[0][0] if ranges else None,
        'line_end': ranges[-1][1] if ranges else None,
        'code_lines': code_lines,
        'params': params,
    }


def loop_note(header_text, cmap):
    """`v7=0, v11=7168, v5=1` for every const-valued var referenced in the header.
    The induction var (assigned in the header) is not a const, so it's skipped."""
    seen, parts = [], []
    for m in RE_VAR.finditer(header_text):
        v = m.group(0)
        if v in cmap and v not in seen:
            seen.append(v)
            parts.append('%s=%s' % (v, cmap[v]))
    return ', '.join(parts)

# ----------------------------------------------------------------------------
# Rename-robust anchors for the filtered per-channel mapping.
#
# These key on the provenance COMMENTS emitted by passdfscheduletoapi.cpp (from
# IR attributes, independent of the C API name) plus the stable runtime RETURN
# TYPES (io / ioevent / XAie_DmaDesc, typedef'd in aie_runtime.h).  They do NOT
# depend on the __Runtime_* wrapper names, so renaming any wrapper changes nothing
# the mapper reads.  Join key everywhere is (col,row,direction,channel).
# ----------------------------------------------------------------------------

# /* Create IO: channel_id=C, bd_id=B, tile=(col,row), direction=DIR */
RE_C_IO = re.compile(
    r'/\*\s*Create IO:\s*channel_id=(\d+),\s*bd_id=(-?\d+),\s*'
    r'tile=\((\d+),\s*(\d+)\),\s*direction=(\w+)')
# /* DMA BD Config: bd_id=B, len=L, ..., acquire_lock_id=A, ..., release_lock_id=R, ... */
RE_C_BD = re.compile(
    r'/\*\s*DMA BD Config:\s*bd_id=(-?\d+),\s*len=(\d+).*?'
    r'acquire_lock_id=(-?\d+),\s*acquire_lock_val=(-?\d+),\s*'
    r'release_lock_id=(-?\d+),\s*release_lock_val=(-?\d+)')
# /* Enable out-of-order BD on tile(col,row) ch=C dir=DIR */
RE_C_OOO = re.compile(
    r'/\*\s*Enable out-of-order BD on tile\((\d+),\s*(\d+)\)\s*ch=(\d+)\s*dir=(\w+)')
# __Runtime_dma_bd_config(dev, tile, BUFVAR, BD_ID, ...) -> buffer var + bd_id
RE_C_BDCALL = re.compile(
    r'__Runtime_dma_bd_config\(\s*\w+\s*,\s*\w+\s*,\s*(\w+)\s*,\s*(-?\d+)')
# void* vN = __runtime_buffer_arg((void*)ADDR);  -> literal DMA-view address
RE_C_BUFARG = re.compile(r'__runtime_buffer_arg\(\s*\(void\s*\*\)\s*(\d+)\s*\)')

# Stable runtime return types -> capture the LHS SSA var of a call statement.
RE_RET_BD = re.compile(r'\bXAie_DmaDesc\s+(v\d+)\s*=')
RE_RET_IO = re.compile(r'\bio\s+(v\d+)\s*=')       # NOT 'ioevent' (no ws after 'io')
RE_RET_SIO = re.compile(r'\bioevent\s+(v\d+)\s*=')
# Per-channel wait on a startio event: __Runtime_wait(device, <event_var>). The
# event var is the LHS of the matching `ioevent vN = __Runtime_startio(...)`.
RE_WAIT_EVENT = re.compile(r'__Runtime_wait\w*\(\s*\w+\s*,\s*(v\d+)\s*\)')
# LockSetValue is an XAie_ driver call (stable, not a __Runtime_ wrapper).
RE_LOCK_FULL = re.compile(
    r'XAie_LockSetValue\(\s*\w+\s*,\s*XAie_TileLoc\((\d+),\s*(\d+)\)\s*,\s*XAie_LockInit\((-?\d+)')

GLOBAL_MARKERS = (
    'load_kernel_group',
    'launch_kernel_group',
    'AieRt_DebugSnapshot',
    '_dbg_',
)


def find_function_range(lines, fname='host_canonicalized'):
    """Return (start_idx, end_idx) 0-based inclusive line indices of the function body."""
    start = None
    for i, ln in enumerate(lines):
        if fname in ln and '(' in ln and ln.rstrip().endswith('{'):
            start = i
            break
    if start is None:
        raise RuntimeError('could not find function %s in host.cc' % fname)
    depth = 0
    for i in range(start, len(lines)):
        depth += lines[i].count('{') - lines[i].count('}')
        if depth == 0 and i > start:
            return start, i
    return start, len(lines) - 1


def attribute_lines(lines, fstart, fend):
    """Return owner[] (same length as lines) for the function body.

    owner[i] is a (col,row) tuple, the string '__global__', or None (outside fn).
    """
    n = len(lines)
    owner = [None] * n
    var2loc = {}
    launch_var = None

    # First pass: collect var->loc and explicit per-line anchors.
    for i in range(fstart, fend + 1):
        ln = lines[i]
        m = RE_TILELOC_DECL.search(ln)
        if m:
            var2loc[m.group(1)] = (int(m.group(2)), int(m.group(3)))
        m = RE_LAUNCH.search(ln)
        if m:
            launch_var = m.group(1)

    for i in range(fstart, fend + 1):
        ln = lines[i]

        # Global markers first (kernel group load/launch, debug snapshot arrays).
        if any(mk in ln for mk in GLOBAL_MARKERS):
            owner[i] = '__global__'
            continue
        # The single outer wait on the launch event.
        if launch_var and '__Runtime_wait' in ln and re.search(
                r'\b%s\b' % re.escape(launch_var), ln):
            owner[i] = '__global__'
            continue

        # Explicit tile-typed function arguments (highest fidelity).
        m = RE_BD_CONFIG.search(ln) or RE_CREATEIO.search(ln) or RE_ENABLE_OOO.search(ln)
        if m and m.group(1) in var2loc:
            owner[i] = var2loc[m.group(1)]
            continue
        m = RE_LOCKSET.search(ln)
        if m:
            owner[i] = (int(m.group(1)), int(m.group(2)))
            continue
        m = RE_TILELOC_DECL.search(ln)
        if m:
            owner[i] = var2loc[m.group(1)]
            continue
        # Comment mentioning a tile (fallback anchor).
        m = RE_COMMENT_TILE.search(ln)
        if m:
            owner[i] = (int(m.group(1)), int(m.group(2)))
            continue

    # Backward-fill: an unanchored helper line belongs to the nearest anchored
    # line *below* it (helpers precede the call they feed).
    current = None
    for i in range(fend, fstart - 1, -1):
        if owner[i] is not None:
            current = owner[i]
        else:
            owner[i] = current if current is not None else '__global__'

    return owner, var2loc, launch_var


def contiguous_ranges(line_nums):
    """Given sorted 1-based line numbers, return list of [start,end] inclusive ranges."""
    ranges = []
    for ln in line_nums:
        if ranges and ln == ranges[-1][1] + 1:
            ranges[-1][1] = ln
        else:
            ranges.append([ln, ln])
    return ranges


# ----------------------------------------------------------------------------
# Rename-robust, filtered line -> schedule mapping.
#
# For each schedule element (bd_config / createio / startio / enable_ooo / lock)
# we discover its directly-implementing host.cc line via provenance COMMENTS and
# stable return TYPES, never via the renameable __Runtime_* wrapper names.
# Result: channel_map[(col,row,dir,channel)] = [ {line,kind,bd_id,comment_line,code}, ... ]
# ----------------------------------------------------------------------------

def _is_comment_or_blank(s):
    t = s.strip()
    return (not t) or t.startswith('/*') or t.startswith('//')


def _next_type_line(lines, i, end, rx):
    """First index in [i,end] whose text matches a return-type regex (call stmt)."""
    for j in range(i, end + 1):
        if rx.search(lines[j]):
            return j
    return None


def _next_stmt_line(lines, i, end):
    """First index in [i,end] that is not a comment/blank (the actual statement)."""
    for j in range(i, end + 1):
        if not _is_comment_or_blank(lines[j]):
            return j
    return None


def map_relevant_lines(host_lines, fstart, fend, prov, loops=None):
    """Map each schedule element to only its directly-implementing host.cc line(s).

    Returns channel_map[(col,row,dir,channel)] -> list of entries
        {line, kind, bd_id, comment_line, code}
    where kind in {createio, bd_config, enable_ooo, startio, wait, lock} and `line` is
    a 1-based host.cc line number.
    """
    # Precompute, per channel key, the bd-id set and lock-id set for exact join.
    chan_bd = {}                 # key -> set(bd_id)
    tile_lock_keys = {}          # (col,row) -> {lock_id: set(key)}
    for t in prov.get('tiles', []):
        col, row = t['col'], t['row']
        for c in t.get('dma_channels', []):
            key = (col, row, c['direction'], c['channel'])
            bds = set()
            locks = set()
            for bd in c.get('bd_chain', []):
                bds.add(bd.get('bd_id'))
                for lk in (bd.get('acquire_lock', []) + bd.get('release_lock', [])):
                    lid = lk.get('id')
                    if lid is not None and lid >= 0:
                        locks.add(lid)
            chan_bd[key] = bds
            for lid in locks:
                tile_lock_keys.setdefault((col, row), {}).setdefault(lid, set()).add(key)

    channel_map = {}

    def emit(key, entry):
        channel_map.setdefault(key, []).append(entry)

    pending_bd = []      # [{comment_line, bd_id, code_line}] since last createio
    io_var2key = {}      # io SSA var -> channel key (for name-free startio linking)

    for i in range(fstart, fend + 1):
        ln = host_lines[i]

        m = RE_C_BD.search(ln)
        if m:
            bd_id = int(m.group(1))
            j = _next_type_line(host_lines, i + 1, fend, RE_RET_BD)
            pending_bd.append({
                'comment_line': i + 1,
                'bd_id': bd_id,
                'code_line': (j + 1) if j is not None else (i + 1),
            })
            continue

        m = RE_C_OOO.search(ln)
        if m:
            # The OOO comment carries the full channel key, so emit directly
            # (its call line is the next statement, the void enable_ooo call).
            col, row = int(m.group(1)), int(m.group(2))
            ch, dr = int(m.group(3)), m.group(4)
            key = (col, row, dr, ch)
            j = _next_stmt_line(host_lines, i + 1, fend)
            code_line = (j + 1) if j is not None else (i + 1)
            emit(key, {
                'line': code_line, 'kind': 'enable_ooo', 'bd_id': None,
                'comment_line': i + 1, 'code': host_lines[code_line - 1].strip(),
            })
            continue

        m = RE_C_IO.search(ln)
        if m:
            ch, bd_id = int(m.group(1)), int(m.group(2))
            col, row, dr = int(m.group(3)), int(m.group(4)), m.group(5)
            key = (col, row, dr, ch)
            # createio call = next stmt whose LHS type is `io` (skips OOO call).
            j = _next_type_line(host_lines, i + 1, fend, RE_RET_IO)
            code_line = (j + 1) if j is not None else (i + 1)
            if j is not None:
                vm = RE_RET_IO.search(host_lines[j])
                if vm:
                    io_var2key[vm.group(1)] = key
            emit(key, {
                'line': code_line, 'kind': 'createio', 'bd_id': bd_id,
                'comment_line': i + 1,
                'code': host_lines[code_line - 1].strip(),
            })
            # Flush every pending BD whose bd_id belongs to this channel.
            keep = []
            for pb in pending_bd:
                if pb['bd_id'] in chan_bd.get(key, set()):
                    emit(key, {
                        'line': pb['code_line'], 'kind': 'bd_config',
                        'bd_id': pb['bd_id'], 'comment_line': pb['comment_line'],
                        'code': host_lines[pb['code_line'] - 1].strip(),
                        'bd_comment': host_lines[pb['comment_line'] - 1].strip(),
                    })
                else:
                    keep.append(pb)
            pending_bd = keep
            continue

    # startio: any `ioevent v = ...` statement referencing a known io var.
    # Also record event_var -> key so the matching __Runtime_wait can be linked.
    event_var2key = {}   # ioevent SSA var -> channel key
    for i in range(fstart, fend + 1):
        ms = RE_RET_SIO.search(host_lines[i])
        if not ms:
            continue
        for var, key in io_var2key.items():
            if re.search(r'\b%s\b' % re.escape(var), host_lines[i]):
                emit(key, {
                    'line': i + 1, 'kind': 'startio', 'bd_id': None,
                    'comment_line': None, 'code': host_lines[i].strip(),
                })
                event_var2key[ms.group(1)] = key
                break

    # wait: __Runtime_wait(device, <event_var>) where event_var is a per-channel
    # startio result. The single outer wait on the launch event is not a startio
    # event (not in event_var2key), so it is correctly excluded here.
    for i in range(fstart, fend + 1):
        m = RE_WAIT_EVENT.search(host_lines[i])
        if not m:
            continue
        key = event_var2key.get(m.group(1))
        if key is None:
            continue
        emit(key, {
            'line': i + 1, 'kind': 'wait', 'bd_id': None,
            'comment_line': None, 'code': host_lines[i].strip(),
        })

    # locks: XAie_LockSetValue(...) -> channel on same tile whose lock set has id.
    for i in range(fstart, fend + 1):
        m = RE_LOCK_FULL.search(host_lines[i])
        if not m:
            continue
        col, row, lid = int(m.group(1)), int(m.group(2)), int(m.group(3))
        keys = tile_lock_keys.get((col, row), {}).get(lid, set())
        if not keys:
            continue
        if len(keys) == 1:
            chosen = next(iter(keys))
        else:
            # Disambiguate by nearest preceding createio line.
            chosen, best = None, -1
            for key in keys:
                for e in channel_map.get(key, []):
                    if e['kind'] == 'createio' and e['line'] <= i + 1 and e['line'] > best:
                        best, chosen = e['line'], key
            if chosen is None:
                chosen = sorted(keys)[0]
        emit(chosen, {
            'line': i + 1, 'kind': 'lock', 'bd_id': None,
            'comment_line': None, 'code': host_lines[i].strip(),
        })

    # Attach enclosing-loop context (for-header + index math + closing brace) to
    # every channel whose captured ops fall inside a loop body. Emits new kinds
    # loop / loopidx / loopend; the header + idx sort ahead of the ops (lowest
    # line numbers) and the closing brace sorts after them (highest line number).
    # The loop header also carries a `note` resolving its SSA const parameters
    # (e.g. v7=0, v11=7168, v5=1).
    cmap = build_const_map(host_lines, fstart, fend)
    for key, entries in list(channel_map.items()):
        lset = {e['line'] for e in entries}                 # 1-based
        for lp in (loops or []):
            body_lo, body_hi = lp['header'] + 2, lp['end'] + 1
            if any(body_lo <= ln <= body_hi for ln in lset):
                hdr_code = host_lines[lp['header']].strip()
                emit(key, {'line': lp['header'] + 1, 'kind': 'loop',
                           'bd_id': None, 'comment_line': None,
                           'code': hdr_code, 'note': loop_note(hdr_code, cmap)})
                for pj in lp['preamble']:
                    emit(key, {'line': pj + 1, 'kind': 'loopidx',
                               'bd_id': None, 'comment_line': None,
                               'code': host_lines[pj].strip()})
                emit(key, {'line': lp['end'] + 1, 'kind': 'loopend',
                           'bd_id': None, 'comment_line': None,
                           'code': host_lines[lp['end']].strip()})

    return channel_map


# ----------------------------------------------------------------------------
# high-level rollup from provenance JSON
# ----------------------------------------------------------------------------

def kernel_for_tile(prov, col, row):
    for grp in prov.get('load_kernel_group', []):
        for t in grp.get('tiles', []):
            if t.get('col') == col and t.get('row') == row:
                return grp.get('callee')
    return None


def tile_role(ttype, channels):
    dirs = {c.get('direction') for c in channels}
    if ttype == 'shim':
        if dirs == {'MM2S'}:
            return 'dma-in (host->array)'
        if dirs == {'S2MM'}:
            return 'dma-out (array->host)'
        return 'shim-io (in+out)'
    return 'compute'


def build_summary(channels):
    out = []
    for c in channels:
        d = c.get('direction')
        ch = c.get('channel')
        bd = c.get('bd_chain', [])
        length = bd[0].get('len') if bd else '?'
        acq = bd[0].get('acquire_lock', [{}])[0] if bd else {}
        rel = bd[0].get('release_lock', [{}])[0] if bd else {}
        verb = 'recv' if d == 'S2MM' else 'send'
        reps = ''
        si = c.get('start_io', [])
        if si:
            rc = si[0].get('repeat_count')
            if rc and rc != 1:
                reps = ' x%d' % rc
        lock = ''
        if acq or rel:
            lock = ' lock%s/%s' % (acq.get('id'), rel.get('id'))
        out.append('%s %sB ch%s %s%s%s' % (verb, length, ch, d, reps, lock))
    return out


# ----------------------------------------------------------------------------
# Kernel code show (core tiles): parse kernel.cc and correlate each core-tile
# DMA channel with the kernel window/argument it feeds or drains.
#
# The generated kernel.cc carries stable, parseable markers:
#   #define LOCK_window_<name>_ACQ <n> / _REL <n>   -> per-window lock ids
#   // window_def window_<name>                       -> window declaration
#   window_init(window_window_<name>, .., LOCK_..ACQ, .., LOCK_..REL, size, ..)
#   // kernel_invoke <fn>
#   <fn>(get_input_async_window_..(window_window_<name>), ...)  -> arg order/dir
# ----------------------------------------------------------------------------

RE_KLOCK = re.compile(r'#define\s+LOCK_window_(\w+)_(ACQ|REL)\s+(\d+)')
RE_KWINDEF = re.compile(r'//\s*window_def\s+window_(\w+)')
RE_KWININIT = re.compile(
    r'window_init\(\s*window_window_(\w+)\s*,.*?LOCK_window_\w+_ACQ\s*,'
    r'.*?LOCK_window_\w+_REL\s*,\s*(\d+)\s*,')
RE_KINVOKE = re.compile(r'//\s*kernel_invoke\s+(\w+)')
RE_KARG = re.compile(r'get_(input|output)_async_window\w*\(\s*window_window_(\w+)\s*\)')
# Ping/pong buffer names referenced by a window_init(...) call (positional
# buffer args) and the .bcf symbol lines that map each buffer name -> address.
RE_KBUF = re.compile(r'\bbuf_\w+')
RE_BCF_SYMBOL = re.compile(r'^\s*_symbol\s+(\S+)\s+(0x[0-9A-Fa-f]+)')


def _win_dir(name):
    """Window direction from its generated name prefix (in_* / out_*)."""
    head = name.split('_', 1)[0]
    if head == 'in':
        return 'input'
    if head == 'out':
        return 'output'
    return 'unknown'


RE_KINCLUDE = re.compile(r'#include\s+"([^"]+\.cc)"')


def _strip_cc_comments(s):
    """Drop // and /* */ comments so structural scanning ignores commas/braces
    that appear inside comments (kernel signatures carry [K, tile_N] comments)."""
    s = re.sub(r'/\*.*?\*/', '', s)
    s = re.sub(r'//.*$', '', s)
    return s


def _split_top_commas(s):
    """Split on top-level commas (ignore commas nested in ()/[]/{})."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        if ch == ',' and depth == 0:
            out.append(''.join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append(''.join(cur))
    return out


def _parse_kernel_func(flines, def_idx, function):
    """Parse the kernel function definition starting at/near line def_idx.

    Returns None if this occurrence is a call/declaration (no '{' body) rather
    than a definition. On success returns {function, start_line, end_line,
    lines[], params[]} where each param is {pos, name, dir, ref_lines[]}.
    """
    stripped = [_strip_cc_comments(l) for l in flines]
    # flatten (line_idx, char) from the def line onward for brace/paren matching
    flat = []
    for li in range(def_idx, len(flines)):
        for ch in stripped[li]:
            flat.append((li, ch))
        flat.append((li, '\n'))
    s = ''.join(ch for _, ch in flat)
    mname = re.search(r'\b' + re.escape(function) + r'\s*\(', s)
    if not mname:
        return None
    popen = s.index('(', mname.start())
    depth, pclose = 0, -1
    for k in range(popen, len(s)):
        if s[k] == '(':
            depth += 1
        elif s[k] == ')':
            depth -= 1
            if depth == 0:
                pclose = k
                break
    if pclose < 0:
        return None
    sig = s[popen + 1:pclose]
    # a definition has '{' after the signature; a call/decl has ';'
    j = pclose + 1
    while j < len(s) and s[j] in ' \t\n':
        j += 1
    if j >= len(s) or s[j] != '{':
        return None
    depth, bclose = 0, -1
    for k in range(j, len(s)):
        if s[k] == '{':
            depth += 1
        elif s[k] == '}':
            depth -= 1
            if depth == 0:
                bclose = k
                break
    if bclose < 0:
        return None
    start_li = def_idx
    end_li = flat[bclose][0]

    params = []
    for pos, part in enumerate(p for p in _split_top_commas(sig) if p.strip()):
        toks = re.findall(r'[A-Za-z_]\w*', part)
        if not toks:
            continue
        if 'output_window' in part or re.search(r'\boutput\b', part):
            d = 'output'
        elif 'input_window' in part or re.search(r'\binput\b', part):
            d = 'input'
        else:
            d = 'unknown'
        params.append({'pos': pos, 'name': toks[-1], 'dir': d, 'ref_lines': []})

    lines = [{'line': n + 1, 'code': flines[n]}
             for n in range(start_li, end_li + 1)]
    for p in params:
        rex = re.compile(r'\b' + re.escape(p['name']) + r'\b')
        p['ref_lines'] = [n + 1 for n in range(start_li, end_li + 1)
                          if rex.search(stripped[n])]
    return {'function': function, 'start_line': start_li + 1,
            'end_line': end_li + 1, 'lines': lines, 'params': params}


def _find_kernel_source(kernel_cc_path, src_lines, function):
    """Locate the file+lines that DEFINE `function`, following #include "*.cc".

    Searches the included .cc files first (that is where the user kernel body
    lives), then kernel.cc itself. Returns the parsed function dict augmented
    with {file, path}, or None when no definition is found (graceful).
    """
    if not function:
        return None
    base = os.path.dirname(kernel_cc_path)
    candidates = []
    for ln in src_lines:
        m = RE_KINCLUDE.search(ln)
        if m:
            p = os.path.join(base, m.group(1))
            if os.path.isfile(p):
                try:
                    with open(p) as f:
                        candidates.append((m.group(1), p, f.read().split('\n')))
                except OSError:
                    pass
    candidates.append((os.path.basename(kernel_cc_path), kernel_cc_path, src_lines))
    defre = re.compile(r'\b' + re.escape(function) + r'\s*\(')
    for fname, fpath, flines in candidates:
        for i, ln in enumerate(flines):
            if defre.search(ln):
                parsed = _parse_kernel_func(flines, i, function)
                if parsed:
                    parsed['file'] = fname
                    parsed['path'] = os.path.abspath(fpath)
                    return parsed
    return None


def parse_kernel(kernel_cc_path):
    """Parse a generated kernel.cc into a window/argument model.

    Returns None if the file is absent (graceful skip). On success returns a
    dict {path, function, windows[], args[], lines[]} where each window has
    {name, dir, acq_lock, rel_lock, buf_size, def_line, init_line} and each arg
    has {arg, window, dir, line}. `lines` holds the marker line records for the
    relevant-slice render.
    """
    if not kernel_cc_path or not os.path.isfile(kernel_cc_path):
        return None
    try:
        with open(kernel_cc_path) as f:
            text = f.read()
    except OSError:
        return None
    src = text.split('\n')

    windows = {}         # name -> window dict
    order = []           # window names in first-seen order
    marker_lines = []    # (line_no, code) records for the render slice

    def _win(name):
        if name not in windows:
            windows[name] = {'name': name, 'dir': _win_dir(name),
                             'acq_lock': None, 'rel_lock': None,
                             'buf_size': None, 'def_line': None,
                             'init_line': None, 'lock_lines': [],
                             'buffers': [], 'buf_decl_lines': []}
            order.append(name)
        return windows[name]

    function = None
    invoke_line = None
    for i, ln in enumerate(src, 1):
        m = RE_KLOCK.search(ln)
        if m:
            w = _win(m.group(1))
            w['acq_lock' if m.group(2) == 'ACQ' else 'rel_lock'] = int(m.group(3))
            w['lock_lines'].append(i)
            marker_lines.append({'line': i, 'code': ln})
            continue
        m = RE_KWINDEF.search(ln)
        if m:
            w = _win(m.group(1))
            w['def_line'] = i
            marker_lines.append({'line': i, 'code': ln})
            continue
        m = RE_KWININIT.search(ln)
        if m:
            w = _win(m.group(1))
            w['buf_size'] = int(m.group(2))
            w['init_line'] = i
            # Positional ping/pong buffer names -> .bcf address correlation.
            for bm in RE_KBUF.finditer(ln):
                if bm.group(0) not in w['buffers']:
                    w['buffers'].append(bm.group(0))
            marker_lines.append({'line': i, 'code': ln})
            continue
        m = RE_KINVOKE.search(ln)
        if m:
            function = m.group(1)
            marker_lines.append({'line': i, 'code': ln})
            continue

    # Buffer declaration lines (`<type> buf_xxx[N];`) per window, for highlight.
    for w in windows.values():
        for buf in w['buffers']:
            decl_re = re.compile(r'\b' + re.escape(buf) + r'\s*\[')
            for i, ln in enumerate(src, 1):
                if decl_re.search(ln) and '=' not in ln.split(buf, 1)[0]:
                    if i not in w['buf_decl_lines']:
                        w['buf_decl_lines'].append(i)

    # Invoke call line: the call to <function>( ... ) with the arg windows.
    args = []
    if function:
        call_re = re.compile(r'\b' + re.escape(function) + r'\s*\(')
        for i, ln in enumerate(src, 1):
            if call_re.search(ln) and 'get_' in ln:
                invoke_line = i
                marker_lines.append({'line': i, 'code': ln})
                for pos, am in enumerate(RE_KARG.finditer(ln)):
                    d = 'input' if am.group(1) == 'input' else 'output'
                    args.append({'arg': pos, 'window': am.group(2), 'dir': d})
                    if am.group(2) in windows:
                        windows[am.group(2)]['dir'] = d
                break

    marker_lines.sort(key=lambda r: r['line'])
    # Resolve and parse the actual kernel function body (follows #include "*.cc").
    source = _find_kernel_source(kernel_cc_path, src, function)
    return {
        'path': os.path.abspath(kernel_cc_path),
        'file': os.path.basename(kernel_cc_path),
        'function': function,
        'invoke_line': invoke_line,
        'windows': [windows[n] for n in order],
        'args': args,
        'lines': marker_lines,
        # Full kernel.cc text (window_init / lock / buffer decls) for the new
        # "kernel.cc" sub-tab; highlighted per focused window.
        'kernel_lines': [{'line': i, 'code': ln} for i, ln in enumerate(src, 1)],
        'source': source,
    }


def find_bcf(workdir):
    """Locate the linker .bcf (buffer symbol map) in workdir; None if absent."""
    matches = sorted(glob.glob(os.path.join(workdir, '*.bcf')))
    return matches[0] if matches else None


def parse_bcf(bcf_path):
    """Parse a .bcf into {path, file, lines[], symbols[]}.

    Each `_symbol <name> <0xADDR>` line maps a kernel buffer name to its tile
    memory address; `symbols` records {name, addr, line} for buffer<->address
    correlation with the focused channel's window. Graceful None if absent.
    """
    if not bcf_path or not os.path.isfile(bcf_path):
        return None
    try:
        with open(bcf_path) as f:
            text = f.read()
    except OSError:
        return None
    src = text.split('\n')
    symbols = []
    for i, ln in enumerate(src, 1):
        m = RE_BCF_SYMBOL.match(ln)
        if m:
            symbols.append({'name': m.group(1), 'addr': m.group(2), 'line': i})
    return {
        'path': os.path.abspath(bcf_path),
        'file': os.path.basename(bcf_path),
        'lines': [{'line': i, 'code': ln} for i, ln in enumerate(src, 1)],
        'symbols': symbols,
    }


# Buffer base offset: a core-tile proc address minus this yields the DMA-view
# address literal that host.cc passes to __runtime_buffer_arg((void*)ADDR); the
# .bcf `_symbol <name> 0x...` addresses are proc addresses (ADDR + WIN_BASE).
WIN_BASE = 0x70000


def match_channels_to_kernel(channels, kernel_view, bcf_view, host_lines, var_def):
    """Correlate each core-tile DMA channel with a kernel window/argument by the
    BD buffer address (deterministic, no lock/positional heuristics).

    Chain per channel BD:
      1. host.cc `__Runtime_dma_bd_config(dev, tile, BUFVAR, bd_id, ...)` names a
         buffer SSA var whose def is
         `void* BUFVAR = __runtime_buffer_arg((void*)ADDR);` where ADDR is the
         DMA-view tile address (proc_addr - WIN_BASE).
      2. ADDR + WIN_BASE == a `.bcf` `_symbol <name> 0x...` address -> buffer
         name (e.g. buf_in_ping_1).
      3. That buffer name appears in a kernel.cc `window_init(...)` -> its window.

    Primary path uses host.cc SSA (`method='addr'`). When the host.cc parse
    fails, falls back to the provenance `bd_chain[i]['buffer_offset']` (already
    the DMA-view ADDR, `method='addr(prov)'`). `method='none'` if no address is
    recoverable. Shim tiles use a symbolic (DDR) buffer arg and are not matched
    here (this runs for core tiles only).
    """
    if not kernel_view:
        return {'matches': []}
    wins = kernel_view.get('windows', [])
    arg_of = {}
    for a in kernel_view.get('args', []):
        arg_of[a['window']] = a['arg']
    # buffer name -> owning window (from kernel.cc window_init buffer lists).
    buf2win = {}
    for w in wins:
        for buf in w.get('buffers', []):
            buf2win.setdefault(buf, w)
    # DMA-view address -> window / symbol name (via .bcf symbol table).
    dmaview2win = {}
    addr2sym = {}
    for s in (bcf_view or {}).get('symbols', []):
        try:
            proc_addr = int(s['addr'], 16)
        except (ValueError, TypeError, KeyError):
            continue
        dmaview = proc_addr - WIN_BASE
        addr2sym[dmaview] = s['name']
        w = buf2win.get(s['name'])
        if w is not None:
            dmaview2win.setdefault(dmaview, w)

    def _bd_addr(entry):
        """Resolve one bd_config entry -> DMA-view ADDR via host.cc SSA, or None.

        Reads the __Runtime_dma_bd_config call line for the buffer SSA var, then
        follows that var's def (`__runtime_buffer_arg((void*)ADDR)`)."""
        ln = host_lines[entry['line'] - 1]
        mc = RE_C_BDCALL.search(ln)
        if not mc:
            return None
        bufvar = mc.group(1)
        def_line = var_def.get(bufvar)
        if not def_line:
            return None
        ma = RE_C_BUFARG.search(host_lines[def_line - 1])
        if not ma:
            return None
        return int(ma.group(1))

    matches = []
    for c in channels:
        direction = c.get('direction')
        addrs = []
        method = 'none'
        # Primary: host.cc SSA (user's path) for every BD line of this channel.
        for e in c.get('host_lines', []) or []:
            if e.get('kind') != 'bd_config':
                continue
            a = _bd_addr(e)
            if a is not None:
                addrs.append(a)
        if addrs:
            method = 'addr'
        else:
            # Fallback: provenance bd_chain buffer_offset (already DMA-view ADDR).
            for bd in c.get('bd_chain', []) or []:
                off = bd.get('buffer_offset')
                if off is not None:
                    addrs.append(int(off))
            if addrs:
                method = 'addr(prov)'
        # Dedupe addresses preserving first-seen order (ping/pong collapse).
        seen = set()
        uaddrs = []
        for a in addrs:
            if a not in seen:
                seen.add(a)
                uaddrs.append(a)
        # Map addresses -> .bcf symbols + the single owning window.
        chosen = None
        bcf_syms = []
        for a in uaddrs:
            bcf_syms.append(addr2sym.get(a, '(unmatched-addr)'))
            w = dmaview2win.get(a)
            if w is not None and chosen is None:
                chosen = w
        matches.append({
            'direction': direction,
            'channel': c.get('channel'),
            'window': chosen['name'] if chosen else None,
            'arg': arg_of.get(chosen['name']) if chosen else None,
            'buf_size': chosen.get('buf_size') if chosen else None,
            'method': method,
            'bcf_syms': bcf_syms,
            'addrs': uaddrs,
            'addrs_hex': ['0x%x' % a for a in uaddrs],
        })
    return {'matches': matches}


# ----------------------------------------------------------------------------
# dfschedule IR slicing (Middle tab): verbatim per-tile / per-channel slice of
# the executable schedule dump 6_BlueprintToSchedulePass.mlir.
#
# The dfschedule dialect is a clean SSA def-use chain, so a scope-aware
# reachability slice extracts a tile's/channel's ops WITHOUT any compiler change.
# SSA names reset per region (each RoutingCreate round reuses %4, %5, ...), so
# operand resolution is scope-aware: a use binds to the nearest prior def whose
# enclosing region is an ancestor-or-equal of the use's region.
# ----------------------------------------------------------------------------

DFSCHED_IR_NAME = '6_BlueprintToSchedulePass.mlir'

# Region-carrying ops in this IR. A line ending with '{' that mentions one of
# these opens a nested-op region; any other line ending with '{' opens an
# attribute dict on a single op (closed later by `} : <types>`).
_MLIR_REGION_KEYWORDS = ('module', 'func.func', 'scf.', 'affine.', 'RoutingCreate')

RE_MLIR_DEF = re.compile(r'^\s*(%[A-Za-z0-9_]+)\s*=')
RE_MLIR_SSA = re.compile(r'%[A-Za-z0-9_]+')
RE_MLIR_DECLTILE = re.compile(
    r'dfschedule\.declaretile\s*\{\s*col\s*=\s*(-?\d+)\s*:\s*i32\s*,\s*'
    r'row\s*=\s*(-?\d+)\s*:\s*i32')


def find_dfschedule_ir(workdir):
    """Locate 6_BlueprintToSchedulePass.mlir; return path or None (graceful)."""
    candidates = []
    env = os.environ.get('SCHEDULE_VIEW_IR_DIR')
    if env:
        candidates.append(os.path.join(env, DFSCHED_IR_NAME))
    wd = os.path.abspath(workdir)
    for up in ('ir/dfschedule', '../ir/dfschedule', '../../ir/dfschedule'):
        candidates.append(os.path.join(wd, up, DFSCHED_IR_NAME))
    candidates.append(os.path.join(os.getcwd(), 'ir/dfschedule', DFSCHED_IR_NAME))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _mlir_is_region_open(s):
    return s.endswith('{') and any(k in s for k in _MLIR_REGION_KEYWORDS)


def _brace_delta(s):
    return (s.count('(') + s.count('[') + s.count('{')
            - s.count(')') - s.count(']') - s.count('}'))


class DfscheduleSlicer:
    """Statement-level def-use slicer over a dfschedule .mlir dump."""

    def __init__(self, text):
        self.stmts = []          # {idx,text,result,operands,scope,region_stack}
        self.def_by_name = {}    # ssa name -> [(idx, scope), ...]
        self.tile_ssa = {}       # (col,row) -> set(declaretile stmt idx)
        self._parse(text)
        self._resolve()

    # -- parsing --------------------------------------------------------------
    def _parse(self, text):
        lines = text.split('\n')
        region_stack = []        # raw header lines (context)
        scope_stack = []         # region ids -> current scope tuple
        rid = [0]
        cur, depth, cur_start = [], 0, 0
        for lineno, raw in enumerate(lines, 1):   # 1-based source line numbers
            s = raw.strip()
            if depth == 0:
                if s == '':
                    continue
                if s == '}':                      # region close
                    if scope_stack:
                        scope_stack.pop()
                    if region_stack:
                        region_stack.pop()
                    continue
                if _mlir_is_region_open(s):        # region open (context header)
                    region_stack.append(raw)
                    rid[0] += 1
                    scope_stack.append(rid[0])
                    continue
                if s.startswith('^') and s.endswith(':'):   # block label
                    continue
                cur = [raw]
                cur_start = lineno
                depth += _brace_delta(s)
                if depth <= 0:
                    depth = 0
                    self._add_stmt(cur, cur_start, tuple(scope_stack), region_stack)
                    cur = []
            else:                                 # inside a multi-line statement
                cur.append(raw)
                depth += _brace_delta(s)
                if depth <= 0:
                    depth = 0
                    self._add_stmt(cur, cur_start, tuple(scope_stack), region_stack)
                    cur = []

    def _add_stmt(self, raw_lines, start_line, scope, region_stack):
        idx = len(self.stmts)
        text = '\n'.join(raw_lines)
        mdef = RE_MLIR_DEF.match(raw_lines[0])
        result = mdef.group(1) if mdef else None
        operands = []
        for j, rl in enumerate(raw_lines):
            body = rl
            if j == 0 and result and '=' in rl:
                body = rl.split('=', 1)[1]
            operands.extend(m.group(0) for m in RE_MLIR_SSA.finditer(body))
        self.stmts.append({
            'idx': idx, 'text': text, 'result': result, 'operands': operands,
            'scope': scope, 'region_stack': list(region_stack),
            'line': start_line, 'lines': list(raw_lines),
        })
        if result:
            self.def_by_name.setdefault(result, []).append((idx, scope))
        m = RE_MLIR_DECLTILE.search(text)
        if m:
            self.tile_ssa.setdefault((int(m.group(1)), int(m.group(2))),
                                     set()).add(idx)

    # -- def-use resolution ---------------------------------------------------
    def _resolve(self):
        self.fwd = {}    # producer idx -> set(consumer idx)
        self.bwd = {}    # consumer idx -> set(producer idx)
        for st in self.stmts:
            pset = set()
            for name in st['operands']:
                p = self._resolve_name(name, st['idx'], st['scope'])
                if p is not None:
                    pset.add(p)
            self.bwd[st['idx']] = pset
            for p in pset:
                self.fwd.setdefault(p, set()).add(st['idx'])

    def _resolve_name(self, name, use_idx, use_scope):
        best = None
        for (didx, dscope) in self.def_by_name.get(name, ()):
            if didx >= use_idx:
                continue
            if len(dscope) <= len(use_scope) and use_scope[:len(dscope)] == dscope:
                if best is None or didx > best:
                    best = didx
        return best

    def _close_backward(self, sel, seeds):
        stack = list(seeds)
        while stack:
            c = stack.pop()
            for p in self.bwd.get(c, ()):
                if p not in sel:
                    sel.add(p)
                    stack.append(p)

    # -- slices ---------------------------------------------------------------
    def tile_slice(self, col, row):
        seeds = self.tile_ssa.get((col, row))
        if not seeds:
            return '(no matching ops)'
        sel = set(seeds)
        stack = list(seeds)                       # forward reachability
        while stack:
            for c in self.fwd.get(stack.pop(), ()):
                if c not in sel:
                    sel.add(c)
                    stack.append(c)
        self._close_backward(sel, list(sel))      # backward completion
        return self._emit(sel)

    def channel_slice(self, col, row, direction, channel):
        tiles = self.tile_ssa.get((col, row), set())
        seeds = []
        for st in self.stmts:
            if 'dfschedule.config.create_io' not in st['text']:
                continue
            if not (self.bwd.get(st['idx'], set()) & tiles):
                continue
            if not re.search(r'channel\s*=\s*%d\b' % channel, st['text']):
                continue
            if not re.search(r'direction\s*=\s*"%s"' % re.escape(direction),
                             st['text']):
                continue
            seeds.append(st['idx'])
        if not seeds:
            return '(no matching ops)'
        sel = set(seeds)
        self._close_backward(sel, seeds)          # bd chain -> buffers -> tile
        frontier = list(seeds)                    # create_io -> start_io -> wait
        while frontier:
            nxt = []
            for f in frontier:
                for c in self.fwd.get(f, ()):
                    txt = self.stmts[c]['text']
                    if ('schedule.start_io' in txt or 'schedule.wait' in txt) \
                            and c not in sel:
                        sel.add(c)
                        nxt.append(c)
                        self._close_backward(sel, [c])   # pull getbdid, etc.
            frontier = nxt
        return self._emit(sel)

    # -- emission -------------------------------------------------------------
    def _emit(self, sel):
        # Returns a list of {line, code} rows preserving the ORIGINAL source line
        # number from 6_BlueprintToSchedulePass.mlir. Context headers, blank
        # separators and '// ...' elision markers carry line=None (no source
        # line). Multi-line statements number each physical line.
        idxs = sorted(sel)
        MAXN = 200
        trunc = len(idxs) > MAXN
        idxs = idxs[:MAXN]
        out, last_ctx, prev = [], None, None
        for idx in idxs:
            st = self.stmts[idx]
            ctx = [h.strip() for h in st['region_stack']
                   if ('scf.' in h or 'RoutingCreate' in h)]
            if ctx != last_ctx:
                if out:
                    out.append({'line': None, 'code': ''})
                out.extend({'line': None, 'code': '// ' + h} for h in ctx)
                last_ctx = ctx
            elif prev is not None and idx != prev + 1:
                out.append({'line': None, 'code': '// ...'})
            for off, code in enumerate(st['lines']):
                out.append({'line': st['line'] + off, 'code': code})
            prev = idx
        if trunc:
            out.append({'line': None,
                        'code': '// ... (slice truncated at %d statements)' % MAXN})
        return out


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def build_view(workdir):
    prov_path = os.path.join(workdir, 'dfscheduleprovenancemap.json')
    host_path = os.path.join(workdir, 'host.cc')
    with open(prov_path) as f:
        prov = json.load(f)
    with open(host_path) as f:
        host_lines = f.read().split('\n')

    # Kernel code show: parse kernel.cc once (graceful skip if absent) so each
    # core tile can correlate its DMA channels to kernel windows/arguments.
    kernel_view = parse_kernel(os.path.join(workdir, 'kernel.cc'))
    # .bcf buffer symbol map (buffer name -> tile address), for the new "*.bcf"
    # sub-tab correlated to the focused channel's window buffers.
    bcf_view = parse_bcf(find_bcf(workdir))

    fstart, fend = find_function_range(host_lines)
    owner, var2loc, _ = attribute_lines(host_lines, fstart, fend)

    # for-loop context (headers + per-iteration index math), computed once.
    loops = find_loops(host_lines, fstart, fend)

    # Rename-robust, filtered per-channel line mapping (comment/type anchored).
    channel_map = map_relevant_lines(host_lines, fstart, fend, prov, loops)

    # SSA value/def maps for the per-channel full-block arg trace-back.
    cmap = build_const_map(host_lines, fstart, fend)
    var_def = build_var_def_map(host_lines, fstart, fend)

    # Middle tab: verbatim dfschedule IR slicer (built once, graceful if absent).
    ir_path = find_dfschedule_ir(workdir)
    slicer = None
    ir_text = None
    if ir_path:
        try:
            with open(ir_path) as f:
                ir_text = f.read()
            slicer = DfscheduleSlicer(ir_text)
        except Exception:
            slicer = None
            ir_text = None
    mid_fallback = ('(dfschedule IR dump not found)' if slicer is None
                    else '(no matching ops)')

    # per-owner 1-based line numbers
    owner_lines = {}
    for i in range(fstart, fend + 1):
        o = owner[i]
        owner_lines.setdefault(o, []).append(i + 1)

    # Full-block view: augment each tile owner's line set with the for-header +
    # index math of any loop that wraps one of its lines, so multi-tile loops
    # show the `for(...)` header for every participating tile, not just the first.
    for okey, lns in list(owner_lines.items()):
        if not isinstance(okey, tuple):        # skip '__global__'
            continue
        lnset = set(lns)
        extra = set()
        for lp in loops:
            body_lo, body_hi = lp['header'] + 2, lp['end'] + 1
            if any(body_lo <= x <= body_hi for x in lnset):
                extra.add(lp['header'] + 1)
                extra.update(p + 1 for p in lp['preamble'])
                extra.add(lp['end'] + 1)
        owner_lines[okey] = sorted(lnset | extra)

    # grid derivation from JSON tiles
    cols = sorted({t['col'] for t in prov['tiles']})
    rows = sorted({t['row'] for t in prov['tiles']})
    shim_rows = sorted({t['row'] for t in prov['tiles'] if t['type'] == 'shim'})
    core_rows = sorted({t['row'] for t in prov['tiles'] if t['type'] == 'core'})

    def code_for(line_nums):
        ordered = sorted(line_nums)
        ranges = contiguous_ranges(ordered)
        chunks = []
        for a, b in ranges:
            chunks.append('\n'.join(host_lines[a - 1:b]))
        # Per-line records (raw text, preserving indentation) so the full-block
        # view can prefix each line with its host.cc line number.
        code_lines = [{'line': n, 'code': host_lines[n - 1]} for n in ordered]
        return ranges, '\n'.join(chunks), code_lines

    tiles_out = []
    for t in prov['tiles']:
        col, row, ttype = t['col'], t['row'], t['type']
        channels = t.get('dma_channels', [])
        lns = owner_lines.get((col, row), [])
        ranges, code, code_lines = code_for(lns) if lns else ([], '', [])

        # Filtered, rename-robust mapping: attach directly-implementing lines to
        # each channel and aggregate a per-tile "relevant only" view.
        line_kinds = {}
        line_bd = {}
        line_note = {}
        for c in channels:
            key = (col, row, c['direction'], c['channel'])
            entries = sorted(channel_map.get(key, []), key=lambda e: e['line'])
            c['host_lines'] = entries
            # Channel-scoped full block (contiguous ops + enclosing loop context
            # + arg-def trace-back lines), plus resolved-value params for inline
            # annotation in the "Full block" sub-tab.
            c['low_level'] = channel_full_block(host_lines, entries, cmap,
                                                var_def, var2loc)
            c['middle_ir'] = (slicer.channel_slice(col, row, c['direction'],
                                                    c['channel'])
                              if slicer else mid_fallback)
            for e in entries:
                line_kinds.setdefault(e['line'], set()).add(e['kind'])
                if e.get('bd_comment'):
                    line_bd[e['line']] = e['bd_comment']
                if e.get('note'):
                    line_note[e['line']] = e['note']
        relevant_lines = sorted(line_kinds)
        relevant_detail = []
        for n in relevant_lines:
            d = {'line': n, 'kinds': sorted(line_kinds[n]), 'code': host_lines[n - 1].strip()}
            if n in line_bd:
                d['bd_comment'] = line_bd[n]
            if n in line_note:
                d['note'] = line_note[n]
            relevant_detail.append(d)
        relevant_code = '\n'.join(
            'L%d: %s' % (n, host_lines[n - 1].strip()) for n in relevant_lines)

        # Kernel channel<->argument correlation (core tiles only). None for
        # shim/other tiles or when kernel.cc is absent.
        kernel_match = (match_channels_to_kernel(channels, kernel_view, bcf_view,
                                                 host_lines, var_def)
                        if ttype == 'core' and kernel_view else None)

        tiles_out.append({
            'loc': [col, row],
            'type': ttype,
            'high_level': {
                'role': tile_role(ttype, channels),
                'kernel': kernel_for_tile(prov, col, row),
                'summary': build_summary(channels),
                'contracts': [c.get('contract') for c in channels if c.get('contract')],
                'kernel_match': kernel_match,
            },
            'low_level': {
                'ranges': ranges,
                'line_start': ranges[0][0] if ranges else None,
                'line_end': ranges[-1][1] if ranges else None,
                'code': code,
                'code_lines': code_lines,
            },
            'middle_ir': (slicer.tile_slice(col, row) if slicer
                          else mid_fallback),
            'relevant_lines': relevant_lines,
            'relevant_detail': relevant_detail,
            'relevant_code': relevant_code,
            'dma_channels': channels,
        })

    # global (non-tile) lines: kernel group load/launch, final wait, dbg snapshot
    glns = owner_lines.get('__global__', [])
    granges, gcode, _ = code_for(glns) if glns else ([], '', [])

    view = {
        'source': {
            'host_cc': os.path.abspath(host_path),
            'provenance': os.path.abspath(prov_path),
            'function': 'host_canonicalized',
            'function_lines': [fstart + 1, fend + 1],
        },
        'grid': {
            'cols': (max(cols) + 1) if cols else 0,
            'rows': (max(rows) + 1) if rows else 0,
            'col_list': cols,
            'row_list': rows,
            'shim_rows': shim_rows,
            'core_rows': core_rows,
            # Partition origin (absolute physical start column) from the
            # provenance map; phys_col = col + startcol. None = no partition.
            'startcol': prov.get('startcol'),
        },
        'module_attrs': prov.get('module_attrs', []),
        'buffers': [
            {'arg': 'v2', 'role': 'aux'},
            {'arg': 'v3', 'role': 'input (A)'},
            {'arg': 'v4', 'role': 'output (C)'},
        ],
        'global': {
            'ranges': granges,
            'code': gcode,
            'note': 'kernel group load/launch, outer wait, debug snapshot',
        },
        'tiles': tiles_out,
        'kernel': kernel_view,
        'bcf': bcf_view,
        'invariant_checks': prov.get('invariant_checks', []),
        'flow_summary': prov.get('flow_summary', []),
        'dfschedule_ir': {
            'name': DFSCHED_IR_NAME,
            'path': os.path.abspath(ir_path) if ir_path else None,
            'text': ir_text or '',
        },
    }
    return view


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AIE Schedule View</title>
<style>
  :root { --shim:#2d6a9f; --core:#3a7d44; --sel:#e0a800; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: -apple-system, Segoe UI, Roboto, sans-serif;
         background:#1e1e1e; color:#ddd; display:flex; height:100vh; }
  #left { flex:0 0 50%; min-width:200px; display:flex; flex-direction:column;
          overflow:hidden; }
  #lefttop { flex:0 1 auto; overflow:auto; padding:16px; min-height:0; }
  /* flex-basis:0 so the bottom frame is sized only by leftover space, never by
     its console content (which would otherwise inflate the frame). */
  #leftbottom { flex:1 1 0; overflow:auto; padding:16px; min-height:80px;
                display:flex; flex-direction:column; }
  #lhsplitter { flex:0 0 6px; cursor:row-resize; background:#333;
                border-top:1px solid #444; border-bottom:1px solid #444; }
  #lhsplitter:hover, #lhsplitter.drag { background:#5a5a5a; }
  #splitter { flex:0 0 6px; cursor:col-resize; background:#333;
              border-left:1px solid #444; border-right:1px solid #444; }
  #splitter:hover, #splitter.drag { background:#5a5a5a; }
  #right { flex:1 1 0; min-width:200px; padding:16px; overflow:hidden;
           display:flex; flex-direction:column; }
  #panel { flex:1 1 0; overflow:auto; min-height:0; }
  #rhsplitter { flex:0 0 6px; cursor:row-resize; background:#333; margin-top:8px;
                border-top:1px solid #444; border-bottom:1px solid #444; }
  #rhsplitter:hover, #rhsplitter.drag { background:#5a5a5a; }
  /* fixed default height + flex column so console output scrolls inside the
     frame instead of inflating it; drag #rhsplitter to resize. */
  #cmdconsole { flex:0 0 260px; border-top:1px solid #333; margin-top:8px;
                padding-top:8px; display:flex; flex-direction:column;
                min-height:0; overflow:hidden; }
  /* keep the header/help at natural height; the terminal box (#conterm)
     flexes and holds both the scrolling output and the inline prompt. */
  #conhdr, #conhelp { flex:0 0 auto; }
  #conhdr { font-weight:700; margin-bottom:4px; }
  #contarget { color:#8ec; }
  #conhelp { color:#789; font-size:10px; margin-bottom:6px; }
  #conreload { margin-left:8px; padding:3px 10px; cursor:pointer; }
  #conterm { flex:1 1 0; display:flex; flex-direction:column; min-height:0;
             margin-top:6px; padding:6px; background:#111; border:1px solid #444;
             border-radius:4px; overflow:hidden; cursor:text; }
  #conout { flex:1 1 auto; min-height:0; overflow:auto; margin:0;
            background:transparent; }
  #conpromptline { flex:0 0 auto; display:flex; align-items:center;
                   margin-bottom:4px; padding-bottom:4px;
                   border-bottom:1px solid #333; }
  #conprompt { color:#8ec; margin-right:6px; white-space:nowrap;
               font-family:monospace; }
  #conin { flex:1 1 auto; background:transparent; border:none; outline:none;
           color:#ddd; font-family:monospace; padding:0; }
  /* highlight of the most recent command + result, pinned at the top of the
     terminal box just under the prompt/input line (before the scrolling log) */
  #conlast { flex:0 0 auto; margin-bottom:6px; padding:6px 8px; background:#161616;
             border:1px solid #444; border-radius:4px; font-family:monospace;
             font-size:11px; font-weight:700; white-space:pre-wrap;
             word-break:break-word; }
  #conlastlbl { color:#789; font-weight:700; margin-right:6px; }
  #conlastcmd { color:#8ec; font-weight:700; }
  #conlastres { color:#f5b942; font-weight:700; }
  body.resizing { cursor:col-resize; user-select:none; }
  body.vresizing { cursor:row-resize; user-select:none; }
  h1 { font-size:16px; margin:0 0 4px; }
  .sub { color:#888; font-size:12px; margin-bottom:12px; }
  #grid { display:grid; gap:6px; }
  .tile { border:1px solid #444; border-radius:6px; padding:8px; cursor:pointer;
          font-size:11px; line-height:1.35; transition:.1s; min-height:64px; }
  .tile.shim { background:var(--shim); }
  .tile.core { background:var(--core); }
  .tile:hover { outline:2px solid #fff6; }
  .tile.sel { outline:3px solid var(--sel); }
  .tile .loc { font-weight:700; font-size:12px; }
  .tile .badge { display:inline-block; background:#0006; border-radius:3px;
                 padding:0 4px; margin:1px 2px 0 0; font-size:10px; }
  .tile .badge { cursor:pointer; }
  .tile .badge:hover { outline:1px solid #fff; }
  /* persistent selection: the clicked channel badge stays highlighted (survives
     mouseout) until another item (tile or channel) is clicked. */
  .tile .badge.selbadge { outline:2px solid var(--sel); background:#0009; }
  /* peer channel badges of the same flow as the clicked channel: partner =
     opposite direction (the source/destination), coop = same direction
     (cooperating group, e.g. broadcast peers / ping-pong). Distinct dotted vs
     dashed lines and colors so they read differently from the solid selbadge. */
  .tile .badge.peerbadge-partner { outline:2px dotted #ffb347; background:#0009; }  /* source/dest = amber dotted */
  .tile .badge.peerbadge-coop    { outline:2px dashed #7ee081; background:#0009; }  /* same group  = green dashed */
  .tile.peer-recv { outline:3px solid #35d0e0; }          /* receivers = cyan  */
  .tile.peer-send { outline:3px dashed #e05fd0; }          /* senders  = magenta*/
  .tile.flowsel   { box-shadow:0 0 0 2px #fff inset; }     /* the clicked badge's tile */
  .legend { margin-top:14px; font-size:12px; color:#aaa; }
  .legend .sw { display:inline-block; width:12px; height:12px; border-radius:3px;
                vertical-align:-2px; margin-right:4px; }
  .panel h2 { font-size:14px; margin:14px 0 6px; border-bottom:1px solid #333;
              padding-bottom:4px; }
  .kv { font-size:13px; margin:2px 0; }
  .kv b { color:#9cc; }
  ul.sum { margin:4px 0; padding-left:18px; font-size:13px; }
  ul.sum li { margin:2px 0; }
  .contract { color:#c9a; font-size:12px; }
  pre.code { background:#111; border:1px solid #333; border-radius:6px; padding:10px;
             overflow:auto; font-size:12px; line-height:1.4; white-space:pre; }
  .lref { color:#888; font-size:12px; margin:4px 0; }
  /* code-piece file path banner shown at the top of the file (host.cc) frame */
  .codepath { color:#8ec; font-size:11px; margin:2px 0 8px; word-break:break-all;
              padding:4px 8px; background:#12252b; border:1px solid #244;
              border-radius:4px; }
  .codepath b { color:#6ab; font-weight:600; }
  .codepath .cpath { color:#bfe; -webkit-user-select:all; user-select:all; }
  .tabs { margin-top:8px; }
  .tab { display:inline-block; padding:4px 10px; border:1px solid #333;
         border-bottom:none; cursor:pointer; background:#252525; border-radius:6px 6px 0 0; }
  .tab.act { background:#111; color:#fff; }
  .tabbody { border:1px solid #333; padding:10px; border-radius:0 6px 6px 6px; }
  .hide { display:none; }
  .kw { color:#569cd6; } .fn { color:#dcdcaa; } .num { color:#b5cea8; }
  .cm { color:#6a9955; }
  .pann { color:#8a8; font-style:italic; margin-left:1ch; }
  .gap { color:#c8905a; font-style:italic; opacity:0.75; }
  .bdpretty { font-family:monospace; font-size:11.5px; line-height:1.4; color:#6a9955;
              background:#0f1a10; border-left:3px solid #4a5a2a; border-radius:4px;
              margin:2px 0 8px 22px; padding:6px 10px; white-space:pre; overflow-x:auto; }
  .placeholder { color:#666; margin-top:40px; }
  /* kernel channel<->argument mapping table (High level tab, core tiles) */
  .kmap { border-collapse:collapse; margin:6px 0 8px; font-size:12px; }
  .kmap th, .kmap td { border:1px solid #333; padding:3px 8px; text-align:left; }
  .kmap th { background:#252525; color:#bbb; font-weight:600; }
  .kmap td.arrow { color:#8ec; text-align:center; }
  .kmap .win { color:#dcdcaa; }
  .kmap .mlock { color:#6a9; }
  .kmap .morder { color:#c93; }
  .kmap .mnone { color:#a55; }
  /* kernel source view (Low level tab, core tiles) */
  .khl { background:#3a3410; border-left:3px solid #e0a800; }
  .kshowall { margin:4px 0 8px; padding:3px 10px; font-size:12px; cursor:pointer;
              background:#252525; color:#ddd; border:1px solid #444; border-radius:4px; }
  .kshowall:hover { background:#333; }
  .kfileref { color:#8ec; font-size:11px; }
  /* merged "kernel code" sub-tab: stacked source/wrapper/bcf sections */
  .ksec { margin:0 0 6px; }
  .ksechdr { color:#8ec; font-size:11px; text-transform:uppercase; letter-spacing:.5px;
             margin:6px 0 2px; opacity:.8; }
  .ksecsep { border-top:1px dashed #444; margin:10px 0; }
  .subtabs { margin:2px 0 8px; }
  .subtab { display:inline-block; padding:2px 9px; border:1px solid #444; cursor:pointer;
            background:#252525; border-radius:4px; margin-right:6px; font-size:12px; }
  .subtab.act { background:#0d3a2a; color:#fff; border-color:#3a7d44; }
  .rline { font-family:monospace; font-size:12px; line-height:1.55; white-space:pre-wrap; }
  .lno { color:#666; user-select:none; margin-right:6px; }
  /* right-aligned gutter for the dfschedule Middle IR original line numbers */
  .mlno { color:#666; user-select:none; display:inline-block; min-width:3.2em;
          text-align:right; margin-right:10px; }
  .midctrls { margin:6px 0; }
  .midctrls button { margin-right:8px; padding:3px 10px; cursor:pointer; }
  .irfull { max-height:420px; overflow:auto; }
  .irln { display:block; }
  .irlno { color:#555; user-select:none; display:inline-block; min-width:3.6em;
           text-align:right; margin-right:10px; }
  .irhi { background:#3a3320; }              /* highlighted tile/channel line */
  .irhi .irlno { color:#e0a800; }
  .irhidden { display:block; }               /* revealed folded lines */
  .irhidden.hide { display:none; }           /* beats global .hide (equal spec) */
  .irfold { display:block; color:#7fb0ff; cursor:pointer; user-select:none;
            background:#1b2330; padding:0 6px; }
  .irfold:hover { background:#243043; }
  .irfold.irfold-open { color:#88cc88; }
  .kb { display:inline-block; border-radius:3px; padding:0 4px; margin-right:5px;
        font-size:10px; font-weight:700; vertical-align:1px; }
  .kb.bd_config{background:#4a5a2a;color:#dfffb0} .kb.createio{background:#2d4a6a;color:#cfe6ff}
  .kb.startio{background:#6a4a2d;color:#ffe0c0} .kb.lock{background:#5a2d5a;color:#f0c0f0}
  .kb.wait{background:#6a2d2d;color:#ffc0c0}
  .kb.enable_ooo{background:#2d5a5a;color:#c0f0f0}
  .kb.loop{background:#3a3a6a;color:#cfd0ff}
  .kb.loopidx{background:#33334d;color:#b0b0e0}
  .kb.loopend{background:#3a3a6a;color:#cfd0ff}
  .rnote{color:#9a9ad0;font-style:italic;margin-left:8px;font-size:11px;}
  /* ── live debug overlay (active only when served by schedule_debug_server) ── */
  #live { font-size:12px; flex:1 1 auto; display:flex; flex-direction:column;
          min-height:0; }
  #live label, #overlayctl label { cursor:pointer; }
  /* overlay controls live in the top-left array/partition view (after #grid) */
  #overlayctl { margin-top:8px; font-size:12px; }
  .ltab { display:inline-block; padding:2px 8px; margin:6px 4px 0 0; border:1px solid #444;
          border-radius:4px; background:#252525; cursor:pointer; font-size:11px; }
  .ltab.act { background:#0d2a3a; color:#fff; border-color:#2d6a9f; }
  #runbtn { margin-left:6px; padding:4px 12px; cursor:pointer; }
  #stopbtn { margin-left:6px; padding:4px 12px; cursor:pointer;
             background:#5a2d2d; color:#fdd; border:1px solid #843; border-radius:4px; }
  #loadlogbtn { margin-left:6px; padding:4px 12px; cursor:pointer; }
  #stopbtn:hover { background:#7a3535; }
  #runbtn:disabled, #stopbtn:disabled { opacity:.4; cursor:not-allowed; }
  #testconn:disabled, #conreload:disabled { opacity:.4; cursor:not-allowed; }
  #conin.disabled, #conin:disabled { opacity:.4; cursor:not-allowed; }
  #live label.disabled, #overlayctl label.disabled { opacity:.5; }
  #boardHost { background:#111; color:#ddd; border:1px solid #444; border-radius:4px;
               padding:3px 6px; font-family:monospace; }
  #livestatus { color:#8ab; font-size:11px; margin-top:6px; min-height:14px; }
  .consolebox { background:#0a0a0a; border:1px solid #333; border-radius:6px; padding:8px;
                font-size:11px; line-height:1.35; max-height:220px; overflow:auto;
                white-space:pre-wrap; word-break:break-word; max-width:100%;
                min-width:0; margin-top:8px; }
  /* the run/log console fills the resizable bottom-left region */
  #console { flex:1 1 auto; max-height:none; min-height:100px; }
  .livebar { display:block; margin-top:3px; font-size:9px; font-weight:700; color:#fff;
             border-radius:3px; padding:0 4px; text-align:center; letter-spacing:.5px; }
  .dbgpanel { margin-top:10px; border:1px solid #333; border-radius:6px; padding:10px; }
  .dbgpanel h2 { margin-top:0; }
  .dbgpresets { margin-bottom:6px; }
  .dbgpresets .ltab { margin-top:0; }
  #dbgcmd { background:#111; color:#ddd; border:1px solid #444; border-radius:4px;
            padding:4px 6px; font-family:monospace; }
  #dbgrun { margin-left:6px; padding:3px 10px; cursor:pointer; }
  /* ── tabbed command console (aiegdb | LLM) ── */
  #contabs { flex:0 0 auto; margin-bottom:6px; }
  .contab { display:inline-block; padding:3px 12px; border:1px solid #444;
            border-bottom:none; cursor:pointer; background:#252525;
            border-radius:6px 6px 0 0; font-size:12px; margin-right:4px; }
  .contab.act { background:#111; color:#fff; }
  #conpane { flex:1 1 0; display:flex; flex-direction:column; min-height:0; }
  #llmpane { flex:1 1 0; display:flex; flex-direction:column; min-height:0;
             position:relative; }
  /* ID rules above (specificity 1,0,0) beat the base .hide (0,1,0); this
     ID-qualified rule (1,1,0) lets the tab switch actually hide a pane. */
  #conpane.hide, #llmpane.hide { display:none; }
  /* LLM (embedded Claude Code) pane — mirrors #conterm/#conout/#conin. */
  #llmterm { flex:1 1 0; display:flex; flex-direction:column; min-height:0;
             margin-top:6px; padding:6px; background:#111; border:1px solid #444;
             border-radius:4px; overflow:hidden; cursor:text; }
  #llmout { flex:1 1 auto; min-height:0; overflow:auto; margin:0;
            background:transparent; white-space:pre-wrap; word-break:break-word;
            font-family:monospace; font-size:12px; line-height:1.4; }
  /* LLM transcript syntax coloring (markers only). */
  #llmout .llm-you       { color:#8ec; font-weight:bold; }
  #llmout .llm-tool      { color:#7aa2f7; }
  #llmout .llm-toolname  { color:#e0af68; font-weight:bold; }
  #llmout .llm-toolresult{ color:#666; }
  #llmout .llm-error     { color:#f7768e; font-weight:bold; }
  #llmout .llm-file      { color:#9ece6a; }
  #llmout .llm-line      { color:#e0af68; }
  /* LLM markdown + fenced-code highlighting. */
  #llmout .md-h        { color:#7dcfff; font-weight:bold; }
  #llmout .md-bullet   { color:#7aa2f7; }
  #llmout strong       { color:#c0caf5; font-weight:bold; }
  #llmout .md-code     { background:#1b1b2b; color:#e0af68; padding:0 3px;
                         border-radius:3px; }
  #llmout .md-block    { background:#0d0d16; border:1px solid #2a2a3a;
                         border-radius:4px; padding:6px 8px; margin:4px 0;
                         overflow:auto; white-space:pre; }
  #llmout .md-block code { background:none; padding:0; color:#c0caf5; }
  #llmout .cm-keyword  { color:#bb9af7; }
  #llmout .cm-string   { color:#9ece6a; }
  #llmout .cm-comment  { color:#565f89; font-style:italic; }
  #llmout .cm-number   { color:#ff9e64; }
  #llminline { flex:0 0 auto; display:flex; align-items:center; margin-top:6px;
               border-top:1px solid #333; padding-top:6px; }
  #llmprompt { color:#8ec; margin-right:6px; white-space:nowrap;
               font-family:monospace; }
  #llmin { flex:1 1 auto; background:transparent; border:none; outline:none;
           color:#ddd; font-family:monospace; padding:0; }
  #llmsend, #llmreset { margin-left:6px; padding:3px 10px; cursor:pointer; }
  /* LLM password modal overlay (matches the #llmout dark palette). */
  #llmauth { position:absolute; inset:0; display:flex; align-items:center;
             justify-content:center; background:rgba(0,0,0,0.7); z-index:50; }
  /* ID rule (1,0,0) beats base .hide (0,1,0); this ID-qualified rule (1,1,0)
     lets the JS actually hide the modal (same trap as #llmpane.hide above). */
  #llmauth.hide { display:none; }
  #llmauthbox { background:#111; border:1px solid #444; border-radius:6px;
                padding:16px 18px; width:280px; font-family:monospace;
                color:#ddd; box-shadow:0 4px 20px rgba(0,0,0,0.6); }
  #llmauthbox .t { color:#8ec; font-weight:bold; margin-bottom:8px; }
  #llmauthin { width:100%; box-sizing:border-box; background:#0d0d16;
               border:1px solid #2a2a3a; border-radius:4px; color:#ddd;
               font-family:monospace; padding:5px 6px; outline:none; }
  #llmauthok { margin-top:10px; padding:4px 12px; cursor:pointer; }
  #llmautherr { color:#f7768e; font-size:11px; margin-top:6px; min-height:14px; }
</style>
</head>
<body>
<div id="left">
  <div id="lefttop">
  <h1>AIE Schedule View</h1>
  <div class="sub" id="meta"></div>
  <div id="grid"></div>
  <!-- live status overlay controls belong to the array/partition view: they
       drive the per-tile overlay rendered onto #grid above. -->
  <div id="overlayctl">
    <label><input type="checkbox" id="liveToggle"> Live status overlay</label>
    <div id="overlaytabs">
      <span class="ltab act" data-w="dma">DMA</span>
      <span class="ltab" data-w="cores">Cores</span>
      <span class="ltab" data-w="events">Events</span>
    </div>
    <div id="livestatus"></div>
  </div>
  <div class="legend">
    <div><span class="sw" style="background:var(--shim)"></span>shim (row 0, host&lt;-&gt;array)</div>
    <div><span class="sw" style="background:var(--core)"></span>core (compute)</div>
    <div>badge = channel dir (S2MM recv / MM2S send)</div>
    <div>click a badge to highlight its flow peers:</div>
    <div><span class="sw" style="background:#35d0e0"></span>receivers (S2MM) of the clicked flow</div>
    <div><span class="sw" style="background:#e05fd0"></span>senders (MM2S) of the clicked flow (dashed)</div>
    <div>peer channel badges of the clicked flow:</div>
    <div><span class="sw" style="border:2px dotted #ffb347;background:transparent"></span>source/destination channel (opposite dir, dotted)</div>
    <div><span class="sw" style="border:2px dashed #7ee081;background:transparent"></span>cooperating channel (same dir group, dashed)</div>
    <button id="gbtn" style="margin-top:10px">Show global / kernel-group code</button>
  </div>
  </div>
  <div id="lhsplitter" title="Drag to resize (top / bottom)"></div>
  <div id="leftbottom">
  <div id="live">
    <div id="devrow" style="margin-bottom:6px;">
      <label>Board:
        <select id="deviceSel">
          <option value="">&mdash; select device &mdash;</option>
<!--__PALMYRA_OPTION__-->          <option value="vek385">vek385</option>
        </select>
      </label>
      <input type="text" id="boardHost" class="hide"
             placeholder="vek385 board hostname" style="margin-left:6px;">
      <button id="testconn" disabled style="margin-left:6px;">Test connect</button>
      <button id="runbtn" disabled>Run test</button>
      <button id="stopbtn" disabled>Force stop</button>
      <button id="loadlogbtn">Load log</button>
      <span id="connstatus" style="margin-left:6px; font-size:11px; color:#8ab;"></span>
      <div id="connhint" class="hide" style="margin-top:6px; padding:6px 8px;
           font-size:11px; color:#f6c177; background:#2a1f14;
           border:1px solid #5a3d1a; border-radius:4px;">
        Connection failed. On the target test board, start the hw_server via xsdb:
        <code style="color:#e0def4; background:#1a1622; padding:1px 4px;
              border-radius:3px;">exec hw_server -stcp:0.0.0.0:3121</code>
      </div>
    </div>
    <pre id="console" class="consolebox hide"></pre>
  </div>
  </div>
</div>
<div id="splitter" title="Drag to resize"></div>
<div id="right">
  <div id="panel" class="panel">
    <div class="placeholder">Click a tile to see its high-level summary and low-level host.cc code.</div>
  </div>
  <div id="rhsplitter" class="hide" title="Drag to resize (panel / console)"></div>
  <div id="cmdconsole" class="hide">
    <div id="contabs">
      <span class="contab act" data-pane="conpane">aiegdb</span>
      <span class="contab" data-pane="llmpane">LLM</span>
    </div>
    <div id="conpane">
    <div id="conhdr">aiegdb &mdash; <span id="contarget">partition</span>
      <button id="conreload" title="kill + restart aiegdb.py (reloads edited code)">Reload aiegdb.py</button></div>
    <div id="conhelp">aiegdb commands: target tile 0 0 | target channel mm2s0 | dma status |
      bd | event | pc | dma counter | reg read OFF | up | top | help</div>
    <div id="conterm">
      <div id="conpromptline"><span id="conprompt">partition&gt;</span><input id="conin"
        placeholder="type an aiegdb command, press Enter (e.g. dma status, help)"></div>
      <div id="conlast"><span id="conlastlbl">last:</span><span id="conlastcmd">&mdash;</span>
        <span id="conlastres"></span></div>
      <pre id="conout" class="code">(aiegdb console &mdash; click a tile or type 'help')</pre>
    </div>
    </div>
    <div id="llmpane" class="hide">
      <div id="conhdr">LLM &mdash; embedded Claude Code (repo-aware AIE debug agent)
        <button id="llmreset" title="kill + respawn claude (new conversation)">New chat</button></div>
      <div id="conhelp">Ask about this schedule, the applog, or the codebase.
        Tool calls appear as [tool: ...] markers. Streams live.</div>
      <div id="llmterm">
        <div id="llmout">(LLM console &mdash; type a question and press Enter)</div>
        <div id="llminline"><span id="llmprompt">you&gt;</span><input id="llmin"
          placeholder="ask Claude Code (e.g. summarize this schedule), press Enter">
          <button id="llmsend">Send</button></div>
      </div>
      <div id="llmauth" class="hide">
        <div id="llmauthbox">
          <div class="t">LLM locked</div>
          <input type="password" id="llmauthin" placeholder="password"
                 autocomplete="off">
          <button id="llmauthok">Unlock</button>
          <div id="llmautherr"></div>
        </div>
      </div>
    </div>
  </div>
</div>
<script>
const DATA = /*__DATA__*/ null;

function esc(s){ return (s==null?'':(''+s)).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
const KIND_LABEL = {bd_config:'BD', createio:'IO', startio:'START', wait:'WAIT',
  lock:'LOCK', enable_ooo:'OOO', loop:'FOR', loopidx:'IDX', loopend:'END'};
function renderRelevant(t){
  const det = t.relevant_detail || [];
  if (!det.length) return '<div class="placeholder">(no relevant lines)</div>';
  return det.map(d => {
    const badges = (d.kinds||[]).map(k =>
      '<span class="kb '+k+'">'+(KIND_LABEL[k]||k)+'</span>').join('');
    const note = d.note ? '<span class="rnote">// '+esc(d.note)+'</span>' : '';
    // Comment-first layout: parameter explanation, then the code line.
    const pp = paramPretty(d.code, d.bd_comment);
    let html = pp ? '<pre class="bdpretty">'+esc(pp)+'</pre>' : '';
    html += '<div class="rline"><span class="lno">L'+d.line+'</span>'+badges+hl(d.code)+note+'</div>';
    return html;
  }).join('');
}
// Kernel channel<->argument mapping (High level tab, core tiles). Renders the
// channel->window/arg correlation table plus the relevant kernel.cc slice so the
// user can match each DMA channel to the kernel input/output parameter.
function renderKernelMatch(t){
  const km = t.high_level && t.high_level.kernel_match;
  const kv = DATA.kernel;
  if(!km || !kv) return '';
  const rows = (km.matches||[]).map(m => {
    const mcls = m.method==='addr'?'mlock':(m.method==='addr(prov)'?'morder':'mnone');
    const win = m.window ? '<span class="win">'+esc(m.window)+'</span>' : '<span class="mnone">(unmatched)</span>';
    const arg = (m.arg!=null) ? ('arg'+m.arg) : '\u2014';
    const adr = (m.addrs_hex||[]).join('/') || '\u2014';
    const bsy = (m.bcf_syms||[]).join('/') || '\u2014';
    return '<tr>'+
      '<td>'+esc(m.direction)+esc(''+m.channel)+'</td>'+
      '<td>'+esc(adr)+'</td>'+
      '<td class="arrow">\u2192</td>'+
      '<td>'+win+'</td>'+
      '<td>'+arg+'</td>'+
      '<td>'+esc(bsy)+'</td>'+
      '<td class="'+mcls+'">'+esc(m.method)+'</td>'+
    '</tr>';
  }).join('');
  const table =
    '<table class="kmap"><thead><tr>'+
      '<th>channel</th><th>bd addr(s)</th><th></th><th>window</th>'+
      '<th>arg</th><th>bcf buffer</th><th>via</th>'+
    '</tr></thead><tbody>'+rows+'</tbody></table>';
  const src = kv.source;
  const fn = kv.function
    ? '<div class="kv"><b>kernel fn:</b> '+esc(kv.function)+
      (src&&src.file?' <span class="lref">('+esc(src.file)+')</span>':'')+'</div>'
    : '';
  const hint = src
    ? '<div class="lref">see the <b>'+esc(src.file)+'</b> sub-tab in Low level for the kernel body; click a channel to isolate its argument\u2019s code.</div>'
    : '';
  return '<div class="kv"><b>channel \u2194 kernel argument (by BD buffer address):</b></div>'+
         table+fn+hint;
}
// Kernel function source (Low level tab, core tiles). For a focused channel it
// isolates the "code piece" = the lines that directly use the mapped parameter
// (win_a/win_b/win_c) and offers a "Show all" toggle that reveals the whole
// function with those lines highlighted. For a tile it shows the full body.
function renderKernelSource(t, ch, focused){
  const src = DATA.kernel && DATA.kernel.source;
  if(!src) return '<div class="placeholder">(no kernel source found)</div>';
  const km = t.high_level && t.high_level.kernel_match;
  let refset = null, param = null, argn = null;
  if(focused && km){
    const m = (km.matches||[]).find(x =>
      x.direction===ch.direction && x.channel===ch.channel);
    if(m && m.arg!=null && src.params && src.params[m.arg]){
      param = src.params[m.arg];
      refset = {}; (param.ref_lines||[]).forEach(n => refset[n]=1);
      argn = m.arg;
    }
  }
  const hdr = '<div class="lref">kernel: <b>'+esc(src.file)+'</b> &mdash; '+
    esc(src.function)+'() lines '+src.start_line+'-'+src.end_line+'</div>';
  const full = (src.lines||[]).map(r => {
    const hlcls = (refset && refset[r.line]) ? ' khl' : '';
    return '<div class="rline'+hlcls+'"><span class="lno">L'+r.line+'</span>'+
      hl(r.code)+'</div>';
  }).join('');
  if(focused && refset && param){
    // Show the related lines; fold the rest into "//line a-b" markers.
    const piece = renderKFolded(src.lines||[], refset, 'src');
    return hdr +
      '<div class="kv">channel <b>'+esc(ch.direction)+esc(''+ch.channel)+
        '</b> \u2192 arg'+argn+' <span class="win">'+esc(param.name)+'</span> ('+
        esc(param.dir)+')</div>' +
      '<button class="kshowall">Show all</button>' +
      '<div id="kern-piece">'+
        (piece||'<div class="placeholder">(no direct uses found)</div>')+'</div>' +
      '<div id="kern-full" class="hide">'+full+'</div>';
  }
  return hdr + '<div id="kern-full">'+full+'</div>';
}
// Resolve the kernel window a focused channel maps to (via kernel_match), or
// null when not focused / no match. Shared by the kernel.cc + .bcf sub-tabs.
function focusedWindow(t, ch, focused){
  if(!focused || !ch) return null;
  const km = t.high_level && t.high_level.kernel_match;
  if(!km) return null;
  const m = (km.matches||[]).find(x =>
    x.direction===ch.direction && x.channel===ch.channel);
  if(!m || !m.window) return null;
  return ((DATA.kernel && DATA.kernel.windows) || [])
    .find(w => w.name===m.window) || null;
}
// "kernel.cc" sub-tab: the generated wrapper (window_init / LOCK #defines /
// buffer decls). When a channel is focused, highlight its window's lines.
function renderKernelCC(t, ch, focused){
  const k = DATA.kernel;
  if(!k || !k.kernel_lines) return '<div class="placeholder">(no kernel.cc found)</div>';
  const win = focusedWindow(t, ch, focused);
  const hset = {};
  if(win){
    (win.lock_lines||[]).forEach(n => hset[n]=1);
    (win.buf_decl_lines||[]).forEach(n => hset[n]=1);
    if(win.def_line) hset[win.def_line]=1;
    if(win.init_line) hset[win.init_line]=1;
  }
  const hdr = '<div class="lref">kernel: <b>'+esc(k.file||'kernel.cc')+'</b>'+
    (win ? ' &mdash; window <span class="win">'+esc(win.name)+'</span>'+
      ' (buffers '+esc((win.buffers||[]).join(', ')||'-')+')' : '')+'</div>';
  // Focused (window matched): show only the window's lines, fold the rest.
  const body = win
    ? renderKFolded(k.kernel_lines||[], hset, 'kcc')
    : (k.kernel_lines||[]).map(r =>
        '<div class="rline"><span class="lno">L'+r.line+'</span>'+
        hl(r.code)+'</div>').join('');
  return hdr + '<div>'+body+'</div>';
}
// "*.bcf" sub-tab: buffer name -> tile address map. When a channel is focused,
// highlight the address lines of that window's ping/pong buffers.
function renderBcf(t, ch, focused){
  const b = DATA.bcf;
  if(!b || !b.lines) return '<div class="placeholder">(no .bcf found)</div>';
  const win = focusedWindow(t, ch, focused);
  const hset = {};
  if(win){
    const bufs = {}; (win.buffers||[]).forEach(x => bufs[x]=1);
    (b.symbols||[]).forEach(s => { if(bufs[s.name]) hset[s.line]=1; });
  }
  const hdr = '<div class="lref">bcf: <b>'+esc(b.file)+'</b>'+
    (win ? ' &mdash; buffers for <span class="win">'+esc(win.name)+'</span>' : '')+'</div>';
  // Focused (window matched): show only this window's buffer lines, fold the rest.
  const body = win
    ? renderKFolded(b.lines||[], hset, 'bcf')
    : (b.lines||[]).map(r =>
        '<div class="rline"><span class="lno">L'+r.line+'</span>'+
        hl(r.code)+'</div>').join('');
  return hdr + '<div>'+body+'</div>';
}
// Render a list of {line,code} rows with only the highlighted (hset) lines shown
// and each run of non-related lines collapsed into a clickable "//line a-b"
// fold marker (default collapsed). `prefix` keeps fold ids unique across the
// stacked kernel-code sections. Reuses the shared fold machinery (wireFolds /
// setFold / foldLabel).
function renderKFolded(lines, hset, prefix){
  const N = lines.length;
  const row = r => '<div class="rline'+(hset[r.line]?' khl':'')+'">'+
    '<span class="lno">L'+r.line+'</span>'+hl(r.code)+'</div>';
  let out = [], i = 0, fid = 0;
  while(i < N){
    const r = lines[i];
    if(hset[r.line]){ out.push(row(r)); i++; continue; }
    let j = i; while(j < N && !hset[lines[j].line]) j++;   // hidden run [i, j)
    let run = ''; for(let k = i; k < j; k++) run += row(lines[k]);
    const cnt = j - i, a = lines[i].line, b = lines[j-1].line;
    const id = prefix + '-' + (++fid);
    out.push('<span class="irfold" data-fold="'+id+'" data-cnt="'+cnt+
             '" data-a="'+a+'" data-b="'+b+'">'+foldLabel(cnt, true, a, b)+'</span>'+
             '<span class="irhidden hide" data-fold="'+id+'">'+run+'</span>');
    i = j;
  }
  return out.join('');
}
// Merged "kernel code" sub-tab content. Sections are stacked in dependency
// order (bcf buffer addresses -> generated kernel.cc wrapper -> conv2d_spatial.cc
// source) so the reader follows the data from where it lands in tile memory up
// to where the kernel body consumes it.
function renderKernelCode(t, ch, focused){
  const sections = [];
  if(DATA.bcf && DATA.bcf.lines)
    sections.push(['buffer address map', renderBcf(t, ch, focused)]);
  if(DATA.kernel && DATA.kernel.kernel_lines)
    sections.push(['generated wrapper', renderKernelCC(t, ch, focused)]);
  if(DATA.kernel && DATA.kernel.source)
    sections.push(['kernel source', renderKernelSource(t, ch, focused)]);
  if(!sections.length)
    return '<div class="placeholder">(no kernel code found)</div>';
  return sections.map(s =>
    '<div class="ksec"><div class="ksechdr">'+esc(s[0])+'</div>'+s[1]+'</div>'
  ).join('<div class="ksecsep"></div>');
}
function hl(code){
  let s = esc(code);
  s = s.replace(/\/\*[\s\S]*?\*\//g, m=>'<span class="cm">'+m+'</span>');
  s = s.replace(/\b(void|int|int32_t|int64_t|size_t|for|return|uint8_t)\b/g,'<span class="kw">$1</span>');
  s = s.replace(/\b(__Runtime_[a-zA-Z0-9_]+|XAie_[A-Za-z0-9_]+|__runtime_[a-z_]+)\b/g,'<span class="fn">$1</span>');
  s = s.replace(/\b(\d+)\b/g,'<span class="num">$1</span>');
  return s;
}
// Reformat a "/* DMA BD Config: k=v, ... */" comment into a friendly multi-line block.
// Returns the block text (to be placed in a <pre>), or '' when code is not a BD comment.
function prettyBd(code){
  const m = code.match(/DMA BD Config:\s*([\s\S]*?)\*\//);
  if(!m) return '';
  const kv = {};
  m[1].split(',').forEach(p=>{
    const i = p.indexOf('=');
    if(i>0) kv[p.slice(0,i).trim()] = p.slice(i+1).trim();
  });
  const g = k => kv[k];
  let o = '/* DMA BD Config \u2014 bd_id='+g('bd_id')+'\n *\n';
  o += ' *   Packet:      enable_packet='+g('enable_packet')+', packet_id='+g('packet_id')+'\n';
  o += ' *   Chaining:    next_bd='+g('next_bd')+', ooo_bd_id='+g('ooo_bd_id')+'\n';
  o += ' *   Locks:       acquire_lock_id='+g('acquire_lock_id')+',  acquire_lock_val='+g('acquire_lock_val')+'\n';
  o += ' *                release_lock_id='+g('release_lock_id')+',  release_lock_val='+g('release_lock_val')+'\n';
  o += ' *\n';
  o += ' *   Transfer:    len='+g('len')+'\n';
  const nd = g('num_dims')!=null ? parseInt(g('num_dims'),10) : 0;
  if(nd>0){
    o += ' *   Dimensions:  num_dims='+nd+'\n';
    const sstr = [];
    for(let i=0;i<nd;i++) sstr.push('stride='+g('d'+i+'_stride')+',');
    const w = Math.max.apply(null, sstr.map(s=>s.length));
    for(let i=0;i<nd;i++)
      o += ' *                d'+i+': '+sstr[i].padEnd(w)+' wrap='+g('d'+i+'_wrap')+'\n';
  }
  if(g('iter_step_size')!=null)
    o += ' *   Iteration:   step_size='+g('iter_step_size')+', wrap='+g('iter_wrap')+'\n';
  o += ' */';
  return o;
}
// Positional parameter names + one-line descriptions for the DMA createio
// runtime calls, so the "Relevant only" view can explain each argument.
const CREATEIO_PARAMS = {
  '__Runtime_dma_createio_4': ['tile_loc','dma_desc','channel_id','bd_id','direction'],
  '__Runtime_dma_createio':   ['tile_loc','dma_desc','channel_id','bd_id','direction','mem'],
};
const CREATEIO_DESC = {
  tile_loc:   'col,row of the owning tile',
  dma_desc:   'pre-built DMA buffer descriptor',
  channel_id: 'DMA channel index on the tile',
  bd_id:      'buffer-descriptor ID for this transfer',
  direction:  'DMA_MM2S: tile mem->stream, DMA_S2MM: stream->tile mem',
  mem:        'optional backing XAie_MemInst (NULL if none)',
};
// Split a call's argument list on top-level commas (ignore nested ()/[]).
function splitTopCommas(s){
  const out=[]; let depth=0, cur='';
  for(const ch of s){
    if(ch==='('||ch==='['||ch==='{') depth++;
    else if(ch===')'||ch===']'||ch==='}') depth--;
    if(ch===','&&depth===0){ out.push(cur); cur=''; } else cur+=ch;
  }
  if(cur.trim()!=='') out.push(cur);
  return out;
}
// Reformat a __Runtime_dma_createio[_4](...) call into a friendly block that
// names each positional argument. Returns '' when code is not such a call.
function createioPretty(code){
  const m = code.match(/(__Runtime_dma_createio(?:_4)?)\s*\(([\s\S]*?)\)\s*;/);
  if(!m) return '';
  const names = CREATEIO_PARAMS[m[1]];
  if(!names) return '';
  const args = splitTopCommas(m[2]).map(a=>a.trim());
  let o = '/* '+m[1]+' \u2014 argument map\n *\n';
  const w = Math.max.apply(null, names.map(n=>n.length));
  for(let i=0;i<names.length;i++){
    const val = (i<args.length) ? args[i] : '\u2014';
    o += ' *   '+names[i].padEnd(w)+' = '+val+
         '   ('+(CREATEIO_DESC[names[i]]||'')+')\n';
  }
  o += ' */';
  return o;
}
// Positional parameter names + one-line descriptions for the startio runtime
// calls (queue a BD chain on a DMA channel and return a wait handle).
// Signature: __Runtime_startio(dev, io, bd_id, repeat).
const STARTIO_PARAMS = {
  '__Runtime_startio':    ['dev','io','bd_id','repeat'],
  '_Runtime_startio_ooo': ['dev','io','bd_id','repeat'],
};
const STARTIO_DESC = {
  dev:    'XAie device instance handle',
  io:     'IO channel handle from createio (tile, channel, direction)',
  bd_id:  'starting buffer-descriptor ID enqueued on the channel',
  repeat: 'number of times the BD (chain) is started (repeat count)',
};
// Reformat a __Runtime_startio[_ooo](...) call into a friendly block that names
// each positional argument. Returns '' when code is not such a call.
function startioPretty(code){
  const m = code.match(/(__Runtime_startio|_Runtime_startio_ooo)\s*\(([\s\S]*?)\)\s*;/);
  if(!m) return '';
  const names = STARTIO_PARAMS[m[1]];
  if(!names) return '';
  const args = splitTopCommas(m[2]).map(a=>a.trim());
  let o = '/* '+m[1]+' \u2014 argument map\n *\n';
  const w = Math.max.apply(null, names.map(n=>n.length));
  for(let i=0;i<names.length;i++){
    const val = (i<args.length) ? args[i] : '\u2014';
    o += ' *   '+names[i].padEnd(w)+' = '+val+
         '   ('+(STARTIO_DESC[names[i]]||'')+')\n';
  }
  o += ' */';
  return o;
}
// First matching parameter-explanation block for a host.cc line: DMA BD Config,
// createio, or startio. `bdComment` is the attached "/* DMA BD Config */" text
// when the entry carries one. Returns '' when no decoder applies. The layout is
// comment-first (this block precedes the code line it explains).
function paramPretty(code, bdComment){
  return prettyBd(bdComment||code) || createioPretty(code) || startioPretty(code);
}
// Render a host.cc block from per-line {line,code} records, prefixing each line
// with its host.cc line number and injecting a friendly bdpretty block BEFORE
// every line that a decoder recognizes (comment explains the parameters, then
// the code line follows).
function renderFullBlock(codeLines, params){
  if(!codeLines || !codeLines.length) return '<pre class="code">(no lines attributed)</pre>';
  const byVar = {};
  (params||[]).forEach(p => { byVar[p.var] = p; });
  let out = '', buf = [], prev = null;
  const flush = () => { if(buf.length){ out += '<pre class="code">'+buf.join('\n')+'</pre>'; buf = []; } };
  codeLines.forEach(item => {
    // Non-contiguous jump: mark the elided host.cc lines with a gap comment.
    if(prev !== null && item.line > prev + 1){
      const a = prev + 1, b = item.line - 1;
      const span = (a === b) ? ('line '+a) : ('lines '+a+'-'+b);
      buf.push('<span class="gap">//... ('+span+' hidden)</span>');
    }
    prev = item.line;
    let row = '<span class="lno">L'+item.line+'</span>'+hl(item.code);
    // Inline arg annotation: resolved values of every vN referenced on this
    // line (deduped, skipping the var's own def line and unresolved vars).
    const seen = {}, anns = [];
    (item.code.match(/\bv\d+\b/g) || []).forEach(v => {
      const p = byVar[v];
      if(!p || p.value == null || seen[v]) return;
      if(p.def_line === item.line) return;
      seen[v] = 1;
      anns.push(v+'='+p.value);
    });
    // __runtime_buffer_arg((void*)<dec>): show the address in hex too.
    let bam;
    const bre = /__runtime_buffer_arg\(\s*(?:\(\s*void\s*\*\s*\)\s*)?(\d+)\s*\)/g;
    while((bam = bre.exec(item.code)) !== null){
      const dec = parseInt(bam[1], 10);
      anns.push(bam[1]+'=0x'+dec.toString(16).toUpperCase());
    }
    if(anns.length) row += '<span class="pann">// '+esc(anns.join(', '))+'</span>';
    // Parameter-explanation block precedes the code line it describes.
    const pb = paramPretty(item.code);
    if(pb){ flush(); out += '<pre class="bdpretty">'+esc(pb)+'</pre>'; }
    buf.push(row);
  });
  flush();
  return out;
}
// Render the dfschedule Middle IR slice. Accepts either a string (fallback /
// '(no matching ops)') or an array of {line, code} rows; array rows show the
// ORIGINAL 6_BlueprintToSchedule.mlir line number in a right-aligned gutter
// (context/separator rows carry line=null -> blank gutter, preserving columns).
function renderMiddleIR(mid){
  if (!mid || typeof mid === 'string')
    return '<pre class="code">' + esc(mid || '(no dfschedule IR)') + '</pre>';
  const body = mid.map(it =>
    '<span class="mlno">' + (it.line != null ? it.line : '') + '</span>'
      + esc(it.code)).join('\n');
  return '<pre class="code">' + body + '</pre>';
}

// Render the FULL 6_BlueprintToSchedule.mlir with the given 1-based source line
// numbers highlighted (amber). Runs of non-highlighted lines are folded into a
// clickable "N lines hidden" marker so the target (highlighted) lines are easy
// to locate. Uses the IR text embedded at generation time.
function renderFullIr(highlightLines){
  const ir = (DATA.dfschedule_ir || {});
  const txt = ir.text || '';
  if (!txt) return '<pre class="code">(full dfschedule IR not embedded)</pre>';
  const hi = new Set(highlightLines || []);
  const lines = txt.split('\n');
  const N = lines.length;
  const rowHtml = (code, n) =>
    '<span class="'+(hi.has(n)?'irln irhi':'irln')+'">' +
    '<span class="irlno">'+n+'</span>'+esc(code)+'</span>';
  const hdr = '<div class="lref">'+esc(ir.name||'')+' &mdash; '+N+
    ' lines, '+hi.size+' highlighted'+(hi.size?' (non-target folded)':'')+'</div>';
  // No highlights -> nothing to locate; show the whole file unfolded.
  if (!hi.size){
    let all=''; for (let i=0;i<N;i++) all += rowHtml(lines[i], i+1);
    return hdr + '<pre class="code irfull">'+all+'</pre>';
  }
  let out=[], i=0, fid=0;
  while (i<N){
    const n=i+1;
    if (hi.has(n)){ out.push(rowHtml(lines[i], n)); i++; continue; }
    let j=i; while (j<N && !hi.has(j+1)) j++;    // hidden run [i, j)
    let run=''; for (let k=i;k<j;k++) run += rowHtml(lines[k], k+1);
    const cnt=j-i; fid++;
    out.push('<span class="irfold" data-fold="'+fid+'" data-cnt="'+cnt+'">' +
             foldLabel(cnt, true) + '</span>' +
             '<span class="irhidden hide" data-fold="'+fid+'">'+run+'</span>');
    i=j;
  }
  return hdr + '<pre class="code irfull">'+out.join('')+'</pre>';
}

// Inline fold marker label: triangle reflects state so the marker itself is the
// expand/collapse control (no separate row). When a line range (a[,b]) is
// given, prefix a "//line a-b" tag (used by the kernel-code sub-tab to show the
// hidden line span for non-related code).
function foldLabel(cnt, collapsed, a, b){
  const s = cnt > 1 ? 's' : '';
  const rng = (a != null)
    ? '//line ' + ((b != null && b !== a) ? (a + '-' + b) : a) + '   ' : '';
  return collapsed
    ? rng + '\u25b6 \u22ef ' + cnt + ' line' + s + ' hidden \u2014 click to expand'
    : rng + '\u25bc ' + cnt + ' line' + s + ' shown \u2014 click to collapse';
}

// Apply a fold state to one marker/hidden pair (classes + visibility + label).
function setFold(f, h, collapsed){
  h.classList.toggle('hide', collapsed);
  f.classList.toggle('irfold-open', !collapsed);
  const a = (f.dataset.a !== undefined) ? +f.dataset.a : null;
  const b = (f.dataset.b !== undefined) ? +f.dataset.b : null;
  f.textContent = foldLabel(+f.dataset.cnt, collapsed, a, b);
}

// Attach inline fold/expand click handlers within a rendered full-IR container.
// onChange (optional) is called after any toggle so callers can refresh an
// Expand-all/Collapse-all label.
function wireFolds(box, onChange){
  box.querySelectorAll('.irfold').forEach(f => {
    const h = box.querySelector('.irhidden[data-fold="'+f.dataset.fold+'"]');
    if (!h) return;
    f.onclick = () => {
      setFold(f, h, !h.classList.contains('hide'));   // toggle to opposite state
      if (onChange) onChange();
    };
  });
}

const g = DATA.grid;
document.getElementById('meta').textContent =
  DATA.source.function + '  (host.cc lines ' + DATA.source.function_lines[0] +
  '-' + DATA.source.function_lines[1] + ')  grid ' + g.cols + '\u00d7' + g.rows +
  ((g.startcol !== null && g.startcol !== undefined)
    ? '  partition startcol=' + g.startcol + ' (phys_col = col + ' + g.startcol + ')'
    : '');

// build a loc->tile map
const byLoc = {};
DATA.tiles.forEach(t => byLoc[t.loc[0]+','+t.loc[1]] = t);

// loc -> DOM cell (filled during grid build)
const cellByLoc = {};
// loc -> live status bar element (filled during grid build)
const liveBar = {};

// flow membership: fi -> {send:[[c,r]], recv:[[c,r]]}
const flowMembers = {};
(DATA.flow_summary||[]).forEach(f => {
  const m = flowMembers[f.flow_index] = {send:[], recv:[]};
  (f.entries||[]).forEach(e => {
    (e.io_direction==='MM2S'?m.send:m.recv).push([e.tile_col,e.tile_row]);
  });
});

function clearPeers(){
  document.querySelectorAll('.tile.peer-recv,.tile.peer-send,.tile.flowsel')
    .forEach(el=>el.classList.remove('peer-recv','peer-send','flowsel'));
  document.querySelectorAll('.badge.peerbadge-partner,.badge.peerbadge-coop')
    .forEach(el=>el.classList.remove('peerbadge-partner','peerbadge-coop'));
}
// Highlight the cooperating channel badges of the same flow as the clicked one:
//   partner = opposite direction (the source for an S2MM / the destination for an
//             MM2S) -> dotted amber
//   coop    = same direction, different badge (broadcast peers / ping-pong)
//             -> dashed green
// The clicked badge itself keeps its solid .selbadge and is excluded here.
function highlightPeerBadges(fi, dir, clickedBadge){
  if(fi==null) return;
  document.querySelectorAll('.badge[data-flow="'+fi+'"]').forEach(b=>{
    if(b===clickedBadge) return;
    b.classList.add(b.dataset.role===dir ? 'peerbadge-coop' : 'peerbadge-partner');
  });
}
function highlightFlow(fi, loc, badgeEl){
  clearPeers();
  highlightPeerBadges(fi, badgeEl ? badgeEl.dataset.role : null, badgeEl);
  const fm = flowMembers[fi]; if(!fm) return;
  const mark=(list,cls)=>list.forEach(([c,r])=>{
    const el=cellByLoc[c+','+r]; if(!el) return;
    if(c===loc[0]&&r===loc[1]) el.classList.add('flowsel');
    else el.classList.add(cls);
  });
  mark(fm.recv,'peer-recv');
  mark(fm.send,'peer-send');
}

// grid: columns left->right, rows top (highest) -> bottom (0)
const grid = document.getElementById('grid');
grid.style.gridTemplateColumns = 'repeat(' + g.cols + ', 1fr)';
const rowsDesc = g.row_list.slice().sort((a,b)=>b-a);
rowsDesc.forEach(r => {
  for (let c=0; c<g.cols; c++){
    const t = byLoc[c+','+r];
    const cell = document.createElement('div');
    if (!t){ cell.style.visibility='hidden'; cell.className='tile'; grid.appendChild(cell); continue; }
    cell.className = 'tile ' + t.type;
    let badges = (t.dma_channels||[]).map((ch,ci) =>
      '<span class="badge" data-flow="'+ch.flow_index+'" data-role="'+ch.direction+'" data-idx="'+ci+'">'
        + ch.direction + ch.channel + '</span>').join('');
    cell.innerHTML = '<div class="loc">('+t.loc[0]+','+t.loc[1]+')</div>' +
                     '<div>'+t.type+'</div>' + badges;
    cell.title = (t.high_level.contracts||[]).join('\n') || t.type;
    cellByLoc[c+','+r] = cell;
    const lbar = document.createElement('div');
    lbar.className = 'livebar hide';
    cell.appendChild(lbar);
    liveBar[c+','+r] = lbar;
    cell.onclick = (ev) => {
      const b = ev.target.closest('.badge');
      if (b && b.dataset.idx !== undefined) {
        const ch = (t.dma_channels||[])[+b.dataset.idx];
        select(t, cell, ch, b);
        highlightFlow(+b.dataset.flow, t.loc, b);
      } else {
        select(t, cell);
        clearPeers();
      }
    };
    grid.appendChild(cell);
  }
});

// One-line summary for a single channel (mirrors build_summary in schedule_view.py).
function chanSummary(ch){
  const bd = (ch.bd_chain||[])[0] || {};
  const len = (bd.len!=null) ? bd.len : '?';
  const acq = (bd.acquire_lock||[])[0] || {};
  const rel = (bd.release_lock||[])[0] || {};
  const verb = ch.direction==='S2MM' ? 'recv' : 'send';
  let reps = '';
  const si = (ch.start_io||[])[0];
  if (si && si.repeat_count && si.repeat_count!==1) reps = ' x'+si.repeat_count;
  let lock = '';
  if (acq.id!=null || rel.id!=null) lock = ' lock'+acq.id+'/'+rel.id;
  return verb+' '+len+'B ch'+ch.channel+' '+ch.direction+reps+lock;
}
// Relevant host.cc lines for a single channel (from its host_lines entries).
function renderChannelLines(ch){
  const es = (ch.host_lines||[]).slice().sort((a,b)=>a.line-b.line);
  if (!es.length) return '<div class="placeholder">(no relevant lines)</div>';
  return es.map(e => {
    const badge = '<span class="kb '+e.kind+'">'+(KIND_LABEL[e.kind]||e.kind)+'</span>';
    const note = e.note ? '<span class="rnote">// '+esc(e.note)+'</span>' : '';
    // Comment-first layout: parameter explanation, then the code line.
    const pp = paramPretty(e.code, e.bd_comment);
    let html = pp ? '<pre class="bdpretty">'+esc(pp)+'</pre>' : '';
    html += '<div class="rline"><span class="lno">L'+e.line+'</span>'+badge+hl(e.code)+note+'</div>';
    return html;
  }).join('');
}

let selEl = null;
let selBadge = null;
function select(t, el, ch, badgeEl){
  if (selEl) selEl.classList.remove('sel');
  selEl = el; el.classList.add('sel');
  // Persistent channel selection: keep the clicked badge highlighted after the
  // cursor moves out, and clear it only when another item (tile or channel) is
  // clicked. A tile-only click (no badgeEl) clears any previous badge highlight.
  if (selBadge) selBadge.classList.remove('selbadge');
  selBadge = badgeEl || null;
  if (selBadge) selBadge.classList.add('selbadge');
  const hlv = t.high_level, lo = t.low_level;
  const focused = !!ch;
  // Full-block source: channel-scoped when a channel is focused, else the tile.
  const flo = (focused && ch.low_level) ? ch.low_level : lo;
  const p = document.getElementById('panel');

  // --- title ---
  const title = focused
    ? 'Tile ('+t.loc[0]+','+t.loc[1]+') &mdash; channel '+ch.direction+ch.channel+
      ' <span class="lref">(flow '+ch.flow_index+')</span>'
    : 'Tile ('+t.loc[0]+','+t.loc[1]+') &mdash; '+t.type;

  // --- high-level body ---
  let hiBody;
  if (focused) {
    const con = ch.contract ? '<div class="contract">'+esc(ch.contract)+'</div>' : '';
    hiBody =
      '<div class="kv"><b>role:</b> '+esc(hlv.role)+'</div>' +
      (hlv.kernel?'<div class="kv"><b>kernel:</b> '+esc(hlv.kernel)+'</div>':'') +
      '<div class="kv"><b>channel:</b> '+ch.direction+ch.channel+' (flow '+ch.flow_index+')</div>' +
      '<div class="kv"><b>transfer:</b> '+esc(chanSummary(ch))+'</div>' +
      (con?'<div class="kv"><b>contract:</b></div>'+con:'');
  } else {
    const sum = (hlv.summary||[]).map(s=>'<li>'+esc(s)+'</li>').join('');
    const con = (hlv.contracts||[]).map(s=>'<div class="contract">'+esc(s)+'</div>').join('');
    hiBody =
      '<div class="kv"><b>role:</b> '+esc(hlv.role)+'</div>' +
      (hlv.kernel?'<div class="kv"><b>kernel:</b> '+esc(hlv.kernel)+'</div>':'') +
      '<div class="kv"><b>transfers:</b></div><ul class="sum">'+sum+'</ul>' +
      (con?'<div class="kv"><b>contracts:</b></div>'+con:'') +
      renderKernelMatch(t);
  }

  // --- relevant lines body ---
  const relCount = focused ? (ch.host_lines||[]).length : (t.relevant_lines||[]).length;
  const relBody = focused ? renderChannelLines(ch) : renderRelevant(t);
  const relLabel = focused
    ? relCount+' relevant line(s) for channel '+ch.direction+ch.channel
    : relCount+' relevant line(s)';

  // --- middle (dfschedule IR) body ---
  const midIR = focused ? (ch.middle_ir||'') : (t.middle_ir||'');

  // --- code-piece file (debugcache/code) for the file-frame header + LLM ---
  const codeFile = focused ? (ch.code_file||'') : (t.code_file||'');
  // Core tiles carry a parsed kernel function body; surface its file name.
  const ksrc = (t.type==='core' && DATA.kernel && DATA.kernel.source)
    ? DATA.kernel.source : null;
  const isCore = !!ksrc;
  // Single "kernel code" sub-tab (core tiles): merges the kernel source
  // (conv2d_spatial.cc), the generated wrapper (kernel.cc) and the buffer
  // address map (.bcf) into one stacked view, each section headed by its file.
  const kcodeOn = (t.type==='core' &&
    ((DATA.kernel && (DATA.kernel.source || DATA.kernel.kernel_lines)) ||
     (DATA.bcf && DATA.bcf.lines)));
  const kfile = ksrc ? ksrc.file : '';
  const kfileTag = isCore
    ? ' <span class="kfileref">+ kernel '+esc(kfile)+'</span>' : '';
  const codePathBanner = codeFile
    ? '<div class="codepath"><b>code piece:</b> <span class="cpath">'+esc(codeFile)+'</span>'+kfileTag+'</div>'
    : (isCore
       ? '<div class="codepath"><b>kernel:</b> <span class="cpath">'+esc(kfile)+'</span></div>'
       : '');

  p.innerHTML =
    '<h2>'+title+'</h2>' +
    '<div class="tabs">' +
      '<span class="tab act" data-t="hi">High level</span>' +
      '<span class="tab" data-t="mid">Middle (dfschedule)</span>' +
      '<span class="tab" data-t="lo">Low level (host.cc)</span>' +
    '</div>' +
    '<div class="tabbody">' +
      '<div id="tab-hi">' + hiBody + '</div>' +
      '<div id="tab-mid" class="hide">' +
        '<div class="lref">dfschedule IR (6_BlueprintToSchedule) &mdash; ' +
          (focused ? 'channel '+ch.direction+ch.channel : 'tile') + ' scope</div>' +
        '<div class="midctrls">' +
          '<button id="loadFullIr">Load full dfschedule IR</button>' +
          '<button id="foldAll" class="hide">Expand all</button>' +
        '</div>' +
        '<div id="midContent">' + renderMiddleIR(midIR) + '</div>' +
      '</div>' +
      '<div id="tab-lo" class="hide">' +
        codePathBanner +
        '<div class="subtabs">' +
          '<span class="subtab act" data-lo="rel">Relevant only</span>' +
          '<span class="subtab" data-lo="full">Full block</span>' +
          (kcodeOn ? '<span class="subtab" data-lo="kern">kernel code</span>' : '') +
        '</div>' +
        '<div id="lo-rel">' +
          '<div class="lref">'+relLabel+'</div>' +
          relBody +
        '</div>' +
        '<div id="lo-full" class="hide">' +
          '<div class="lref">host.cc lines '+ (flo.line_start||'?') +'-'+ (flo.line_end||'?') +
            '  ('+(flo.ranges||[]).length+' range(s))' +
            (focused ? ' &mdash; channel '+ch.direction+ch.channel+' scope' : '') + '</div>' +
          renderFullBlock(flo.code_lines, focused ? ((ch.low_level||{}).params||null) : null) +
        '</div>' +
        (kcodeOn ? '<div id="lo-kern" class="hide">'+renderKernelCode(t, ch, focused)+'</div>' : '') +
      '</div>' +
    '</div>';
  const panes = {hi:'#tab-hi', mid:'#tab-mid', lo:'#tab-lo'};
  p.querySelectorAll('.tab').forEach(tab => tab.onclick = () => {
    p.querySelectorAll('.tab').forEach(x=>x.classList.remove('act'));
    tab.classList.add('act');
    Object.entries(panes).forEach(([k,sel]) =>
      p.querySelector(sel).classList.toggle('hide', tab.dataset.t!==k));
  });
  p.querySelectorAll('.subtab').forEach(st => st.onclick = () => {
    p.querySelectorAll('.subtab').forEach(x=>x.classList.remove('act'));
    st.classList.add('act');
    p.querySelector('#lo-rel').classList.toggle('hide', st.dataset.lo!=='rel');
    p.querySelector('#lo-full').classList.toggle('hide', st.dataset.lo!=='full');
    const kp = p.querySelector('#lo-kern');
    if (kp) kp.classList.toggle('hide', st.dataset.lo!=='kern');
  });
  // Kernel source "Show all" toggle: swap the isolated param code-piece for the
  // whole function body (with the param's lines highlighted), and back.
  const kbtn = p.querySelector('.kshowall');
  if (kbtn) {
    const kpiece = p.querySelector('#kern-piece');
    const kfull  = p.querySelector('#kern-full');
    kbtn.onclick = () => {
      const showAll = kfull.classList.contains('hide');
      kfull.classList.toggle('hide', !showAll);
      if (kpiece) kpiece.classList.toggle('hide', showAll);
      kbtn.textContent = showAll ? 'Show piece only' : 'Show all';
    };
  }
  // Kernel-code sub-tab: make the "//line a-b" fold markers (non-related code,
  // collapsed by default) clickable to expand/collapse in place.
  const klo = p.querySelector('#lo-kern');
  if (klo) wireFolds(klo);

  // --- full dfschedule IR (embedded) load/highlight/fold ---
  // The slice rows carry the ORIGINAL 6_BlueprintToSchedule.mlir line numbers;
  // reuse them as the highlight set for the full-file view. The button toggles
  // the SAME window (#midContent) between the slice piece and the full IR (with
  // non-target runs folded); a second button expands/collapses all folds.
  const midLines = Array.isArray(midIR)
    ? midIR.filter(r => r.line != null).map(r => r.line) : [];
  const loadBtn    = p.querySelector('#loadFullIr');
  const foldAllBtn = p.querySelector('#foldAll');
  const midContent = p.querySelector('#midContent');
  if (loadBtn && midContent) {
    const sliceHtml = midContent.innerHTML;     // cache the slice view
    let showingFull = false;
    const refreshFoldAll = () => {
      if (!foldAllBtn) return;
      foldAllBtn.textContent =
        midContent.querySelector('.irhidden.hide') ? 'Expand all' : 'Collapse all';
    };
    loadBtn.onclick = () => {
      showingFull = !showingFull;
      if (showingFull) {
        midContent.innerHTML = renderFullIr(midLines);
        wireFolds(midContent, refreshFoldAll);
        loadBtn.textContent = 'Show slice only';
        if (foldAllBtn) foldAllBtn.classList.remove('hide');
        refreshFoldAll();
        const first = midContent.querySelector('.irhi');
        if (first) first.scrollIntoView({block: 'center'});
      } else {
        midContent.innerHTML = sliceHtml;
        loadBtn.textContent = 'Load full dfschedule IR';
        if (foldAllBtn) foldAllBtn.classList.add('hide');
      }
    };
    if (foldAllBtn) {
      foldAllBtn.onclick = () => {
        const collapse = foldAllBtn.textContent.indexOf('Collapse') === 0;
        midContent.querySelectorAll('.irfold').forEach(f => {
          const h = midContent.querySelector('.irhidden[data-fold="'+f.dataset.fold+'"]');
          if (h) setFold(f, h, collapse);
        });
        refreshFoldAll();
      };
    }
  }

  // --- aiegdb console (only after a passing connection test) ---
  // Clicking a tile runs `target tile <col> <row>`; a channel click chains
  // `target channel <dir_ch>` so the console scope follows the UI.
  if (LIVE.connected) setConTarget(t, ch);

  // --- local-LLM cooperation: prepare (do NOT send) the tile/file location
  // context for the clicked tile/channel. It is attached to the NEXT user chat
  // message, and only if it differs from the last one attached (see llmSend). ---
  setLLMContext(t, ch, codeFile);
}

// Prepare the pending LLM context string from the clicked tile/channel and its
// code-piece file. Nothing is sent here; llmSend attaches it to the next user
// message (deduped against the last attached context).
function setLLMContext(t, ch, codeFile){
  if (!codeFile){ LLM.ctx = null; return; }
  const where = ch
    ? 'tile ('+t.loc[0]+','+t.loc[1]+') channel '+ch.direction+ch.channel
    : 'tile ('+t.loc[0]+','+t.loc[1]+')';
  LLM.ctx = '[context] Currently inspecting '+where+'. The related code is '+codeFile;
}

// ─── aiegdb console (right-bottom, drives aiegdb.py --server) ─────────────────
// A terminal-style console over the daemon's persistent aiegdb REPL subprocess.
// The user types any aiegdb command; scope (partition->tile->channel) is kept by
// the subprocess and reported back via r.scope. Tile/channel clicks send
// `target ...` so the console follows the UI. "Reload aiegdb.py" respawns the
// subprocess (reloads edited code), resetting to partition scope.
const CON = { scope:'partition' };
function conAppend(text){
  const out = document.getElementById('conout');
  if (!out) return;
  out.textContent += (out.textContent && !out.textContent.endsWith('\n') ? '\n' : '') + text;
  out.scrollTop = out.scrollHeight;
}
function conSetScope(s){
  CON.scope = s || 'partition';
  const tgt = document.getElementById('contarget');
  if (tgt) tgt.textContent = CON.scope;
  const pr = document.getElementById('conprompt');
  if (pr) pr.textContent = CON.scope + '>';
}
function conSetLast(cmd, res){
  const c = document.getElementById('conlastcmd');
  const r = document.getElementById('conlastres');
  if (c) c.textContent = CON.scope + '> ' + cmd;
  if (r) r.textContent = res == null ? '' : (' \u2192 ' + res);
}
// applyScope=false lets a caller keep an already-set (optimistic) scope instead
// of adopting this command's returned r.scope — used for the intermediate
// `target tile` step of a channel selection so it doesn't downgrade the prompt.
function conSend(cmd, echo, applyScope){
  if (echo !== false) conAppend(CON.scope + '> ' + cmd);
  conSetLast(cmd, null);
  return api('/aiegdb', {method:'POST', headers:{'Content-Type':'application/json'},
                         body: JSON.stringify({cmd:cmd})})
    .then(r => {
      const out = r.output ? r.output.replace(/\n+$/,'') : '';
      if (out) conAppend(out);
      if (applyScope !== false && r.scope) conSetScope(r.scope);
      conSetLast(cmd, out || '(no output)');
    })
    .catch(() => { conAppend('daemon offline (static mode)');
                   conSetLast(cmd, 'daemon offline (static mode)'); });
}
// Tile/channel click: run `target tile c r`, then chain `target channel dir_ch`.
// The prompt is updated optimistically from the UI selection first, so the
// console scope follows the click immediately — independent of daemon latency,
// and even if the chained `target channel` request is slow or fails (its .catch
// would otherwise leave the prompt stuck at tile scope). A successful daemon
// response still confirms/refines the scope via r.scope. The optimistic string
// mirrors aiegdb's prompt format exactly: partition(startcol=N)/tile(c,r)/dir_ch
// (direction lowercased, matching _parse_dir_ch).
function setConTarget(t, ch){
  const box = document.getElementById('cmdconsole');
  if (box) box.classList.remove('hide');
  const rsp = document.getElementById('rhsplitter');
  if (rsp) rsp.classList.remove('hide');
  const sc = (g.startcol !== null && g.startcol !== undefined) ? g.startcol : 0;
  let optScope = 'partition(startcol=' + sc + ')/tile(' + t.loc[0] + ',' + t.loc[1] + ')';
  if (ch) optScope += '/' + ch.direction.toLowerCase() + ch.channel;
  conSetScope(optScope);
  // For a channel selection, suppress the intermediate tile step's scope apply
  // (applyScope=false) so it can't downgrade the optimistic channel prompt; the
  // chained `target channel` then confirms it. A tile-only click applies scope.
  const p = conSend('target tile ' + t.loc[0] + ' ' + t.loc[1], undefined, !ch);
  if (ch) p.then(() => conSend('target channel ' +
                               ch.direction.toLowerCase() + ch.channel));
}
document.getElementById('conin').addEventListener('keydown', e => {
  if (e.key === 'Enter'){ const v = e.target.value.trim();
    if (v){ conSend(v); e.target.value=''; } }
});
// Clicking anywhere in the terminal focuses the inline prompt, but don't
// steal focus mid-drag while the user is selecting text in the output.
document.getElementById('conterm').onclick = () => {
  const sel = window.getSelection();
  if (sel && sel.toString()) return;
  document.getElementById('conin').focus();
};
document.getElementById('conreload').onclick = () => {
  api('/aiegdb/reload', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'})
    .then(r => { conAppend('[reloaded aiegdb.py]'); if (r.scope) conSetScope(r.scope); })
    .catch(() => conAppend('daemon offline (static mode)'));
};

// ─── LLM console (embedded Claude Code, drives claude -p streaming) ───────────
// The daemon runs one persistent `claude -p --output-format stream-json` process
// in the repo root. llmSend writes one user turn; llmPoll tails the decoded
// transcript buffer (reusing the applog-tail idiom) until the turn ends.
// ctx: pending tile/file context prepared on the last tile/channel click.
// ctxSent: the context string last attached to a message (for change-only dedup).
const LLM = { off:0, poll:null, busy:false, text:'', ctx:null, ctxSent:null };
// Escape HTML so model output (which may contain <, >, &, code) is inert before
// we inject it as innerHTML for coloring.
function llmEscape(s){
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
// Colorize an ALREADY-escaped prose string: tool calls, tool results, you>
// prompts, error/offline markers, and file / file:line references.
function llmColorizeMarkers(s){
  s = s.replace(/^you&gt;.*$/gm, m => '<span class="llm-you">' + m + '</span>');
  s = s.replace(/\[tool: ([^\s\]]+)([^\]]*)\]/g,
    (m,name,rest) => '<span class="llm-tool">[tool: <span class="llm-toolname">'
      + name + '</span>' + rest + ']</span>');
  s = s.replace(/\[tool result\]/g, '<span class="llm-toolresult">[tool result]</span>');
  s = s.replace(/\[(llm error[^\]]*|daemon offline[^\]]*)\]/g,
    (m,inner) => '<span class="llm-error">[' + inner + ']</span>');
  s = s.replace(/\b([A-Za-z0-9_./-]+\.(?:py|cc|cpp|cxx|h|hpp|c|md|mlir|sh|json|txt|inc|td|html|elf|log))(?::(\d+))?\b/g,
    (m,file,line) => '<span class="llm-file">' + file + '</span>'
      + (line ? ':<span class="llm-line">' + line + '</span>' : ''));
  return s;
}
// Minimal offline highlighter for one fenced code block. `raw` is UNESCAPED
// code; escape first, then one alternation regex colors comments/strings/
// numbers/keywords. First-match-wins means a keyword inside a string or comment
// is left alone (the string/comment alt starts earlier and consumes it).
const LLM_TOK = /(\/\*[\s\S]*?\*\/|\/\/[^\n]*|#\s[^\n]*)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|\b(0x[0-9a-fA-F]+|\d+\.?\d*)\b|\b(if|else|elif|for|while|do|return|break|continue|switch|case|default|goto|int|float|double|char|bool|void|long|short|unsigned|signed|const|static|struct|class|public|private|protected|namespace|template|typename|typedef|enum|union|new|delete|sizeof|this|nullptr|true|false|NULL|auto|using|include|define|import|from|def|lambda|None|True|False|self|and|or|not|in|is|print|std)\b/g;
function llmHighlightCode(raw){
  return llmEscape(raw.replace(/\n$/,'')).replace(LLM_TOK, (m,c,s,n,k) =>
    c ? '<span class="cm-comment">' + c + '</span>' :
    s ? '<span class="cm-string">'  + s + '</span>' :
    n ? '<span class="cm-number">'  + n + '</span>' :
    k ? '<span class="cm-keyword">' + k + '</span>' : m);
}
// Render one prose (non-code) segment: escape, apply light markdown (headings,
// bullets, inline code, bold), then the marker colorizer.
function llmProse(raw){
  let s = llmEscape(raw);
  s = s.replace(/^(#{1,6})\s+(.*)$/gm, (m,h,t) => '<span class="md-h">' + t + '</span>');
  s = s.replace(/^(\s*)[-*+]\s+/gm, '$1<span class="md-bullet">\u2022 </span>');
  s = s.replace(/`([^`\n]+)`/g, (m,c) => '<code class="md-code">' + c + '</code>');
  s = s.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
  return llmColorizeMarkers(s);
}
// Split the transcript on ``` fences: even segments are prose, odd are code
// blocks (an unclosed trailing fence mid-stream is treated as code, so partial
// /llm/poll chunks still render safely).
function llmRenderText(raw){
  const parts = raw.split('```');
  let html = '';
  for (let i = 0; i < parts.length; i++){
    if (i % 2 === 0){ html += llmProse(parts[i]); continue; }
    const blk = parts[i], nl = blk.indexOf('\n');
    const body = nl >= 0 ? blk.slice(nl + 1) : blk;   // drop optional ```lang line
    html += '<pre class="md-block"><code>' + llmHighlightCode(body) + '</code></pre>';
  }
  return html;
}
function llmRender(){
  const out = document.getElementById('llmout');
  if (!out) return;
  const atBottom = (out.scrollHeight - out.scrollTop - out.clientHeight) < 4;
  out.innerHTML = llmRenderText(LLM.text);
  if (atBottom) out.scrollTop = out.scrollHeight;
}
function llmAppend(text){ LLM.text += text; llmRender(); }
function llmStopPoll(){ if (LLM.poll){ clearInterval(LLM.poll); LLM.poll = null; } }
function llmPollOnce(){
  if (LLM.busy) return;
  LLM.busy = true;
  api('/llm/poll?offset=' + LLM.off).then(r => {
    if (r.auth){ llmLock(); return; }
    if (r.error){ llmStopPoll(); return; }
    if (r.data) llmAppend(r.data);
    if (r.next != null) LLM.off = r.next;
    if (r.active === false) llmStopPoll();   // turn finished
  }).catch(() => { llmStopPoll(); llmAppend('\n[daemon offline (static mode)]\n'); })
    .finally(() => { LLM.busy = false; });
}
function llmSend(prompt){
  prompt = (prompt || '').trim();
  if (!prompt) return;
  // Attach the prepared tile/file context ONLY when it changed since the last
  // attach (first click, or a different tile/channel). Unchanged context is not
  // re-sent, so repeat messages about the same selection stay clean.
  let toSend = prompt;
  if (LLM.ctx && LLM.ctx !== LLM.ctxSent){
    toSend = LLM.ctx + '\n' + prompt;
    LLM.ctxSent = LLM.ctx;
    llmAppend((LLM.text.endsWith('\n') || LLM.text === '' ? '' : '\n')
              + LLM.ctx + '\n');
  }
  llmAppend((LLM.text.endsWith('\n') || LLM.text === '' ? '' : '\n')
            + 'you> ' + prompt + '\n');
  api('/llm', {method:'POST', headers:{'Content-Type':'application/json'},
               body: JSON.stringify({prompt:toSend})})
    .then(r => {
      if (r.auth){ llmLock(); return; }
      if (!r.ok){ llmAppend('\n[llm error: ' + (r.error || 'unknown') + ']\n'); return; }
      if (r.offset != null) LLM.off = r.offset;
      llmStopPoll();
      llmPollOnce();
      LLM.poll = setInterval(llmPollOnce, 700);
    })
    .catch(() => llmAppend('\n[daemon offline: LLM tab needs schedule_debug_server]\n'));
}
function llmReset(){
  llmStopPoll();
  api('/llm/reset', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'})
    .then(r => {
      if (r && r.auth){ llmLock(); return; }
      // New chat: forget which context was attached so it re-attaches next send.
      LLM.ctxSent = null;
      LLM.off = 0; LLM.text = '(new chat \u2014 context reset)\n'; llmRender();
    })
    .catch(() => llmAppend('\n[daemon offline: cannot reset]\n'));
}
// ── LLM password modal ──────────────────────────────────────────────────────
function llmAuthShow(){ const m=document.getElementById('llmauth');
  if (m){ m.classList.remove('hide');
    const i=document.getElementById('llmauthin'); if (i){ i.value=''; i.focus(); } } }
function llmAuthHide(){ const m=document.getElementById('llmauth');
  if (m) m.classList.add('hide');
  const e=document.getElementById('llmautherr'); if (e) e.textContent=''; }
// Lock the LLM tab: drop the (bad/absent) token, stop polling, show the modal.
function llmLock(){
  sessionStorage.removeItem('LLM_AUTH');
  llmStopPoll();
  llmAppend('\n[locked: enter password]\n');
  llmAuthShow();
}
// Ask the daemon whether auth is required; prompt if so and no token stored yet.
function llmCheckAuth(){
  api('/llm/auth').then(r => {
    if (r && r.required && !llmToken()) llmAuthShow();
  }).catch(() => {});
}
(function(){
  const inp = document.getElementById('llmin');
  const snd = document.getElementById('llmsend');
  const rst = document.getElementById('llmreset');
  const term = document.getElementById('llmterm');
  if (inp) inp.addEventListener('keydown', e => {
    if (e.key === 'Enter'){ const v = e.target.value.trim();
      if (v){ llmSend(v); e.target.value=''; } }
  });
  if (snd) snd.onclick = () => { const v = inp.value.trim();
    if (v){ llmSend(v); inp.value=''; } };
  if (rst) rst.onclick = llmReset;
  if (term) term.onclick = () => {
    const sel = window.getSelection();
    if (sel && sel.toString()) return;
    if (inp) inp.focus();
  };
  // Password modal: store the entered value as the session token, then hide.
  // Validation is implicit on the next /llm call (a 401 re-locks).
  const authin = document.getElementById('llmauthin');
  const authok = document.getElementById('llmauthok');
  const doUnlock = () => {
    const v = authin ? authin.value : '';
    if (!v){ const e=document.getElementById('llmautherr');
      if (e) e.textContent='enter a password'; return; }
    sessionStorage.setItem('LLM_AUTH', v);
    llmAuthHide();
  };
  if (authok) authok.onclick = doUnlock;
  if (authin) authin.addEventListener('keydown', e => {
    if (e.key === 'Enter') doUnlock();
  });
  // Prompt on first load if the daemon requires a password.
  llmCheckAuth();
})();
// Console tab switching: toggle .hide on #conpane/#llmpane + .act on the tabs.
document.querySelectorAll('#contabs .contab').forEach(tab => tab.onclick = () => {
  document.querySelectorAll('#contabs .contab').forEach(x => x.classList.remove('act'));
  tab.classList.add('act');
  const pane = tab.dataset.pane;
  const con = document.getElementById('conpane');
  const llm = document.getElementById('llmpane');
  if (con) con.classList.toggle('hide', pane !== 'conpane');
  if (llm) llm.classList.toggle('hide', pane !== 'llmpane');
  if (pane === 'llmpane') llmCheckAuth();
});

document.getElementById('gbtn').onclick = () => {
  clearPeers();
  if (selEl){ selEl.classList.remove('sel'); selEl=null; }
  const gl = DATA.global;
  document.getElementById('panel').innerHTML =
    '<h2>Global / kernel-group</h2>' +
    '<div class="kv">'+esc(gl.note)+'</div>' +
    '<pre class="code">'+hl(gl.code||'')+'</pre>';
};

// ─── Live debug overlay (talks to schedule_debug_server on the same origin) ───
// All additive: when opened as a static file:// (no daemon) every fetch fails
// and the page silently stays in static mode.
const LIVE = { enabled:false, connected:false, what:'dma', gridTimer:null,
               conTimer:null, logoff:0, gridBusy:false, logBusy:false,
               runActive:false, device:'', host:'',
               hwsrvOff:0, hwsrvTimer:null, hwsrvBusy:false };
const LSTATE = {
  running:['#2e7d32','RUN'], stalled:['#b8860b','STALL'], error:['#c62828','ERR'],
  idle:['#3a3a3a','idle'], unreachable:['#5a2d2d','n/a'], unknown:['#3a3a3a','?']
};
function llmToken(){ return sessionStorage.getItem('LLM_AUTH') || ''; }
function api(path, opts){
  opts = opts || {};
  const tok = llmToken();
  if (tok) opts.headers = Object.assign({}, opts.headers, {'X-LLM-Auth': tok});
  return fetch(path, opts).then(r => r.json());
}
function setStatus(msg){ const e=document.getElementById('livestatus'); if(e) e.textContent=msg; }

function clearBars(){
  Object.values(liveBar).forEach(b => { b.className='livebar hide'; b.textContent=''; });
}
function applyGrid(res){
  if (res.error){ setStatus('live: '+res.error); }
  else setStatus('live '+LIVE.what+' @ '+new Date().toLocaleTimeString());
  const cells = res.cells || {};
  Object.keys(liveBar).forEach(k => {
    const b = liveBar[k], c = cells[k];
    if (!c){ b.className='livebar hide'; return; }
    const m = LSTATE[c.state] || LSTATE.unknown;
    b.className = 'livebar';
    b.style.background = m[0];
    b.textContent = m[1];
    b.title = JSON.stringify(c, null, 2);
  });
}
function pollGridOnce(){
  // Belt-and-suspenders: never poll the JTAG while a board run is live; the
  // aiedbg reads would collide with apppaltest's device program/reset/download.
  if (LIVE.runActive) return;
  // Skip if a /grid is still in flight: aiedbg reads can take longer than the
  // 2s tick, and piled-up requests would exhaust the browser's ~6-connection
  // limit and starve /applog (freezing the run console).
  if (LIVE.gridBusy) return;
  LIVE.gridBusy = true;
  const dev = deviceSel ? deviceSel.value : '';
  const host = boardHost ? boardHost.value.trim() : '';
  const qs = '/grid?what='+LIVE.what +
             '&device='+encodeURIComponent(dev) +
             '&host='+encodeURIComponent(host);
  api(qs).then(applyGrid)
    .catch(() => setStatus('daemon offline (static mode)'))
    .finally(() => { LIVE.gridBusy = false; });
}
function startGridPoll(){ pollGridOnce(); LIVE.gridTimer = setInterval(pollGridOnce, 2000); }
function stopGridPoll(){ if(LIVE.gridTimer){ clearInterval(LIVE.gridTimer); LIVE.gridTimer=null; } }
function hideConsole(){
  CON.scope = 'partition';
  const box = document.getElementById('cmdconsole');
  if (box) box.classList.add('hide');
  const rsp = document.getElementById('rhsplitter');
  if (rsp) rsp.classList.add('hide');
  const tgt = document.getElementById('contarget');
  if (tgt) tgt.textContent = 'partition';
}
function setLive(on){
  LIVE.enabled = on;
  const cb = document.getElementById('liveToggle'); if(cb) cb.checked = on;
  if (on) startGridPoll();
  else { stopGridPoll(); clearBars(); setStatus(''); }
  // Console visibility is tied to LIVE.connected (the connection test), NOT to
  // the overlay toggle, so it persists while the overlay is switched off.
}
// Disable ALL aiedbg-driven features while a board run is live so their JTAG
// reads don't collide with apppaltest's device program/reset/dow -force over
// the single serialized xsdb://<PALIP>:3121 link (intermittent download fail).
function setDebugEnabled(on){
  const cb = document.getElementById('liveToggle');
  const tc = document.getElementById('testconn');
  const cr = document.getElementById('conreload');
  const ci = document.getElementById('conin');
  if (!on){
    LIVE.runActive = true;
    stopGridPoll(); clearBars();
    if (cb){ cb.checked = false; cb.disabled = true;
      cb.closest('label').classList.toggle('disabled', true); }
    if (tc) tc.disabled = true;
    if (cr) cr.disabled = true;
    if (ci){ ci.disabled = true; ci.classList.add('disabled'); }
    setStatus('debug disabled during run');
  } else {
    LIVE.runActive = false;
    if (tc) tc.disabled = false;
    if (cr) cr.disabled = false;
    if (ci){ ci.disabled = false; ci.classList.remove('disabled'); }
    // Re-enable the overlay toggle only if the connection is still valid
    // (mirrors the testConnect gating).
    if (cb){ cb.disabled = !LIVE.connected;
      cb.closest('label').classList.toggle('disabled', !LIVE.connected); }
    setStatus('');
  }
}

// Device selection gates the live controls (item #4/#5). A device must be
// chosen before the "Live status overlay" checkbox and "Run test" button work;
// picking vek385 also reveals the board-hostname text box.
const deviceSel = document.getElementById('deviceSel');
const boardHost = document.getElementById('boardHost');
const liveToggle = document.getElementById('liveToggle');
const runbtn = document.getElementById('runbtn');
const stopbtn = document.getElementById('stopbtn');
const testconn = document.getElementById('testconn');
function setConnStatus(msg){ const e=document.getElementById('connstatus'); if(e) e.textContent=msg; }
// Show/hide the "start hw_server on the target board" hint (shown on failure).
function setConnHint(show){ const e=document.getElementById('connhint'); if(e) e.classList.toggle('hide', !show); }
// Selecting a device only enables the "Test connect" button. The live overlay
// checkbox + the drill-down console stay locked until a connection test passes
// (LIVE.connected). Changing the device invalidates any prior test.
function updateDeviceUI(){
  const dev = deviceSel ? deviceSel.value : '';
  const has = !!dev;
  LIVE.connected = false;
  setLive(false);                              // uncheck overlay + stop poll
  hideConsole();                               // connection invalidated
  if (liveToggle){ liveToggle.disabled = true;
    liveToggle.closest('label').classList.toggle('disabled', true); }
  // Run test / Force stop stay gray until "Test connect" passes; changing the
  // device invalidates any prior test (LIVE.connected cleared above).
  if (runbtn) runbtn.disabled = true;
  if (stopbtn) stopbtn.disabled = true;
  if (testconn) testconn.disabled = !has;
  if (boardHost) boardHost.classList.toggle('hide', dev !== 'vek385');
  setConnStatus(has ? 'click "Test connect" to enable live features' : '');
  setConnHint(false);                            // clear stale failure hint
}
if (deviceSel) deviceSel.onchange = updateDeviceUI;
// Apply a successful connection: unlock Run test / Force stop / overlay and
// reveal the aiegdb console. Shared by the direct test and the auto-launch path.
function applyConnected(r){
  LIVE.connected = true;
  if (runbtn) runbtn.disabled = false;    // unlock Run test now
  if (stopbtn) stopbtn.disabled = false;  // unlock Force stop now
  if (liveToggle){ liveToggle.disabled = false;
    liveToggle.closest('label').classList.remove('disabled'); }
  const box = document.getElementById('cmdconsole');
  if (box) box.classList.remove('hide');   // reveal the aiegdb console
  const rsp = document.getElementById('rhsplitter');
  if (rsp) rsp.classList.remove('hide');
  setConnHint(false);
  // Only now that the JTAG connect succeeded do we tell the daemon to switch the
  // live target to this host and respawn the aiegdb console against it. Without
  // this the console stays pinned to the daemon's startup target and every
  // read times out even though the host was reachable.
  setConnStatus('connected \u2014 ' + ((r && r.detail) || 'ok') + '; setting target\u2026');
  api('/settarget', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({device:LIVE.device, host:LIVE.host})})
    .then(sr => {
      if (sr && sr.ok){
        setConnStatus('connected \u2014 ' + (sr.target || ((r && r.detail) || 'ok')));
        conSend('', false);   // spawn aiegdb with the new target; shows scope
      } else {
        setConnStatus('target switch failed: ' + ((sr && sr.detail) || 'unknown'));
        conSend('', false);   // still spawn so the console is usable
      }
    })
    .catch(() => {   // daemon offline / static mode: fall back to a plain spawn
      setConnStatus('connected \u2014 ' + ((r && r.detail) || 'ok'));
      conSend('', false);
    });
}
// Mark disconnected: re-gray Run test / Force stop / overlay and hide console.
function markDisconnected(){
  LIVE.connected = false;
  if (runbtn) runbtn.disabled = true;      // keep Run test gray
  if (stopbtn) stopbtn.disabled = true;    // keep Force stop gray
  if (liveToggle) liveToggle.disabled = true;
  hideConsole();
}
// Recovery: when the JTAG connect fails but the daemon is up, have the daemon
// ssh to the board, start hw_server (ssh -> systest -> xsdb -> exec hw_server),
// then re-probe once. The launch runs async on the daemon; we tail its per-step
// progress into the left-bottom console (#console) via /hwsrv_log so the user
// sees each step live. On success we're connected; otherwise show the hint.
function autoLaunchHwServer(dev, host, why){
  markDisconnected();
  // Keep the selection so applyConnected (reached via pollHwSrv on success)
  // switches the aiegdb console target to this same host.
  LIVE.device = dev; LIVE.host = host;
  setConnStatus('connect failed (' + why + ') \u2014 starting hw_server on board\u2026');
  setConnHint(false);
  // Reveal + reset the left-bottom console so the launch steps stream in.
  const con = document.getElementById('console');
  if (con){ con.classList.remove('hide'); con.textContent =
    '[auto-starting hw_server on board \u2014 live progress below]\n'; }
  LIVE.hwsrvOff = 0;
  api('/launch_hwserver', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({device:dev, host:host})})
    .then(r => {
      if (r && r.started){
        // Begin tailing the daemon's hw_server session log into #console.
        if (LIVE.hwsrvTimer) clearInterval(LIVE.hwsrvTimer);
        LIVE.hwsrvTimer = setInterval(pollHwSrv, 1000);
        pollHwSrv();   // immediate first tail so steps show without a 1s delay
      } else {
        markDisconnected();
        setConnStatus('auto-start failed: ' + ((r && r.detail) || 'unknown'));
        setConnHint(true);   // fall back to the manual start-hw_server guidance
      }
    })
    .catch(() => {
      markDisconnected();
      setConnStatus('daemon offline: cannot auto-start hw_server.');
      setConnHint(true);
    });
}
// Tail the daemon's hw_server launch session into #console and, once the
// background worker is done, apply the connect result (single-retry outcome).
function pollHwSrv(){
  if (LIVE.hwsrvBusy) return;           // guard against overlapping tails
  LIVE.hwsrvBusy = true;
  api('/hwsrv_log?offset='+LIVE.hwsrvOff).then(r => {
    const con = document.getElementById('console');
    if (con){
      const atBottom = (con.scrollHeight - con.scrollTop - con.clientHeight) < 4;
      if (r.data){ con.textContent += r.data; }
      if (atBottom) con.scrollTop = con.scrollHeight;
    }
    if (r.next != null) LIVE.hwsrvOff = r.next;
    setConnStatus('hw_server: ' + (r.status || 'starting') + '\u2026');
    if (r.done){
      if (LIVE.hwsrvTimer){ clearInterval(LIVE.hwsrvTimer); LIVE.hwsrvTimer=null; }
      if (r.ok){ applyConnected(r); }    // connected on the single retry
      else {
        // Final retry still failed. Re-gray Run test / Force stop, but KEEP the
        // console visible (don't call markDisconnected -> hideConsole) so the
        // user still sees the launch steps, and append the original
        // "connection failed -> start hw_server manually" guidance in place.
        LIVE.connected = false;
        if (runbtn) runbtn.disabled = true;
        if (stopbtn) stopbtn.disabled = true;
        if (liveToggle) liveToggle.disabled = true;
        if (con){
          con.textContent += '\n[auto-start failed \u2014 connection still down: '
            + ((r && r.detail) || 'unknown') + ']\n'
            + 'Connection failed. On the target hw board, start hw_server via xsdb:\n'
            + '    exec hw_server -stcp:0.0.0.0:3121\n';
          con.scrollTop = con.scrollHeight;
        }
        setConnStatus('connection failed \u2014 run "exec hw_server -stcp:0.0.0.0:3121" on the target board');
        setConnHint(true);   // also show the persistent manual-start hint banner
      }
    }
  }).catch(() => {
    if (LIVE.hwsrvTimer){ clearInterval(LIVE.hwsrvTimer); LIVE.hwsrvTimer=null; }
    markDisconnected();
    setConnStatus('daemon offline: cannot auto-start hw_server.');
    setConnHint(true);
  }).finally(() => { LIVE.hwsrvBusy = false; });
}
// Test the daemon + JTAG target; on success unlock the overlay + reveal console.
function testConnect(){
  const dev = deviceSel ? deviceSel.value : '';
  if (!dev){ setConnStatus('select a device first'); return; }
  const host = boardHost ? boardHost.value.trim() : '';
  if (dev === 'vek385' && !host){ setConnStatus('enter vek385 board hostname'); return; }
  // Remember the selection so applyConnected can tell the daemon which target
  // to switch the aiegdb console to (mirrors autoLaunchHwServer's closure).
  LIVE.device = dev; LIVE.host = host;
  setConnStatus('testing\u2026');
  const qs = '?device='+encodeURIComponent(dev)+'&host='+encodeURIComponent(host);
  api('/ping'+qs).then(r => {
    if (r && r.ok){
      applyConnected(r);
    } else {
      // Daemon answered but the JTAG connect failed → try to auto-start
      // hw_server on the board and retry once.
      autoLaunchHwServer(dev, host, (r && r.detail) || 'no response');
    }
  }).catch(() => {
    // Daemon itself is unreachable (static mode) → can't ssh from a dead
    // daemon, so don't attempt auto-launch; just guide the user.
    markDisconnected();
    setConnStatus('daemon offline (static mode)');
    setConnHint(true);   // guide user to start hw_server on the target board
  });
}
if (testconn) testconn.onclick = testConnect;

document.getElementById('liveToggle').onchange = e => setLive(e.target.checked);
document.querySelectorAll('#overlaytabs .ltab').forEach(tab => tab.onclick = () => {
  document.querySelectorAll('#overlaytabs .ltab').forEach(x => x.classList.remove('act'));
  tab.classList.add('act');
  LIVE.what = tab.dataset.w;
  if (LIVE.enabled) pollGridOnce();
});

function pollLog(){
  // Guard against overlapping tails so a slow response can't pile up.
  if (LIVE.logBusy) return;
  LIVE.logBusy = true;
  api('/applog?offset='+LIVE.logoff).then(r => {
    const con = document.getElementById('console');
    const atBottom = (con.scrollHeight - con.scrollTop - con.clientHeight) < 4;
    if (r.data){ con.textContent += r.data; }
    if (r.next != null) LIVE.logoff = r.next;
    if (atBottom) con.scrollTop = con.scrollHeight;
    setStatus('run: ' + r.status);
    // Stop only when the subprocess has actually exited — NOT on a derived
    // pass/fail. apppaltest keeps writing (summary/cleanup/reboot) long after
    // the teardown marker first appears mid-run.
    if (r.running === false){
      if (LIVE.conTimer){ clearInterval(LIVE.conTimer); LIVE.conTimer=null; }
      // Run ended (force-stop OR natural completion) → re-enable aiedbg features.
      setDebugEnabled(true);
    }
  }).catch(() => {}).finally(() => { LIVE.logBusy = false; });
}
document.getElementById('runbtn').onclick = () => {
  const dev = deviceSel ? deviceSel.value : '';
  if (!dev){ setStatus('select a device first'); return; }
  const host = boardHost ? boardHost.value.trim() : '';
  if (dev === 'vek385' && !host){ setStatus('enter vek385 board hostname'); return; }
  const con = document.getElementById('console');
  con.classList.remove('hide'); con.textContent = '';
  LIVE.logoff = 0;
  // Stop aiedbg polling/console BEFORE the download starts so its JTAG reads
  // don't collide with device program/reset/dow -force. Re-enabled in pollLog
  // when the run ends (force-stop or natural completion).
  setDebugEnabled(false);
  api('/run', {method:'POST', headers:{'Content-Type':'application/json'},
               body: JSON.stringify({device:dev, board_host:host})})
    .then(r => {
      // Run didn't actually start → re-enable debug (no pollLog will run).
      if (r.error){ con.textContent = 'run error: ' + r.error; setDebugEnabled(true); return; }
      con.textContent = '[run ' + r.run_id + ' started \u2192 ' + (r.applog||'applog') + ']\n';
      if (LIVE.conTimer) clearInterval(LIVE.conTimer);
      LIVE.conTimer = setInterval(pollLog, 1000);
    })
    .catch(() => { con.textContent = 'daemon offline: cannot start a run (open via schedule_debug_server).'; setDebugEnabled(true); });
};
document.getElementById('stopbtn').onclick = () => {
  api('/stop', {method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'})
    .then(r => {
      if (r.error){ setStatus('stop: ' + r.error); return; }
      setStatus('run: stopped (pid ' + r.pid + ')');
      // The next pollLog tick sees running=false, drains the force-stop line,
      // and clears the timer. Force one immediate tail so it shows at once.
      if (typeof pollLog === 'function') pollLog();
    })
    .catch(() => { setStatus('daemon offline: cannot stop a run.'); });
};
// Load the existing applog from the start (one-shot, no live tail). Useful for
// inspecting the last run without launching a new one.
document.getElementById('loadlogbtn').onclick = () => {
  const con = document.getElementById('console');
  con.classList.remove('hide'); con.textContent = '';
  LIVE.logoff = 0;
  // Cancel any active tail so the one-shot load isn't fought by polling.
  if (LIVE.conTimer){ clearInterval(LIVE.conTimer); LIVE.conTimer=null; }
  api('/applog?offset=0').then(r => {
    con.textContent = r.data || '(applog is empty)';
    if (r.next != null) LIVE.logoff = r.next;
    con.scrollTop = con.scrollHeight;
    setStatus('loaded existing log (' + (r.status || 'idle') + ')');
  }).catch(() => { con.textContent = 'daemon offline: cannot load applog.'; });
};

// Draggable splitter: resize left/right panes by dragging (default 50/50).
(function(){
  const sp = document.getElementById('splitter');
  const left = document.getElementById('left');
  if (!sp || !left) return;
  let dragging = false;
  sp.addEventListener('mousedown', e => {
    dragging = true;
    sp.classList.add('drag');
    document.body.classList.add('resizing');
    e.preventDefault();
  });
  document.addEventListener('mousemove', e => {
    if (!dragging) return;
    // Clamp so neither pane collapses below its min-width.
    const min = 200, max = window.innerWidth - 200 - sp.offsetWidth;
    let w = Math.max(min, Math.min(max, e.clientX));
    left.style.flex = '0 0 ' + w + 'px';
  });
  document.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    sp.classList.remove('drag');
    document.body.classList.remove('resizing');
  });
})();

// Draggable horizontal splitter: resize the top-left / bottom-left regions.
(function(){
  const sp = document.getElementById('lhsplitter');
  const top = document.getElementById('lefttop');
  const left = document.getElementById('left');
  if (!sp || !top || !left) return;
  let dragging = false;
  sp.addEventListener('mousedown', e => {
    dragging = true;
    sp.classList.add('drag');
    document.body.classList.add('vresizing');
    e.preventDefault();
  });
  document.addEventListener('mousemove', e => {
    if (!dragging) return;
    // Clamp so neither region collapses below a usable height.
    const rect = left.getBoundingClientRect();
    const min = 80, max = rect.height - 80 - sp.offsetHeight;
    let h = Math.max(min, Math.min(max, e.clientY - rect.top));
    top.style.flex = '0 0 ' + h + 'px';
  });
  document.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    sp.classList.remove('drag');
    document.body.classList.remove('vresizing');
  });
})();

// Draggable splitter for the right pane: resize the command-console frame.
(function(){
  const sp = document.getElementById('rhsplitter');
  const con = document.getElementById('cmdconsole');
  const right = document.getElementById('right');
  if (!sp || !con || !right) return;
  let dragging = false;
  sp.addEventListener('mousedown', e => {
    dragging = true;
    sp.classList.add('drag');
    document.body.classList.add('vresizing');
    e.preventDefault();
  });
  document.addEventListener('mousemove', e => {
    if (!dragging) return;
    // Console height = distance from cursor to the bottom of the right pane.
    const rect = right.getBoundingClientRect();
    const min = 80, max = rect.height - 120;
    let h = Math.max(min, Math.min(max, rect.bottom - e.clientY));
    con.style.flex = '0 0 ' + h + 'px';
  });
  document.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    sp.classList.remove('drag');
    document.body.classList.remove('vresizing');
  });
})();

// Served over HTTP by the daemon: leave live controls gated on device choice
// (updateDeviceUI disables Run/overlay until a device is picked). On file://
// there is no daemon, so stay static.
if (location.protocol === 'http:' || location.protocol === 'https:') {
  updateDeviceUI();   // device empty ⇒ controls disabled, overlay off
} else {
  setStatus('static mode — open via schedule_debug_server.py for live status');
  updateDeviceUI();
}

// LLM tab is independent of the JTAG connection: probe /llm/poll on load and,
// if the daemon answers (and the tab is enabled), reveal the console and
// default-select the LLM tab. The aiegdb tab stays gated on LIVE.connected.
function probeLLM(){
  api('/llm/poll?offset=0').then(r => {
    if (!r || r.error) return;   // disabled (--no-llm) or non-daemon response
    const box = document.getElementById('cmdconsole');
    if (box) box.classList.remove('hide');
    const rsp = document.getElementById('rhsplitter');
    if (rsp) rsp.classList.remove('hide');
    // Default-select the LLM tab.
    const tab = document.querySelector('#contabs .contab[data-pane="llmpane"]');
    if (tab) tab.click();
    // Show any transcript already buffered (e.g. after a page reload).
    if (r.data){ LLM.text = r.data; llmRender(); }
    if (r.next != null) LLM.off = r.next;
  }).catch(() => {});   // no daemon → stay static
}
if (location.protocol === 'http:' || location.protocol === 'https:') probeLLM();
</script>
</body>
</html>
"""


def _palmyra_enabled():
    """The palmyra board option is shown only when ~/USERENV/ENABLEPAL holds '1'.
    Missing/unreadable file (or any other value) hides it."""
    try:
        with open(os.path.expanduser('~/USERENV/ENABLEPAL')) as f:
            return f.read().strip() == '1'
    except OSError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Per-tile / per-channel code-piece cache
# ─────────────────────────────────────────────────────────────────────────────
# For local-LLM cooperation we materialize the host.cc code that implements each
# tile and each DMA channel into standalone files under ./debugcache/code/. Each
# tile/channel in the view gets a `code_file` (absolute path) so the browser can
# (a) show the code-piece path at the top of the file frame and (b) auto-tell the
# LLM "the related code is <path>" on click.

def _tile_cache_content(t):
    col, row = t['loc']
    hl = t.get('high_level', {}) or {}
    lo = t.get('low_level', {}) or {}
    out = ['// AIE Schedule debug code piece',
           '// Tile (%d,%d)  type=%s' % (col, row, t.get('type'))]
    if hl.get('role'):
        out.append('// role: %s' % hl['role'])
    if hl.get('kernel'):
        out.append('// kernel: %s' % hl['kernel'])
    for s in hl.get('summary', []) or []:
        out.append('//   transfer: %s' % s)
    km = hl.get('kernel_match')
    if km and km.get('matches'):
        out.append('// channel <-> kernel argument (by BD buffer address):')
        for m in km['matches']:
            adr = '/'.join(m.get('addrs_hex') or []) or '-'
            bsy = '/'.join(m.get('bcf_syms') or []) or '-'
            out.append('//   %s%s -> window %s arg %s  [bd %s = %s] via %s'
                       % (m.get('direction'), m.get('channel'),
                          m.get('window'),
                          ('arg%s' % m['arg']) if m.get('arg') is not None else '-',
                          adr, bsy, m.get('method')))
    rngs = lo.get('ranges') or []
    if rngs:
        out.append('// host.cc lines %s-%s (%d range(s))'
                   % (lo.get('line_start'), lo.get('line_end'), len(rngs)))
    out.append('')
    code_lines = lo.get('code_lines') or []
    if not code_lines:
        out.append('// (no attributed host.cc lines for this tile)')
    for rec in code_lines:
        out.append('/* L%-5d */ %s' % (rec['line'], rec['code']))
    return '\n'.join(out) + '\n'


def _channel_cache_content(t, c):
    col, row = t['loc']
    d, ch = c['direction'], c['channel']
    out = ['// AIE Schedule debug code piece',
           '// Tile (%d,%d) channel %s%s (flow %s)'
           % (col, row, d, ch, c.get('flow_index'))]
    if c.get('contract'):
        out.append('// contract: %s' % c['contract'])
    out.append('')
    entries = sorted(c.get('host_lines', []) or [], key=lambda e: e['line'])
    if not entries:
        out.append('// (no relevant host.cc lines for this channel)')
    for e in entries:
        if e.get('bd_comment'):
            out.append(e['bd_comment'])
        if e.get('note'):
            out.append('// %s' % e['note'])
        out.append('/* L%-5d %s */ %s' % (e['line'], e.get('kind', ''), e['code']))
    return '\n'.join(out) + '\n'


def write_code_cache(view, cache_dir='debugcache/code'):
    """Write one code-piece file per tile and per channel; annotate the view
    with absolute `code_file` paths. Returns (cache_dir_abs, file_count)."""
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    count = 0
    for t in view.get('tiles', []):
        col, row = t['loc']
        tpath = os.path.join(cache_dir, 'tile_c%d_r%d.cc' % (col, row))
        with open(tpath, 'w') as f:
            f.write(_tile_cache_content(t))
        t['code_file'] = tpath
        count += 1
        for c in t.get('dma_channels', []) or []:
            cpath = os.path.join(
                cache_dir,
                'tile_c%d_r%d_%s%s.cc' % (col, row, c['direction'].lower(),
                                          c['channel']))
            with open(cpath, 'w') as f:
                f.write(_channel_cache_content(t, c))
            c['code_file'] = cpath
            count += 1
    return cache_dir, count


def write_html(view, out_path):
    data_json = json.dumps(view, indent=None)
    html = HTML_TEMPLATE.replace('/*__DATA__*/ null', data_json)
    palmyra_opt = ('          <option value="palmyra">palmyra</option>\n'
                   if _palmyra_enabled() else '')
    html = html.replace('<!--__PALMYRA_OPTION__-->', palmyra_opt)
    with open(out_path, 'w') as f:
        f.write(html)


def main():
    workdir = sys.argv[1] if len(sys.argv) > 1 else 'aout/worklocal'
    view = build_view(workdir)
    # Materialize per-tile/channel code pieces (annotates view with code_file
    # paths) BEFORE serializing so both the JSON and the embedded HTML DATA carry
    # the paths for the file-frame header + LLM auto-notify.
    cache_dir, ncode = write_code_cache(view)
    json_out = os.path.join(workdir, 'schedule_view.json')
    html_out = os.path.join(workdir, 'host_schedule.html')
    with open(json_out, 'w') as f:
        json.dump(view, f, indent=2)
    write_html(view, html_out)
    ntiles = len(view['tiles'])
    covered = sum(len(t['low_level']['ranges']) for t in view['tiles'])
    print('wrote %s (%d tiles, %d line-ranges attributed)' % (json_out, ntiles, covered))
    print('wrote %s' % html_out)
    print('wrote %d code pieces to %s' % (ncode, cache_dir))


if __name__ == '__main__':
    main()
