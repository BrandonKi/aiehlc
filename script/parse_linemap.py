#!/usr/bin/env python3
###############################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
###############################################################################
"""parse_linemap.py - Convert readelf --debug-dump=decodedline output to JSON.

Parses the DWARF line table (.debug_line) of the AIE kernel ELF into a sorted
address -> {file, line} map, written as kernel.linemap.json. aiediag loads this
map to translate a stuck core's program counter (PC) back to kernel source.

Input  : kernel.decodedline.txt (output of `readelf --debug-dump=decodedline`)
Output : kernel.linemap.json
  {
    "version": 1,
    "elf": "build/kernel",
    "entries": [
      {"addr": "0x1234", "addr_int": 4660, "file": "kernel.cc", "line": 71},
      ...
    ]
  }

readelf's decodedline format looks like:

    Decoded dump of debug contents of section .debug_line:

    CU: ./kernel.cc:
    File name      Line number    Starting address    View    Stmt
    kernel.cc               71              0x1234               x
    kernel.cc               72              0x1240
    ...

Lines with a line number and a starting address become entries. The leftmost
column (file name) is carried over to continuation rows that omit it. The "View"
and "Stmt" trailing columns are ignored.
"""

import argparse
import json
import os
import re
import sys

# A row carrying a filename, a line number and a hex starting address.
#   file  line  0xADDR  [view] [x]
_ROW_WITH_FILE = re.compile(
    r"^(?P<file>\S+)\s+(?P<line>\d+)\s+(?P<addr>0x[0-9a-fA-F]+)"
)
# A continuation row that omits the (repeated) filename: line  0xADDR
_ROW_NO_FILE = re.compile(
    r"^\s+(?P<line>\d+)\s+(?P<addr>0x[0-9a-fA-F]+)"
)
# CU header line: "CU: ./kernel.cc:" — gives us a default current file.
_CU_HEADER = re.compile(r"^CU:\s+(?P<file>.+?):?\s*$")


def parse_decodedline(text):
    """Parse readelf decodedline text into a list of entry dicts.

    Returns entries sorted by integer address, de-duplicated on address
    (last writer wins for a given address).
    """
    by_addr = {}
    current_file = None

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            continue

        cu = _CU_HEADER.match(line)
        if cu:
            current_file = os.path.basename(cu.group("file").strip())
            continue

        # Skip the column header row.
        if line.lstrip().startswith("File name"):
            continue
        # Skip the section banner.
        if line.startswith("Decoded dump"):
            continue

        m = _ROW_WITH_FILE.match(line)
        if m:
            current_file = os.path.basename(m.group("file"))
            line_no = int(m.group("line"))
            addr = int(m.group("addr"), 16)
            by_addr[addr] = (current_file, line_no)
            continue

        m = _ROW_NO_FILE.match(line)
        if m:
            line_no = int(m.group("line"))
            addr = int(m.group("addr"), 16)
            by_addr[addr] = (current_file, line_no)
            continue

    entries = []
    for addr in sorted(by_addr):
        fname, line_no = by_addr[addr]
        entries.append({
            "addr": f"0x{addr:X}",
            "addr_int": addr,
            "file": fname if fname is not None else "",
            "line": line_no,
        })
    return entries


def build_linemap(decoded_path, elf_path):
    with open(decoded_path) as f:
        text = f.read()
    entries = parse_decodedline(text)
    return {
        "version": 1,
        "elf": elf_path if elf_path else "",
        "entries": entries,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="parse_linemap.py",
        description="Convert readelf decodedline output to kernel.linemap.json",
    )
    parser.add_argument("decodedline",
                        help="Path to kernel.decodedline.txt (readelf decodedline output)")
    parser.add_argument("--elf", default="",
                        help="Path of the kernel ELF (stored in JSON for reference)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output JSON path (default: kernel.linemap.json next to input)")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.decodedline):
        print(f"Error: input not found: {args.decodedline}", file=sys.stderr)
        return 1

    linemap = build_linemap(args.decodedline, args.elf)

    out_path = args.output
    if out_path is None:
        out_path = os.path.join(os.path.dirname(os.path.abspath(args.decodedline)),
                                "kernel.linemap.json")

    with open(out_path, "w") as f:
        json.dump(linemap, f, indent=2)
        f.write("\n")

    n = len(linemap["entries"])
    print(f"Wrote {out_path}: {n} line entries")
    # Print a small sample (sorted by address) for standalone inspection.
    for e in linemap["entries"][:10]:
        print(f"  {e['addr']:>10s}  {e['file']}:{e['line']}")
    if n > 10:
        print(f"  ... ({n - 10} more)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
