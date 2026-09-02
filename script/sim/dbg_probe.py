#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
# SPDX-License-Identifier: MIT
#
# Usage: dbg_probe.py <dbg_dir> [col row offset]
import json
import os
import socket
import struct
import sys
import time

PING, WRITE32, READ32 = 0x01, 0x10, 0x11
REQ = "<BxxxQI"
RESP = "<BxxxxxxxQ"


def _recvall(s, n):
    buf = b""
    while len(buf) < n:
        c = s.recv(n - len(buf))
        if not c:
            return None
        buf += c
    return buf


def txn(path, cmd, arg1=0, arg2=0):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(5)
    s.connect(path)
    s.sendall(struct.pack(REQ, cmd, arg1, arg2))
    raw = _recvall(s, struct.calcsize(RESP))
    s.close()
    status, value = struct.unpack(RESP, raw)
    return status, value


def main():
    dbg_dir = sys.argv[1]
    col = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    row = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    off = int(sys.argv[4], 0) if len(sys.argv) > 4 else 0x00032004  # core status

    info_path = os.path.join(dbg_dir, "dbg_info.json")
    for _ in range(600):
        socks = [f for f in os.listdir(dbg_dir) if f.endswith(".sock.dbg")] \
            if os.path.isdir(dbg_dir) else []
        if socks and os.path.isfile(info_path):
            break
        time.sleep(0.5)
    else:
        print("FAIL: no socket / dbg_info.json appeared")
        return 1

    sock_path = os.path.join(dbg_dir, socks[0])
    info = json.load(open(info_path))
    print("dbg_info:", info)

    st, _ = txn(sock_path, PING)
    print("PING status", st)
    if st != 0:
        print("FAIL: ping")
        return 1

    base = info["base_address"]
    cs, rs = info["column_shift"], info["row_shift"]
    addr = base + (col << cs) + (row << rs) + off
    st, val = txn(sock_path, READ32, addr)
    print(f"READ32 col={col} row={row} off=0x{off:x} addr=0x{addr:x} "
          f"-> status={st} value=0x{val:08x}")
    if st != 0:
        print("FAIL: read")
        return 1

    stw, _ = txn(sock_path, WRITE32, addr, 0)
    print(f"WRITE32 -> status={stw} (1=ERR_PROTO expected when writes disabled, "
          f"writes_enabled={info.get('writes_enabled')})")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
