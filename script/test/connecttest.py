#!/usr/bin/env python3
"""Copyright 2025-2026 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: Apache-2.0
"""
"""
Minimal Palboard connection test.

Only exercises the connect logic (nonreboot / systest-client path):
    ssh -> systest-client -> become <board> -> source Vitis -> xsdb
        -> connect -url TCP:<PALIP>:3121 -> tar 1 -> exit

It does NOT power-cycle the board, program BOOT.BIN, or download an ELF.
Use it to check whether the xsdb -> hw_server -> JTAG connection is alive.

Prerequisites:
- SSH public key auth to PALIP
- Environment: USERNAME, PALIP, BOARDNAME (or a script/test/envlocal.sh that exports them)
- pip install pexpect
"""

import os
import sys
import subprocess

try:
    import pexpect
except ImportError:
    print("Error: pexpect module not found. Install with: pip install pexpect")
    sys.exit(1)

# --- Resolve USERNAME / PALIP / BOARDNAME (mirror apppaltest.py) ---
username = os.environ.get("USERNAME")
palip = os.environ.get("PALIP")
boardname = os.environ.get("BOARDNAME")

if not username or not palip or not boardname:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    envlocal_path = os.path.join(script_dir, "envlocal.sh")
    if os.path.exists(envlocal_path):
        print("Environment variables not set. Sourcing envlocal.sh...")
        result = subprocess.run(
            ['bash', '-c', f'source "{envlocal_path}" && env'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if '=' in line:
                    key, _, value = line.partition('=')
                    os.environ[key] = value
            username = os.environ.get("USERNAME")
            palip = os.environ.get("PALIP")
            boardname = os.environ.get("BOARDNAME")

if not username or not palip or not boardname:
    print("Error: Please set USERNAME, PALIP, and BOARDNAME environment variables")
    sys.exit(1)

host = f"{username}@{palip}"

# Configuration (kept in sync with apppaltest.py)
XSDB_ALT_PATH = "/proj/xbuilds/2025.2_daily_latest/installs/lin64/HEAD/Vitis/bin/xsdb"
VITIS_SETTINGS = "/proj/xbuilds/2025.2_daily_latest/installs/lin64/HEAD/Vitis/settings64.sh"


def connect_test():
    """SSH in, start xsdb (nonreboot), connect to hw_server, verify targets."""
    print(f"[connecttest] Connecting to {host}...")
    child = pexpect.spawn(f"ssh -X {host}", encoding='utf-8', timeout=60)
    child.logfile_read = sys.stdout

    child.expect([r'\$\s*$', r'#\s*$', r'>\s*$'], timeout=60)

    print("[connecttest] Starting systest-client (nonreboot mode)...")
    child.sendline("/opt/systest/common/bin/systest-client")
    child.expect(r'Systest[#>]', timeout=60)

    print(f"[connecttest] become {boardname}...")
    child.sendline(f'become "{boardname}"')
    child.expect(r'Systest[#>]', timeout=60)

    print("[connecttest] Sourcing Vitis and starting xsdb...")
    child.sendline(f"source {VITIS_SETTINGS}")
    child.expect(r'Systest[#>]', timeout=30)

    child.sendline("xsdb")
    index = child.expect([r'xsdb%', r'command not found', r'Unrecognized', pexpect.TIMEOUT], timeout=15)
    if index != 0:
        print("[connecttest] xsdb not found, trying alternative path...")
        child.sendline(XSDB_ALT_PATH)
        child.expect(r'xsdb%', timeout=60)

    print(f"[connecttest] connect -url TCP:{palip}:3121 ...")
    child.sendline(f"connect -url TCP:{palip}:3121")
    idx = child.expect([r'xsdb%', r'no such file', r'Connection refused', pexpect.TIMEOUT], timeout=60)
    if idx != 0:
        print("\n[connecttest] FAIL: could not connect to hw_server.")
        _quit(child)
        return 1

    # Show the JTAG scan chain / debug targets — this is what actually drives JTAG.
    print("[connecttest] Listing targets...")
    child.sendline("targets")
    child.expect(r'xsdb%', timeout=60)

    print("[connecttest] tar 1 (select device)...")
    child.sendline("tar 1")
    tidx = child.expect([r'xsdb%', r'no targets found', r'Invalid', pexpect.TIMEOUT], timeout=60)
    if tidx != 0:
        print("\n[connecttest] FAIL: JTAG target not reachable (board off / cable / locked).")
        _quit(child)
        return 1

    print("\n[connecttest] PASS: hw_server connected and JTAG target selected.")
    _quit(child)
    return 0


def _quit(child):
    try:
        child.sendline("exit")
        child.expect(r'Systest[#>]', timeout=10)
        child.sendline("exit")
    except Exception:
        pass
    try:
        child.close()
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(connect_test())
