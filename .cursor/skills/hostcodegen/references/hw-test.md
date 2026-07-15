<!-- Copyright (C) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
     SPDX-License-Identifier: Apache-2.0 -->

# HW test: apppaltest.py and verify_host.sh

## Verify host script

**script/test/verify_host.sh** runs the host ELF on HW and checks console output:

- **Usage**: `./script/test/verify_host.sh [--compile] [elf_path]`
- **--compile**: Build host first (hostcompile.sh from worklocal).
- **elf_path**: Default is worklocal/build/host (from repo root).
- **Exit**: 0 if no "AIE ERROR" / "Invalid Tile Type" and runtime teardown seen; 1 otherwise.
- **Log**: Console output is saved to script/test/.verify_host_console.log.

Example from repo root:
```bash
./script/test/verify_host.sh --compile
# or
./script/test/verify_host.sh src/mlir/mlirfront/tilinglinalg/pass/unitest/worklocal/build/host
```

## Environment

Set before running HW test (or use envlocal.sh):

| Variable | Meaning |
|----------|--------|
| `USERNAME` | SSH login on PAL host. |
| `PALIP` | IP of PAL host (e.g. 10.23.x.x). |
| `BOARDNAME` | Board name for systest `become` (e.g. pal***). |

**Optional**: Create `script/test/envlocal.sh` with:

```bash
export USERNAME=your_username
export PALIP=10.23.x.x
export BOARDNAME=pal***
```

apppaltest.py will try to source it if the vars are not set.

## Running apppaltest.py

- **Location**: `script/test/apppaltest.py`
- **Usage**: `python3 script/test/apppaltest.py [path-to-ELF]`

Examples:

- No arg: looks for default `aout/main.elf` or asks.
- Full path: `python3 script/test/apppaltest.py /path/to/host`.
- Relative path: `python3 script/test/apppaltest.py ./worklocal/build/host` (from a cwd where that path exists).

The script (1) SSHs to PALIP, (2) runs systest and becomes BOARDNAME, (3) programs device via xsdb, (4) copies ELF to remote, (5) downloads ELF and continues execution, (6) opens second connection to com0 and captures console output.

## What to look for in console

- **Success**: Application prints (e.g. "Host built", XAie init messages, your test output). No assertion or XAie error messages.
- **Failure**: XAie_* errors, assertion failures, hang (no output after "Execution started"), or board/connection errors in the script output.

## Common HW/test failures

| Symptom | What to check |
|---------|----------------|
| `USERNAME`, `PALIP`, `BOARDNAME` not set | Export them or add script/test/envlocal.sh and re-run. |
| SSH / connection timeout | Network, PALIP, SSH keys. |
| ELF file not found | Pass correct path (absolute or relative to cwd); for unitest host use worklocal/build/host. |
| xsdb not found / wrong path | Script tries default then XSDB_ALT_PATH; ensure xsdb is on remote PATH or set path in script. |
| Device program / dow fails | Board state, BOOT.BIN/PALBOARD_BIN path on remote; ensure device is programmed and target (tar 1, tar 20) is correct. |
| No console output | com0 connection; second SSH must connect to com0; first connection must run ELF (dow, con). |
| Application crash / XAie error on board | Runtime or generated host bug; check aie_runtime.c, device init, and generated host.cc; re-run with same ELF and inspect console. |

## Prerequisites (from apppaltest.py)

- SSH access to PALIP with key auth.
- `pexpect`: `pip install pexpect`
- Remote: systest, xsdb, BOOT.BIN at expected paths; com0 available for console.
