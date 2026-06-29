#!/usr/bin/env bash
# Run one performance experiment end-to-end and classify the outcome.
#   usage: run_experiment.sh <exp-id> [--rebuild]
# Tees the full console to doc/profiling/experiments/logs/<exp-id>.log and
# prints a final "RESULT:" line. Exit codes:
#   0 = ran + PASS (device_teardown done, no AIE ERROR)
#   1 = ran but FAIL (AIE ERROR / no teardown / mismatch)
#   2 = build or codegen failed
#   3 = board / SSH unreachable (infra, not an experiment failure)
set -uo pipefail

EXP_ID="${1:?usage: run_experiment.sh <exp-id> [--rebuild]}"
REBUILD=0
[[ "${2:-}" == "--rebuild" ]] && REBUILD=1

cd "$(git rev-parse --show-toplevel)"

SRC="${PERF_SRC:-./example/tileprogram/ccode/simplematmul2_prof.cc}"
AIE_VERSION="${PERF_AIE_VERSION:-5}"
LOGDIR="doc/profiling/experiments/logs"
LOG="$LOGDIR/${EXP_ID}.log"
mkdir -p "$LOGDIR"

: "${USERNAME:?export USERNAME first}"
: "${VEK385IP:?export VEK385IP (reserved board) first}"

finish() { echo "RESULT: $1"; exit "$2"; }

{
  echo "=== experiment $EXP_ID @ $(date -u +%FT%TZ) ==="
  echo "src=$SRC rebuild=$REBUILD board=$VEK385IP user=$USERNAME"

  if [[ $REBUILD -eq 1 ]]; then
    echo "--- [1/3] rebuild aiehlc ---"
    if ! ( cd build && make -j"$(nproc)" ); then
      echo "BUILD_FAIL: aiehlc make failed"; exit 2
    fi
  fi

  echo "--- [2/3] generate host+kernel ELF ---"
  if ! bash script/aiehlc.sh --aie-version "$AIE_VERSION" --runtime-source-file "$SRC"; then
    echo "CODEGEN_FAIL: aiehlc.sh failed"; exit 2
  fi
  if [[ ! -f aout/main.elf ]]; then
    echo "CODEGEN_FAIL: aout/main.elf not produced"; exit 2
  fi

  echo "--- [3/3] run on board ---"
  python3 script/test/appvek385.py -y aout/main.elf
  echo "--- run wrapper exit: $? ---"
} 2>&1 | tee "$LOG"

# Classify from the captured log (pipefail-safe: inspect the file, not $?).
if grep -qiE "connection refused|no route to host|permission denied|ssh: connect|could not resolve|timed out" "$LOG"; then
  finish "INFRA (board/ssh unreachable) — see $LOG" 3
fi
if grep -qE "BUILD_FAIL|CODEGEN_FAIL" "$LOG"; then
  finish "FAIL (build/codegen) — see $LOG" 2
fi
if grep -qi "AIE ERROR" "$LOG"; then
  finish "FAIL (AIE ERROR) — see $LOG" 1
fi
if grep -qi "device_teardown done" "$LOG"; then
  finish "PASS (candidate) — parse metrics from $LOG" 0
fi
finish "FAIL (no teardown sentinel) — see $LOG" 1
