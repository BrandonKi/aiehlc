#!/usr/bin/env bash
# Establish the BEST checkpoint for an auto-performance-agent session.
# Commits the CURRENT working tree (harness + profiling docs) onto a dedicated
# branch so keep/revert is relative to this point, not to HEAD.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

BRANCH="perf/auto-$(date +%Y%m%d-%H%M%S)"

# Make sure the experiments log dir exists and is tracked-friendly.
mkdir -p doc/profiling/experiments/logs

git checkout -b "$BRANCH"
git add -A
git commit -m "perf: session baseline checkpoint (BEST)" --no-verify

echo "RESULT: session ready"
echo "BRANCH: $BRANCH"
echo "BEST:   $(git rev-parse --short HEAD)"
echo
echo "Board vars expected: USERNAME=${USERNAME:-<unset>} VEK385IP=${VEK385IP:-<unset>}"
