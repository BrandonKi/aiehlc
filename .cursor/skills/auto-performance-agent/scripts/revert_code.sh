#!/usr/bin/env bash
# Revert CODE changes back to the BEST checkpoint while PRESERVING the
# profiling log (doc/profiling/**). Run only after the experiment doc already
# captured the full changeset.
set -uo pipefail

cd "$(git rev-parse --show-toplevel)"

# Restore tracked modifications outside doc/profiling to the last commit (BEST).
mapfile -t MODIFIED < <(git diff --name-only -- . ':(exclude)doc/profiling/**')
if [[ ${#MODIFIED[@]} -gt 0 ]]; then
  git checkout -- "${MODIFIED[@]}"
fi

# Remove untracked code files (but keep docs, build artifacts, aout, logs).
git clean -fd \
  -e 'doc/profiling/**' \
  -- src include example script CMakeLists.txt >/dev/null 2>&1 || true

echo "RESULT: code reverted to BEST $(git rev-parse --short HEAD); docs preserved"
git status --short | grep -vE '^\?\? (build|aout)/' || true
