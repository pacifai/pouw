#!/usr/bin/env bash
# Measure patches/0001 on an H100: proofs on with and without it, against
# proofs off and the plain multiply. Runs on the pod after setup_pod.sh, in
# about 45 minutes.
#
# 1. Build the proof library with and without patches/0001 (build_ab_wheels.sh).
# 2. One warm-up run, so kernels compile before any timed run. Discard it.
# 3. Three rounds of: proofs on with the patch, proofs on without it, proofs
#    off, and the plain multiply. 200 prompts of 2,048 tokens, sent at once.
#
# The patch changes the plugin's Python code as well as the library, so each
# switch applies or reverts it in /opt/upstream and installs the matching
# library. The verifier runs in its own environment with the patched library,
# because pouw builds its trees with MerkleTree.from_buffer. The proofs-on
# runs are named ab-patched-* and ab-unpatched-*, which the demo's --timing
# option reads. To rerun only the proofs-on runs, set PROOFS_ON_ONLY=1 and
# REPS to the rounds, such as REPS="1 2 3 4". Logs go to /root/pouw/runs/<label>/ and
# /root/session-<label>.log.
set -uo pipefail

UPSTREAM=/opt/upstream
RUN=/root/pouw/scripts/gpu/run_e2e.sh
PY=$UPSTREAM/.venv/bin/python
PATCH=$(ls /root/pouw/patches/0001-*.patch)
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

run() {
  local label=$1; shift
  echo "== $(date +%T) $label"
  env LABEL="$label" "$@" bash "$RUN" > "/root/session-$label.log" 2>&1
  echo "   exit $? ; $(grep -E 'Benchmark duration' "/root/session-$label.log" || echo 'no benchmark line')"
}

use() {
  if [ "$1" = patched ]; then
    git -C "$UPSTREAM" apply --check "$PATCH" 2>/dev/null && git -C "$UPSTREAM" apply "$PATCH"
  else
    git -C "$UPSTREAM" apply -R --check "$PATCH" 2>/dev/null && git -C "$UPSTREAM" apply -R "$PATCH"
  fi
  uv pip install --python "$PY" --reinstall --no-deps /root/wheels/"$1"/*.whl > /dev/null 2>&1
  echo "   $1: $("$PY" -c 'import inspect, pearl_mining, miner_base.matrix_merkle_tree as m
print("library has from_buffer:", hasattr(pearl_mining.MerkleTree, "from_buffer"), "; plugin uses it:", "from_buffer" in inspect.getsource(m))')"
}

echo "== $(date +%T) building both libraries"
bash /root/pouw/scripts/gpu/build_ab_wheels.sh > /root/session-wheels.log 2>&1 || { tail /root/session-wheels.log; exit 1; }
VERIFIER_ENV=/root/verifier-venv
uv venv -q --python "$PY" "$VERIFIER_ENV"
uv pip install -q --python "$VERIFIER_ENV/bin/python" /root/wheels/patched/*.whl numpy safetensors blake3
export VERIFIER_PY=$VERIFIER_ENV/bin/python

use patched
run warm
for rep in ${REPS:-1 2 3}; do
  use patched
  run "ab-patched-$rep"
  use unpatched
  run "ab-unpatched-$rep"
  [ -n "${PROOFS_ON_ONLY:-}" ] && continue
  use patched
  run "nosubmit-$rep" MINER_SKIP_BLOCK_SUBMISSION=true
  run "plain-$rep" NO_MINING=1
done
echo "== $(date +%T) done"
