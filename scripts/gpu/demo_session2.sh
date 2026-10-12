#!/usr/bin/env bash
# The second demo session: a fair test of the GIL patch, and runs with cheats.
# Runs on the pod after setup_pod.sh, in about 40 minutes.
#
# 1. Build the proof library with and without patches/0001 (build_ab_wheels.sh).
# 2. One warm-up run, so kernels compile before any timed run. Discard it.
# 3. Three rounds of: proofs on with the patch, proofs on without it, proofs
#    off, and the plain multiply. Same load as before: 200 prompts of 2,048
#    tokens, sent at once. Before each proofs-on run, tests/test_gil.py
#    confirms which library is installed.
# 4. A recorded run with pouw.cheat_relay between the plugin and the verifier.
# 5. A recorded run serving a checkpoint with one weight changed in every
#    int8 matrix, against a verifier that declares the original.
#
# Logs go to /root/pouw/runs/<label>/ and /root/session-<label>.log.
set -uo pipefail

RUN=/root/pouw/scripts/gpu/run_e2e.sh
PY=/opt/upstream/.venv/bin/python
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

run() {
  local label=$1; shift
  echo "== $(date +%T) $label"
  env LABEL="$label" "$@" bash "$RUN" > "/root/session-$label.log" 2>&1
  echo "   exit $? ; $(grep -E 'Benchmark duration' "/root/session-$label.log" || echo 'no benchmark line')"
}

use() {
  uv pip install --python "$PY" --reinstall --no-deps /root/wheels/"$1"/*.whl > /dev/null 2>&1
  local gil
  gil=$(cd /root/pouw && "$PY" -m pytest -q -s tests/test_gil.py 2>&1 | grep -E "builds took|passed|failed" | tr '\n' ' ')
  echo "   library: $1 ; $gil"
}

echo "== $(date +%T) building both libraries"
bash /root/pouw/scripts/gpu/build_ab_wheels.sh > /root/session-wheels.log 2>&1 || { tail /root/session-wheels.log; exit 1; }
uv pip install --python "$PY" pytest > /dev/null 2>&1

use patched
run warm
for rep in 1 2 3; do
  use patched
  run "ab-patched-$rep"
  use unpatched
  run "ab-unpatched-$rep"
  use patched
  run "nosubmit-$rep" MINER_SKIP_BLOCK_SUBMISSION=true
  run "plain-$rep" NO_MINING=1
done

use patched
run cheats RELAY=1 RECORD=1 ATTRIBUTE=1 SHAPES=1 RATE=3 PROMPTS=540 WIN=2e-8 \
  RELAY_ARGS="--every 8 --start-after 4 --spoof-window 90 120" \
  DESCRIPTION="H100 run with cheats from a relay between the prover and the verifier, 2026-10-11"

echo "== $(date +%T) substituted model"
"$PY" /root/pouw/scripts/gpu/make_substituted_model.py /root/model /root/model-sub
run model-sub MODEL=/root/model-sub VERIFY_WEIGHTS=/root/model RECORD=1 ATTRIBUTE=1 SHAPES=1 RATE=3 PROMPTS=180 WIN=2e-8 \
  DESCRIPTION="H100 run serving a checkpoint with one weight changed per int8 matrix, 2026-10-11"
echo "== $(date +%T) done"
