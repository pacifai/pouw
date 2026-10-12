#!/usr/bin/env bash
# The H100 session behind the demo: timing runs, a recorded run, and a
# decode-heavy run. Runs on the pod after setup_pod.sh, in about 45 minutes.
#
# 1. Probe the GPU's identity and attestation data.
# 2. One warm-up run, so kernels compile before any timed run. Discard it.
# 3. Three rounds of: plain multiply, noisy multiply with proofs off, and
#    noisy multiply with proofs on. The load is the same as the first H100
#    session: 200 prompts of 2,048 tokens, sent at once.
# 4. The demo recording: GPU attribution on, every job and proof recorded,
#    every GEMM shape logged, and requests arriving at 3 per second.
# 5. A decode-heavy run with shapes logged: short prompts, long answers.
#
# Logs go to /root/pouw/runs/<label>/ and /root/session-<label>.log.
set -uo pipefail

RUN=/root/pouw/scripts/gpu/run_e2e.sh
UPSTREAM_PY=/opt/upstream/.venv/bin/python

run() {
  local label=$1; shift
  echo "== $(date +%T) $label"
  env LABEL="$label" "$@" bash "$RUN" > "/root/session-$label.log" 2>&1
  echo "   exit $? ; $(grep -E 'Benchmark duration' "/root/session-$label.log" || echo 'no benchmark line')"
}

echo "== $(date +%T) attestation probe"
"$UPSTREAM_PY" /root/pouw/scripts/gpu/probe_attestation.py > /root/session-attestation.log 2>&1
cat /root/session-attestation.log

run warm
for rep in 1 2 3; do
  run "plain-$rep" NO_MINING=1
  run "nosubmit-$rep" MINER_SKIP_BLOCK_SUBMISSION=true
  run "noisy-$rep"
done
run demo RECORD=1 ATTRIBUTE=1 SHAPES=1 RATE=3 PROMPTS=540 WIN=2e-8
run decode SHAPES=1 INPUT_LEN=256 OUTPUT_LEN=512 PROMPTS=200
echo "== $(date +%T) done"
