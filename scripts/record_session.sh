#!/usr/bin/env bash
# Record the demo's three runs on this pod, then write them into one session file.
#
# Runs on the pod after setup_pod.sh, in about 30 minutes:
#   honest         the plugin and the verifier, nothing in between
#   relay          the cheat relay sends a cheated copy after every eighth honest proof
#   changed-model  vLLM serves a copy of the model with one weight changed per int8 matrix,
#                  against a verifier that declares the original
# Prompts are 2,048 tokens, sent at 3 per second: 540 in the first two runs
# and 180 in the changed-model run. GPU attribution is on, and each tile has a
# 1 in 50 million chance to win.
#
# Usage: bash scripts/record_session.sh OUT [--shapes]
# --shapes logs every int7 GEMM's shape (patches/0003), and the session file
# then holds each run's proven share. Exit: 0 every run recorded, 1 a run failed.
set -uo pipefail
cd "$(dirname "$0")/../.."

OUT="$1"
SHAPES=$([ "${2:-}" = --shapes ] && echo 1 || echo 0)
POUW="env PYTHONPATH=src /opt/upstream/.venv/bin/python -m pouw"
export RUNS=/root/runs
rm -rf "$RUNS"
failed=0

run() {
  local label=$1; shift
  echo "== $(date +%T) $label"
  if ! env LABEL="$label" RECORD=1 ATTRIBUTE=1 SHAPES="$SHAPES" RATE=3 PROMPTS=540 WIN=2e-8 "$@" \
      bash scripts/run_e2e.sh > "$RUNS-$label.log" 2>&1; then
    echo "   FAILED; the end of its log:"; tail -20 "$RUNS-$label.log"; failed=1
  fi
  grep -E "submissions:|Benchmark duration" "$RUNS-$label.log" | sed 's/^/   /'
}

run honest DESCRIPTION="The prover sends every proof unchanged"
run relay RELAY=1 RELAY_ARGS="--every 8 --start-after 4" \
  DESCRIPTION="A relay between the prover and the verifier sends cheated copies of real proofs"
[ -d /root/model-changed ] || $POUW changed-model /root/model /root/model-changed
run changed-model MODEL=/root/model-changed VERIFY_WEIGHTS=/root/model PROMPTS=180 \
  DESCRIPTION="vLLM serves a copy of the model with one weight changed in every int8 matrix"

echo "== $(date +%T) session file"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
$POUW session "$RUNS"/honest "$RUNS"/relay "$RUNS"/changed-model --gpu "$GPU" --out "$OUT" || failed=1
exit "$failed"
