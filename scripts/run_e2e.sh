#!/usr/bin/env bash
# Serve the int7 Llama with the upstream plugin, connected to this repo's
# verifier, and drive it with vLLM's benchmark.
#
# Runs on the pod after setup_pod.sh. It starts the verifier on the socket the
# plugin connects to, starts vLLM, sends long prompts (the plugin only runs
# the noisy multiply on batches of at least 1,024 tokens), and then stops both.
# Logs go to $RUNS/<label>/, by default /root/pouw/runs/<label>/.
#
# Usage:
#   bash scripts/run_e2e.sh                    # noisy multiply, proofs on
#   NO_MINING=1 bash scripts/run_e2e.sh        # plain multiply, for comparison
#   WIN=1e-8 PROMPTS=400 bash scripts/run_e2e.sh
#   RECORD=1 ATTRIBUTE=1 SHAPES=1 RATE=3 bash scripts/run_e2e.sh
#
# RECORD=1 saves every job and proof for the session file, ATTRIBUTE=1
# turns on GPU attribution on both sides, SHAPES=1 logs every int7 GEMM's
# shape (patches/0003), and RATE sets requests per second (default: all at
# once). RELAY=1 puts the cheat relay between the plugin and the verifier,
# with RELAY_ARGS passed to it. MAX_AGE sets the oldest job a proof may use,
# in seconds, for both the verifier and the relay. VERIFY_WEIGHTS sets the checkpoint the
# verifier declares, if it differs from the one vLLM serves, and CHANGED_MODEL=1
# marks vLLM's checkpoint as a changed copy of it. DESCRIPTION, both
# checkpoints, CHANGED_MODEL, and the versions of the code and the declared
# model go into run.json, which the session file reads. RUNS sets the folder
# that holds the run's folder.
set -euo pipefail

MODEL="${MODEL:-/root/model}"
UPSTREAM=/opt/upstream
PY="$UPSTREAM/.venv/bin/python"
POUW="env PYTHONPATH=/root/pouw/src $PY -m pouw"
WIN="${WIN:-1e-8}"
PROMPTS="${PROMPTS:-200}"
INPUT_LEN="${INPUT_LEN:-2048}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
NO_MINING="${NO_MINING:-0}"
RECORD="${RECORD:-0}"
ATTRIBUTE="${ATTRIBUTE:-0}"
SHAPES="${SHAPES:-0}"
RATE="${RATE:-inf}"
RELAY="${RELAY:-0}"
RELAY_ARGS="${RELAY_ARGS:-}"
MAX_AGE="${MAX_AGE:-}"
VERIFY_WEIGHTS="${VERIFY_WEIGHTS:-$MODEL}"
CHANGED_MODEL="${CHANGED_MODEL:-0}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="${LABEL:-$([ "$NO_MINING" = 1 ] && echo plain || echo noisy)-$(date +%H%M%S)}"
OUT="${RUNS:-/root/pouw/runs}/$LABEL"
SOCKET=/tmp/pearlgw.sock
mkdir -p "$OUT"
# record.py ships the repository without .git, so it writes the commit to .commit.
POUW_COMMIT=$(cat "$REPO/.commit" 2>/dev/null || git -C "$REPO" describe --always --dirty --abbrev=40 2>/dev/null || true)
UPSTREAM_COMMIT=$(git -C "$UPSTREAM" rev-parse HEAD 2>/dev/null || true)
# hf download --local-dir keeps the checkpoint's commit on the first line of each file's metadata.
MODEL_REVISION=$(head -1 "$VERIFY_WEIGHTS/.cache/huggingface/download/config.json.metadata" 2>/dev/null || true)
"$PY" - "$OUT/run.json" "$REPO/patches" "${DESCRIPTION:-$LABEL}" "$MODEL" "$VERIFY_WEIGHTS" "$CHANGED_MODEL" \
  "$POUW_COMMIT" "$UPSTREAM_COMMIT" "$MODEL_REVISION" <<'PY'
import hashlib, json, pathlib, sys
out, patch_dir, description, model, declared, changed, pouw, upstream, revision = sys.argv[1:]
patches = sorted(pathlib.Path(patch_dir).glob("*.patch"))
meta = {
    "description": description,
    "model": model,
    "declared_model": declared,
    "changed_model": changed == "1",
    "pouw_commit": pouw or None,
    "upstream_commit": upstream or None,
    "patches_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in patches)).hexdigest() if patches else None,
    "declared_model_revision": revision or None,
}
pathlib.Path(out).write_text(json.dumps(meta, indent=2) + "\n")
PY
export PATH="$UPSTREAM/.venv/bin:/usr/local/cuda/bin:$PATH"

pids=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup EXIT

echo "== verifier (win probability $WIN)"
VERIFIER_SOCKET=$([ "$RELAY" = 1 ] && echo /tmp/verifier.sock || echo "$SOCKET")
rm -f "$SOCKET" /tmp/verifier.sock  # a socket file left by an earlier run would look ready
max_age_args=()
[ -n "$MAX_AGE" ] && max_age_args=(--max-age "$MAX_AGE")
verifier_args=("${max_age_args[@]}")
[ "$RECORD" = 1 ] && verifier_args+=(--record "$OUT/recording.jsonl")
[ "$ATTRIBUTE" = 1 ] && verifier_args+=(--attribute-gpus)
$POUW verify --socket "$VERIFIER_SOCKET" --weights "$VERIFY_WEIGHTS"/*.safetensors \
  --win-probability "$WIN" --log "$OUT/verifier.jsonl" "${verifier_args[@]}" > "$OUT/verifier.out" 2>&1 &
pids+=($!)
verifier_pid=$!
for _ in $(seq 1 600); do
  grep -q "^listening" "$OUT/verifier.out" 2>/dev/null && break
  kill -0 "$verifier_pid" 2>/dev/null || break
  sleep 1
done
grep -q "^listening" "$OUT/verifier.out" || { echo "verifier never started listening"; tail "$OUT/verifier.out"; exit 1; }
head -2 "$OUT/verifier.out"
if [ "$RELAY" = 1 ]; then
  echo "== cheat relay"
  # shellcheck disable=SC2086  # RELAY_ARGS is a list of flags
  $POUW relay --listen "$SOCKET" --verifier "$VERIFIER_SOCKET" --weights "$VERIFY_WEIGHTS"/*.safetensors \
    --log "$OUT/cheats.jsonl" "${max_age_args[@]}" $RELAY_ARGS > "$OUT/relay.out" 2>&1 &
  pids+=($!)
  relay_pid=$!
  for _ in $(seq 1 60); do
    grep -q "^listening" "$OUT/relay.out" 2>/dev/null && break
    kill -0 "$relay_pid" 2>/dev/null || break
    sleep 1
  done
  grep -q "^listening" "$OUT/relay.out" || { echo "relay never started listening"; tail "$OUT/relay.out"; exit 1; }
fi

echo "== vLLM (MINER_NO_MINING=$NO_MINING)"
MINER_NO_MINING=$([ "$NO_MINING" = 1 ] && echo true || echo false) \
  MINER_SEND_GPU_ID=$([ "$ATTRIBUTE" = 1 ] && echo true || echo false) \
  MINER_LOG_GEMM_SHAPES=$([ "$SHAPES" = 1 ] && echo "$OUT/gemm_shapes.jsonl" || true) \
  vllm serve "$MODEL" --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.85 \
  --enforce-eager > "$OUT/vllm.log" 2>&1 &
pids+=($!)
vllm_pid=$!
for _ in $(seq 1 900); do
  curl -sf localhost:8000/health >/dev/null && break
  kill -0 "$vllm_pid" 2>/dev/null || break  # vLLM exited during startup
  sleep 2
done
curl -sf localhost:8000/health >/dev/null || { echo "vLLM never became healthy"; tail -30 "$OUT/vllm.log"; exit 1; }
date +%s.%N > "$OUT/serving_since"  # proofs before this come from vLLM's warm-up

echo "== one prompt"
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d "{\"model\": \"$MODEL\", \"prompt\": \"The capital of France is\", \"max_tokens\": 12, \"temperature\": 0}" \
  | "$PY" -c 'import json,sys; print(repr(json.load(sys.stdin)["choices"][0]["text"]))'

echo "== benchmark: $PROMPTS prompts at $RATE/s, $INPUT_LEN tokens in, $OUTPUT_LEN out"
vllm bench serve --model "$MODEL" --dataset-name random --random-input-len "$INPUT_LEN" \
  --random-output-len "$OUTPUT_LEN" --num-prompts "$PROMPTS" --request-rate "$RATE" \
  --save-result --result-dir "$OUT" > "$OUT/bench.log" 2>&1
grep -E "Successful requests|Benchmark duration|throughput" "$OUT/bench.log" || tail -20 "$OUT/bench.log"

# Let proofs from the last batches arrive and get checked: wait until the
# verifier's log stops growing for 20 s, up to 3 minutes.
last=-1; still=0
for _ in $(seq 1 180); do
  n=$(wc -l < "$OUT/verifier.jsonl" 2>/dev/null || echo 0)
  if [ "$n" = "$last" ]; then still=$((still + 1)); else still=0; last=$n; fi
  [ "$still" -ge 20 ] && break
  sleep 1
done
echo "== verifier log"
$POUW summarize "$OUT/verifier.jsonl"
echo "exception lines in vllm.log: $(grep -ciE "traceback|exception" "$OUT/vllm.log" || true)"
