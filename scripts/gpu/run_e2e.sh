#!/usr/bin/env bash
# Serve the int7 Llama with the upstream plugin, connected to this repo's
# verifier, and drive it with vLLM's benchmark.
#
# Runs on the pod after setup_pod.sh. It starts the verifier on the socket the
# plugin connects to, starts vLLM, sends long prompts (the plugin only runs
# the noisy multiply on batches of at least 1,024 tokens), and then stops both.
# Logs go to /root/pouw/runs/<label>/.
#
# Usage:
#   bash scripts/gpu/run_e2e.sh                    # noisy multiply, proofs on
#   NO_MINING=1 bash scripts/gpu/run_e2e.sh        # plain multiply, for comparison
#   WIN=1e-8 PROMPTS=400 bash scripts/gpu/run_e2e.sh
#   RECORD=1 ATTRIBUTE=1 SHAPES=1 RATE=3 bash scripts/gpu/run_e2e.sh
#
# RECORD=1 saves every job and proof for replay (pouw.recording), ATTRIBUTE=1
# turns on GPU attribution on both sides, SHAPES=1 logs every int7 GEMM's
# shape (patches/0003), and RATE sets requests per second (default: all at
# once). RELAY=1 puts pouw.cheat_relay between the plugin and the verifier,
# with RELAY_ARGS passed to it. LIVE=1 runs the demo server in live mode as
# the verifier, on port 8080. VERIFY_WEIGHTS sets the checkpoint the verifier
# declares, if it differs from the one vLLM serves. DESCRIPTION is saved for
# the demo's run picker.
set -euo pipefail

MODEL="${MODEL:-/root/model}"
UPSTREAM=/opt/upstream
PY="$UPSTREAM/.venv/bin/python"
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
LIVE="${LIVE:-0}"
VERIFY_WEIGHTS="${VERIFY_WEIGHTS:-$MODEL}"
LABEL="${LABEL:-$([ "$NO_MINING" = 1 ] && echo plain || echo noisy)-$(date +%H%M%S)}"
OUT=/root/pouw/runs/$LABEL
SOCKET=/tmp/pearlgw.sock
mkdir -p "$OUT"
[ -n "${DESCRIPTION:-}" ] && echo "$DESCRIPTION" > "$OUT/description.txt"
export PATH="$UPSTREAM/.venv/bin:/usr/local/cuda/bin:$PATH"

pids=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup EXIT

echo "== verifier (win probability $WIN)"
VERIFIER_SOCKET=$([ "$RELAY" = 1 ] && echo /tmp/verifier.sock || echo "$SOCKET")
rm -f "$SOCKET" /tmp/verifier.sock  # a socket file left by an earlier run would look ready
verifier_args=()
[ "$RECORD" = 1 ] && verifier_args+=(--record "$OUT/recording.jsonl")
[ "$ATTRIBUTE" = 1 ] && verifier_args+=(--attribute-gpus)
if [ "$LIVE" = 1 ]; then
  PYTHONPATH=/root/pouw/src "$PY" -m pouw.demo.server --live --socket "$VERIFIER_SOCKET" \
    --weights "$VERIFY_WEIGHTS"/*.safetensors --win-probability "$WIN" --log "$OUT/verifier.jsonl" \
    --shapes "$OUT/gemm_shapes.jsonl" --serving-since "$OUT/serving_since" "${verifier_args[@]}" \
    > "$OUT/verifier.out" 2>&1 &
else
  PYTHONPATH=/root/pouw/src "$PY" -m pouw.verifier --socket "$VERIFIER_SOCKET" --weights "$VERIFY_WEIGHTS"/*.safetensors \
    --win-probability "$WIN" --log "$OUT/verifier.jsonl" "${verifier_args[@]}" > "$OUT/verifier.out" 2>&1 &
fi
pids+=($!)
for _ in $(seq 1 600); do grep -q "^listening" "$OUT/verifier.out" 2>/dev/null && break; sleep 1; done
grep -q "^listening" "$OUT/verifier.out" || { echo "verifier never started listening"; tail "$OUT/verifier.out"; exit 1; }
head -2 "$OUT/verifier.out"
if [ "$RELAY" = 1 ]; then
  echo "== cheat relay"
  # shellcheck disable=SC2086  # RELAY_ARGS is a list of flags
  PYTHONPATH=/root/pouw/src "$PY" -m pouw.cheat_relay --listen "$SOCKET" --upstream "$VERIFIER_SOCKET" \
    --log "$OUT/cheats.jsonl" $RELAY_ARGS > "$OUT/relay.out" 2>&1 &
  pids+=($!)
  for _ in $(seq 1 60); do grep -q "^listening" "$OUT/relay.out" 2>/dev/null && break; sleep 1; done
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
if [ -s "$OUT/verifier.jsonl" ]; then
  "$PY" /root/pouw/scripts/summarize_log.py "$OUT/verifier.jsonl"
else
  echo "no submissions"
fi
echo "exception lines in vllm.log: $(grep -ciE "traceback|exception" "$OUT/vllm.log" || true)"
