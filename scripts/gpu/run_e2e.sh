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
set -euo pipefail

MODEL=/root/model
UPSTREAM=/opt/upstream
PY="$UPSTREAM/.venv/bin/python"
WIN="${WIN:-1e-8}"
PROMPTS="${PROMPTS:-200}"
INPUT_LEN="${INPUT_LEN:-2048}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"
NO_MINING="${NO_MINING:-0}"
LABEL="${LABEL:-$([ "$NO_MINING" = 1 ] && echo plain || echo noisy)-$(date +%H%M%S)}"
OUT=/root/pouw/runs/$LABEL
SOCKET=/tmp/pearlgw.sock
mkdir -p "$OUT"
export PATH="$UPSTREAM/.venv/bin:/usr/local/cuda/bin:$PATH"

pids=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup EXIT

echo "== verifier (win probability $WIN)"
rm -f "$SOCKET"  # a socket file left by an earlier run would look ready
PYTHONPATH=/root/pouw/src "$PY" -m pouw.verifier --weights "$MODEL"/*.safetensors \
  --win-probability "$WIN" --log "$OUT/verifier.jsonl" > "$OUT/verifier.out" 2>&1 &
pids+=($!)
for _ in $(seq 1 600); do grep -q "^listening" "$OUT/verifier.out" 2>/dev/null && break; sleep 1; done
grep -q "^listening" "$OUT/verifier.out" || { echo "verifier never started listening"; tail "$OUT/verifier.out"; exit 1; }
head -2 "$OUT/verifier.out"

echo "== vLLM (MINER_NO_MINING=$NO_MINING)"
MINER_NO_MINING=$([ "$NO_MINING" = 1 ] && echo true || echo false) \
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

echo "== one prompt"
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d "{\"model\": \"$MODEL\", \"prompt\": \"The capital of France is\", \"max_tokens\": 12, \"temperature\": 0}" \
  | "$PY" -c 'import json,sys; print(repr(json.load(sys.stdin)["choices"][0]["text"]))'

echo "== benchmark: $PROMPTS prompts, $INPUT_LEN tokens in, $OUTPUT_LEN out"
vllm bench serve --model "$MODEL" --dataset-name random --random-input-len "$INPUT_LEN" \
  --random-output-len "$OUTPUT_LEN" --num-prompts "$PROMPTS" --request-rate inf \
  --save-result --result-dir "$OUT" > "$OUT/bench.log" 2>&1
grep -E "Successful requests|Benchmark duration|throughput" "$OUT/bench.log" || tail -20 "$OUT/bench.log"

sleep 15  # let proofs from the last batches arrive
echo "== verifier log"
if [ -s "$OUT/verifier.jsonl" ]; then
  "$PY" /root/pouw/scripts/summarize_log.py "$OUT/verifier.jsonl"
else
  echo "no submissions"
fi
echo "exception lines in vllm.log: $(grep -ciE "traceback|exception" "$OUT/vllm.log" || true)"
