# Contributing

This guide covers where things are, how to build and test, and the rules that
keep results trustworthy. For what pouw is, start with the
[README](README.md).

## Layout

| Path | Contents |
|---|---|
| `src/pouw/verifier.py` | The verifier: the five checks, and the JSON-RPC server that stands in for upstream's gateway |
| `src/pouw/jobs.py` | Jobs: the verifier's random value in a block header, and the target's chance to win |
| `src/pouw/weights.py` | The declared weights, fused the way vLLM holds them, and their Merkle roots |
| `src/pouw/cpu_prover.py` | A CPU stand-in for the GPU, for tests |
| `src/pouw/cheat_relay.py` | A relay between the GPU and the verifier that sends cheated copies of real proofs |
| `src/pouw/recording.py` | Records every job and proof of a run, so it can be checked again later |
| `src/pouw/session.py` | Checks recorded runs again and writes them into one session file |
| `src/pouw/rpc.py` | The newline-delimited JSON-RPC that upstream's plugin speaks |
| `tests/` | The tests |
| `scripts/fetch_tensors.py` | Downloads single tensors from a checkpoint on Hugging Face |
| `scripts/summarize_log.py` | Counts a verifier log's verdicts and the work they imply |
| `scripts/gpu/` | Run on the pod: setup, one end-to-end run, the session's three runs, and the changed model |
| `runpod/record.py` | Rents an H100, records a session there, and releases it |
| `patches/` | The changes this repository makes to upstream's code |
| `docker/` | The image that builds upstream's proof library and runs the tests |
| `docs/decisions/` | The decision records behind the design and its claims |

## Build and test

Everything that doesn't need a GPU runs in Docker:

```bash
docker build -t pouw-local -f docker/Dockerfile .
docker run --rm -v "$PWD":/work pouw-local python -m pytest -q
```

`tests/test_real_weights.py` needs four tensors from the real checkpoint and
skips itself without them. The README's quickstart fetches them.

To run the verifier and the CPU stand-in as separate processes, inside the
image:

```bash
python -m pouw.verifier --weights data/llama-sample.safetensors --win-probability 0.02 --log runs/local.jsonl &
python -m pouw.cpu_prover --weights data/llama-sample.safetensors \
  --matrix model.layers.0.self_attn.o_proj.weight --proofs 3
```

### GPU

Upstream's kernels need a Hopper GPU, such as an H100, on a host whose driver
supports CUDA 13. `uv run runpod/record.py` rents one and runs everything. To
work on a pod by hand, copy this repository to `/root/pouw`, then:

```bash
bash /root/pouw/scripts/gpu/setup_pod.sh     # builds upstream and downloads the model
bash /root/pouw/scripts/gpu/run_e2e.sh       # one run: the verifier, vLLM, and a benchmark
```

`run_e2e.sh` takes its settings from environment variables, listed at its top.
For example, `NO_MINING=1` runs plain inference, and
`MINER_SKIP_BLOCK_SUBMISSION=true` runs the noisy multiplication without
sending proofs. The first run on a pod compiles kernels, so don't compare its
time with later runs.

## Rules

- **Change upstream's code only through `patches/`.** This repository replaces
  upstream's gateway. Any other change to upstream's code is a `git diff`
  against the pinned commit, saved in `patches/` with a numbered name and
  explained in `patches/README.md`. Keep patches small, and don't change the
  proof format or the verify function.
- **Move the upstream pin on purpose.** `docker/Dockerfile` and
  `scripts/gpu/setup_pod.sh` build one upstream commit. To move it, change both,
  check that every patch still applies, and rerun the tests.
- **Name upstream only as the source.** Credit Pearl Research Labs in the
  README, and call their code *upstream* everywhere else, or say what the part
  is: the plugin, the gateway, the verify function.
- **Say prover, not miner.** The GPU that does the work is the prover. Keep
  mining terms only in upstream names you can't change, such as
  `getMiningInfo`.
- **Version the session file.** The demo site reads it. When its shape changes,
  change `FORMAT` in `src/pouw/session.py`.
- **Keep data out of git.** Checkpoint tensors go in `data/`, run logs in
  `runs/`, and session files in `results/`. Git ignores all three.
- **Compare within a session.** Pods differ in speed, so times don't carry
  between them.
- **Report what you measure.** A cheat that passes is a finding. Report
  negative or mixed results plainly.
- **Keep `NEXT_STEPS.md` to open work.** When an item ships, delete it there and
  record the result in the README or a decision record.
