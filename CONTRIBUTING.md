# Contributing

This guide covers the repository layout, the build and tests, and the rules
for changes. The [README](README.md) describes what pouw is.

## Layout

The `pouw` package has three parts. The core is the verifier and an honest
CPU prover. `pouw.cheats` and `pouw.record` build the demo's runs on top of
it, and the core never imports them.

| Path | Contents |
|---|---|
| `src/pouw/cli.py` | The single entry point, `python -m pouw <command>` |
| `src/pouw/verifier.py` | The five checks, and the service that answers the plugin's calls |
| `src/pouw/jobs.py` | Jobs and the issuer that hands them out |
| `src/pouw/weights.py` | The declared weights, fused as vLLM holds them, and their Merkle roots |
| `src/pouw/proof.py` | The GPU prover's tile pattern and noise rank, and the job key |
| `src/pouw/target.py` | The target's compact encoding and a tile's chance to win |
| `src/pouw/protocol.py` | The JSON-RPC client and server shared by the verifier and the relay |
| `src/pouw/prover.py` | The CPU prover |
| `src/pouw/summary.py` | Verdict counts and estimated work from a verifier log |
| `src/pouw/checkpoint.py` | Downloads single tensors from a checkpoint on Hugging Face |
| `src/pouw/cheats/` | One class per cheat, and the relay that sends them |
| `src/pouw/record/` | The recorder, the session file, and the changed model |
| `tests/` | Tests |
| `scripts/` | Shell scripts for the GPU host: setup, one run, and a session's three runs |
| `runpod/record.py` | Rents an H100, records a session, and releases the pod |
| `patches/` | Changes to upstream's code |
| `docker/` | The image that builds upstream's proof library and runs the tests |
| `docs/decisions/` | Architecture decision records |

## Commands

| Command | Action |
|---|---|
| `verify` | Issue jobs and check every proof |
| `prove` | Send proofs from the CPU prover |
| `relay` | Pass calls to the verifier, and send cheated copies of real proofs |
| `session` | Check recorded runs again and write the session file |
| `summarize` | Count a verifier log's verdicts and the work they imply |
| `fetch-tensors` | Download single tensors from a Hugging Face checkpoint |
| `changed-model` | Copy a checkpoint with one weight changed in every int8 matrix |

`python -m pouw <command> --help` lists each command's options.

## Build and test

Everything except the GPU runs in Docker:

```bash
docker build -t pouw-local -f docker/Dockerfile .
docker run --rm -v "$PWD":/work pouw-local python -m pytest -q
```

`tests/test_real_weights.py` needs four tensors from the real checkpoint and
skips itself without them. The README's commands fetch them.

To run the verifier and the CPU prover as separate processes inside the
image:

```bash
python -m pouw verify --weights data/llama-sample.safetensors --win-probability 0.02 --log runs/local.jsonl &
python -m pouw prove --weights data/llama-sample.safetensors \
  --matrix model.layers.0.self_attn.o_proj.weight --proofs 3
```

### GPU

Upstream's kernels require a Hopper GPU, such as an H100, and a driver that
supports CUDA 13. `uv run runpod/record.py` rents one and runs everything. To
work on a pod by hand, copy the repository to `/root/pouw` and run:

```bash
bash /root/pouw/scripts/setup_pod.sh     # builds upstream and downloads the model
bash /root/pouw/scripts/run_e2e.sh       # one run: the verifier, vLLM, and a benchmark
```

`run_e2e.sh` reads its settings from environment variables, listed at its top.
`NO_MINING=1` runs plain inference, and `MINER_SKIP_BLOCK_SUBMISSION=true`
runs the noisy multiplication without sending proofs. The first run on a pod
compiles kernels, so its time is not comparable with later runs.

## Rules

- Change upstream's code only through `patches/`. Each change is a `git diff`
  against the pinned commit, saved with a numbered name and described in
  `patches/README.md`. Keep patches small, and leave the proof format and the
  verify function unchanged.
- Move the upstream pin deliberately. `docker/Dockerfile` and
  `scripts/setup_pod.sh` build the same commit. To move it, change both, check
  that every patch applies, and rerun the tests.
- Credit Pearl Research Labs in the README only. Elsewhere, call their code
  *upstream*, or name the part: the plugin, the gateway, the verify function.
- Use *prover*, not *miner*, for the GPU that does the work. Mining terms stay
  only in upstream names, such as `getMiningInfo`.
- Version the session file. The demo site reads it, so change `FORMAT` in
  `src/pouw/record/session.py` whenever its shape changes.
- Keep data out of git: checkpoint tensors in `data/`, run logs in `runs/`,
  and session files in `results/`.
- Compare times only within one session. Pods differ in speed.
- Report results as measured, including cheats that pass and mixed results.
- When an item in `NEXT_STEPS.md` ships, delete it there and record the result
  in the README or a decision record.
