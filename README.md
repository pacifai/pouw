# pouw

A verifier that checks a GPU did inference matrix multiplies with a declared
model's weights, using proof of useful work (PoUW).

The proof scheme comes from the paper
[Proofs of Useful Work from Arbitrary Matrix Multiplication](https://arxiv.org/abs/2504.09971).
The vLLM plugin, its GPU kernels, and the proof library this repo uses are
Pearl Research Labs' [open-source implementation](https://github.com/pearl-research-labs/pearl),
called *upstream* in this repo. Credit for that implementation goes to them.

This is one part of a larger compute-verification effort: showing that a GPU
runs only its declared inference workload. A companion project,
[memchal](../memchal), checks the GPU's free memory.

## How it works

The prover is the GPU running inference with the upstream vLLM plugin.
(Upstream calls it a miner, and some names it expects, such as
`getMiningInfo`, still say so.) The plugin replaces the matrix multiply in each
linear layer. For
`C = A·B`, where `A` is the batch's activations and `B` is a weight matrix:

1. It hashes `A` and `B` together with a job header.
2. From that hash, it generates two random low-rank noise matrices, `E` and
   `F`.
3. It computes `(A+E)(B+F)` on the GPU, hashing the running sums of each small
   output tile along the way.
4. It subtracts the noise terms to recover `A·B`, and inference continues as
   normal.

The low rank keeps step 4 cheap. Occasionally a tile's hash falls below a
difficulty target. The prover then sends a proof containing the tile's rows of
`A`, its columns of `B`, and Merkle proofs tying them to the full matrices.

The prover gets its job header from a separate process, the gateway, over a
local socket. This repo replaces that gateway. Each job carries a fresh random
value from the verifier, so a proof can only come from work done after the
verifier issued it. Each proof must pass four checks:

| Check | What it rules out |
|---|---|
| The job was issued here within the last `--max-age` seconds | Precomputed or replayed work |
| The upstream verify function accepts the tile | Fake or losing tiles |
| `B`'s Merkle root matches a declared weight matrix under the same job | Multiplies with any other matrix, including a compressed or edited copy of the weights |
| The tile wasn't submitted before | Counting one tile twice |

The verifier recomputes each declared matrix's root under the current job's
key. The check therefore covers every byte of the matrix, not only the rows
the tile opened.

### From proofs to an amount of work

Each accepted proof records `win_probability`, the chance that one tile hash
wins. Summing `1 / win_probability` over accepted proofs estimates how many
tiles the GPU hashed. With the GPU prover's settings, one tile is 2 rows by 64
columns of output over the full inner dimension `k`. That's `128 × k`
multiply-adds, and the tiles cover each output matrix exactly once.

## Status

### On an H100

Measured on 2026-10-10 on one H100 SXM, with upstream's vLLM plugin (vLLM
0.20.2) serving `pearl-ai/Llama-3.1-8B-Instruct-pearl`, connected to this
verifier. The load was 200 prompts of 2,048 random tokens with 64 output
tokens each, sent at once. Logs are in `runs/h100-2026-10-10/`, which git
ignores.

- **Proofs from real inference pass every check.** All 39 proofs from the
  first run were accepted, covering 27 different weight matrices. Each
  matched a checkpoint matrix, so vLLM holds those weights byte for byte as
  stored, fused the way `pouw.weights` assumes.
- **The work estimate matches the expected work.** The 39 proofs imply
  2.04 × 10¹⁵ multiply-adds. The prompt tokens times the proven matrix shapes
  give 1.93 × 10¹⁵.
- **The noisy multiply alone costs about a third more time.** With proofs
  off, the run took 12.1 s against 9.2 s for the plain multiply (median of
  three and two runs). Measured alone at 8,192 tokens, one noisy multiply
  takes 2.0 times as long as the plain one for `gate_up_proj`, 3.4 times
  for `qkv_proj`, and 4.3 times for `o_proj`.
- **Building proofs stalls inference.** With proofs on, at a 1-in-10⁸ chance
  per tile, the run took about 28 s, three times the plain multiply. The
  plugin builds each proof inside vLLM's engine process, and the build holds
  Python's global lock for 0.1–0.4 s, freezing inference. Wins also arrive
  faster than proofs get built, and the queued ones were lost when the run
  stopped, so later runs undercounted the work.
- **vLLM's startup warm-up produces valid proofs.** Before serving requests,
  vLLM runs forward passes on dummy inputs. These went through the noisy
  multiply, and two of three runs produced proofs from them. The verifier
  accepted them, because the work was real. Two identical warm-up passes
  also produced the same tile twice, which the verifier rejected as a
  duplicate.
- **Small batches earn no proofs.** The plugin runs the noisy multiply only
  when a batch has at least 1,024 tokens (`min_m` in its `config.yaml`).
  Decode steps with fewer tokens use the plain multiply.

### On a laptop

- **The verifier works against real proofs.** A CPU stand-in for the prover
  (`pouw.cpu_prover`) builds proofs the way the GPU prover does. The tests show
  each check catching the case it targets: an honest proof is accepted, while
  a changed weight, a losing tile, a replay, a job the verifier never issued,
  a stale job, and a loosened target are each rejected.
- **The weight check works on a real checkpoint.** Proofs built from
  real `o_proj` and fused `qkv_proj` tensors of
  `pearl-ai/Llama-3.1-8B-Instruct-pearl` are accepted. Changing one weight out
  of 16.7 million gets the proof rejected.
- **Only part of the model earns proofs.** The int7 Llama checkpoint stores
  these layers as 7-bit integers, which run the noisy multiply: every
  `o_proj`, `gate_proj`, and `up_proj`, plus `q/k/v_proj` in layers 16–31.
  Every `down_proj`, `q/k/v_proj` in layers 0–15, and `lm_head` use fp8 or
  bf16 and earn no proofs. That's about 67% of the multiply-adds in the
  transformer blocks, or 63% if you include `lm_head`. This figure comes from
  the checkpoint's config, not a measurement. Attention over the KV cache is
  additional unproven work on top.
- **Weight-check cost:** about 2.2 GiB/s of hashing, measured in Docker on an
  Apple M-series laptop. For the 4.4 GiB of proven weights, that's roughly 2
  seconds per new random value.

## Run it

The upstream Rust library, `pearl_mining`, isn't published as a package, so
the Docker image builds it from a pinned upstream commit:

```bash
docker build -t pouw-local -f docker/Dockerfile .
```

To fetch four real tensors (about 40 MB) and run the tests:

```bash
docker run --rm -v "$PWD":/work pouw-local sh -c "
  python scripts/fetch_tensors.py data/llama-sample.safetensors \
    model.layers.0.self_attn.o_proj.weight model.layers.20.self_attn.q_proj.weight \
    model.layers.20.self_attn.k_proj.weight model.layers.20.self_attn.v_proj.weight
  python -m pytest -q"
```

To run the verifier and the CPU stand-in as separate processes:

```bash
python -m pouw.verifier --weights data/llama-sample.safetensors --win-probability 0.02 --log runs/local.jsonl &
python -m pouw.cpu_prover --weights data/llama-sample.safetensors \
  --matrix model.layers.0.self_attn.o_proj.weight --proofs 3
```

To run on an H100, rent a pod with the `nvidia/cuda:13.0.3-devel-ubuntu24.04`
image on a host whose driver supports CUDA 13. Copy this repo to
`/root/pouw` and download the checkpoint to `/root/model`. Then run:

```bash
bash /root/pouw/scripts/gpu/setup_pod.sh       # builds upstream; takes a while
bash /root/pouw/scripts/gpu/run_e2e.sh         # noisy multiply, proofs on
NO_MINING=1 bash /root/pouw/scripts/gpu/run_e2e.sh                    # plain multiply
MINER_SKIP_BLOCK_SUBMISSION=true bash /root/pouw/scripts/gpu/run_e2e.sh   # noisy, proofs off
```

`run_e2e.sh` starts the verifier on `/tmp/pearlgw.sock`, where the plugin
connects by default, then vLLM and its benchmark. Run each setup more than
once: the first run compiles kernels and is slower.

## Layout

- `src/pouw/jobs.py`: job headers carrying the random value, and the
  conversions between difficulty and win probability.
- `src/pouw/weights.py`: the declared weights, vLLM-style fusing of `qkv` and
  `gate_up`, and the Merkle-root check.
- `src/pouw/verifier.py`: the four checks and the JSON-RPC server that stands
  in for the upstream gateway.
- `src/pouw/cpu_prover.py`: the CPU stand-in for the GPU prover, with
  options to misbehave.
- `scripts/fetch_tensors.py`: downloads individual tensors from a Hugging Face
  checkpoint.
- `scripts/summarize_log.py`: counts accepted and rejected proofs in a
  verifier log and estimates the work they imply.
- `scripts/gpu/`: pod setup, the end-to-end run, and a benchmark of one noisy
  multiply against a plain one.
- `docs/glossary.md`: the terms used here, and where each comes from.
- `upstream/`: unmodified copies of the upstream files `src/pouw` adapts,
  with a table that pairs each function with its original.
