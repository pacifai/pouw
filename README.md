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

Everything below runs on a laptop. Nothing has run on a GPU yet.

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

On a GPU box, start `pouw.verifier` with the full checkpoint's safetensors
files, then start vLLM with the upstream plugin and its gateway connection
enabled. The plugin connects to `/tmp/pearlgw.sock` by default, which is where the verifier
listens.

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
- `upstream/`: unmodified copies of the upstream files `src/pouw` adapts,
  with a table that pairs each function with its original.
