# pouw

pouw verifies that a GPU serving an AI model ran the model's matrix
multiplications with the declared weights. It is the verifier for a **proof
of useful work (PoUW)** scheme: while the GPU serves the model, it emits
small proofs, and each accepted proof stands for a known amount of
multiplication.

The proof scheme is from
[Proofs of Useful Work from Arbitrary Matrix Multiplication](https://arxiv.org/abs/2504.09971).
The GPU side, a [vLLM](https://github.com/vllm-project/vllm) plugin with its
kernels and proof library, is Pearl Research Labs'
[implementation](https://github.com/pearl-research-labs/pearl), called
*upstream* in this repository. pouw adds the verifier.

## Motivation

Training and inference run on the same chips. A limit on training can be
checked only if the chips declared as serving a model can be shown to spend
their time on it. Proven multiplication with the declared weights, compared
with the GPU's peak rate, bounds the compute left for other work. A companion
project, [memchal](https://github.com/pacifai/memchal), bounds the GPU's spare
memory.

pouw is a research prototype. [NEXT_STEPS.md](NEXT_STEPS.md) lists open work.

## Installation

The upstream proof library has no published package, so the Docker image
builds it from a pinned upstream commit.

```bash
docker build -t pouw-local -f docker/Dockerfile .
```

## How to run

Run the tests on four real weight matrices from the checkpoint (about 40 MB):

```bash
docker run --rm -v "$PWD":/work pouw-local sh -c "
  python -m pouw fetch-tensors data/llama-sample.safetensors \
    model.layers.0.self_attn.o_proj.weight model.layers.20.self_attn.q_proj.weight \
    model.layers.20.self_attn.k_proj.weight model.layers.20.self_attn.v_proj.weight &&
  python -m pytest -q"
```

The tests run the verifier against a CPU prover, which makes real proofs
slowly, and against each cheat.

### Record a session on a GPU

Recording requires [uv](https://docs.astral.sh/uv/) and a
[RunPod](https://www.runpod.io) account with an
[API key](https://docs.runpod.io/get-started/api-keys) and a registered
[SSH key](https://docs.runpod.io/pods/configuration/use-ssh). Set
`RUNPOD_API_KEY` and `POUW_SSH_KEY` in `.env` (see `.env.example`), then run:

```bash
uv run runpod/record.py
```

The command shows the hourly price and waits for approval. It rents an NVIDIA
H100, builds upstream's plugin, and records three runs: honest, with cheated
proofs, and with a changed model. It saves a session file to
`results/sessions/` and the raw runs to `runs/`, then terminates the pod. A
session takes about 1.5 hours. Add `--shapes` to also measure the share of the
GPU's work that earned proofs.

## How it works

Each transformer layer multiplies a batch of activations `A` by a weight
matrix `B`. With upstream's plugin, the GPU computes each large product in
three steps:

1. Add noise to `A` and `B`, derived from the verifier's current job. Each
   result then depends on the job, so the work can't be done in advance or
   reused.
2. Multiply in small blocks of the output, called tiles, and hash each tile's
   running sums.
3. Remove the noise. Inference continues with the exact product.

A tile whose hash falls below the job's target is a **win**: about 1 in 50
million tiles in the recorded runs. For each win, the GPU sends a proof with
the tile's rows of `A`, its columns of `B`, and Merkle proofs that bind them
to the full matrices. A Merkle root is a single hash that commits to every
byte of a matrix. Wins are random, so their count estimates the total work.

The verifier replaces upstream's gateway, the process the plugin asks for
jobs and sends proofs to. A proof is accepted only if it passes five checks:

| Check | Rejects |
|---|---|
| Issued job: this verifier issued the job, unchanged | Made-up jobs and edited job fields |
| On time: the job is at most 15 seconds old | Work done in advance, and old proofs |
| Valid proof: upstream's verify function recomputes the tile, and it wins | Forged or losing tiles, and proofs moved to another job |
| Declared weights: `B`'s Merkle root matches a declared matrix | Other weights, including an edited copy |
| New tile: the tile wasn't submitted before | The same tile counted twice |

The [decision records](docs/decisions/) give the reasoning behind each check.

## Cheats

The cheat relay (`python -m pouw relay`) sits between the GPU and the
verifier and sends cheated copies of real proofs. The changed model is a copy
of the checkpoint with one weight changed in every matrix the plugin proves.

| Cheat | Prover's action | Rejected by | H100 run |
|---|---|---|---|
| Changed model | Serves a copy of the model with one weight changed per matrix | Declared weights | 72 of 72 |
| Edited proof | Changes one activation in a real proof by one | Valid proof | 5 of 5 |
| Easier target | Rewrites the job's target to one 256 times easier | Issued job | 5 of 5 |
| Another GPU's job | Submits a real proof under a job issued to another GPU | Valid proof | 5 of 5 |
| Replay | Sends a proof a second time | New tile, or on time | 6 of 6 |
| Late proof | Holds a proof until its job expires | On time | 5 of 5 |

Two cheats pass. Both are known limits:

- Made-up inputs. A proof shows that a multiplication used the declared
  weights, not that its activations came from real requests. The work is
  still done, which is enough for counting compute.
- A fake GPU ID. With `--attribute-gpus`, each GPU gets its own jobs, so a
  proof counts for one GPU. The GPU reports its own ID, and nothing signs it.

## Results

Measured on rented NVIDIA H100 SXM GPUs serving
`pearl-ai/Llama-3.1-8B-Instruct-pearl`, a Llama 3.1 8B checkpoint with 7-bit
integer weights. Times are comparable only within one session.

Cost, for 200 prompts of 2,048 tokens sent at once (median of four runs, one
session):

| Setup | Time |
|---|---:|
| Plain inference | 9.2 s |
| Noisy multiplication, no proofs sent | 12.3 s |
| Proofs on, with [`patches/0001`](patches/README.md) | 21.5 s |
| Proofs on, upstream unchanged | 32.4 s |

Work estimate: in one run, 197 accepted proofs implied 5.16 × 10¹⁵
multiply-adds. The plugin's own count was 5.29 × 10¹⁵, 2.4% higher and within
the estimate's expected spread of about 7%.

Coverage: only layers stored as 7-bit integers use the noisy multiplication,
about 67% of the model's multiply-adds, and the plugin skips batches under
1,024 tokens. In H100 runs, 97% of the 7-bit work earned proofs with
2,048-token prompts, and 40% with 256-token prompts and 512-token answers.

## Session files

A session file holds every proof of each recorded run, checked again by the
verifier, with the check that rejected it and the cheat behind it. The
[demo site](https://pacifai.github.io/compute-verification-demo/) replays one.
`python -m pouw session` writes one from recorded runs.

## Security

The verifier trusts every client that connects to its socket. Run it only
where access to the socket is controlled. To report a vulnerability, see
[SECURITY.md](SECURITY.md).

## Further reading

- [CONTRIBUTING.md](CONTRIBUTING.md): layout, build, tests, and rules for
  changes.
- [Architecture decision records](docs/decisions/): design decisions and the
  measurements behind them.
- [patches/](patches/README.md): changes to upstream's code.
- [NEXT_STEPS.md](NEXT_STEPS.md): open work.

## License

[MIT](LICENSE). The upstream code that `patches/` changes is under its own
[ISC license](https://github.com/pearl-research-labs/pearl/blob/master/LICENSE).
