# pouw

pouw checks that a GPU spent its time running a declared AI model. It uses
**proof of useful work (PoUW)**: while the GPU serves the model, it produces
small proofs that a verifier can check, and each accepted proof stands for a
known amount of the model's matrix multiplication.

The proof scheme comes from the paper
[Proofs of Useful Work from Arbitrary Matrix Multiplication](https://arxiv.org/abs/2504.09971).
The GPU side, a plugin for the [vLLM](https://github.com/vllm-project/vllm)
inference server with its GPU kernels and proof library, is Pearl Research Labs'
[open-source implementation](https://github.com/pearl-research-labs/pearl).
This repository calls that code *upstream* and adds the verifier.

## Why

Training a new AI model and serving an existing one use the same chips. A rule
that limits training holds only if someone can check what the chips declared as
serving are doing.

If a GPU proves how much multiplication it did with the declared model's
weights, the proven work, compared with what the GPU can do, bounds the time it
had left for anything else. pouw is the verifier for those proofs. A companion
project, [memchal](https://github.com/pacifai/memchal), checks the GPU's spare
memory.

pouw is a research prototype. The [open work](NEXT_STEPS.md) includes making
the proofs cheaper for the GPU and covering more of the model.

## Quickstart

### Try it on your computer

You need [Docker](https://docs.docker.com/get-docker/). The upstream proof
library isn't published as a package, so the image builds it from a pinned
upstream commit. Build the image, fetch four real weight matrices (about
40 MB), and run the tests:

```bash
docker build -t pouw-local -f docker/Dockerfile .
docker run --rm -v "$PWD":/work pouw-local sh -c "
  python scripts/fetch_tensors.py data/llama-sample.safetensors \
    model.layers.0.self_attn.o_proj.weight model.layers.20.self_attn.q_proj.weight \
    model.layers.20.self_attn.k_proj.weight model.layers.20.self_attn.v_proj.weight
  python -m pytest -q"
```

The tests run the verifier against a CPU stand-in for the GPU, which makes real
proofs slowly, and against each cheat.

### Record a session on a GPU

You need [uv](https://docs.astral.sh/uv/) and a [RunPod](https://www.runpod.io)
account with an [API key](https://docs.runpod.io/get-started/api-keys) and a
registered [SSH key](https://docs.runpod.io/pods/configuration/use-ssh).

```bash
cp .env.example .env    # then set RUNPOD_API_KEY and POUW_SSH_KEY in .env
uv run runpod/record.py
```

The command shows the hourly price and waits for your yes. It then rents an
NVIDIA H100, builds upstream's plugin, and records three runs: an honest one,
one with cheated proofs, and one serving a changed model. It saves a session
file to `results/sessions/` and the raw runs to `runs/`, and terminates the
pod, which ends the billing. A session takes about 1.5 hours.

To also measure what share of the GPU's work earned proofs, add `--shapes`.

## How it works

Every layer of a transformer multiplies a batch of activations `A` by a weight
matrix `B`. With upstream's plugin, the GPU does each large multiplication a
different way:

1. It adds random noise to `A` and `B`, derived from the verifier's latest job.
   The noise makes each result depend on the job, so work can't be done ahead
   of time or reused.
2. It multiplies in small blocks of the output, called tiles, and hashes each
   tile's running sums.
3. It removes the noise, and inference continues with the right answer.

A tile whose hash falls below the job's target is a **win**, about 1 in 50
million tiles in the recorded runs. For each win, the GPU sends a proof: the
tile's rows of `A`, its columns of `B`, and Merkle proofs that tie them to the
full matrices. Wins are random, so their count estimates the total work, much
as winning lottery tickets estimate how many tickets were bought.

The verifier stands in for upstream's gateway, the process the plugin asks for
jobs and sends proofs to. Each proof must pass five checks:

| Check | What it rules out |
|---|---|
| **Issued job:** this verifier issued the job, unchanged | Made-up jobs and easier targets |
| **On time:** the job is at most 15 seconds old | Work done ahead of time, and old proofs |
| **Valid proof:** upstream's verify function recomputes the tile, and it wins | Forged or losing tiles, and proofs moved to another job |
| **Declared weights:** `B`'s Merkle root matches one of the declared model's matrices | Any other weights, including an edited copy |
| **New tile:** nobody submitted this tile before | Counting one tile twice |

The [decision records](docs/decisions/) explain each choice.

## Cheats

The cheat relay, `pouw.cheat_relay`, sits between the GPU and the verifier and
sends cheated copies of real proofs. The changed model is a copy of the
checkpoint with one weight changed in every matrix the plugin proves.

| Cheat | What the prover does | Check that rejects it | H100 run |
|---|---|---|---|
| Changed model | Serves a copy of the model with one weight changed per matrix | Declared weights | 72 of 72 rejected |
| Edited proof | Changes one activation in a real proof by one | Valid proof | 5 of 5 |
| Easier target | Rewrites the job's target to one 256 times easier | Issued job | 5 of 5 |
| Another GPU's job | Submits a real proof under a job issued to another GPU | Valid proof | 5 of 5 |
| Replay | Sends a proof a second time | New tile, or on time | 6 of 6 |
| Late proof | Holds a proof until its job expires | On time | 5 of 5 |

Two cheats pass, and are known limits:

- **Made-up inputs.** A proof shows that a multiplication used the declared
  weights. It doesn't show that the activations came from real requests. The
  work is still done, so for counting compute this is acceptable.
- **A fake GPU ID.** With `--attribute-gpus`, each GPU gets its own jobs, so a
  proof counts for one GPU. The GPU reports its own ID, and nothing signs it.

## Results

Measured on rented NVIDIA H100 SXM GPUs, serving
`pearl-ai/Llama-3.1-8B-Instruct-pearl`, a Llama 3.1 8B checkpoint with 7-bit
integer weights. Compare times within one session only.

**Cost.** 200 prompts of 2,048 tokens, sent at once. Medians of four runs, from
one session.

| Setup | Time |
|---|---:|
| Plain inference | 9.2 s |
| Noisy multiplication, no proofs sent | 12.3 s |
| Proofs on, with [`patches/0001`](patches/README.md) | 21.5 s |
| Proofs on, upstream as is | 32.4 s |

**Work estimate.** In one run, 197 accepted proofs implied 5.16 × 10¹⁵
multiply-adds. The plugin's own count was 5.29 × 10¹⁵, 2.4% more, within the
estimate's expected spread of about 7%.

**Coverage.** Only layers stored as 7-bit integers use the noisy
multiplication, about 67% of the model's multiply-adds. The plugin also skips
batches under 1,024 tokens. In H100 runs, 97% of the 7-bit work earned
proofs with 2,048-token prompts, and 40% with 256-token prompts and 512-token
answers, where most batches are small.

## Session files

A session file holds every proof of each recorded run, checked again by the
verifier, with the check that rejected it and the cheat behind it. The
[demo site](https://pacifai.github.io/compute-verification-demo/) replays one.
To write one from recorded runs, use `python -m pouw.session`.

## Security

The verifier trusts whoever connects to its socket. Run it only where you
control who connects. To report a vulnerability, see [SECURITY.md](SECURITY.md).

## Learn more

- [CONTRIBUTING.md](CONTRIBUTING.md): the repository layout, how to build and
  test, and the rules for changes.
- [Decision records](docs/decisions/): why the design is the way it is, and the
  measurements behind each claim.
- [patches/](patches/README.md): the small changes this repository makes to
  upstream's code.
- [NEXT_STEPS.md](NEXT_STEPS.md): open work.

## License

[MIT](LICENSE). The upstream code that `patches/` changes is under its own
[ISC license](https://github.com/pearl-research-labs/pearl/blob/master/LICENSE).
