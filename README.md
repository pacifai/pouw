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

### Attributing proofs to a GPU

Attribution is off by default. To credit each proof to a GPU, start the
verifier with `--attribute-gpus`. Each prover then sends its GPU's ID in
`getMiningInfo` and gets its own jobs. The verifier hashes the ID into the
header's Merkle-root field, so each GPU's noise and tile hashes differ. A
proof is credited to the ID of the job it used, and the log records it. A
prover can't move a proof to another GPU, because the ID is part of the
header that seeded the noise. A request without an ID gets the shared job,
and its proofs are logged with `gpu_id` set to `null`.

The ID is the UUID the driver reports, and nothing signs it. A dishonest host
can report any ID, or several IDs for one GPU. Attribution shows which ID
each proof was made for, not which physical GPU made it.

To send the ID from the upstream plugin, set `MINER_SEND_GPU_ID=true`; see
`patches/README.md`. From the CPU stand-in, pass `--gpu-id <id>`. Each GPU's
jobs use their own hash key, so the weight check hashes the declared weights
once per GPU for each random value.

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
  `patches/0001-release-gil-in-merkle-tree.patch` now makes the proof library
  release the lock while it hashes. The second session below measures it.
- **vLLM's startup warm-up produces valid proofs.** Before serving requests,
  vLLM runs forward passes on dummy inputs. These went through the noisy
  multiply, and two of three runs produced proofs from them. The verifier
  accepted them, because the work was real. Two identical warm-up passes
  also produced the same tile twice, which the verifier rejected as a
  duplicate.
- **Small batches earn no proofs.** The plugin runs the noisy multiply only
  when a batch has at least 1,024 tokens (`min_m` in its `config.yaml`).
  Decode steps with fewer tokens use the plain multiply.

### Second H100 session

Measured on 2026-10-11 on another H100 SXM pod, with all three patches and
the same model. Logs are in `runs/h100-2026-10-11/`, which git ignores, and
`scripts/gpu/demo_session.sh` reproduces them. This pod ran the same load
about twice as slowly as the first one, so compare its runs with each other,
not with the first session's.

- **This pod's timings were misleading.** On the same load as above, the
  medians of three runs were 21.0 s for the plain multiply, 24.1 s with
  proofs off, and 31.6 s with proofs on. That made proofs look cheap, but
  the third session below, on a faster pod, refutes it: this slow host
  shrank every GPU-side cost as a share of run time.
- **Attribution works with the GPU's UUID.** With `MINER_SEND_GPU_ID=true`
  and `--attribute-gpus`, every proof of the recorded run came from a job
  for the GPU's UUID, `GPU-425a2592-…`.
- **The live verifier fell behind.** The recorded run served 540 prompts of
  2,048 tokens at 3 per second. Verdicts lagged submissions by up to 20 s,
  because the verifier hashes every declared weight matrix again for each
  new job, on the pod's 6.8 CPU cores. The run script stopped it 15 s after
  the load ended, so 14 of the 198 submitted proofs never got a verdict. The
  script now waits until the verifier's log stops growing.
- **Replaying the recording reproduces every verdict.** On a laptop,
  `pouw.recording` rechecked all 198 recorded proofs at their arrival times.
  It reached the same verdict for each of the 184 the live verifier checked,
  and accepted the other 14.
- **The work estimate matches the plugin's own count.** The 197 accepted
  proofs imply 5.16 × 10¹⁵ multiply-adds. The plugin's GEMM log
  (`patches/0003-log-gemm-shapes.patch`) counts 5.29 × 10¹⁵ multiply-adds in
  noisy multiplies, including warm-up: 2.4% more. With about 200 proofs, the
  estimate's expected spread is about 7%.
- **The proven share depends on the load.** In the recorded run, 97% of the
  int7 GEMM work used the noisy multiply. In a decode-heavy run, 200 prompts
  of 256 tokens with 512 output tokens each, only 40% did, because most
  decode batches had fewer than 1,024 tokens.
- **The GPU can't attest outside CC mode.** NVML reported confidential
  computing off (`ccFeature=0`), and both the certificate and attestation
  report calls failed with `NVML_ERROR_UNKNOWN`. The UUID and serial number
  were readable, and both are unsigned.

### Third H100 session

Measured on 2026-10-11 on a third H100 SXM pod, as fast as the first. Logs
are in `runs/h100-2026-10-11b/`, which git ignores, and
`scripts/gpu/demo_session2.sh` reproduces most of them. Timings are medians
of three runs on the same load as the first session.

| Run | Median |
|---|---|
| Plain multiply | 9.2 s |
| Noisy multiply, proofs off | 12.1 s |
| Noisy multiply, proofs on, nothing can win (two runs) | 12.4 s |
| Proofs on, with the GIL patch | 22.4 s |
| Proofs on, without the GIL patch | 28.5 s |

- **The noisy multiply costs about 30%.** 12.1 s against 9.2 s, the same as
  the first session (+32%).
- **The GIL patch helps, but most of the proof cost remains.** The patch and
  unpatched libraries were built on the same pod and alternated, with
  `tests/test_gil.py` confirming which was installed before each run. Proofs
  on took 22.4 s with the patch and 28.5 s without it, and the unpatched
  runs varied far more (27–38 s). Either way, proofs on costs 85–135% more
  than proofs off.
- **Handling wins is the cost, not watching for them.** The plugin queues a
  Python callback for every noisy GEMM. With a target nothing can win, proofs
  on took 12.4 s, the same as proofs off. So the remaining cost comes from
  what happens on a win: copying the activations and the weight matrix to
  the CPU, and building the proof.
- **The verifier's lag makes honest proofs stale.** The plugin waits for each
  verdict before sending its next proof. While the verifier hashes weights
  for each new job, proofs queue in the plugin, and some reach the verifier
  more than 15 s after their job was issued. In the recorded relay run, 28
  honest proofs were rejected as stale. Each extra GPU ID adds a job, and
  its own pass of weight hashing.
- **Cheats sent through a relay fail the expected checks.** In a recorded
  run, `pouw.cheat_relay` sent cheated copies of real proofs. Replays were
  rejected as duplicates (4) or stale (2), easier targets at job binding
  (5), changed activations and moved proofs at proof verification (5 each),
  and held-back proofs as stale (5). Proofs under a made-up GPU ID were
  accepted and credited to it.
- **A changed checkpoint fails every proof.** vLLM served a copy of the
  checkpoint with one weight changed in each int8 matrix, against a verifier
  declaring the original. All 72 proofs failed the weight check.
- **Replay matches live.** Replaying each recording reproduced every verdict
  the live verifier gave.

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

To try attribution, add `--attribute-gpus` to the verifier command and
`--gpu-id GPU-test-0` to the prover command.

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

## Demo

The demo replays a recorded H100 run on your laptop and lets you run attacks
against it. Every recorded proof goes through the verifier again, at its
recorded arrival time, so each verdict on the page is a fresh check. The
attacks send real submissions through the same verifier:

- **Detected:** replay, precomputation, difficulty tampering, a forged
  proof (one activation changed), model substitution (one weight changed),
  and proof theft (the H100's proof moved onto another GPU's job). Each
  fails the check named on its card.
- **Undetected:** synthetic inputs (random activations with the declared
  weights) and GPU ID spoofing (the laptop claims the H100's UUID). Both are
  accepted. Covert training and relaying jobs to remote GPUs are listed
  without a Run button, because the demo can't stage them honestly.

Model substitution, synthetic inputs, and spoofing need a new proof, which
the laptop builds on the CPU. A CPU can't find wins at the H100's target, so
those jobs use a target of 1 win in 50 tiles. The work estimate accounts for
each job's own target.

To run it, download the checkpoint once, then start the server:

```sh
uvx --from huggingface_hub hf download pearl-ai/Llama-3.1-8B-Instruct-pearl --local-dir data/model
docker run --rm -p 8080:8080 -v "$PWD":/work pouw-local python -m pouw.demo.server \
  --run runs/h100-2026-10-11/demo runs/h100-2026-10-11b/cheats3 runs/h100-2026-10-11b/model-sub \
  --timing runs/h100-2026-10-11b --weights data/model/*.safetensors
```

Then open http://localhost:8080. The run directory comes from
`scripts/gpu/demo_session.sh`, which records it on an H100 pod. To pick
between several recorded runs on the page, pass several directories to
`--run`.

**Recorded cheats.** `scripts/gpu/demo_session2.sh` records two more runs.
In one, `pouw.cheat_relay` sits between the plugin and the verifier, and
after every eighth honest proof sends a cheated copy: a replay, an easier
target, one changed activation, another GPU's job, or a proof held past its
job's expiry. For 30 seconds it also fakes the GPU ID. The relay logs each
change, and the demo labels those submissions in the replay. In the other
run, vLLM serves a copy of the checkpoint with one weight changed in every
int8 matrix, against a verifier that declares the original.

**Live mode.** With `--live`, the demo server is also the verifier: it
listens on the socket the plugin connects to, and the page shows each proof
as it arrives, with how long it waited for its verdict. On a pod, run
`LIVE=1 bash scripts/gpu/run_e2e.sh` and forward port 8080 over SSH. Live
mode also records the session, so you can replay it later.

## Layout

- `src/pouw/jobs.py`: job headers carrying the random value, and the
  conversions between difficulty and win probability.
- `src/pouw/weights.py`: the declared weights, vLLM-style fusing of `qkv` and
  `gate_up`, and the Merkle-root check.
- `src/pouw/verifier.py`: the four checks and the JSON-RPC server that stands
  in for the upstream gateway.
- `src/pouw/cpu_prover.py`: the CPU stand-in for the GPU prover, with
  options to misbehave.
- `src/pouw/recording.py`: records a verifier session and rebuilds a verifier
  that replays it.
- `src/pouw/demo/`: the demo server, its replay, live mode, attacks, and
  the page.
- `src/pouw/cheat_relay.py`: a relay between the prover and the verifier that
  sends cheated copies of real proofs on a schedule.
- `scripts/fetch_tensors.py`: downloads individual tensors from a Hugging Face
  checkpoint.
- `scripts/summarize_log.py`: counts accepted and rejected proofs in a
  verifier log and estimates the work they imply.
- `scripts/gpu/`: pod setup, the end-to-end run, and a benchmark of one noisy
  multiply against a plain one.
- `patches/`: the changes this repo makes to upstream code, applied at build
  time, with `patches/README.md` explaining each.
- `docs/glossary.md`: the terms used here, and where each comes from.
- `upstream/`: unmodified copies of the upstream files `src/pouw` adapts,
  with a table that pairs each function with its original.
