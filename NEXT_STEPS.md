# Next steps

The goal is to measure, on one H100 running real vLLM inference, three things:

- how much the noisy multiply slows inference;
- what share of the GPU's work earns verified proofs;
- whether a separate verifier can check that work end to end, with fresh
  random values.

Attacks, multi-GPU setups, and the KV cache come after this.

## Done

- [x] **Verifier, tested locally.** It stands in for the upstream gateway, issues
  fresh random values, and runs four checks on each proof. A CPU stand-in for
  the prover produces real proofs, and tests cover each rejection case,
  including on real tensors from the int7 Llama checkpoint.
- [x] **Upstream plugin running on an H100.** `scripts/gpu/setup_pod.sh`
  builds it on a CUDA 13 pod. Upstream's `test_pearl_noisy_gemm` passes, and
  vLLM serves the int7 Llama.
- [x] **Prover connected to the verifier.** Proofs from real inference arrive
  and pass every check. Each one's weight matched a checkpoint matrix, which
  confirms that vLLM holds those weights as stored and fuses them as assumed
  (27 of the 80 matrices in the first run).
- [x] **First measurements under load.** `README.md` has the numbers: the
  noisy multiply alone adds about a third to run time, building proofs
  stalls inference, and the work estimate matches the expected work when no
  proofs are lost.
- [x] **Demo.** `pouw.demo` replays the recorded H100 run through the
  verifier and runs six attacks it detects and two it doesn't. See the Demo
  section of `README.md`.

## To do

- [ ] **Stop proof building from stalling inference.** The plugin builds each
  proof inside vLLM's engine process, and the proof library held Python's
  global lock (GIL) the whole time. `patches/0001-release-gil-in-merkle-tree.patch`
  makes the library release it while it hashes. In Docker on a laptop, a
  second Python thread now gets 61–67% of its possible turns during a tree
  build, up from 6% (`tests/test_gil.py`), and proofs are unchanged. The
  plugin's own copies of each matrix still hold the lock, and on the laptop
  they take about as long as the hash. On one H100, alternating the two
  libraries, proofs on took 22.4 s with the patch and 28.5 s without it,
  against 12.1 s with proofs off (medians of three runs). With a target
  nothing can win, proofs on took 12.4 s, so the per-GEMM callbacks cost
  almost nothing: the remaining cost is handling wins. Next, in order of
  effort: keep one host copy of the weights so a win copies only the
  activations, build proofs in a separate process, and have the GPU's
  existing Merkle tree kernel supply the proof paths. The win rate is also a
  lever: fewer wins cost less but make the work estimate noisier. *Done
  when:* overhead with proofs on is within a few percent of overhead with
  proofs off.
- [ ] **Measure at more load levels and targets.** Proven share now exists
  for two loads: 97% of int7 GEMM work for 2,048-token prompts at 3 per
  second, and 40% for 256-token prompts with 512-token answers. Next,
  measure overhead for the decode-heavy load, and compare the estimated
  work with the H100's int8 peak. *Done when:* proven share and overhead
  exist for at least two loads.
- [ ] **Attribute proofs to a GPU.** Version 1 runs locally: with
  `--attribute-gpus`, each GPU ID gets its own jobs, the ID is hashed into
  the header, and every accepted proof names the ID whose job it used.
  `patches/0002-send-gpu-id.patch` makes the upstream plugin send its GPU's
  UUID when `MINER_SEND_GPU_ID=true`. The UUID is unsigned, so a host can
  fake it or claim several IDs for one GPU. Each new ID also costs the
  verifier a pass of weight hashing, so a cap on IDs is needed before
  untrusted provers can connect. On an H100 on 2026-10-11, the plugin sent
  its GPU's UUID and every proof was credited to it. Next, replace the
  reported ID with a signed one. An H100 can sign an attestation report over
  a verifier-supplied 32-byte nonce with a device-unique key that chains to
  NVIDIA. According to NVIDIA's open driver source, that works only in
  confidential computing (CC) mode, which needs a confidential VM (Intel TDX
  or AMD SEV-SNP), not a RunPod container. Even then, the report shows that
  a genuine GPU answered after the nonce was issued, not which GPU ran a
  kernel. Only full CC mode, with the verifier attesting the VM, keeps a host
  from attesting with one GPU and computing on another. On a RunPod H100
  outside CC mode, NVML's certificate and attestation report calls both
  failed (`scripts/gpu/probe_attestation.py`). Next, find a provider that
  offers H100s in CC mode. *Done when:* proofs carry a GPU identity that
  a host can't fake, or it's documented which parts a host can still fake.

  Hashing the ID into the job stays useful once the ID is signed, because
  attestation and proofs show different things. Attestation shows a genuine
  GPU answered, but not how much work it did. Proofs show how much work was
  done, but not where. The ID in the job makes the work specific to one ID:
  each ID gets different noise, so a proof counts for exactly one ID, and
  claiming 10 IDs takes 10 times the work. With a signed ID, that gives
  per-GPU numbers, which are useful for three things:
  - **Bounding other work per GPU.** Total proven work only bounds the
    whole cluster. Per-GPU proven work, compared with each GPU's peak rate,
    bounds how much time each GPU had left for anything else.
  - **Checking a registry.** A declared GPU that produces no proofs is idle
    or doing something else.
  - **Location.** The ID gives no location itself. A separate delay-based
    check can: verifier machines at known places send nonces, the GPU signs
    each one, and the round-trip time bounds its distance (about 100 km per
    millisecond in fiber). The ID in the job then ties the proven work to
    that located GPU. Proofs take seconds and vary by milliseconds, so their
    own timing can't do this. The check fails against a relay, where a local
    GPU answers the pings and remote GPUs compute that ID's jobs, unless
    attested CC software rules out forwarding. A stolen device key, as in
    TEE.fail-style attacks, defeats it entirely.
- [ ] **Write up the results.**

- [ ] **Keep the verifier up with the prover.** The verifier hashes every
  declared weight matrix again for each new job, about 2 s per job in
  Docker on a laptop. On an H100 pod with 6.8 CPU cores, verdicts lagged
  proofs by up to 20 s. Because the plugin waits for each verdict before
  sending its next proof, the lag backs up the plugin's proof queue, and in
  one recorded run 28 honest proofs arrived stale. Every GPU ID adds jobs,
  so the cost grows with the number of GPUs. Options: compare the opened
  weight columns with the declared matrix instead of rehashing the whole
  model per job, with full checks on a sample of jobs; hash weights for the
  next job before it's issued; or reply to the prover before checking. *Done
  when:* verdicts keep up with a full-speed H100.

## Open questions

- **Low-rank inputs: does upstream's noise match the paper's argument?** The
  paper (Section 2) covers a prover that picks easy inputs, such as `A = 0`.
  If only the output were hashed, that would work: the product would reduce
  to the noise product `E·F`, which is cheap because both are low rank. So
  the scheme hashes the running sums at every `r`-wide step of the inner
  dimension instead. Each step multiplies blocks of `A+E` and `B+F` that are
  uniformly random on their own, whatever `A` and `B` are. A shortcut would
  have to exploit how the noise is correlated across steps, and the paper
  conjectures this is as hard as multiplying (Assumption 6.4). That's an
  assumption, not a proof. The paper's argument assumes noise that's
  uniformly random of rank `r` over a finite field. Upstream's noise is
  different: in `noise_generation.py`, one factor is a sparse matrix with one
  +1 and one −1 per column, the other holds small integers, and arithmetic
  is on plain integers. A rough count suggests that zero or low-rank `A`
  still saves nothing per tile, because each step is only `r` wide. Nobody
  has checked whether the paper's argument carries over to this noise.
- **Proving real requests.** The proofs show that multiplies with the
  declared weights happened. They don't show that `A` came from real
  requests. That's acceptable when the goal is to account for compute, but
  not if you need to prove the served requests were real.
- **fp8 and bf16 layers.** Coverage is set by which layers a checkpoint
  stores as 7-bit integers. A model served in fp8 earns no proofs for its fp8
  layers.
- **Proofs from warm-up and made-up inputs.** vLLM's startup warm-up runs the
  noisy multiply on dummy inputs, and those proofs pass every check. The
  verifier can't tell them apart from proofs of real requests, the same gap
  as above. For compute accounting this is correct, since the work happened.
- **Small batches.** The plugin skips the noisy multiply for batches under
  1,024 tokens, so work done in small batches earns no proofs. Proofs then
  account for less of the GPU's time, and a verifier comparing proven work
  with the GPU's capacity needs to know the batch sizes.
