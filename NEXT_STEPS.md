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

## To do

- [ ] **Stop proof building from stalling inference.** The plugin builds each
  proof inside vLLM's engine process while holding Python's global lock.
  Moving that work to a separate process, or copying `A` and `B` out and
  hashing them elsewhere, needs a change to the upstream plugin, which
  `CLAUDE.md` says needs a deliberate decision. Until then, measure the
  overhead with proofs off, and pick a target that keeps wins rare.
  *Done when:* overhead with proofs on is within a few percent of overhead
  with proofs off.
- [ ] **Measure at more load levels and targets.** So far there's one load:
  200 prompts of 2,048 tokens, sent at once. Add a decode-heavy load, where
  most batches have fewer than 1,024 tokens and earn no proofs, to measure
  what share of real serving work gets proven. Compare the estimated work
  with the H100's int8 peak. *Done when:* proven share and overhead exist
  for at least two loads.
- [ ] **Write up the results.**

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
