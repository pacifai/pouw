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

## To do

- [ ] **Bring up the upstream plugin on an H100.** Its kernels compile only for
  Hopper (`sm_90a`). Install vLLM and the upstream packages directly on the
  pod, because RunPod doesn't allow Docker inside a pod. Then run the
  upstream test `test_pearl_noisy_gemm` and serve `pearl-ai/Llama-3.1-8B-Instruct-pearl`.
  *Done when:* the test passes and the server answers prompts.
- [ ] **Confirm that vLLM's weights match the checkpoint.** The weight check
  assumes vLLM holds each weight exactly as stored, fusing `q/k/v` and
  `gate/up` by stacking rows in that order. *Done when:* the Merkle root of
  each fused tensor in GPU memory matches the one `pouw.weights` computes from
  the checkpoint.
- [ ] **Connect the prover to the verifier.** Start `pouw.verifier` with the
  full checkpoint, then start vLLM with the plugin's gateway connection
  enabled.
  *Done when:* proofs from real inference arrive and are accepted.
- [ ] **Measure under load.** Drive the server with vLLM's benchmark script,
  keeping at least 128 tokens per batch: the noise needs `A` to have at
  least 128 rows. Record:
  - tokens per second with the noisy multiply versus upstream's plain int8
    multiply;
  - tiles estimated from proofs versus tiles expected from the matrix shapes
    and token counts;
  - estimated multiply work versus the H100's int8 peak.

  Repeat at a few targets, because each win copies the full `A` and `B` to
  CPU memory, and that copying could distort the overhead figure. *Done
  when:* these numbers exist for at least two load levels.
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
