# Next steps

Open work, most important first. Each item says what is open and why it
matters. When an item is done, delete it here and record the result in the
README or a decision record.

- [ ] **Make proofs cheaper for the GPU.** With proofs on, inference takes 21.5 s
  against 12.3 s with the noisy multiplication alone. The plugin builds each
  proof inside vLLM's engine process, and each win copies both matrices from
  the GPU and hashes them on the CPU, which stalls inference. With a target
  nothing can win, proofs on cost almost nothing, so the cost comes from
  handling wins. Options, in order of effort: keep one host copy of the
  weights so a win copies only the activations, build proofs in a separate
  process, and have the GPU's own Merkle tree kernel supply the proof paths.
  `patches/0001` is proposed upstream as pearl-research-labs/pearl#473.

- [ ] **Keep the verifier up with the GPU.** The verifier hashes every declared
  weight matrix again for each new job, and on a pod its verdicts lagged proofs
  by up to 20 s, so honest proofs failed the on-time check. Every GPU ID adds
  jobs, so the cost grows with the number of GPUs. Options: compare the opened
  columns with the declared matrix and run full checks on a sample of jobs, or
  hash the weights for the next job before issuing it.

- [ ] **Sign the GPU's ID.** The GPU reports its own ID, so a host can fake it or
  claim several IDs for one GPU. An H100 can sign an attestation report only in
  confidential computing mode, inside a confidential virtual machine. Open
  questions: which providers offer H100s in that mode, and what the report shows
  about which GPU did the work.

- [ ] **Measure more loads.** Proven share and overhead exist for long prompts.
  Measure them for a decode-heavy load, and compare the estimated work with the
  H100's peak rate.

- [ ] **Does upstream's noise match the paper's argument?** The paper assumes
  noise that is uniformly random of a fixed rank over a finite field. Upstream's
  noise is different: one factor has one +1 and one −1 per column, the other
  holds small integers, and the arithmetic is on plain integers. Whether the
  paper's argument still holds is unchecked.

- [ ] **Cover more of the model.** Only layers stored as 7-bit integers use the
  noisy multiplication, about 67% of the multiply-adds, and batches under 1,024
  tokens earn no proofs. A verifier comparing proven work with a GPU's capacity
  has to account for the rest. Upstream's work on 8-bit floating point,
  pearl-research-labs/pearl#311, may change this.

- [ ] **Can a proof show its inputs came from real requests?** A proof shows that
  a multiplication used the declared weights, not that its activations came
  from real requests. vLLM's warm-up on dummy inputs also earns valid proofs.
  This is fine for counting compute, but not for proving that the served
  requests were real.
