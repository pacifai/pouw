# Next steps

Open work, most important first. When an item is done, delete it here and
record the result in the README or a decision record.

- [ ] **Make proofs cheaper for the GPU.** With proofs on, inference takes
  21.5 s, against 12.3 s for the noisy multiplication alone. The plugin builds
  each proof inside vLLM's engine process: each win copies both matrices from
  the GPU and hashes them on the CPU, which stalls inference. With a target no
  tile can meet, proofs cost almost nothing, so the cost is in handling wins.
  Options, cheapest first: keep one host copy of the weights so a win copies
  only the activations, build proofs in a separate process, or take the proof
  paths from the GPU's own Merkle tree kernel. `patches/0001` is proposed
  upstream as pearl-research-labs/pearl#473.

- [ ] **Keep the verifier up with the GPU.** The verifier hashes every
  declared weight matrix again for each new job. On a pod, its verdicts lagged
  proofs by up to 20 s, and honest proofs failed the on-time check. Each GPU
  ID adds jobs, so the cost grows with the number of GPUs. Options: compare
  the opened columns with the declared matrix and run full checks on a sample
  of jobs, or hash the weights for the next job before issuing it.

- [ ] **Sign the GPU's ID.** The GPU reports its own ID, so a host can fake it
  or claim several IDs for one GPU. An H100 signs attestation reports only in
  confidential computing mode, inside a confidential virtual machine. Open
  questions: which providers offer H100s in that mode, and what the report
  shows about which GPU did the work.

- [ ] **Measure more loads.** Proven share and overhead are measured only for
  long prompts. Measure them for a decode-heavy load, and compare the
  estimated work with the H100's peak rate.

- [ ] **Does upstream's noise satisfy the paper's assumptions?** The paper
  assumes uniformly random noise of a fixed rank over a finite field.
  Upstream's noise has one factor with a single +1 and −1 per column, another
  with small integers, and uses plain integer arithmetic. Whether the paper's
  argument still holds is unchecked.

- [ ] **Cover more of the model.** Only layers stored as 7-bit integers use
  the noisy multiplication, about 67% of the multiply-adds, and batches under
  1,024 tokens earn no proofs. A comparison of proven work with a GPU's
  capacity has to account for the rest. Upstream's 8-bit floating-point work,
  pearl-research-labs/pearl#311, may change this.

- [ ] **Can a proof show that its inputs came from real requests?** A proof
  shows that a multiplication used the declared weights, not where its
  activations came from. vLLM's warm-up on dummy inputs also earns valid
  proofs. That is enough for counting compute, but not for showing that the
  served requests were real.
