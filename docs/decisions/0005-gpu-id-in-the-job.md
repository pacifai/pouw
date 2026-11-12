# ADR 0005: Each GPU's ID is hashed into its own jobs, and the ID is self-reported

**Status:** accepted

## Context

Total proven work bounds what a whole cluster could do besides inference. To
bound each GPU, the verifier has to credit each proof to one GPU, in a way the
prover can't move afterward.

## Decision

With `--attribute-gpus`, the plugin sends its GPU's ID with each job request
(`patches/0002`), and each ID gets its own jobs. The verifier hashes the ID into
the job's header, in the Merkle-root field, so each GPU's noise and tile hashes
differ. A proof is credited to the ID of the job it used.

The ID is the UUID the driver reports. Nothing signs it.

## Evidence

- On an H100, every proof was credited to the GPU's UUID.
- A relay moved real proofs onto jobs issued to another ID. All 5 failed the
  valid-proof check, because the noise no longer matched.
- The same relay replaced the reported ID with a made-up one for 30 seconds.
  Those proofs were accepted and credited to the made-up ID.
- Outside confidential computing mode, the H100's attestation calls failed, so
  it couldn't sign anything.

## Consequences

- A proof counts for exactly one ID, and claiming two IDs takes twice the work.
- A dishonest host can report any ID, or several IDs for one GPU. Attribution
  shows which ID a proof was made for, not which GPU made it.
- Every ID costs the verifier its own pass of weight hashing
  ([ADR 0003](0003-check-every-weight.md)), so untrusted provers need a cap on
  IDs.

## Alternatives considered

- **A signed ID from the GPU's attestation.** An H100 can sign a report over a
  verifier's nonce, but only in confidential computing mode, inside a
  confidential virtual machine. Even then, it shows that a genuine GPU answered,
  not which GPU did the work. Open.
