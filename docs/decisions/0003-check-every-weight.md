# ADR 0003: The verifier recomputes the Merkle root of every declared weight matrix for each job

**Status:** accepted

## Context

A proof opens one tile's columns of the weight matrix `B`, and carries the
Merkle root of all of `B`. A Merkle root is a hash that commits to every byte
of the matrix: change one byte, and the root changes. The proof format keys
the tree with a hash of the job, so the same matrix has a different root under
each job.

Checking only the opened columns would let a prover use different weights
everywhere else.

## Decision

The verifier holds the declared model's 7-bit weight matrices, fused the way
vLLM holds them. For each job and matrix shape, it computes every declared
matrix's root under that job's key, once, and caches it. A proof passes only if
its root equals one of them.

## Evidence

- On an H100, vLLM served a copy of the model with one weight changed in every
  7-bit matrix, against a verifier that declared the original. All 72 proofs
  failed the declared-weights check.
- On a laptop, changing one weight out of 16.7 million in a real matrix got the
  proof rejected.
- Every proof from honest runs matched a checkpoint matrix, so vLLM holds the
  weights byte for byte as stored.

## Consequences

- The check covers every byte of every proven matrix.
- It costs the verifier one pass of hashing over the declared weights for each
  new job: about 2 seconds for the 4.4 GiB of 7-bit weights in Docker on a
  laptop. With GPU attribution, every GPU's jobs add their own pass. On a pod,
  this made the verifier fall behind the GPU.

## Alternatives considered

- **Check only the opened columns against the declared matrix.** Much cheaper,
  but blind to the rest of the matrix. Combined with full checks on a sample of
  jobs, it is an open option.
- **Hash the weights for the next job before issuing it.** It hides the cost
  without reducing it. Also open.
