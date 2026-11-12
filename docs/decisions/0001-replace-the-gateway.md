# 0001: Verifier in place of upstream's gateway

**Status:** accepted

## Context

Upstream's vLLM plugin already does the GPU's side of proof of useful work: the
noisy multiplication, the tile hashes, and the proofs. It talks to a separate
process, the gateway, over a local socket with two JSON-RPC calls:
`getMiningInfo` asks for a job, and `submitPlainProof` sends a proof. Upstream's
gateway forwards proofs to a blockchain node. pouw needs a verifier that checks
proofs itself and counts the work.

## Decision

The verifier speaks the gateway's two calls and replaces it. The plugin
connects to it as it would to the gateway, with no change to how it makes
proofs. The verifier checks each proof with upstream's own verify function from
the proof library, and adds its own checks around it.

Any other change to upstream's code is a small patch in `patches/`, applied to
one pinned upstream commit. No patch changes the proof format or the verify
function.

## Consequences

- A proof that passes is a proof upstream would accept. pouw doesn't have to
  argue that its own proof code is sound.
- The verifier's own checks, on jobs, weights, and repeats, sit outside the
  proof format, so they can change without touching the GPU side.
- Each move of the upstream pin needs every patch rechecked.
- pouw inherits upstream's limits: only layers stored as 7-bit integers use the
  noisy multiplication, and the plugin skips batches under 1,024 tokens.

## Alternatives considered

- Write a prover from the paper. Rejected: it would take GPU kernels pouw
  doesn't have, and its soundness would rest on new code.
- Fork upstream and change the plugin freely. Rejected: every difference
  from upstream weakens the claim that the proofs are upstream's.
