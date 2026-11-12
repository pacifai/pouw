# ADR 0002: Every job carries a fresh random value and expires after 15 seconds

**Status:** accepted

## Context

A proof shows that some multiplication happened. To count it as work the GPU
did during a given time, the verifier has to know the work couldn't have been
done earlier, or reused from another proof.

The plugin seeds its noise from a block header it gets from the gateway. Every
noise value and tile hash depends on that header.

## Decision

The verifier issues a new job every 5 seconds. Each job is a block header with
32 fresh random bytes in its previous-block-hash field. A proof passes only if:

- this verifier issued its job, with the target and version unchanged;
- the job is at most 15 seconds old when the proof arrives;
- its tile wasn't submitted before.

## Evidence

On an H100, a relay sent cheated copies of real proofs. Proofs held until
their job expired failed the on-time check (5 of 5). Replays failed the
new-tile check (4) or arrived late (2). Easier targets failed the issued-job
check (5 of 5).

## Consequences

- Work behind an accepted proof happened at most 15 seconds before it arrived.
- A slow verifier makes honest proofs late. The plugin waits for each verdict
  before it sends the next proof, so when the verifier falls behind, proofs
  queue. In one recorded run, 33 honest proofs failed the on-time check this way.
- The verifier keeps every job for a while, and every tile it accepted, in
  memory.

## Alternatives considered

- **A longer maximum age.** It would forgive a slow verifier but give a
  cheater more time to do work early.
- **A random value per proof, sent by the verifier on request.** The plugin
  asks for jobs on its own schedule, so this would need a larger change to
  upstream.
