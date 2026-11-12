# 0004: Work estimate from accepted proofs

**Status:** accepted

## Context

The GPU sends a proof only for winning tiles, a tiny share of all tiles. The
verifier needs the total amount of work, not the number of proofs.

## Decision

For each accepted proof, the verifier computes `p`, the chance that one tile
wins under the proof's job, from the target and the noise rank the way the
proof library does. The proof stands for `1 / p` tiles. A tile is 2 rows by 64
columns of output, each a dot product over the inner dimension `k`, so the
proof stands for `128 × k / p` multiply-adds. The verifier records this as the
proof's work.

## Evidence

- In one H100 run, 197 accepted proofs implied 5.16 × 10¹⁵ multiply-adds. A log
  of every multiplication the plugin ran (`patches/0003`) counted 5.29 × 10¹⁵,
  2.4% more. With about 200 proofs, the estimate's expected spread is about 7%.
- In an earlier run, 39 proofs implied 2.04 × 10¹⁵ multiply-adds, against
  1.93 × 10¹⁵ from the prompt tokens and the proven matrix shapes.

## Consequences

- The estimate is unbiased, and its spread shrinks with more proofs. A higher
  chance to win gives a tighter estimate but costs the GPU more proofs to build.
- Proofs lost before they reach the verifier, such as ones still queued when a
  run stops, make the estimate low.
