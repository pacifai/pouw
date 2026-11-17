# Architecture decision records

Each architecture decision record (ADR) explains one design decision: the
problem, the choice, and its cost. Read the relevant record before changing a
check, the job format, or a claim about what the proofs show.

A record is kept as evidence. When a decision changes, a new record replaces
it, and the old one says so at the top.

| # | Decision | Why it matters |
|---|---|---|
| [0001](0001-replace-the-gateway.md) | Verifier in place of upstream's gateway | The proofs are upstream's own, checked by upstream's own verify function |
| [0002](0002-fresh-jobs-with-a-maximum-age.md) | Fresh jobs with a 15-second maximum age | It bounds when the work behind a proof happened |
| [0003](0003-check-every-weight.md) | Merkle root check on every declared weight matrix | It shows the GPU used the declared weights, every byte of them |
| [0004](0004-estimate-work-from-wins.md) | Work estimate from accepted proofs | It turns proofs into an amount of work that can be compared with a GPU's capacity |
| [0005](0005-gpu-id-in-the-job.md) | Self-reported GPU ID in each job | It credits work to one GPU, with a known gap until the ID is signed |
