# Decision records

Each record explains one design decision: the problem, what was chosen, and
what it costs. Read the relevant record before you change a check, the job
format, or a claim about what the proofs show.

A record is kept as evidence. When a decision changes, a new record replaces
it, and the old one says so at the top.

| # | Decision | Why it matters |
|---|---|---|
| [0001](0001-replace-the-gateway.md) | The verifier replaces upstream's gateway and leaves the proof format alone | The proofs are upstream's own, checked by upstream's own verify function |
| [0002](0002-fresh-jobs-with-a-maximum-age.md) | Every job carries a fresh random value and expires after 15 seconds | It bounds when the work behind a proof happened |
| [0003](0003-check-every-weight.md) | The verifier recomputes the Merkle root of every declared weight matrix for each job | It shows the GPU used the declared weights, every byte of them |
| [0004](0004-estimate-work-from-wins.md) | Proven work is estimated from the accepted proofs and their chance to win | It turns proofs into an amount of work that can be compared with a GPU's capacity |
| [0005](0005-gpu-id-in-the-job.md) | Each GPU's ID is hashed into its own jobs, and the ID is self-reported | It credits work to one GPU, with a known gap until the ID is signed |
