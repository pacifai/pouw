# AGENTS.md: pouw

pouw is a verifier for proof of useful work on inference. It checks that a
GPU's matrix multiplications used a declared model's weights.

The layout, build, tests, and rules in [CONTRIBUTING.md](CONTRIBUTING.md)
apply to agents. The rules below apply to agents only.

- Ask before renting a GPU. It costs money. This includes running
  `runpod/record.py`.
- Keep `NEXT_STEPS.md` a plain backlog of the project's own work, most
  important first: a bold task name and a short description per item. Frame
  research items as open questions, with at most a one- or two-sentence
  starting idea.
- Write Markdown in a plain, technical register. State what a thing is and
  why it matters, briefly. Explain field terms such as GEMM, tile, and Merkle
  root where they first appear. Implementation detail belongs in the code and
  the decision records.
- Keep the README to what pouw is, why, installation and usage, then short
  reference tables.
- Keep code comments short and sparse. The code explains itself. Comments
  are for non-obvious reasons behind a choice, and docstrings for functions
  where they help. Reasons for a change go in its commit message, and project
  history stays out of the code.
- Keep `CONTRIBUTING.md` current when the layout, the build, or the tests
  change.
