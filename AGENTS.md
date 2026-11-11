# AGENTS.md: pouw

pouw is a verifier for proof of useful work on inference. It checks that a
GPU's matrix multiplications used a declared model's weights.

Read [CONTRIBUTING.md](CONTRIBUTING.md) first. Its layout, build and test steps,
and rules all apply to you. The rules below are for agents only.

- **Ask before you spend.** Renting a GPU costs money. Ask before you run
  `runpod/record.py` or rent a pod any other way.
- **Keep `NEXT_STEPS.md` a plain backlog** of the project's own work, most
  important first, each item a bold task name and a short description. Frame
  research items as open questions, with at most a one- or two-sentence starting
  idea.
- **Write every Markdown file for a newcomer.** Say what a thing is and why it
  matters in plain words, and keep it short. Use the field's terms, such as
  GEMM, tile, and Merkle root, and explain each one where it first appears.
  Implementation detail belongs in the code and the decision records.
- **Keep the README light:** what pouw is, why, a quickstart, then short
  reference tables.
- **Keep comments to what the code does.** Explain reasons in commit messages,
  and keep project history out of the code.
- **Keep `CONTRIBUTING.md` current** when you change the layout, the build, or
  the tests.
