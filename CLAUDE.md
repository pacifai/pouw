# CLAUDE.md: pouw

<!-- living-files: CLAUDE.md NEXT_STEPS.md README.md -->

A verifier for proof of useful work (PoUW) on inference. It checks that a GPU's inference
matrix multiplies ran on a declared model's weights. `README.md` explains how
it works and what runs today. `NEXT_STEPS.md` holds the plan and the open
questions.

## Working here

- **Name Pearl only as the source.** The vLLM plugin, GPU kernels, and proof
  library come from Pearl Research Labs' open-source repo. Credit them in
  `README.md` and `upstream/README.md`, but don't use Pearl's name in code,
  identifiers, or descriptions. Call their code *upstream*, or say what the
  part is: the vLLM plugin, the gateway, the verify function.
- **Say prover, not miner.** The GPU running inference is the prover. Keep
  mining terms only in upstream names you can't change, such as
  `getMiningInfo` and `pearl_mining`.
- **The upstream code is pinned.** `docker/Dockerfile` builds the upstream
  Rust library, `pearl_mining`, from one commit. The proof format and verify
  function come from that commit. To move to a newer one, change the pin on
  purpose and rerun the tests.
- **`upstream/` holds unmodified upstream files.** They're for comparison
  only. Don't edit them. When the pin moves, recopy them from the new commit.
- **Run tests in Docker.** `pearl_mining` isn't published as a package, and
  the image keeps Rust off the host. Run
  `docker run --rm -v "$PWD":/work pouw-local python -m pytest -q`.
- **Change upstream code only through `patches/`.** This repo replaces
  upstream's gateway: the two JSON-RPC calls `getMiningInfo` and
  `submitPlainProof`. Any other change to upstream code is a `git diff`
  against the pinned commit, saved in `patches/` with a numbered name and
  explained in `patches/README.md`. `docker/Dockerfile` and
  `scripts/gpu/setup_pod.sh` apply every patch after checkout. Keep patches
  small, and don't change the proof format or the verify function. When the
  pin moves, check that each patch still applies, and rerun the tests.
- **GPU work costs money.** The upstream kernels need a Hopper GPU (H100 class)
  on a host whose driver supports CUDA 13. Ask before renting one. Arm a hard
  time cap in the pod's boot command, keep the pod ID in `.gpu_state` (ignored
  by git), and delete the pod when the run ends. The first run of each setup
  compiles kernels, so don't compare timings from it.
- **Use plain language.** Use the field's terms (GEMM, tile, KV cache,
  Merkle root), and don't coin new ones. Explain a mechanism in words before
  naming it.
- **Keep data and runs out of git.** Fetched tensors go in `data/` and run
  logs in `runs/`. Both are ignored by git and can be regenerated.
