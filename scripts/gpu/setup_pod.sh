#!/usr/bin/env bash
# Install the upstream vLLM plugin on a fresh H100 pod.
#
# Runs on the pod, not your laptop. It installs uv and Rust, clones the
# upstream repo at the same commit docker/Dockerfile pins, applies this repo's
# patches/, and builds the whole workspace: the CUDA kernels, the proof
# library, and vLLM. The kernel build takes a while. You can rerun it safely.
#
# Expects an nvidia/cuda:13.0.x-devel image on a host whose driver supports
# CUDA 13, because upstream builds against PyTorch for CUDA 13.0.
set -euo pipefail

UPSTREAM_COMMIT="${UPSTREAM_COMMIT:-2f8b770cac8f8b74be05c1fc51c2baeb76ce0701}"
UPSTREAM_DIR=/opt/upstream
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../patches" && pwd)"

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
if ! command -v cargo >/dev/null && [ ! -x "$HOME/.cargo/bin/cargo" ]; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"

if [ ! -d "$UPSTREAM_DIR/.git" ]; then
  git init -q "$UPSTREAM_DIR"
  git -C "$UPSTREAM_DIR" remote add origin https://github.com/pearl-research-labs/pearl
fi
git -C "$UPSTREAM_DIR" fetch -q --depth 1 origin "$UPSTREAM_COMMIT"
# Discard changes from an earlier run, so the patches apply to clean files.
git -C "$UPSTREAM_DIR" checkout -q --force FETCH_HEAD
git -C "$UPSTREAM_DIR" apply "$PATCH_DIR"/*.patch

cd "$UPSTREAM_DIR"
uv python install 3.12
# uv doesn't notice changed Rust sources, so rebuild the proof library every
# time to pick up the patches.
UV_TORCH_BACKEND=cu130 uv sync --all-packages --all-groups --reinstall-package py-pearl-mining

uv run python -c "import torch, vllm, pearl_gemm, pearl_mining; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(0)); print('vllm', vllm.__version__)"
