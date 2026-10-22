#!/usr/bin/env bash
# Build the proof library twice, with and without the GIL patch, for a fair
# comparison on one pod. Runs on the pod after setup_pod.sh.
#
# Leaves /root/wheels/patched/*.whl and /root/wheels/unpatched/*.whl. Install
# one into the upstream venv with:
#   uv pip install --python /opt/upstream/.venv/bin/python --reinstall --no-deps /root/wheels/<which>/*.whl
set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
UPSTREAM=/opt/upstream
PATCH=$(ls /root/pouw/patches/0001-*.patch)
cd "$UPSTREAM"
rm -rf /root/wheels && mkdir -p /root/wheels
uv build --wheel --package py-pearl-mining -o /root/wheels/patched
git apply -R "$PATCH"
uv build --wheel --package py-pearl-mining -o /root/wheels/unpatched
git apply "$PATCH"
sha256sum /root/wheels/*/*.whl
