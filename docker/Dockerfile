# Local test image: Python 3.12 with the upstream Rust library, pearl_mining,
# built from source. It has no published wheel, and building it in a
# container keeps Rust off the host. No GPU is needed for anything in here.
FROM python:3.12-slim-bookworm

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential ca-certificates curl git \
 && rm -rf /var/lib/apt/lists/*

RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH=/root/.cargo/bin:$PATH

# Check out upstream at the pinned commit, so the proof format and verify
# function can't move, and apply this repo's patches.
COPY patches /opt/pouw/patches
COPY scripts/fetch_upstream.sh /opt/pouw/scripts/fetch_upstream.sh
RUN bash /opt/pouw/scripts/fetch_upstream.sh /opt/upstream

RUN pip install --no-cache-dir uv maturin \
 && cd /opt/upstream/py-pearl-mining \
 && maturin build --release -o /wheels

COPY pyproject.toml /work/pyproject.toml
RUN uv pip install --system /wheels/*.whl numpy safetensors blake3 pytest

WORKDIR /work
ENV PYTHONPATH=/work/src
