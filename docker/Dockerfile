# Local test image: Python 3.12 with the upstream Rust library, pearl_mining,
# built from source. It has no published wheel, and building it in a
# container keeps Rust off the host. No GPU is needed for anything in here.
FROM python:3.12-slim-bookworm

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential ca-certificates curl git \
 && rm -rf /var/lib/apt/lists/*

RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH=/root/.cargo/bin:$PATH

# Pin the upstream code so the proof format and verify function can't move.
ARG UPSTREAM_COMMIT=2f8b770cac8f8b74be05c1fc51c2baeb76ce0701
RUN git init /opt/upstream \
 && cd /opt/upstream \
 && git remote add origin https://github.com/pearl-research-labs/pearl \
 && git fetch --depth 1 origin "$UPSTREAM_COMMIT" \
 && git checkout FETCH_HEAD

# Apply this repo's changes to the upstream code. patches/README.md says what
# each one does.
COPY patches /opt/patches
RUN cd /opt/upstream && git apply /opt/patches/*.patch

RUN pip install --no-cache-dir uv maturin \
 && cd /opt/upstream/py-pearl-mining \
 && maturin build --release -o /wheels

COPY pyproject.toml /work/pyproject.toml
RUN uv pip install --system /wheels/*.whl numpy safetensors blake3 pytest

WORKDIR /work
ENV PYTHONPATH=/work/src
