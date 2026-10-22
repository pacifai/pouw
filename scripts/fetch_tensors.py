"""Download selected tensors from a Hugging Face safetensors checkpoint.

A safetensors file starts with a JSON header giving each tensor's byte range,
so you can fetch a few tensors with HTTP range requests instead of the whole
checkpoint. The result is a small safetensors file with only those tensors.

Example:
    python scripts/fetch_tensors.py data/llama-layer0.safetensors \
        model.layers.0.self_attn.o_proj.weight model.layers.20.self_attn.q_proj.weight
"""

from __future__ import annotations

import argparse
import json
import struct
import urllib.request
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

REPO = "pearl-ai/Llama-3.1-8B-Instruct-pearl"
DTYPES = {"I8": np.int8, "BF16": np.uint16, "F8_E4M3": np.uint8}  # raw bytes for the float types


def _get(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req) as r:
        return r.read()


def fetch(repo: str, names: list[str]) -> dict[str, np.ndarray]:
    base = f"https://huggingface.co/{repo}/resolve/main/"
    with urllib.request.urlopen(base + "model.safetensors.index.json") as r:
        shard_of = json.load(r)["weight_map"]
    out = {}
    for name in names:
        url = base + shard_of[name]
        header_len = struct.unpack("<Q", _get(url, 0, 7))[0]
        info = json.loads(_get(url, 8, 8 + header_len - 1))[name]
        start, end = info["data_offsets"]
        data = _get(url, 8 + header_len + start, 8 + header_len + end - 1)
        out[name] = np.frombuffer(data, dtype=DTYPES[info["dtype"]]).reshape(info["shape"]).copy()
        print(f"{name}: {info['dtype']} {info['shape']}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out", type=Path)
    p.add_argument("names", nargs="+")
    p.add_argument("--repo", default=REPO)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(fetch(args.repo, args.names), str(args.out))


if __name__ == "__main__":
    main()
