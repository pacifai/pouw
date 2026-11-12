"""Download single tensors from a safetensors checkpoint on Hugging Face, with HTTP range requests."""

from __future__ import annotations

import json
import struct
import urllib.request
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

MODEL_REPO = "pearl-ai/Llama-3.1-8B-Instruct-pearl"
HEADER_LEN_BYTES = 8
# NumPy has no bf16 or fp8, so those keep their raw bytes.
DTYPES = {"I8": np.int8, "BF16": np.uint16, "F8_E4M3": np.uint8}


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
        header_len = struct.unpack("<Q", _get(url, 0, HEADER_LEN_BYTES - 1))[0]
        data_start = HEADER_LEN_BYTES + header_len
        info = json.loads(_get(url, HEADER_LEN_BYTES, data_start - 1))[name]
        start, end = info["data_offsets"]
        data = _get(url, data_start + start, data_start + end - 1)
        out[name] = np.frombuffer(data, dtype=DTYPES[info["dtype"]]).reshape(info["shape"]).copy()
        print(f"{name}: {info['dtype']} {info['shape']}")
    return out


def save(out: Path, names: list[str], repo: str = MODEL_REPO) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(fetch(repo, names), str(out))
