"""Copy a checkpoint with one weight changed in every int8 weight matrix.

Runs on the pod, with the upstream venv's Python (it needs PyTorch to read
the checkpoint's bf16 and fp8 tensors). Each 2-D int8 tensor gets its [0, 0]
entry moved by one, staying inside the int7 range. Served by vLLM, this model
does the same work as the original, but every proof's weight matrix differs
from the declared one, so the verifier should reject each proof at the
weight check.

Example:
    /opt/upstream/.venv/bin/python scripts/gpu/make_substituted_model.py /root/model /root/model-sub
"""

import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    changed = 0
    for f in sorted(src.iterdir()):
        if f.suffix != ".safetensors":
            if f.is_file():
                shutil.copy2(f, dst / f.name)
            continue
        with safe_open(str(f), framework="pt") as h:
            metadata = h.metadata()
        tensors = load_file(str(f))
        for name, t in tensors.items():
            if t.dtype == torch.int8 and t.ndim == 2:
                v = int(t[0, 0])
                t[0, 0] = v + 1 if v < 63 else v - 1
                changed += 1
        save_file(tensors, str(dst / f.name), metadata=metadata)
    print(f"changed one weight in each of {changed} int8 matrices")


if __name__ == "__main__":
    main()
