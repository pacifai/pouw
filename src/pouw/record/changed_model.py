"""A copy of a checkpoint with one weight changed in every int8 matrix.

Served by vLLM, it does the same work as the original, but every proof's
weight matrix differs from the declared one. It needs PyTorch to read the
checkpoint's bf16 and fp8 tensors, so run it with the upstream venv's Python.
"""

import shutil
from pathlib import Path

from safetensors import safe_open

INT7_MAX = 63


def write(src: Path, dst: Path) -> int:
    """Writes the copy to dst, and returns the number of matrices changed."""
    import torch
    from safetensors.torch import load_file, save_file

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
                t[0, 0] = v + 1 if v < INT7_MAX else v - 1
                changed += 1
        save_file(tensors, str(dst / f.name), metadata=metadata)
    return changed
