"""The declared model's weights, and the check that a proof used one of them.

A proof commits to its whole weight matrix through a Merkle root keyed by the
job, so a matching root covers every byte of the matrix.
"""

from __future__ import annotations

import re
import threading
from concurrent.futures import Future
from pathlib import Path

import numpy as np
from pearl_mining import MerkleTree
from safetensors import safe_open

# Projections vLLM fuses into one tensor, with the parts' rows stacked in this order.
FUSED = {
    "self_attn.qkv_proj": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    "mlp.gate_up_proj": ["mlp.gate_proj", "mlp.up_proj"],
}
# Room for many GPUs, each with a few live jobs and matrix shapes.
CACHED_JOB_SHAPES = 1024
_LAYER = re.compile(r"^(model\.layers\.\d+)\.(.+)\.weight$")


def merkle_root(matrix: np.ndarray, key: bytes) -> bytes:
    """The proof format's Merkle root of an int8 matrix, rows laid end to end."""
    return bytes(MerkleTree.from_buffer(np.ascontiguousarray(matrix), key).root)


def fuse_like_vllm(tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Replace each complete group of int8 parts with the fused tensor vLLM builds, in place."""
    layers = {m.group(1) for name in tensors if (m := _LAYER.match(name))}
    for layer in sorted(layers):
        for fused, parts in FUSED.items():
            names = [f"{layer}.{p}.weight" for p in parts]
            if all(n in tensors for n in names):
                tensors[f"{layer}.{fused}.weight"] = np.concatenate([tensors.pop(n) for n in names], axis=0)
    return tensors


class DeclaredWeights:
    """Int8 weight matrices, each stored as vLLM holds it: (output features, input features)."""

    def __init__(self, tensors: dict[str, np.ndarray]):
        self.tensors = {
            name: np.ascontiguousarray(t) for name, t in tensors.items() if t.dtype == np.int8 and t.ndim == 2
        }
        self._shapes = {t.shape for t in self.tensors.values()}
        self._roots: dict[tuple[bytes, int, int], Future[dict[bytes, str]]] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_safetensors(cls, paths: list[Path], fuse: bool = True) -> DeclaredWeights:
        # Read only the int8 tensors: NumPy can't hold the checkpoint's bf16 and fp8 ones.
        tensors: dict[str, np.ndarray] = {}
        for path in paths:
            with safe_open(str(path), framework="np") as f:
                for name in f.keys():
                    if f.get_slice(name).get_dtype() == "I8":
                        tensors[name] = f.get_tensor(name)
        return cls(fuse_like_vllm(tensors) if fuse else tensors)

    def has_shape(self, n: int, k: int) -> bool:
        return (n, k) in self._shapes

    def match(self, key: bytes, n: int, k: int, root: bytes) -> str | None:
        """Name of the declared (n, k) matrix whose root under key is root, if any."""
        cache_key = (key, n, k)
        with self._lock:
            roots = self._roots.get(cache_key)
            first = roots is None
            if first:
                roots = self._roots[cache_key] = Future()
                self._forget_old_jobs()
        if first:
            try:
                roots.set_result({merkle_root(t, key): name for name, t in self.tensors.items() if t.shape == (n, k)})
            except BaseException as e:
                with self._lock:
                    if self._roots.get(cache_key) is roots:
                        del self._roots[cache_key]
                roots.set_exception(e)
                raise
        return roots.result().get(root)

    def _forget_old_jobs(self) -> None:
        while len(self._roots) > CACHED_JOB_SHAPES:
            del self._roots[next(iter(self._roots))]
