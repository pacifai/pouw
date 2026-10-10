"""The declared model's weights, and the check that a proof used them.

A proof commits to the full weight matrix B through a Merkle root: a
keyed hash over all of B's bytes. The key depends on the job, so the root
changes with every random value the verifier issues. To check that a proof
multiplied with a declared weight matrix, the verifier recomputes that
matrix's root under the same key and compares. This covers every byte of the
matrix, not only the rows the winning tile opened.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path

import numpy as np
from blake3 import blake3
from pearl_mining import (
    MerkleTree,
    MiningConfiguration,
    MMAType,
    PeriodicPattern,
    pad_to_chunk_boundary,
)
from safetensors import safe_open

# vLLM stores some projections as one fused tensor: rows of the parts stacked in
# this order. The GEMM, and so the proof, sees the fused tensor.
FUSED = {
    "self_attn.qkv_proj": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
    "mlp.gate_up_proj": ["mlp.gate_proj", "mlp.up_proj"],
}
_LAYER = re.compile(r"^(model\.layers\.\d+)\.(.+)\.weight$")


def config_from_indices(k: int, rank: int, rows: list[int], cols: list[int]) -> MiningConfiguration:
    """Rebuild the proof configuration a proof's tile indices imply (as upstream does)."""
    r0, c0 = min(rows), min(cols)
    return MiningConfiguration(
        common_dim=k,
        rank=rank,
        mma_type=MMAType.Int7xInt7ToInt32,
        rows_pattern=PeriodicPattern.from_list([r - r0 for r in rows]),
        cols_pattern=PeriodicPattern.from_list([c - c0 for c in cols]),
        moe=None,
    )


def job_key(header_bytes: bytes, config: MiningConfiguration) -> bytes:
    """The key the proof format uses for every Merkle tree in a job."""
    return blake3(header_bytes + bytes(config.to_bytes())).digest()


def merkle_root(matrix: np.ndarray, key: bytes) -> bytes:
    """The proof format's Merkle root of an int8 matrix, rows laid end to end."""
    data = pad_to_chunk_boundary(np.ascontiguousarray(matrix).tobytes())
    return bytes(MerkleTree(data=bytes(data), key=key).root)


def fuse_like_vllm(tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Replace each complete group of int8 parts with the fused tensor vLLM builds."""
    out = dict(tensors)
    layers = {m.group(1) for name in tensors if (m := _LAYER.match(name))}
    for layer in sorted(layers):
        for fused, parts in FUSED.items():
            names = [f"{layer}.{p}.weight" for p in parts]
            if all(n in out for n in names):
                out[f"{layer}.{fused}.weight"] = np.concatenate([out.pop(n) for n in names], axis=0)
    return out


class DeclaredWeights:
    """Int8 weight matrices, each stored as vLLM holds it: (output features, input features)."""

    def __init__(self, tensors: dict[str, np.ndarray]):
        self.tensors = {
            name: np.ascontiguousarray(t) for name, t in tensors.items() if t.dtype == np.int8 and t.ndim == 2
        }
        self._roots: dict[tuple[bytes, int, int], dict[bytes, str]] = {}
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

    def match(self, key: bytes, n: int, k: int, root: bytes) -> str | None:
        """Name of the declared (n, k) matrix whose root under `key` is `root`, if any.

        Roots are computed once per job and shape, then reused for every proof
        in that job.
        """
        cache_key = (key, n, k)
        with self._lock:
            roots = self._roots.get(cache_key)
            if roots is None:
                roots = {
                    merkle_root(t, key): name for name, t in self.tensors.items() if t.shape == (n, k)
                }
                self._roots[cache_key] = roots
                self._forget_old_jobs(keep=cache_key)
        return roots.get(root)

    def _forget_old_jobs(self, keep: tuple[bytes, int, int], limit: int = 64) -> None:
        while len(self._roots) > limit:
            oldest = next(iter(self._roots))
            if oldest == keep:
                break
            del self._roots[oldest]
