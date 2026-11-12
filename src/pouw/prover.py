"""The CPU prover: an honest prover that runs without a GPU.

It speaks the plugin's JSON-RPC and packages proofs as upstream's create_proof
does. Instead of hashing every tile of a GPU multiply, it tries tiles one at a
time until upstream's verify says one wins, so it's slow but its proofs are
real. The tests use it, and so can a local run of the verifier.
"""

from __future__ import annotations

import base64
import json
import random
import time
from pathlib import Path

import numpy as np
from pearl_mining import (
    IncompleteBlockHeader,
    MatrixMerkleProof,
    MerkleTree,
    PeriodicPattern,
    PlainProof,
    verify_plain_proof_for_cert_version,
)

from .proof import COLS_PATTERN, NOISE_RANK, ROWS_PATTERN, gpu_config, job_key
from .protocol import RpcClient
from .weights import DeclaredWeights

INT7_MIN, INT7_MAX = -63, 63
MAX_TILES = 100_000


def tile_starts(pattern: list[int], dim: int) -> list[int]:
    """Offsets along one dimension where a tile with this pattern may start."""
    periodic = PeriodicPattern.from_list(pattern)
    return [i for i in range(dim) if periodic.offset_is_valid(i) and i + max(pattern) < dim]


def random_activations(m: int, k: int, seed: int | None = None) -> np.ndarray:
    return np.random.default_rng(seed).integers(INT7_MIN, INT7_MAX + 1, size=(m, k), dtype=np.int8)


def prove(
    header_bytes: bytes,
    cert_version: int,
    A: np.ndarray,
    Bt: np.ndarray,
    max_tiles: int = MAX_TILES,
    want_win: bool = True,
    seed: int | None = None,
) -> tuple[PlainProof | None, int]:
    """Tries tiles of A·B until one wins, or with want_win=False until one loses.

    Returns the proof, or None, and the number of tiles tried.
    """
    m, k = A.shape
    n, _ = Bt.shape
    key = job_key(header_bytes, gpu_config(k))
    tree_a = MerkleTree.from_buffer(np.ascontiguousarray(A), key)
    tree_b = MerkleTree.from_buffer(np.ascontiguousarray(Bt), key)
    header = IncompleteBlockHeader.from_bytes(header_bytes)

    tiles = [(r, c) for r in tile_starts(ROWS_PATTERN, m) for c in tile_starts(COLS_PATTERN, n)]
    random.Random(seed).shuffle(tiles)
    for tried, (r, c) in enumerate(tiles[:max_tiles], start=1):
        rows = [r + d for d in ROWS_PATTERN]
        cols = [c + d for d in COLS_PATTERN]
        proof = _package(A.shape, Bt.shape, rows, cols, tree_a, tree_b)
        ok, _ = verify_plain_proof_for_cert_version(cert_version, header, proof)
        if ok == want_win:
            return proof, tried
    return None, min(len(tiles), max_tiles)


def _package(
    a_shape: tuple[int, int], bt_shape: tuple[int, int], rows: list[int], cols: list[int], tree_a, tree_b
) -> PlainProof:
    (m, k), (n, _) = a_shape, bt_shape
    a = MatrixMerkleProof(tree_a.get_multileaf_proof(MerkleTree.compute_leaf_indices_from_rows(rows, (m, k))), rows)
    b = MatrixMerkleProof(tree_b.get_multileaf_proof(MerkleTree.compute_leaf_indices_from_rows(cols, (n, k))), cols)
    return PlainProof(m, n, k, NOISE_RANK, a, b, None)


def run(
    weights: list[Path],
    matrix: str,
    proofs: int,
    m: int,
    gpu_id: str | None = None,
    socket_path: str | None = None,
    tcp: tuple[str, int] | None = None,
) -> None:
    """Asks the verifier for jobs, and sends it one proof per job, with random activations."""
    Bt = DeclaredWeights.from_safetensors(weights).tensors[matrix]
    client = RpcClient(socket_path, tcp)
    try:
        for i in range(proofs):
            job = client.call("getMiningInfo", {"gpu_id": gpu_id} if gpu_id else None)
            started = time.time()
            proof, tried = prove(
                base64.b64decode(job["incomplete_header_bytes"]),
                job["cert_version"],
                random_activations(m, Bt.shape[1]),
                Bt,
            )
            if proof is None:
                print(json.dumps({"proof": i, "result": "no winning tile", "tiles_tried": tried}))
                continue
            result = client.call("submitPlainProof", {"plain_proof": proof.to_base64(), "mining_job": job})
            print(json.dumps({"proof": i, "tiles_tried": tried, "prove_s": round(time.time() - started, 2), **result}))
    finally:
        client.close()
