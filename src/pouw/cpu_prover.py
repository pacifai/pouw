"""A CPU stand-in for the GPU prover, for testing the verifier without a GPU.

It speaks the plugin's JSON-RPC and packages proofs the way upstream's
create_proof does. Instead of hashing every tile during a GPU multiply, it
tries tiles one at a time and asks the library's verify function whether each
one wins. That is slow, but its proofs are real.

Its options make it cheat, so tests can show each check catching it.
"""

from __future__ import annotations

import argparse
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
from safetensors.numpy import load_file

from .jobs import COLS_PATTERN, NOISE_RANK, ROWS_PATTERN, gpu_proof_config
from .rpc import RpcClient
from .weights import fuse_like_vllm, job_key


def tile_starts(pattern: list[int], dim: int) -> list[int]:
    """Valid tile offsets along one dimension, using the library's own validity check."""
    pp = PeriodicPattern.from_list(pattern)
    return [i for i in range(dim) if pp.offset_is_valid(i) and i + max(pattern) < dim]


def build_proof(
    A: np.ndarray, Bt: np.ndarray, rows: list[int], cols: list[int], tree_a: MerkleTree, tree_b: MerkleTree
) -> PlainProof:
    """Package one tile as a PlainProof, as upstream's create_proof does."""
    m, k = A.shape
    n, _ = Bt.shape
    a = MatrixMerkleProof(tree_a.get_multileaf_proof(MerkleTree.compute_leaf_indices_from_rows(rows, (m, k))), rows)
    b = MatrixMerkleProof(tree_b.get_multileaf_proof(MerkleTree.compute_leaf_indices_from_rows(cols, (n, k))), cols)
    return PlainProof(m, n, k, NOISE_RANK, a, b, None)


def prove(
    header_bytes: bytes,
    cert_version: int,
    A: np.ndarray,
    Bt: np.ndarray,
    max_tiles: int = 100_000,
    want_win: bool = True,
    seed: int | None = None,
) -> tuple[PlainProof | None, int]:
    """Try tiles of A·B until one wins (or, with want_win=False, until one loses).

    Returns the proof and the number of tiles tried.
    """
    m, k = A.shape
    n, _ = Bt.shape
    key = job_key(header_bytes, gpu_proof_config(k))
    tree_a = MerkleTree.from_buffer(np.ascontiguousarray(A), key)
    tree_b = MerkleTree.from_buffer(np.ascontiguousarray(Bt), key)
    header = IncompleteBlockHeader.from_bytes(header_bytes)

    tiles = [(r, c) for r in tile_starts(ROWS_PATTERN, m) for c in tile_starts(COLS_PATTERN, n)]
    random.Random(seed).shuffle(tiles)
    for tried, (r, c) in enumerate(tiles[:max_tiles], start=1):
        rows = [r + d for d in ROWS_PATTERN]
        cols = [c + d for d in COLS_PATTERN]
        proof = build_proof(A, Bt, rows, cols, tree_a, tree_b)
        ok, _ = verify_plain_proof_for_cert_version(cert_version, header, proof)
        if ok == want_win:
            return proof, tried
    return None, min(len(tiles), max_tiles)


def random_activations(m: int, k: int, seed: int | None = None) -> np.ndarray:
    """Int7 activations in the library's range, standing in for a real batch."""
    return np.random.default_rng(seed).integers(-63, 64, size=(m, k), dtype=np.int8)


def main() -> None:
    p = argparse.ArgumentParser(description="CPU stand-in for the GPU prover.")
    p.add_argument("--weights", type=Path, required=True, help="safetensors file holding the weight matrix")
    p.add_argument("--matrix", required=True, help="tensor name, after vLLM-style fusing")
    p.add_argument("--socket", default="/tmp/pearlgw.sock", help="Unix socket path (upstream's default)")
    p.add_argument("--tcp", metavar="HOST:PORT")
    p.add_argument("--m", type=int, default=128, help="rows of activations (tokens in the batch)")
    p.add_argument("--proofs", type=int, default=3, help="winning tiles to submit")
    p.add_argument("--tamper-weights", action="store_true", help="change one weight before proving")
    p.add_argument("--replay", action="store_true", help="submit each proof twice")
    p.add_argument("--delay", type=float, default=0.0, help="seconds to wait before submitting")
    p.add_argument(
        "--gpu-id",
        help="GPU ID to send in getMiningInfo; a verifier run with --attribute-gpus issues this ID its own jobs",
    )
    args = p.parse_args()

    Bt = fuse_like_vllm(load_file(str(args.weights)))[args.matrix].copy()
    if args.tamper_weights:
        Bt[0, 0] = Bt[0, 0] + 1 if Bt[0, 0] < 63 else Bt[0, 0] - 1
    tcp = None
    if args.tcp:
        host, port = args.tcp.rsplit(":", 1)
        tcp = (host, int(port))

    client = RpcClient(args.socket, tcp)
    for i in range(args.proofs):
        job = client.call("getMiningInfo", {"gpu_id": args.gpu_id} if args.gpu_id else None)
        header_bytes = base64.b64decode(job["incomplete_header_bytes"])
        A = random_activations(args.m, Bt.shape[1])
        t0 = time.time()
        proof, tried = prove(header_bytes, job["cert_version"], A, Bt)
        if proof is None:
            print(json.dumps({"proof": i, "result": "no winning tile", "tiles_tried": tried}))
            continue
        time.sleep(args.delay)
        for _ in range(2 if args.replay else 1):
            result = client.call("submitPlainProof", {"plain_proof": proof.to_base64(), "mining_job": job})
            print(json.dumps({"proof": i, "tiles_tried": tried, "mine_s": round(time.time() - t0, 2), **result}))
    client.close()


if __name__ == "__main__":
    main()
